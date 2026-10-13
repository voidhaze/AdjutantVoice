"""
Core TTS engine — model loading and synthesis.

This module is the single source of truth for OmniVoice inference.
Nothing here depends on FastAPI, FastMCP, or any transport layer.
"""

from __future__ import annotations

import io
import logging
import pickle
import threading
import types
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
from omnivoice import OmniVoice

from adjutantvoice.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_model: Optional[OmniVoice] = None
_voice_clone_prompt = None
_loaded: bool = False
_inference_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class _CPUUnpickler(pickle.Unpickler):
    """Unpickler that maps tensor storages to CPU.

    Voice-clone pickles created on a CUDA machine embed CUDA storages, which
    fail to load on CPU / XPU / MPS hosts. OmniVoice moves the prompt tokens
    to the model's device during generation, so CPU is always safe here.
    """

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def _log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x - torch.logsumexp(x, dim=dim, keepdim=True)


def _xpu_predict_tokens_with_scoring(self, c_logits, u_logits, gen_config):
    """Copy of ``OmniVoice._predict_tokens_with_scoring`` without ``log_softmax``.

    On Intel XPU (torch 2.14 / Arc 140V), ``softmax`` / ``log_softmax`` over a
    last dim of 1025 — OmniVoice's audio vocab size — return wrong values,
    which makes every generation come out as noise. ``logsumexp`` is correct.
    """
    from omnivoice.models.omnivoice import _filter_top_k, _gumbel_sample

    if gen_config.guidance_scale != 0:
        c_log_probs = _log_softmax(c_logits)
        u_log_probs = _log_softmax(u_logits)
        log_probs = _log_softmax(
            c_log_probs + gen_config.guidance_scale * (c_log_probs - u_log_probs)
        )
    else:
        log_probs = _log_softmax(c_logits)

    log_probs[..., self.config.audio_mask_id] = -float("inf")

    if gen_config.class_temperature > 0.0:
        filtered_probs = _filter_top_k(log_probs, ratio=0.1)
        pred_tokens = _gumbel_sample(
            filtered_probs, gen_config.class_temperature
        ).argmax(dim=-1)
    else:
        pred_tokens = log_probs.argmax(dim=-1)

    confidence_scores = log_probs.max(dim=-1)[0]

    return pred_tokens, confidence_scores


def _load_model() -> OmniVoice:
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(settings.dtype, torch.float16)

    logger.info("Loading OmniVoice model (%s) …", settings.model_id)
    model = OmniVoice.from_pretrained(
        settings.model_id,
        device_map=settings.device,
        dtype=dtype,
    )
    if settings.device.startswith("xpu"):
        model._predict_tokens_with_scoring = types.MethodType(
            _xpu_predict_tokens_with_scoring, model
        )
    return model


def load(
    voice_clone_path: Optional[Path] = None,
) -> None:
    """Load the OmniVoice model and voice-clone prompt into memory.

    Safe to call multiple times — subsequent calls are no-ops if already loaded.

    Args:
        voice_clone_path: Override the default voice-clone pickle path.
    """
    global _model, _voice_clone_prompt, _loaded

    if _loaded:
        return  # already loaded

    _model = _load_model()

    # Resolution order: explicit override > user-generated clone >
    # bundled default clone > instruct-mode fallback.
    clone_path = voice_clone_path or settings.voice_clone_path
    if not clone_path.exists() and settings.bundled_voice_clone_path.exists():
        clone_path = settings.bundled_voice_clone_path

    if clone_path.exists():
        logger.info("Loading voice clone from %s …", clone_path)
        with open(clone_path, "rb") as fh:
            _voice_clone_prompt = _CPUUnpickler(fh).load()
    else:
        logger.info(
            "No voice clone found at %s — falling back to the default '%s' "
            "OmniVoice voice. Run `av voice create-clone` to generate one.",
            clone_path,
            settings.default_voice_instruct,
        )
        _voice_clone_prompt = None

    _loaded = True
    logger.info("AdjutantVoice: ready.")


def unload() -> None:
    """Release model references (called on server shutdown)."""
    global _model, _voice_clone_prompt, _loaded
    _model = None
    _voice_clone_prompt = None
    _loaded = False


def is_loaded() -> bool:
    """Return True if the model is loaded and ready."""
    return _loaded


def using_voice_clone() -> bool:
    """Return True if a voice-clone prompt is active (vs. the fallback voice)."""
    return _voice_clone_prompt is not None


def get_model() -> OmniVoice:
    """Return the loaded OmniVoice model, loading it first if necessary.

    Lets callers that need direct model access (e.g.
    :func:`adjutantvoice.voice.create_voice_clone`) reuse the same
    singleton-loading logic as the server/CLI synthesis path instead of
    duplicating the dtype resolution and ``from_pretrained`` call — and, if
    a model is already loaded in this process, avoids loading a second copy.
    """
    if not _loaded:
        load()
    return _model


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synthesize(text: str) -> bytes:
    """Synthesize *text* and return raw MP3 bytes.

    Args:
        text: The text to convert to speech.

    Returns:
        MP3 audio data as bytes.

    Raises:
        RuntimeError: If the model has not been loaded yet.
        ValueError: If *text* is blank.
    """
    if not is_loaded():
        raise RuntimeError("TTS model is not loaded. Call tts.load() first.")

    text = text.strip()
    if not text:
        raise ValueError("text must not be empty")

    with _inference_lock:
        if _voice_clone_prompt is not None:
            audio = _model.generate(text=text, voice_clone_prompt=_voice_clone_prompt)
        else:
            audio = _model.generate(text=text, instruct=settings.default_voice_instruct)

    buf = io.BytesIO()
    sf.write(buf, audio[0], settings.sample_rate, format="MP3")
    buf.seek(0)
    return buf.getvalue()


def synthesize_to_buffer(text: str) -> io.BytesIO:
    """Like :func:`synthesize` but returns a seeked ``BytesIO`` buffer."""
    data = synthesize(text)
    buf = io.BytesIO(data)
    buf.seek(0)
    return buf


def synthesize_with_duration(text: str) -> tuple[bytes, float]:
    """Like :func:`synthesize`, but also returns the exact audio duration.

    Duration is computed from the raw sample count and configured sample
    rate *before* MP3 encoding, so — unlike estimating it from the encoded
    byte size afterwards — it isn't affected by bitrate/compression and is
    exact rather than approximate.

    Returns:
        A ``(mp3_bytes, duration_seconds)`` tuple.

    Raises:
        RuntimeError: If the model has not been loaded yet.
        ValueError: If *text* is blank.
    """
    if not is_loaded():
        raise RuntimeError("TTS model is not loaded. Call tts.load() first.")

    text = text.strip()
    if not text:
        raise ValueError("text must not be empty")

    with _inference_lock:
        if _voice_clone_prompt is not None:
            audio = _model.generate(text=text, voice_clone_prompt=_voice_clone_prompt)
        else:
            audio = _model.generate(text=text, instruct=settings.default_voice_instruct)

    samples = audio[0]
    duration_s = round(len(samples) / settings.sample_rate, 1)

    buf = io.BytesIO()
    sf.write(buf, samples, settings.sample_rate, format="MP3")
    buf.seek(0)
    return buf.getvalue(), duration_s
