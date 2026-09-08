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

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(settings.dtype, torch.float16)

    logger.info("Loading OmniVoice model (%s) …", settings.model_id)
    _model = OmniVoice.from_pretrained(
        settings.model_id,
        device_map=settings.device,
        dtype=dtype,
    )

    # Resolution order: explicit override > user-generated clone >
    # bundled default clone > instruct-mode fallback.
    clone_path = voice_clone_path or settings.voice_clone_path
    if not clone_path.exists() and settings.bundled_voice_clone_path.exists():
        clone_path = settings.bundled_voice_clone_path

    if clone_path.exists():
        logger.info("Loading voice clone from %s …", clone_path)
        with open(clone_path, "rb") as fh:
            _voice_clone_prompt = pickle.load(fh)
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
