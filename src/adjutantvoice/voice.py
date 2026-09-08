"""
Voice clone utilities — create and manage voice-clone prompts.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

from adjutantvoice import tts
from adjutantvoice.config import settings

logger = logging.getLogger(__name__)


def create_voice_clone(
    ref_audio: Path,
    output_path: Path | None = None,
) -> Path:
    """Generate and save a voice-clone prompt pickle from a reference audio file.

    Args:
        ref_audio: Path to the reference MP3/WAV file. 
        output_path: Where to write the ``.pkl`` file. Defaults to
            ``settings.voice_clone_path``.

    Returns:
        Absolute path to the saved pickle file.
    """
    output_path = output_path or settings.voice_clone_path

    # Reuse the same singleton-loading path as the server/CLI synthesis
    # flow (tts.load) instead of duplicating the dtype resolution and
    # `OmniVoice.from_pretrained` call here. If a model is already loaded
    # in this process, this returns it directly rather than loading a
    # second copy.
    model = tts.get_model()

    logger.info("Creating voice clone from %s …", ref_audio)
    voice_clone_prompt = model.create_voice_clone_prompt(ref_audio=str(ref_audio))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        pickle.dump(voice_clone_prompt, fh)

    logger.info("Voice clone saved to %s", output_path)
    return output_path.resolve()
