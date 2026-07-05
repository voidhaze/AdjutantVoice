"""
Voice clone utilities — create and manage voice-clone prompts.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from omnivoice import OmniVoice

from adjutantvoice.config import settings


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
    # old funcationailty, need to clear the legal with Blizzard before we can use this. 
    # ref_audio = ref_audio or settings.ref_audio_path
    output_path = output_path or settings.voice_clone_path

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(settings.dtype, torch.float16)

    print(f"Loading OmniVoice model to create voice clone …")
    model = OmniVoice.from_pretrained(
        settings.model_id,
        device_map=settings.device,
        dtype=dtype,
    )

    print(f"Creating voice clone from {ref_audio} …")
    voice_clone_prompt = model.create_voice_clone_prompt(ref_audio=str(ref_audio))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        pickle.dump(voice_clone_prompt, fh)

    print(f"Voice clone saved to {output_path}")
    return output_path.resolve()
