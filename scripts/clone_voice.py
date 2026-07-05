#!/usr/bin/env python3
"""Thin wrapper — delegates to `av voice create-clone`."""
import argparse
from pathlib import Path

from adjutantvoice.voice import create_voice_clone

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ref_audio",
        type=Path,
        help="Path to a short, clean reference audio sample (WAV recommended) of the voice to clone.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Where to save the resulting clone prompt (.pkl). Defaults to a path under the AdjutantVoice config directory.",
    )
    args = parser.parse_args()

    saved = create_voice_clone(ref_audio=args.ref_audio, output_path=args.output)
    print(f"Voice clone saved to: {saved}")
