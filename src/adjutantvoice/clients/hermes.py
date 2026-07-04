"""
Hermes TTS command client.

Invoked by Hermes as:
  python -m adjutantvoice.clients.hermes <input_path> <output_path>

Reads text from <input_path>, POSTs to the AdjutantVoice server,
and writes the MP3 response to <output_path>.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests
import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    input_path: Path = typer.Argument(..., help="Path to the text file to synthesise."),
    output_path: Path = typer.Argument(..., help="Path to write the MP3 output."),
    server_url: str = typer.Option("http://localhost:8000", "--server", help="TTS server base URL."),
):
    text = input_path.read_text(encoding="utf-8")
    resp = requests.post(f"{server_url}/synthesize", json={"text": text}, timeout=120)
    resp.raise_for_status()
    output_path.write_bytes(resp.content)


if __name__ == "__main__":
    app()
