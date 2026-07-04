"""
AdjutantVoice MCP Server — OmniVoice TTS exposed via FastMCP.

Tools:
  tts_file    — synthesise a text file → returns an audio file path on disk
  tts_stream  — synthesise a text snippet → returns raw MP3 bytes
  tts_speak   — synthesise text and play it aloud on the server machine

Run via the unified CLI:
  av mcp start                                # stdio (default)
  av mcp start --transport streamable-http    # HTTP on the configured port
  av mcp start [--transport stdio|streamable-http] [--port PORT]

Run directly:
  python -m adjutantvoice.mcp                 # stdio
  python -m adjutantvoice.mcp streamable-http # HTTP

MCP port is configured via AV_MCP_PORT (default: see config.py).

For default HTTP remote clients use:
  av mcp start --transport streamable-http --port 8001

"""

from __future__ import annotations

import subprocess
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastmcp import FastMCP, Context

from adjutantvoice import tts
from adjutantvoice.config import settings


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(server: FastMCP):
    settings.tts_output_dir.mkdir(parents=True, exist_ok=True)
    tts.load()
    try:
        yield
    finally:
        tts.unload()


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="AdjutantVoice TTS",
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool(
    name="tts_file",
    description=(
        "Read plain text from a local file and synthesise it with the "
        "Adjutant voice clone. Returns the absolute path to the resulting "
        "MP3 file written to disk."
    ),
)
def tts_file(text_file_path: str, ctx: Context) -> str:
    """Synthesise the contents of a text file and save the audio to disk.

    The output MP3 is written to ``settings.tts_output_dir`` with a random
    hex filename.

    Args:
        text_file_path: Absolute or relative path to a UTF-8 text file.

    Returns:
        Absolute path to the generated MP3 file.

    Raises:
        FileNotFoundError: If ``text_file_path`` does not exist.
    """
    src = Path(text_file_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Text file not found: {src}")

    text = src.read_text(encoding="utf-8")
    mp3_bytes = tts.synthesize(text)

    out_path = settings.tts_output_dir / f"{uuid.uuid4().hex}.mp3"
    out_path.write_bytes(mp3_bytes)
    return str(out_path.resolve())


@mcp.tool(
    name="tts_stream",
    description=(
        "Synthesise a short section of text with the Adjutant voice clone "
        "and return the audio as raw MP3 bytes. Useful when the caller wants "
        "to handle the audio data directly rather than write it to disk."
    ),
)
def tts_stream(text: str, ctx: Context) -> bytes:
    """Synthesise text and return raw MP3 bytes.

    Args:
        text: The text to convert to speech.

    Returns:
        Raw MP3 audio data as bytes.
    """
    return tts.synthesize(text)


@mcp.tool(
    name="tts_speak",
    description=(
        "Synthesise text with the Adjutant voice clone and play the audio "
        "aloud on the server machine using the system's default audio output. "
        "Returns a confirmation message once playback completes."
    ),
)
def tts_speak(text: str, ctx: Context) -> str:
    """Synthesise text and play it through the server's speakers.

    Delegates playback to ``ffplay`` (bundled with ffmpeg). Audio is written
    to a temporary file, played, then deleted.

    Args:
        text: The text to speak aloud.

    Returns:
        A short confirmation string, e.g. ``'Played ~3.2 s of audio.'``
        Duration is estimated from the MP3 byte length and may not be exact.

    Raises:
        RuntimeError: If ``ffplay`` is not found on the server's PATH.
    """
    mp3_bytes = tts.synthesize(text)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp_path.write_bytes(mp3_bytes)

    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(tmp_path)],
            check=True,
            capture_output=True,
        )
        duration_s = round(len(mp3_bytes) / 16_000, 1)
        return f"Played ~{duration_s} s of audio."
    except FileNotFoundError:
        raise RuntimeError(
            "ffplay not found. Install ffmpeg on the server machine "
            "(`sudo apt install ffmpeg` or `brew install ffmpeg`)."
        ) from None
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def start(transport: str = "streamable-http", port: int = settings.mcp_port):
    """Start the MCP server.

    Called by ``__main__`` when the module is run directly. The normal
    entry point for end users is ``av mcp`` via the CLI (``cli.py``).

    Args:
        transport: FastMCP transport name. ``'stdio'`` for
            Claude Desktop / Claude Code; ``'streamable-http'`` (default) for
            remote HTTP clients.
        port: Port to listen on when using an HTTP transport. Defaults
            to ``settings.mcp_port`` (``AV_MCP_PORT`` env var).
    """
    if transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=transport, port=port)


if __name__ == "__main__":
    start()