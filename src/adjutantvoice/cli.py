"""
AdjutantVoice CLI — single entry point for all functionality.

Usage:
  av server start [--host HOST] [--port PORT] [--reload]
  av mcp start [--transport stdio|streamable-http] [--port PORT]
  av speak <text>
  av speak-file <path>
  av voice create-clone --ref-audio PATH [--output PATH]
  av install claude
  av install hermes
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import typer

from adjutantvoice.config import settings

app = typer.Typer(
    name="av",
    help="AdjutantVoice — [bold]OmniVoice[/bold] TTS toolkit.",
    rich_markup_mode="rich",
    no_args_is_help=True,
    epilog=(
        "Common workflows:\n\n"
        "  av server start                 Start the HTTP TTS server locally.\n\n"
        "  av speak \"hello there\"           Speak text using the running server.\n\n"
        "  av voice create-clone --ref-audio me.wav   Build a reusable voice clone.\n\n"
        "  av install claude                Register AdjutantVoice with Claude Desktop.\n\n"
        "  av install hermes                Register AdjutantVoice as the default TTS provider for Hermes agents.\n\n"
        "Run [bold]av COMMAND --help[/bold] for details on any command.\n\n"
        "\n\n"
        "To get started run \"av voice create-clone\" to build a voice clone, fire up the server with \"av server start\" then \"av speak\" to test it out."
    ),
)


# ---------------------------------------------------------------------------
# server sub-app
# ---------------------------------------------------------------------------

server_app = typer.Typer(
    help="Manage the FastAPI TTS server.",
    no_args_is_help=True,
)
app.add_typer(server_app, name="server")


@server_app.command("start")
def server_start(
    host: str = typer.Option(settings.host, help="Interface to bind to (use 0.0.0.0 to expose beyond localhost)."),
    port: int = typer.Option(settings.port, help="TCP port for the HTTP API."),
    reload: bool = typer.Option(
        False, "--reload", help="Restart automatically on code changes. Development only — do not use in production."
    ),
):
    """
    Start the TTS HTTP server, including the Open WebUI-compatible /v1/ endpoints.

    The server loads the OmniVoice model on first request and keeps it resident
    in memory (singleton, thread-locked) for subsequent calls, so the first
    synthesis after startup will be slower than the rest.

    Examples:

      av server start

      av server start --host 0.0.0.0 --port 9000

      av server start --reload
    """
    from adjutantvoice.server import start
    start(host=host, port=port, reload=reload)


# ---------------------------------------------------------------------------
# mcp sub-app
# ---------------------------------------------------------------------------

mcp_app = typer.Typer(
    help="Manage the FastMCP TTS server.",
    no_args_is_help=True,
)
app.add_typer(mcp_app, name="mcp")


@mcp_app.command("start")
def mcp_start(
    transport: str = typer.Option(
        "streamable-http",
        help="Transport protocol. Use [bold]stdio[/bold] when launched by an MCP client (e.g. Claude Desktop), "
        "or [bold]streamable-http[/bold] to run as a standalone HTTP MCP server.",
    ),
    port: int = typer.Option(settings.mcp_port, help="Port to listen on. Ignored when --transport stdio is used."),
):
    """
    Start the MCP server exposing TTS tools over the Model Context Protocol.

    This is the entry point used by 'av install claude' and by Hermes agents
    configured to call AdjutantVoice as an MCP tool provider.

    Examples:

      av mcp start --transport stdio

      av mcp start --transport streamable-http --port 8765
    """
    from adjutantvoice.mcp import start
    start(transport=transport, port=port)


# ---------------------------------------------------------------------------
# speak commands
# ---------------------------------------------------------------------------

@app.command(
    epilog=(
        "Examples:\n\n"
        "  av speak \"Hello, world\"\n\n"
        "  av speak \"Reminder: standup in five minutes\" --output reminder.mp3\n\n"
        "  av speak \"Status update\" --server http://lucy:{settings.port}"
    )
)
def speak(
    text: str = typer.Argument(..., help="Text to synthesise and play."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write the resulting MP3 to this path instead of playing it aloud."
    ),
    server_url: str = typer.Option(
        f"http://localhost:{settings.port}", "--server", help="Base URL of a running 'av server start' instance."
    ),
):
    """
    Synthesise text via the running server and play it, or save it to a file.

    Requires an AdjutantVoice server to already be running (see 'av server start').
    Playback uses ffplay; if it's not installed, use --output to save the MP3
    instead of playing it.
    """
    content = _synthesize(server_url, text, timeout=60)

    if output:
        output.write_bytes(content)
        typer.echo(f"Saved to {output}")
    else:
        _play_mp3(content)


@app.command(
    "speak-file",
    epilog=(
        "Examples:\n\n"
        "  av speak-file notes.txt\n\n"
        "  av speak-file script.txt --output narration.mp3"
    ),
)
def speak_file(
    input_file: Path = typer.Argument(..., help="Path to a UTF-8 text file to read and synthesise."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write the resulting MP3 to this path instead of playing it aloud."
    ),
    server_url: str = typer.Option(
        f"http://localhost:{settings.port}", "--server", help="Base URL of a running 'av server start' instance."
    ),
):
    """
    Read a text file in full and synthesise it via the running server.

    Useful for longer content (scripts, notes, articles) where typing the text
    directly into 'av speak' would be unwieldy. Allows a longer request timeout
    than 'av speak' since larger files take longer to synthesise.
    """
    if not input_file.exists():
        typer.echo(f"File not found: {input_file}", err=True)
        raise typer.Exit(code=1)

    text = input_file.read_text(encoding="utf-8")
    content = _synthesize(server_url, text, timeout=120)

    if output:
        output.write_bytes(content)
        typer.echo(f"Saved to {output}")
    else:
        _play_mp3(content)


# ---------------------------------------------------------------------------
# voice sub-app
# ---------------------------------------------------------------------------

voice_app = typer.Typer(
    help="Voice clone utilities.",
    no_args_is_help=True,
    rich_help_panel="Integrations",
)
app.add_typer(voice_app, name="voice", rich_help_panel="Integrations")


@voice_app.command(
    "create-clone",
    epilog=(
        "Examples:\n\n"
        "  av voice create-clone --ref-audio sample.wav\n\n"
        "  av voice create-clone --ref-audio sample.wav --output ~/.adjutantvoice/clones/me.pkl"
    ),
)
def voice_create_clone(
    ref_audio: Path = typer.Option(
        ..., "--ref-audio", help="A short, clean audio sample (WAV recommended) of the voice to clone."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Where to save the resulting clone prompt (.pkl). Defaults to a path under the AdjutantVoice config directory."
    ),
):
    """
    Build a reusable voice-clone prompt from a reference audio file.

    The generated .pkl file can be passed to the server/MCP tools to synthesise
    speech in the cloned voice. If the reference audio is unclear or a clone
    can't be reliably built, AdjutantVoice falls back to the default voice —
    check the command output to confirm which voice was actually used.
    """
    from adjutantvoice.voice import create_voice_clone
    saved = create_voice_clone(ref_audio=ref_audio, output_path=output)
    typer.echo(f"Voice clone saved to: {saved}")


# ---------------------------------------------------------------------------
# install sub-app
# ---------------------------------------------------------------------------

install_app = typer.Typer(
    help="Install AdjutantVoice integrations.",
    no_args_is_help=True,
    rich_help_panel="Integrations",
)
app.add_typer(install_app, name="install", rich_help_panel="Integrations")


@install_app.command(
    "claude",
    epilog=(
        "Examples:\n\n"
        "  av install claude\n\n"
        "  av install claude --name \"My Voice\""
    ),
)
def install_claude(
    name: str = typer.Option(
        "AdjutantVoice TTS", "--name", help="Display name shown for this server in Claude Desktop's MCP server list."
    ),
):
    """
    Register AdjutantVoice as an MCP server in Claude Desktop.

    Runs 'fastmcp install' against this package's mcp.py entry point, adding
    the necessary config so Claude Desktop can launch AdjutantVoice's TTS
    tools automatically. Restart Claude Desktop after running this for the
    new server to appear.
    """
    mcp_path = Path(__file__).parent / "mcp.py"
    subprocess.run(
        ["fastmcp", "install", str(mcp_path), "--name", name],
        check=True,
    )


@install_app.command("hermes")
def install_hermes():
    """
    Register AdjutantVoice as the default TTS provider for Hermes agents.

    Writes a 'type: command' provider entry to ~/.hermes/config.yaml that
    invokes AdjutantVoice with {input_path}/{output_path} placeholders — no
    Hermes plugin required. Existing TTS provider config, if any, is replaced.
    """
    from adjutantvoice.integrations.hermes import install
    install()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _synthesize(server_url: str, text: str, timeout: float) -> bytes:
    """POST text to the running TTS server's /synthesize endpoint.

    Exits with a clear, actionable message instead of an unhandled
    traceback if the server is unreachable, slow, or returns an error.
    """
    import requests

    try:
        resp = requests.post(f"{server_url}/synthesize", json={"text": text}, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        typer.echo(
            f"Could not reach the AdjutantVoice server at {server_url}.\n"
            f"Is it running? Start it with: av server start",
            err=True,
        )
        raise typer.Exit(code=1)
    except requests.exceptions.Timeout:
        typer.echo(
            f"Timed out waiting for a response from {server_url} after {timeout:.0f}s.\n"
            f"The server may be overloaded or still loading the model.",
            err=True,
        )
        raise typer.Exit(code=1)
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = f" — {resp.json().get('detail', '')}"
        except Exception:
            pass
        typer.echo(f"Server returned an error ({resp.status_code}){detail}", err=True)
        raise typer.Exit(code=1)
    except requests.exceptions.RequestException as exc:
        typer.echo(f"Request to {server_url} failed: {exc}", err=True)
        raise typer.Exit(code=1)

    return resp.content


def _play_mp3(mp3_bytes: bytes) -> None:
    """Write bytes to a temp file and play with ffplay."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp_path.write_bytes(mp3_bytes)
    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(tmp_path)],
            check=True,
        )
    except FileNotFoundError:
        typer.echo(
            "ffplay not found. Install ffmpeg (`sudo apt install ffmpeg`) "
            "or save with --output instead.",
            err=True,
        )
        sys.exit(1)
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    app()