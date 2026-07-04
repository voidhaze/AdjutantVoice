"""Tests for the adjutantvoice.mcp tool functions.

`@mcp.tool(...)` leaves the underlying function directly callable, so these
tests call `tts_file` / `tts_stream` / `tts_speak` as plain functions rather
than going through a real MCP transport. `tts.synthesize` itself is mocked
directly (see test_tts.py for coverage of that layer).
"""

import subprocess

import pytest

from adjutantvoice import mcp as mcp_module
from adjutantvoice import tts


@pytest.fixture(autouse=True)
def fake_synthesize(monkeypatch):
    monkeypatch.setattr(tts, "synthesize", lambda text: b"FAKE_MP3")


# ---------------------------------------------------------------------------
# tts_file
# ---------------------------------------------------------------------------

def test_tts_file_writes_mp3_to_output_dir(tmp_settings, tmp_path):
    # In production, mcp.py's lifespan creates tts_output_dir on startup
    # (see _lifespan in mcp.py) — replicate that here since we call the
    # tool function directly without going through the real lifespan.
    tmp_settings.tts_output_dir.mkdir(parents=True, exist_ok=True)

    text_file = tmp_path / "notes.txt"
    text_file.write_text("hello there", encoding="utf-8")

    out_path_str = mcp_module.tts_file(str(text_file), ctx=None)

    out_path = tmp_path
    assert out_path_str.endswith(".mp3")
    from pathlib import Path
    written = Path(out_path_str)
    assert written.exists()
    assert written.read_bytes() == b"FAKE_MP3"
    assert written.parent == tmp_settings.tts_output_dir.resolve()


def test_tts_file_raises_if_missing(tmp_settings):
    with pytest.raises(FileNotFoundError):
        mcp_module.tts_file("/nonexistent/path/notes.txt", ctx=None)


# ---------------------------------------------------------------------------
# tts_stream
# ---------------------------------------------------------------------------

def test_tts_stream_returns_raw_bytes():
    result = mcp_module.tts_stream("hello there", ctx=None)
    assert result == b"FAKE_MP3"


# ---------------------------------------------------------------------------
# tts_speak
# ---------------------------------------------------------------------------

def test_tts_speak_plays_and_confirms(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mcp_module.subprocess,
        "run",
        lambda *a, **k: calls.append((a, k)) or subprocess.CompletedProcess(a, 0),
    )

    result = mcp_module.tts_speak("hello there", ctx=None)

    assert "Played" in result
    assert len(calls) == 1
    assert calls[0][0][0][0] == "ffplay"


def test_tts_speak_raises_runtime_error_if_ffplay_missing(monkeypatch):
    def _raise(*a, **k):
        raise FileNotFoundError()

    monkeypatch.setattr(mcp_module.subprocess, "run", _raise)

    with pytest.raises(RuntimeError, match="ffplay"):
        mcp_module.tts_speak("hello there", ctx=None)
