"""Tests for adjutantvoice.cli using Typer's CliRunner.

HTTP calls (`requests`) and audio playback (`subprocess` / ffplay) are
mocked throughout — these tests exercise CLI argument handling and error
messaging, not the real network or audio stack.
"""

from __future__ import annotations

import subprocess as real_subprocess

import pytest
import requests
from typer.testing import CliRunner

from adjutantvoice import cli

runner = CliRunner()


class _FakeResponse:
    def __init__(self, content: bytes = b"FAKE_MP3", status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return {"detail": "server exploded"}


# ---------------------------------------------------------------------------
# av speak
# ---------------------------------------------------------------------------

def test_speak_saves_to_output_file(monkeypatch, tmp_path):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse())
    out_file = tmp_path / "out.mp3"

    result = runner.invoke(cli.app, ["speak", "hello there", "--output", str(out_file)])

    assert result.exit_code == 0
    assert out_file.read_bytes() == b"FAKE_MP3"
    assert "Saved to" in result.stdout


def test_speak_plays_audio_when_no_output(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse())
    play_calls = []
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *a, **k: play_calls.append(a) or real_subprocess.CompletedProcess(a, 0)
    )

    result = runner.invoke(cli.app, ["speak", "hello there"])

    assert result.exit_code == 0
    assert len(play_calls) == 1
    assert play_calls[0][0][0] == "ffplay"


def test_speak_connection_error_shows_actionable_hint(monkeypatch):
    def _raise(*a, **k):
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr(requests, "post", _raise)

    result = runner.invoke(cli.app, ["speak", "hello there"])

    assert result.exit_code == 1
    assert "av server start" in result.output


def test_speak_timeout_shows_message(monkeypatch):
    def _raise(*a, **k):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(requests, "post", _raise)

    result = runner.invoke(cli.app, ["speak", "hello there"])

    assert result.exit_code == 1
    assert "Timed out" in result.output


def test_speak_http_error_shows_status_and_detail(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(status_code=500))

    result = runner.invoke(cli.app, ["speak", "hello there"])

    assert result.exit_code == 1
    assert "500" in result.output


# ---------------------------------------------------------------------------
# av speak-file
# ---------------------------------------------------------------------------

def test_speak_file_missing_file_errors_cleanly(tmp_path):
    missing = tmp_path / "nope.txt"

    result = runner.invoke(cli.app, ["speak-file", str(missing)])

    assert result.exit_code == 1
    assert "File not found" in result.output


def test_speak_file_reads_and_synthesizes(monkeypatch, tmp_path):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse())
    text_file = tmp_path / "notes.txt"
    text_file.write_text("some notes", encoding="utf-8")
    out_file = tmp_path / "out.mp3"

    result = runner.invoke(cli.app, ["speak-file", str(text_file), "--output", str(out_file)])

    assert result.exit_code == 0
    assert out_file.read_bytes() == b"FAKE_MP3"


# ---------------------------------------------------------------------------
# av voice create-clone
# ---------------------------------------------------------------------------

def test_voice_create_clone_invokes_library_function(monkeypatch, tmp_path):
    saved_path = tmp_path / "clone.pkl"
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"fake audio bytes")
    monkeypatch.setattr(
        "adjutantvoice.voice.create_voice_clone", lambda ref_audio, output_path: saved_path
    )

    result = runner.invoke(cli.app, ["voice", "create-clone", "--ref-audio", str(ref_audio)])

    assert result.exit_code == 0
    assert str(saved_path) in result.stdout


def test_voice_create_clone_requires_ref_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "adjutantvoice.voice.create_voice_clone", lambda ref_audio, output_path: tmp_path
    )

    result = runner.invoke(cli.app, ["voice", "create-clone"])

    assert result.exit_code != 0
    assert "--ref-audio" in result.stdout or "--ref-audio" in (result.stderr or "")


# ---------------------------------------------------------------------------
# av install
# ---------------------------------------------------------------------------

def test_install_claude_calls_fastmcp_install(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *a, **k: calls.append((a, k)) or real_subprocess.CompletedProcess(a, 0)
    )

    result = runner.invoke(cli.app, ["install", "claude"])

    assert result.exit_code == 0
    assert len(calls) == 1
    args = calls[0][0][0]
    assert args[0] == "fastmcp"
    assert args[1] == "install"
    assert "--name" in args


def test_install_hermes_calls_integration_install(monkeypatch):
    called = []
    monkeypatch.setattr("adjutantvoice.integrations.hermes.install", lambda: called.append(True))

    result = runner.invoke(cli.app, ["install", "hermes"])

    assert result.exit_code == 0
    assert called == [True]
