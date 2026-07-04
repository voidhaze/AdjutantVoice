"""Tests for adjutantvoice.clients.hermes — the command invoked by Hermes."""

import requests
from typer.testing import CliRunner

from adjutantvoice.clients import hermes as hermes_client

runner = CliRunner()


class _FakeResponse:
    def __init__(self, content: bytes = b"FAKE_MP3"):
        self.content = content

    def raise_for_status(self):
        pass


def test_reads_input_writes_output(monkeypatch, tmp_path):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    input_path = tmp_path / "in.txt"
    input_path.write_text("read this aloud", encoding="utf-8")
    output_path = tmp_path / "out.mp3"

    result = runner.invoke(hermes_client.app, [str(input_path), str(output_path)])

    assert result.exit_code == 0
    assert output_path.read_bytes() == b"FAKE_MP3"
    assert captured["json"] == {"text": "read this aloud"}
    assert captured["url"] == "http://localhost:8000/synthesize"


def test_respects_custom_server_url(monkeypatch, tmp_path):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    input_path = tmp_path / "in.txt"
    input_path.write_text("hi", encoding="utf-8")
    output_path = tmp_path / "out.mp3"

    result = runner.invoke(
        hermes_client.app, [str(input_path), str(output_path), "--server", "http://lucy:9000"]
    )

    assert result.exit_code == 0
    assert captured["url"] == "http://lucy:9000/synthesize"
