"""Tests for adjutantvoice.config.Settings."""

from pathlib import Path

from adjutantvoice.config import Settings


def test_defaults_are_sane():
    s = Settings(_env_file=None)
    assert s.host == "0.0.0.0"
    assert s.port == 8111
    assert s.mcp_port == 8222
    assert s.dtype == "float16"
    assert s.sample_rate == 24_000


def test_voice_clone_path_derived_from_dir_and_name():
    s = Settings(_env_file=None, voice_clone_dir=Path("/tmp/voices"), default_voice_clone_name="my-voice")
    assert s.voice_clone_path == Path("/tmp/voices/my-voice.pkl")


def test_voice_clone_path_updates_if_dir_changes():
    s = Settings(_env_file=None)
    s.voice_clone_dir = Path("/tmp/other-voices")
    assert s.voice_clone_path == Path("/tmp/other-voices") / f"{s.default_voice_clone_name}.pkl"


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("AV_PORT", "9999")
    monkeypatch.setenv("AV_HOST", "127.0.0.1")
    s = Settings(_env_file=None)
    assert s.port == 9999
    assert s.host == "127.0.0.1"


def test_unknown_env_vars_are_ignored(monkeypatch):
    # extra="ignore" in model_config — this should not raise.
    monkeypatch.setenv("AV_SOME_UNRELATED_SETTING", "whatever")
    Settings(_env_file=None)
