"""Tests for adjutantvoice.tts — model lifecycle and synthesis, fully mocked."""

import pickle

import pytest

from adjutantvoice import tts


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_not_loaded_by_default():
    assert tts.is_loaded() is False
    assert tts.using_voice_clone() is False


def test_load_without_clone_file_falls_back_to_default_voice(monkeypatch, fake_omnivoice_cls, tmp_settings):
    monkeypatch.setattr(tts, "OmniVoice", fake_omnivoice_cls)
    # tmp_settings points voice_clone_dir at an empty tmp_path — no clone exists yet.

    tts.load()

    assert tts.is_loaded() is True
    assert tts.using_voice_clone() is False
    fake_omnivoice_cls.from_pretrained.assert_called_once()


def test_load_with_existing_clone_file_uses_it(monkeypatch, fake_omnivoice_cls, tmp_settings):
    monkeypatch.setattr(tts, "OmniVoice", fake_omnivoice_cls)

    clone_path = tmp_settings.voice_clone_path
    clone_path.parent.mkdir(parents=True, exist_ok=True)
    with open(clone_path, "wb") as fh:
        pickle.dump({"prompt": "data"}, fh)

    tts.load()

    assert tts.using_voice_clone() is True


def test_load_is_idempotent(monkeypatch, fake_omnivoice_cls, tmp_settings):
    monkeypatch.setattr(tts, "OmniVoice", fake_omnivoice_cls)

    tts.load()
    tts.load()  # second call should be a no-op, not reload the model

    fake_omnivoice_cls.from_pretrained.assert_called_once()


def test_unload_resets_state(monkeypatch, fake_omnivoice_cls, tmp_settings):
    monkeypatch.setattr(tts, "OmniVoice", fake_omnivoice_cls)
    tts.load()

    tts.unload()

    assert tts.is_loaded() is False
    assert tts.using_voice_clone() is False


def test_load_respects_explicit_clone_path_override(monkeypatch, fake_omnivoice_cls, tmp_settings, tmp_path):
    monkeypatch.setattr(tts, "OmniVoice", fake_omnivoice_cls)

    override_path = tmp_path / "custom" / "clone.pkl"
    override_path.parent.mkdir(parents=True, exist_ok=True)
    with open(override_path, "wb") as fh:
        pickle.dump({"prompt": "custom"}, fh)

    tts.load(voice_clone_path=override_path)

    assert tts.using_voice_clone() is True


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def test_synthesize_raises_if_not_loaded():
    with pytest.raises(RuntimeError, match="not loaded"):
        tts.synthesize("hello")


def test_synthesize_raises_on_blank_text(monkeypatch, fake_omnivoice_cls, tmp_settings):
    monkeypatch.setattr(tts, "OmniVoice", fake_omnivoice_cls)
    tts.load()

    with pytest.raises(ValueError, match="empty"):
        tts.synthesize("   ")


def test_synthesize_uses_default_instruct_when_no_clone(monkeypatch, fake_omnivoice_cls, fake_model, tmp_settings):
    monkeypatch.setattr(tts, "OmniVoice", fake_omnivoice_cls)
    monkeypatch.setattr(tts.sf, "write", lambda buf, audio, rate, format: buf.write(b"FAKE_MP3"))
    tts.load()

    result = tts.synthesize("hello world")

    assert result == b"FAKE_MP3"
    _, kwargs = fake_model.generate.call_args
    assert kwargs["text"] == "hello world"
    assert "instruct" in kwargs
    assert "voice_clone_prompt" not in kwargs


def test_synthesize_uses_voice_clone_when_available(monkeypatch, fake_omnivoice_cls, fake_model, tmp_settings):
    monkeypatch.setattr(tts, "OmniVoice", fake_omnivoice_cls)
    monkeypatch.setattr(tts.sf, "write", lambda buf, audio, rate, format: buf.write(b"FAKE_MP3"))

    clone_path = tmp_settings.voice_clone_path
    clone_path.parent.mkdir(parents=True, exist_ok=True)
    with open(clone_path, "wb") as fh:
        pickle.dump({"prompt": "data"}, fh)

    tts.load()
    tts.synthesize("hello world")

    _, kwargs = fake_model.generate.call_args
    assert "voice_clone_prompt" in kwargs
    assert "instruct" not in kwargs


def test_synthesize_to_buffer_returns_seeked_bytesio(monkeypatch, fake_omnivoice_cls, tmp_settings):
    monkeypatch.setattr(tts, "OmniVoice", fake_omnivoice_cls)
    monkeypatch.setattr(tts.sf, "write", lambda buf, audio, rate, format: buf.write(b"FAKE_MP3"))
    tts.load()

    buf = tts.synthesize_to_buffer("hello")

    assert buf.tell() == 0
    assert buf.read() == b"FAKE_MP3"
