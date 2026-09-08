"""Tests for adjutantvoice.voice.create_voice_clone."""

import pickle

import pytest

from adjutantvoice import tts, voice


def test_create_voice_clone_writes_pickle_and_returns_resolved_path(
    monkeypatch, fake_omnivoice_cls, fake_model, tmp_settings, tmp_path
):
    monkeypatch.setattr(tts, "OmniVoice", fake_omnivoice_cls)

    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"fake audio bytes")
    output_path = tmp_path / "clones" / "mine.pkl"

    result = voice.create_voice_clone(ref_audio=ref_audio, output_path=output_path)

    assert result == output_path.resolve()
    assert output_path.exists()
    with open(output_path, "rb") as fh:
        assert pickle.load(fh) == {"fake": "clone-prompt"}

    fake_model.create_voice_clone_prompt.assert_called_once_with(ref_audio=str(ref_audio))


def test_create_voice_clone_uses_settings_default_output_path(
    monkeypatch, fake_omnivoice_cls, tmp_settings, tmp_path
):
    monkeypatch.setattr(tts, "OmniVoice", fake_omnivoice_cls)

    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"fake audio bytes")

    result = voice.create_voice_clone(ref_audio=ref_audio)

    assert result == tmp_settings.voice_clone_path.resolve()
    assert tmp_settings.voice_clone_path.exists()


def test_create_voice_clone_creates_missing_parent_dirs(
    monkeypatch, fake_omnivoice_cls, tmp_settings, tmp_path
):
    monkeypatch.setattr(tts, "OmniVoice", fake_omnivoice_cls)

    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"fake audio bytes")
    output_path = tmp_path / "a" / "b" / "c" / "clone.pkl"
    assert not output_path.parent.exists()

    voice.create_voice_clone(ref_audio=ref_audio, output_path=output_path)

    assert output_path.exists()


def test_create_voice_clone_requires_ref_audio(monkeypatch, fake_omnivoice_cls, tmp_settings):
    monkeypatch.setattr(tts, "OmniVoice", fake_omnivoice_cls)

    with pytest.raises(TypeError):
        voice.create_voice_clone()


def test_create_voice_clone_reuses_already_loaded_model(
    monkeypatch, fake_omnivoice_cls, fake_model, tmp_settings, tmp_path
):
    """If a model is already loaded in-process (e.g. inside a long-running
    server), create_voice_clone must reuse it rather than loading a second
    copy via a duplicated from_pretrained call."""
    monkeypatch.setattr(tts, "OmniVoice", fake_omnivoice_cls)
    tts.load()  # simulate a server that already has the model resident
    assert fake_omnivoice_cls.from_pretrained.call_count == 1

    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"fake audio bytes")
    voice.create_voice_clone(ref_audio=ref_audio, output_path=tmp_path / "clone.pkl")

    # Still exactly one call — create_voice_clone did not reload the model.
    assert fake_omnivoice_cls.from_pretrained.call_count == 1
    fake_model.create_voice_clone_prompt.assert_called_once_with(ref_audio=str(ref_audio))


def test_create_voice_clone_loads_model_when_none_loaded(
    monkeypatch, fake_omnivoice_cls, tmp_settings, tmp_path
):
    """When called standalone (the common CLI case, no server running),
    create_voice_clone should still trigger a model load via tts.get_model."""
    monkeypatch.setattr(tts, "OmniVoice", fake_omnivoice_cls)
    assert tts.is_loaded() is False

    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"fake audio bytes")
    voice.create_voice_clone(ref_audio=ref_audio, output_path=tmp_path / "clone.pkl")

    assert tts.is_loaded() is True
    fake_omnivoice_cls.from_pretrained.assert_called_once()
