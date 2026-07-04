"""Tests for adjutantvoice.voice.create_voice_clone."""

import pickle

from adjutantvoice import voice


def test_create_voice_clone_writes_pickle_and_returns_resolved_path(
    monkeypatch, fake_omnivoice_cls, fake_model, tmp_settings, tmp_path
):
    monkeypatch.setattr(voice, "OmniVoice", fake_omnivoice_cls)

    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"fake audio bytes")
    output_path = tmp_path / "clones" / "mine.pkl"

    result = voice.create_voice_clone(ref_audio=ref_audio, output_path=output_path)

    assert result == output_path.resolve()
    assert output_path.exists()
    with open(output_path, "rb") as fh:
        assert pickle.load(fh) == {"fake": "clone-prompt"}

    fake_model.create_voice_clone_prompt.assert_called_once_with(ref_audio=str(ref_audio))


def test_create_voice_clone_uses_settings_defaults_when_no_args(
    monkeypatch, fake_omnivoice_cls, tmp_settings
):
    monkeypatch.setattr(voice, "OmniVoice", fake_omnivoice_cls)

    result = voice.create_voice_clone()

    assert result == tmp_settings.voice_clone_path.resolve()
    assert tmp_settings.voice_clone_path.exists()


def test_create_voice_clone_creates_missing_parent_dirs(
    monkeypatch, fake_omnivoice_cls, tmp_settings, tmp_path
):
    monkeypatch.setattr(voice, "OmniVoice", fake_omnivoice_cls)

    output_path = tmp_path / "a" / "b" / "c" / "clone.pkl"
    assert not output_path.parent.exists()

    voice.create_voice_clone(output_path=output_path)

    assert output_path.exists()
