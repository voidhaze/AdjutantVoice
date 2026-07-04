"""
Shared pytest fixtures for the AdjutantVoice test suite.

Design note — no GPU / real model in CI
----------------------------------------
`torch` and `omnivoice` are heavy, GPU-oriented dependencies that aren't
installed in CI or on most contributors' machines. Before any test module
imports anything from `adjutantvoice` (which happens at collection time),
this file installs lightweight stand-ins for both into `sys.modules`. This
lets `adjutantvoice.tts` / `adjutantvoice.voice` do their normal
`import torch` / `from omnivoice import OmniVoice` without error.

Individual tests then swap in their own `MagicMock` model (via the
`fake_omnivoice_cls` fixture below or a local monkeypatch) to control what
`OmniVoice.from_pretrained(...)` / `model.generate(...)` return, so no test
ever touches real model weights or a GPU.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub out torch / omnivoice at import time (see module docstring)
# ---------------------------------------------------------------------------

if "torch" not in sys.modules:
    _torch_stub = MagicMock(name="torch_stub")
    # tts.py / voice.py only ever reference these three dtype attributes.
    _torch_stub.float16 = "float16"
    _torch_stub.bfloat16 = "bfloat16"
    _torch_stub.float32 = "float32"
    sys.modules["torch"] = _torch_stub

if "omnivoice" not in sys.modules:
    _omnivoice_stub = MagicMock(name="omnivoice_module_stub")
    _omnivoice_stub.OmniVoice = MagicMock(name="OmniVoice_class_stub")
    sys.modules["omnivoice"] = _omnivoice_stub


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_model():
    """A MagicMock standing in for a loaded OmniVoice model instance.

    `.generate(...)` returns a one-element list containing a tiny fake
    "audio array" (a plain list of floats is enough — nothing in the code
    under test cares about real audio samples once `sf.write` itself is
    mocked in the tests that need it).
    """
    model = MagicMock(name="fake_omnivoice_model")
    model.generate.return_value = [[0.0, 0.1, 0.2]]
    model.create_voice_clone_prompt.return_value = {"fake": "clone-prompt"}
    return model


@pytest.fixture
def fake_omnivoice_cls(fake_model):
    """A stand-in for the `OmniVoice` class with `.from_pretrained(...)` wired up."""
    cls = MagicMock(name="fake_OmniVoice_cls")
    cls.from_pretrained.return_value = fake_model
    return cls


@pytest.fixture(autouse=True)
def reset_tts_state():
    """Ensure `adjutantvoice.tts` module-level state never leaks between tests.

    `tts.py` intentionally uses module-level globals (_model, _loaded, ...)
    as a process-wide singleton — that's correct for the real server, but
    without this fixture one test's `tts.load()` would leave the model
    "loaded" for every test that runs after it.
    """
    from adjutantvoice import tts

    tts.unload()
    yield
    tts.unload()


@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    """Patch `adjutantvoice.config.settings` to use isolated tmp_path directories.

    Prevents tests from touching the real ~/.adjutantvoice or ~/.hermes
    directories on the machine running the suite.
    """
    from adjutantvoice.config import settings

    monkeypatch.setattr(settings, "voice_clone_dir", tmp_path / "voices")
    monkeypatch.setattr(settings, "tts_output_dir", tmp_path / "tts_output")
    return settings
