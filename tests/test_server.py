"""Tests for the adjutantvoice.server FastAPI app.

We deliberately avoid triggering the app's real `lifespan` (which calls
`tts.load()` against a real model). Instead each test uses a plain
`TestClient(app)` (no `with` block, so startup/shutdown never fire) and
loads/unloads `tts` state itself the same way `tests/test_tts.py` does,
via the mocked `OmniVoice` class.
"""

from fastapi.testclient import TestClient

from adjutantvoice import tts
from adjutantvoice.server import app

client = TestClient(app)


def _load_fake_model(monkeypatch, fake_omnivoice_cls, tmp_settings):
    monkeypatch.setattr(tts, "OmniVoice", fake_omnivoice_cls)
    monkeypatch.setattr(tts.sf, "write", lambda buf, audio, rate, format: buf.write(b"FAKE_MP3"))
    tts.load()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_when_model_not_loaded():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"status": "ok", "model_loaded": False, "voice_mode": "default"}


def test_health_when_model_loaded(monkeypatch, fake_omnivoice_cls, tmp_settings):
    _load_fake_model(monkeypatch, fake_omnivoice_cls, tmp_settings)

    resp = client.get("/health")

    assert resp.json() == {"status": "ok", "model_loaded": True, "voice_mode": "default"}


def test_health_reports_clone_voice_mode(monkeypatch, fake_omnivoice_cls, tmp_settings):
    import pickle

    clone_path = tmp_settings.voice_clone_path
    clone_path.parent.mkdir(parents=True, exist_ok=True)
    with open(clone_path, "wb") as fh:
        pickle.dump({"prompt": "x"}, fh)

    _load_fake_model(monkeypatch, fake_omnivoice_cls, tmp_settings)

    resp = client.get("/health")

    assert resp.json()["voice_mode"] == "clone"


# ---------------------------------------------------------------------------
# Open WebUI endpoints
# ---------------------------------------------------------------------------

def test_owui_models():
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json() == {"data": [{"id": "omnivoice", "name": "OmniVoice"}]}


def test_owui_voices():
    resp = client.get("/v1/audio/voices")
    assert resp.status_code == 200
    assert resp.json() == {"voices": ["adjutant"]}


def test_owui_speech_success(monkeypatch, fake_omnivoice_cls, tmp_settings):
    _load_fake_model(monkeypatch, fake_omnivoice_cls, tmp_settings)

    resp = client.post("/v1/audio/speech", json={"input": "hello there"})

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert resp.content == b"FAKE_MP3"


def test_owui_speech_503_when_model_not_loaded():
    resp = client.post("/v1/audio/speech", json={"input": "hello there"})
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Legacy /synthesize endpoint
# ---------------------------------------------------------------------------

def test_synthesize_success(monkeypatch, fake_omnivoice_cls, tmp_settings):
    _load_fake_model(monkeypatch, fake_omnivoice_cls, tmp_settings)

    resp = client.post("/synthesize", json={"text": "hello there"})

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert resp.content == b"FAKE_MP3"


def test_synthesize_503_when_model_not_loaded():
    resp = client.post("/synthesize", json={"text": "hello there"})
    assert resp.status_code == 503
    assert "not loaded" in resp.json()["detail"].lower()


def test_synthesize_400_on_blank_text(monkeypatch, fake_omnivoice_cls, tmp_settings):
    _load_fake_model(monkeypatch, fake_omnivoice_cls, tmp_settings)

    resp = client.post("/synthesize", json={"text": "   "})

    assert resp.status_code == 400
