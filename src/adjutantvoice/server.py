"""
AdjutantVoice FastAPI server.

Exposes:
  POST /synthesize               — legacy endpoint (Hermes / CLI clients)
  GET  /v1/models                — Open WebUI Custom TTS
  GET  /v1/audio/voices          — Open WebUI Custom TTS
  POST /v1/audio/speech          — Open WebUI Custom TTS
  GET  /health                   — liveness probe

Configure in Open WebUI:
  Admin → Settings → Audio → TTS Engine: Custom TTS
  API Base URL : http://<host>:<port>/v1
  API Key      : (leave blank)

Run directly:
  python -m adjutantvoice.server
  # or via the CLI:
  av server start
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from adjutantvoice import tts
from adjutantvoice.config import settings


# ---------------------------------------------------------------------------
# Lifespan — delegate model loading/unloading to tts module
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    tts.load()
    yield
    tts.unload()


app = FastAPI(title="AdjutantVoice TTS Server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Open WebUI Custom TTS endpoints (/v1/…)
# ---------------------------------------------------------------------------

@app.get("/v1/models")
def owui_models():
    """Return the model list Open WebUI expects."""
    return {"data": [{"id": settings.model_label, "name": "OmniVoice"}]}


@app.get("/v1/audio/voices")
def owui_voices():
    """Return the voice list Open WebUI expects."""
    return {"voices": settings.available_voices}


class SpeechRequest(BaseModel):
    input: str
    voice: str = "adjutant"
    model: str = settings.model_label


@app.post("/v1/audio/speech")
def owui_speech(req: SpeechRequest):
    """Synthesize speech — called by Open WebUI for every TTS chunk."""
    return _synthesis_response(req.input)


# ---------------------------------------------------------------------------
# Legacy endpoint — keeps existing Hermes / CLI clients working unchanged
# ---------------------------------------------------------------------------

class SynthesizeRequest(BaseModel):
    text: str


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    return _synthesis_response(req.text)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": tts.is_loaded(),
        "voice_mode": "clone" if tts.using_voice_clone() else "default",
    }


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _synthesis_response(text: str) -> StreamingResponse:
    if not tts.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        buf = tts.synthesize_to_buffer(text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StreamingResponse(buf, media_type="audio/mpeg")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def start(host: str = settings.host, port: int = settings.port, reload: bool = False):
    """Start the uvicorn server (called by CLI or __main__)."""
    uvicorn.run(
        "adjutantvoice.server:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    start()
