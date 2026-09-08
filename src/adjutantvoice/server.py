"""
AdjutantVoice FastAPI server.

Exposes:
  POST /synthesize               — legacy endpoint (Hermes / CLI clients)
  GET  /v1/models                — OpenAI-compatible audio API
  GET  /v1/audio/voices          — OpenAI-compatible audio API
  POST /v1/audio/speech          — OpenAI-compatible audio API
  GET  /health                   — liveness probe

The /v1/ endpoints implement the OpenAI audio/speech API, so any
OpenAI-compatible TTS client can talk to this server. Point the client at:
  API Base URL : http://<host>:<port>/v1
  API Key      : (any non-empty value; not checked)

Run directly:
  python -m adjutantvoice.server
  # or via the CLI:
  av server start
"""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from adjutantvoice import tts
from adjutantvoice.config import settings

logger = logging.getLogger(__name__)


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
# OpenAI-compatible audio API endpoints (/v1/…)
# ---------------------------------------------------------------------------

@app.get("/v1/models")
def openai_models():
    """Return the model list in the OpenAI API shape."""
    return {"data": [{"id": settings.model_label, "name": "OmniVoice"}]}


@app.get("/v1/audio/voices")
def openai_voices():
    """Return the available voices (OpenAI-compatible audio API extension)."""
    return {"voices": settings.available_voices}


class SpeechRequest(BaseModel):
    input: str
    voice: str = "adjutant"
    model: str = settings.model_label


@app.post("/v1/audio/speech")
def openai_speech(req: SpeechRequest):
    """Synthesize speech — the OpenAI audio/speech endpoint."""
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
        # ValueError here is always our own deliberate, user-facing
        # validation message (e.g. "text must not be empty") — safe to
        # pass through as-is.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        # Anything else is unexpected (model/inference failure, etc.).
        # Log the full detail server-side but don't leak internal paths,
        # config, or tracebacks to the client.
        logger.exception("Unexpected error during synthesis")
        raise HTTPException(status_code=500, detail="Synthesis failed unexpectedly") from None
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
