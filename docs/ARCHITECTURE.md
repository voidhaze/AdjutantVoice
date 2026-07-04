# Architecture

AdjutantVoice is a small, layered TTS toolkit built around a single OmniVoice
model instance. This document describes how the pieces fit together.

## Layers

```
┌─────────────────────────────────────────────────────────────────┐
│  Entry points                                                    │
│  av CLI (cli.py)   •   av-server (server.py)   •   av-mcp (mcp.py) │
└───────────────┬───────────────────────────────┬─────────────────┘
                │                               │
      ┌─────────▼─────────┐         ┌───────────▼───────────┐
      │  server.py         │         │  mcp.py                │
      │  FastAPI app        │         │  FastMCP server         │
      │  /synthesize         │         │  tts_file / tts_stream   │
      │  /v1/audio/speech      │         │  tts_speak                │
      │  /health                 │         │                            │
      └─────────┬───────────────┘         └───────────┬────────────────┘
                │                                     │
                └───────────────┬─────────────────────┘
                                │
                        ┌───────▼────────┐
                        │  tts.py          │
                        │  model lifecycle   │
                        │  load/unload/synth   │
                        └───────┬────────────┘
                                │
                        ┌───────▼────────┐
                        │  OmniVoice model │
                        │  (torch, GPU)      │
                        └────────────────────┘
```

`tts.py` is the single source of truth for model state and inference. Both
transport layers (`server.py`'s FastAPI app and `mcp.py`'s FastMCP server)
call into it rather than touching the model directly — this is what let the
test suite mock `tts.py` once and reuse that mock across CLI, HTTP, and MCP
tests.

## Model lifecycle

The OmniVoice model is a **process-wide singleton** (module-level globals in
`tts.py`, guarded by a `threading.Lock` during inference). It is loaded once,
on startup, by whichever transport is running:

- `server.py` loads it in a FastAPI `lifespan` context manager.
- `mcp.py` loads it in a FastMCP `lifespan` context manager.
- The CLI never loads the model itself — `av speak` / `av speak-file` are
  thin HTTP clients that expect a server to already be running.

`tts.load()` is idempotent: calling it when the model is already loaded is a
no-op. This matters because both lifespans call it unconditionally on
startup rather than checking `is_loaded()` themselves.

## Voice: clone vs. fallback

On load, `tts.py` looks for a voice-clone prompt at
`settings.voice_clone_path` (`~/.adjutantvoice/voices/default.pkl` by
default):

- **If found**, it's unpickled and used for every synthesis call
  (`model.generate(text=..., voice_clone_prompt=...)`).
- **If not found** (e.g. a fresh clone with no clone generated yet),
  `tts.py` falls back to OmniVoice's Voice Design mode, passing
  `instruct=settings.default_voice_instruct` instead.

`tts.using_voice_clone()` reports which mode is active, and it's surfaced
directly in the `/health` endpoint's `voice_mode` field (`"clone"` vs.
`"default"`) so it's easy to tell at a glance which voice a running server
is actually using.

Run `av voice create-clone` to generate the pickle from a reference audio
file (`voice.py`); this only needs to be done once per machine.

## Transports

### FastAPI server (`server.py`)

Two API surfaces on the same app:

- **Legacy**: `POST /synthesize` — used by the CLI and the Hermes command
  client (`clients/hermes.py`).
- **Open WebUI Custom TTS**: `GET /v1/models`, `GET /v1/audio/voices`,
  `POST /v1/audio/speech` — lets AdjutantVoice be configured directly as a
  TTS engine in Open WebUI.

Both surfaces funnel through the same `_synthesis_response()` helper, which
translates `tts.py`'s exceptions into HTTP status codes: `503` if the model
isn't loaded yet, `400` for blank/invalid text.

### MCP server (`mcp.py`)

Exposes three tools over the Model Context Protocol:

| Tool | Returns | Use case |
|---|---|---|
| `tts_file` | Absolute path to a written MP3 | Caller wants a file on disk |
| `tts_stream` | Raw MP3 bytes | Caller wants to handle audio data directly |
| `tts_speak` | Confirmation string | Caller wants audio played on the server machine |

Supports both `stdio` (for Claude Desktop / Claude Code) and
`streamable-http` (for remote clients) transports. `av install claude`
automates registering this server with Claude Desktop via `fastmcp install`.

### CLI (`cli.py`)

A `typer` app that's purely a client — it doesn't import `tts.py` at all.
`av speak` / `av speak-file` POST to a running server's `/synthesize`
endpoint and either play the result (via `ffplay`) or save it with
`--output`. Connection errors, timeouts, and HTTP errors are all caught and
turned into actionable one-line messages rather than raw tracebacks.

## Integrations

- **Claude Desktop**: `av install claude` runs `fastmcp install` against
  `mcp.py`, registering AdjutantVoice as an MCP server Claude Desktop can
  launch automatically.
- **Hermes agents**: `av install hermes` patches `~/.hermes/config.yaml`
  (via `integrations/hermes.py`) to add AdjutantVoice as a `type: command`
  TTS provider. The command it registers invokes
  `adjutantvoice.clients.hermes` — a minimal script that reads a text file,
  POSTs it to a running AdjutantVoice server, and writes the MP3 response —
  so no Hermes-specific plugin is required.
- **Open WebUI**: point Open WebUI's Custom TTS engine at
  `http://<host>:<port>/v1` — no extra integration code needed, since
  `server.py` implements the expected endpoints directly.

## Configuration

All tunables live in `config.py` as a single `pydantic-settings` `Settings`
object, overridable via `AV_*` environment variables (or a `.env` file). See
[CONFIGURATION.md](./CONFIGURATION.md) for the full reference.
