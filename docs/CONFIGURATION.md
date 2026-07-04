# Configuration reference

All settings are defined in `src/adjutantvoice/config.py` as a single
`pydantic-settings` `Settings` class, instantiated once as the module-level
`settings` singleton and imported directly wherever it's needed
(`from adjutantvoice.config import settings`).

Every field can be overridden with an environment variable prefixed
`AV_`, or by placing values in a `.env` file in the working directory.
Unrecognized `AV_*` variables are silently ignored rather than raising an
error (`extra="ignore"`).

## Precedence

1. Environment variable (`AV_<FIELD_NAME>`, upper snake case)
2. `.env` file in the current working directory
3. Default value below

## Settings

| Field | Env var | Default | Notes |
|---|---|---|---|
| `host` | `AV_HOST` | `0.0.0.0` | Bind address for `av server start`. |
| `port` | `AV_PORT` | `8111` | HTTP port for the FastAPI server. |
| `mcp_port` | `AV_MCP_PORT` | `8222` | Port for the MCP server when using the `streamable-http` transport. Ignored for `stdio`. |
| `model_id` | `AV_MODEL_ID` | `k2-fsa/OmniVoice` | HuggingFace-style model identifier passed to `OmniVoice.from_pretrained()`. |
| `device` | `AV_DEVICE` | `cuda:0` | Torch device string for model placement. |
| `dtype` | `AV_DTYPE` | `float16` | One of `float16`, `bfloat16`, `float32`. Unrecognized values silently fall back to `float16` (see `tts.py` / `voice.py`). |
| `voice_clone_dir` | `AV_VOICE_CLONE_DIR` | `~/.adjutantvoice/voices` | Directory where clone `.pkl` files live. |
| `default_voice_clone_name` | `AV_DEFAULT_VOICE_CLONE_NAME` | `default` | Base filename (without `.pkl`) for the clone `tts.load()` looks for automatically. |
| `ref_audio_path` | `AV_REF_AUDIO_PATH` | `<package>/assets/adjutant-terran-advisor-quotes.mp3` | Default reference audio for `av voice create-clone` when `--ref-audio` isn't given. |
| `default_voice_instruct` | `AV_DEFAULT_VOICE_INSTRUCT` | `female` | OmniVoice Voice Design instruct string used when no clone is found. |
| `sample_rate` | `AV_SAMPLE_RATE` | `24000` | Sample rate (Hz) used when encoding synthesized audio to MP3. |
| `available_voices` | `AV_AVAILABLE_VOICES` | `["adjutant"]` | Voice list reported by `GET /v1/audio/voices` (Open WebUI). |
| `model_label` | `AV_MODEL_LABEL` | `omnivoice` | Model id reported by `GET /v1/models` and used as the default `model` field in `/v1/audio/speech` requests. |
| `tts_output_dir` | `AV_TTS_OUTPUT_DIR` | `tts_output` (relative to CWD) | Directory the MCP `tts_file` tool writes generated MP3s into. Created automatically on MCP server startup. |

## Derived value: `voice_clone_path`

Not itself an env-configurable field — it's a computed property:

```python
settings.voice_clone_path == settings.voice_clone_dir / f"{settings.default_voice_clone_name}.pkl"
```

So overriding either `AV_VOICE_CLONE_DIR` or `AV_DEFAULT_VOICE_CLONE_NAME`
changes where `tts.load()` looks for a clone, without needing a third
setting for the combined path.

## Example `.env`

```dotenv
AV_HOST=127.0.0.1
AV_PORT=9000
AV_DEVICE=cuda:1
AV_DTYPE=bfloat16
AV_VOICE_CLONE_DIR=/mnt/models/adjutantvoice/voices
```

## Notes for contributors

- `dtype` validation is lenient by design — an invalid string doesn't raise,
  it just falls back to `float16` at the point of use in `tts.py` / `voice.py`
  (`.get(settings.dtype, torch.float16)`). If you're debugging an
  unexpectedly-`float16` model load, check for typos in `AV_DTYPE` first.
- `available_voices` is a `list[str]`; when overriding via environment
  variable, pydantic-settings expects JSON (e.g.
  `AV_AVAILABLE_VOICES='["adjutant","narrator"]'`), not a comma-separated
  string.
