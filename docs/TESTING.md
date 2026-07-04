# Testing

## Running the suite

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Note that `torch` and `omnivoice` are listed as required dependencies in
`[project.dependencies]`, so a plain `pip install -e ".[dev]"` will pull
both in. That's fine on a GPU machine (e.g. Lucy), but if you want a fast,
GPU-free install just for running tests — on a laptop or in CI — install the
runtime deps individually instead and skip those two:

```bash
pip install fastapi "uvicorn[standard]" pydantic pydantic-settings \
    soundfile typer requests fastmcp "ruamel.yaml" pytest pytest-mock httpx httpx2
pip install -e . --no-deps
```

The test suite mocks `torch` and `omnivoice` regardless (see below), so it
runs identically either way.

## Why no GPU is required to run tests

OmniVoice inference needs a GPU and real model weights — neither of which
are available in CI or on most contributors' laptops. Rather than skip
model-dependent tests entirely, the suite mocks the model at two levels:

1. **Import-time stubs** (`tests/conftest.py`): before any test module
   imports `adjutantvoice.tts` or `adjutantvoice.voice` (both of which do
   `import torch` and `from omnivoice import OmniVoice` unconditionally),
   `conftest.py` inserts lightweight `MagicMock` stand-ins into
   `sys.modules["torch"]` / `sys.modules["omnivoice"]`. This lets those
   modules import cleanly with neither package actually installed.

2. **Per-test model mocks** (`fake_model` / `fake_omnivoice_cls` fixtures):
   individual tests monkeypatch `tts.OmniVoice` / `voice.OmniVoice` with a
   `MagicMock` whose `.from_pretrained()` returns a fake model. The fake
   model's `.generate()` returns a small placeholder "audio array," and
   `sf.write` is monkeypatched too where the test cares about the resulting
   bytes — so no real audio encoding happens either.

This means the tests verify AdjutantVoice's own logic (model lifecycle,
clone-vs-fallback selection, error handling, HTTP status codes, CLI
argument parsing, config file patching) without ever touching a GPU or real
model weights. It does **not** verify that OmniVoice itself behaves as
expected, or that the model actually sounds right — that still needs manual
testing against a real GPU machine.

## Test layout

| File | Covers |
|---|---|
| `test_config.py` | `Settings` defaults, env var overrides, derived `voice_clone_path` |
| `test_tts.py` | Model lifecycle (`load`/`unload`/`is_loaded`), clone-vs-fallback selection, `synthesize` error paths |
| `test_voice.py` | `create_voice_clone` pickle output, default paths, directory creation |
| `test_server.py` | FastAPI endpoints (`/health`, `/v1/*`, `/synthesize`), status codes for unloaded model / blank text |
| `test_mcp.py` | MCP tool functions (`tts_file`, `tts_stream`, `tts_speak`) called directly |
| `test_cli.py` | `av speak` / `speak-file` / `voice create-clone` / `install` via Typer's `CliRunner` |
| `test_hermes_client.py` | The Hermes command-client script (`clients/hermes.py`) |
| `test_hermes_integration.py` | Hermes config file patching (`integrations/hermes.py`) — install/uninstall, preserving unrelated config |

## Fixtures (in `conftest.py`)

- `reset_tts_state` *(autouse)* — calls `tts.unload()` before and after every
  test, since `tts.py`'s model state is process-wide module globals. Without
  this, one test's `tts.load()` would leak into every test that runs after it.
- `tmp_settings` — points `settings.voice_clone_dir` and
  `settings.tts_output_dir` at `tmp_path` so tests never touch a real
  `~/.adjutantvoice` directory.
- `fake_model` / `fake_omnivoice_cls` — the mocked model described above.

## Gotchas if you're adding tests

- **Typer errors go to stderr.** `CliRunner`'s `result.stdout` will be empty
  for messages printed via `typer.echo(..., err=True)` (which is how most of
  the CLI's error paths work). Use `result.output` instead — it merges
  stdout and stderr.
- **`server.py`'s `lifespan` isn't triggered** in `test_server.py` — the
  tests use a plain `TestClient(app)` without a `with` block, so
  startup/shutdown events never fire, and each test loads/unloads `tts`
  state explicitly instead. If you add a test using
  `with TestClient(app) as client:`, be aware it *will* trigger a real
  `tts.load()` against whatever `tts.OmniVoice` currently is at that point.
- **`@mcp.tool(...)` returns the plain function** (verified against the
  installed `fastmcp` version), so MCP tools can be called directly in
  tests without spinning up a real MCP transport.
