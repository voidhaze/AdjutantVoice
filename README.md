# AdjutantVoice

AdjutantVoice is a local OmniVoice text-to-speech (TTS) server with voice cloning, an OpenAI-compatible speech API, FastMCP/MCP support, and integrations for Hermes Agent, Claude Desktop, Claude Code and other AI applications.

Run high-quality voice synthesis locally using the OmniVoice model without relying on a cloud TTS provider.

AdjutantVoice provides:

- Local TTS using OmniVoice and PyTorch
- Voice cloning from reference audio
- Persistent TTS server so the model is loaded once and reused
- FastAPI HTTP server
- OpenAI-compatible /v1/audio/speech API
- Model Context Protocol (MCP) server via FastMCP
- Claude Desktop and Claude Code integration
- Hermes Agent custom TTS integration
- Command-line interface (av)
- MP3 generation, streaming and server-side playback
- NVIDIA CUDA acceleration for local inference

AdjutantVoice is designed for developers who want a self-hosted, local TTS backend for AI agents and applications.

Instead of coupling an application directly to an individual TTS implementation, AdjutantVoice keeps the OmniVoice model running in a persistent server and exposes several standard interfaces:

AI application → OpenAI-compatible TTS API → AdjutantVoice → OmniVoice

or

AI agent → MCP → AdjutantVoice → OmniVoice

This makes the same local voice synthesis engine usable from multiple AI clients and agent frameworks.

## Structure

```
src/adjutantvoice/
├── __init__.py
├── config.py          # all settings, overrideable via AV_* env vars
├── tts.py             # core synthesis engine (model loading, inference)
├── voice.py           # voice-clone creation utilities
├── server.py          # FastAPI server (legacy + OpenAI-compatible /v1 audio endpoints)
├── mcp.py             # FastMCP server (tts_file, tts_stream, tts_speak)
├── cli.py             # unified `av` CLI entry point
├── assets/
│   └── models/
│       └── default.pkl  # bundled voice clone — used out of the box until you make your own
├── clients/
│   └── hermes.py      # Hermes command client (invoked by Hermes agent)
└── integrations/
    └── hermes.py      # install/uninstall Hermes TTS config
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the layers fit together (transports → `tts.py` → model)
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) — full `AV_*` settings reference and precedence rules
- [docs/TESTING.md](docs/TESTING.md) — running the suite without a GPU, test layout, fixtures

## Supported Platforms

Currently this is only supported on Linux / Ubuntu

## Install

```bash

# From PyPI (stable release)
sudo apt update
sudo apt install pipx
pipx ensurepath

pipx install adjutantvoice

av voice create-clone --ref-audio path/to/my_reference_voice.mp3

```

## Uninstall

```bash

# Using Pipx
pipx uninstall adjutantvoice


```

## Development

```bash

# Install
git clone https://github.com/voidhaze/AdjutantVoice.git

sudo apt install python3-venv

python3 -m venv .venv/adjutantvoice

source .venv/adjutantvoice/bin/activate

pip install ".[dev]"


# Uninstall
cd ~/your_dev_git_dir/AdjutantVoice

source .venv/adjutantvoice/bin/activate

pip uninstall adjutantvoice

# deactivate venv
deactivate

```

## CLI

```bash
# Print a guided getting-started walkthrough
av tutorial

# Start the TTS HTTP server
av server start [--host HOST] [--port PORT]

# Start the MCP server
# Use 'stdio' for Claude Desktop / Claude Code, 
# Use 'streamable-http' (default) for remote HTTP clients.
av mcp start
av mcp start [--transport stdio]
av mcp start [--transport streamable-http] [--port PORT]

# Synthesise text (requires server running)
av speak "The fleet is prepped and ready, Commander."
av speak "Hello" --output hello.mp3

# Synthesise from a text file
av speak-file script.txt --output script.mp3

# Create a voice clone from reference audio (--ref-audio is required)
av voice create-clone --ref-audio path/to/audio.mp3 [--output path/to/clone.pkl]

# Register AdjutantVoice as an MCP server in Claude Desktop
av install claude [--name "AdjutantVoice TTS"]

# Install as Hermes TTS provider
av install hermes
```

### Shell completion

`av` supports tab completion for commands, sub-commands (`server`, `voice`,
`mcp`, `install`, …), and option names, powered by Typer/Click.

There's no such thing as a pip/pipx "post-install hook" for wheels, so this
can't be wired up automatically at install time. Instead, the first time you
run any real `av` command from an interactive terminal, it offers to set
this up for you (a one-time prompt, recorded so it won't ask again):

```
Enable tab completion for the `av` command in your shell? [Y/n]
```

Answering yes runs the same thing as `av --install-completion` below —
detects your shell and installs a completion script into the appropriate
rc/config file. Restart your shell (or re-source the rc file) afterwards.

You can also trigger or manage this manually at any point:

```bash
# Auto-detects your shell (bash, zsh, fish, PowerShell) and installs a
# completion script into the appropriate rc/config file.
av --install-completion

# Print the completion script instead of installing it, e.g. to review it
# first or wire it up yourself.
av --show-completion
```

If shell auto-detection fails, pass the shell explicitly:
`av --install-completion bash` (or `zsh` / `fish` / `powershell`).

Set `AV_SKIP_COMPLETION_PROMPT=1` to suppress the one-time prompt entirely
(e.g. in scripted/non-interactive environments where it wouldn't fire
anyway, or if you just don't want to be asked).

## Configuration

All settings can be overridden via environment variables prefixed `AV_` (or a
`.env` file in the working directory). The common ones:

| Variable             | Default            | Description                  |
|---------------------|--------------------|------------------------------|
| `AV_HOST`           | `0.0.0.0`          | FastAPI server bind host      |
| `AV_PORT`           | `8111`             | FastAPI server port          |
| `AV_MCP_PORT`       | `8222`             | MCP server port (`streamable-http` transport) |
| `AV_MODEL_ID`       | `k2-fsa/OmniVoice` | HuggingFace-style model identifier |
| `AV_DEVICE`         | `cuda:0`           | Torch device                 |
| `AV_DTYPE`          | `float16`          | Model dtype (`float16` / `bfloat16` / `float32`) |
| `AV_VOICE_CLONE_DIR` | `~/.adjutantvoice/voices` | Directory for user-generated voice clones |
| `AV_DEFAULT_VOICE_CLONE_NAME` | `default` | Clone filename (without `.pkl`) loaded automatically |
| `AV_DEFAULT_VOICE_INSTRUCT` | `female`   | Voice Design instruct used when no clone is found |
| `AV_SAMPLE_RATE`    | `24000`            | Sample rate (Hz) for MP3 encoding |
| `AV_TTS_OUTPUT_DIR` | `tts_output`       | Where the MCP `tts_file` tool writes MP3s |
| `AV_SKIP_COMPLETION_PROMPT` | (unset)   | Set to skip the one-time "enable tab completion?" prompt on first interactive `av` use |

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the complete list
(including `AV_AVAILABLE_VOICES`, `AV_MODEL_LABEL`, `AV_BUNDLED_VOICE_CLONE_PATH`,
`AV_COMPLETION_PROMPT_MARKER`), the derived `voice_clone_path`, and the
clone-resolution order.

## MCP (Claude Desktop / Claude Code)

```bash
# Registers the MCP server with Claude Desktop by running `fastmcp install`
# against adjutantvoice/mcp.py. Restart Claude Desktop afterwards.
av install claude [--name "AdjutantVoice TTS"]
```

For `stdio` clients that launch the server themselves (e.g. Claude Code),
point them at `av mcp start --transport stdio`.

## MCP (General)

The standard MCP server should work with most applications, take care to configure this correctly in your application settings.

## Hermes Agent TTS integration

Install via CLI, run the server and then enable "/voice on" and "/voice tts" inside hermes agent.

## Use cases

AdjutantVoice can be used as a local voice backend for:

- AI coding agents — give Hermes Agent, Claude Code and other agents spoken responses.
- Claude Desktop — expose local text-to-speech through MCP.
- OpenAI-compatible applications — use AdjutantVoice anywhere an application supports the OpenAI speech API.
- Open WebUI — connect a local OmniVoice backend as a custom TTS provider.
- Voice assistants — use the persistent HTTP API as a local speech synthesis service.
- Voice cloning experiments — create and reuse a voice clone from reference audio.
- Local/private AI systems — perform TTS locally without sending text or voice data to a cloud provider.
