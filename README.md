# AdjutantVoice

OmniVoice TTS toolkit — persistent FastAPI server, Open WebUI Custom TTS compatibility, FastMCP server, and a unified CLI.

## Structure

```
src/adjutantvoice/
├── __init__.py
├── config.py          # all settings, overrideable via AV_* env vars
├── tts.py             # core synthesis engine (model loading, inference)
├── voice.py           # voice-clone creation utilities
├── server.py          # FastAPI server (legacy + Open WebUI /v1 endpoints)
├── mcp.py             # FastMCP server (tts_file, tts_stream, tts_speak)
├── cli.py             # unified `av` CLI entry point
├── clients/
│   └── hermes.py      # Hermes command client (invoked by Hermes agent)
└── integrations/
    └── hermes.py      # install/uninstall Hermes TTS config
```

## Install

```bash

# From PyPI (stable release)
sudo apt install pipx

pipx install adjutantvoice

# First thing you should do after install if you want to use a custom voice is to run the clone, then decide which service you want to use.
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

# Create a voice clone from reference audio
av voice create-clone [--ref-audio path/to/audio.mp3] [--output path/to/clone.pkl]

# Install as Hermes TTS provider
av install hermes
```

## Configuration

All settings can be overridden via environment variables prefixed `AV_`:

| Variable             | Default            | Description                  |
|---------------------|--------------------|------------------------------|
| `AV_HOST`           | `0.0.0.0`          | Server bind host             |
| `AV_PORT`           | `8111`             | Server bind port             |
| `AV_MCP_PORT`       | `8222`             | MCP server port (HTTP mode)  |
| `AV_DEVICE`         | `cuda:0`           | Torch device                 |
| `AV_DTYPE`          | `float16`          | Model dtype                  |
| `AV_VOICE_CLONE_PATH` | (bundled .pkl)   | Path to voice clone pickle   |

## Open WebUI

Set **TTS Engine** to `Custom TTS` and **API Base URL** to `http://<host>:<port>/v1`.

## MCP (Claude Desktop / Claude Code)

```bash
fastmcp install adjutantvoice.mcp --name "AdjutantVoice TTS"
```

## MCP (General)

The standard MCP server should work with most applications, take care to configure this correctly in your application settings.

## Hermes Agent TTS integration

Install via CLI, run the server and then enable "/voice on" and "/voice tts" inside hermes agent.
