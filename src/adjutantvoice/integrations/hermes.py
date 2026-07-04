"""
Hermes agent integration — install AdjutantVoice as the default TTS provider.
"""

from __future__ import annotations

import sys
from pathlib import Path

from adjutantvoice.config import settings

HERMES_CONFIG = Path.home() / ".hermes" / "config.yaml"


def install(hermes_config: Path = HERMES_CONFIG) -> None:
    """Patch ~/.hermes/config.yaml to use AdjutantVoice as the TTS provider.

    Args:
        hermes_config: Path to the Hermes config file.
    """
    try:
        from ruamel.yaml import YAML
    except ImportError:
        sys.exit("ERROR: ruamel.yaml is required. Run: pip install ruamel.yaml")

    # Resolve the installed CLI script path
    python = Path(sys.executable)
    # The hermes client script reads a file path and posts to the server
    client_module = "adjutantvoice.clients.hermes"

    yaml = YAML()
    yaml.preserve_quotes = True
    hermes_config.parent.mkdir(parents=True, exist_ok=True)
    config = yaml.load(hermes_config) if hermes_config.exists() else {}
    if config is None:
        config = {}

    config.setdefault("tts", {})
    config["tts"]["provider"] = "adjutantvoice"
    config["tts"].setdefault("providers", {})
    config["tts"]["providers"]["adjutantvoice"] = {
        "type": "command",
        "command": f"{python} -m {client_module} {{input_path}} {{output_path}}",
        "output_format": "mp3",
        "timeout": 60,
    }

    with open(hermes_config, "w") as fh:
        yaml.dump(config, fh)

    print(f"✓ Hermes config updated: {hermes_config}")
    print(f"  TTS provider → adjutantvoice")
    print(f"  Python: {python}")


def uninstall(hermes_config: Path = HERMES_CONFIG) -> None:
    """Remove AdjutantVoice from the Hermes TTS providers config."""
    try:
        from ruamel.yaml import YAML
    except ImportError:
        sys.exit("ERROR: ruamel.yaml is required. Run: pip install ruamel.yaml")

    if not hermes_config.exists():
        print("No Hermes config found — nothing to do.")
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    config = yaml.load(hermes_config) or {}

    providers = config.get("tts", {}).get("providers", {})
    if "adjutantvoice" in providers:
        del providers["adjutantvoice"]

    if config.get("tts", {}).get("provider") == "adjutantvoice":
        del config["tts"]["provider"]

    with open(hermes_config, "w") as fh:
        yaml.dump(config, fh)

    print(f"✓ AdjutantVoice removed from Hermes config: {hermes_config}")
