"""Tests for adjutantvoice.integrations.hermes — Hermes config patching."""

from ruamel.yaml import YAML

from adjutantvoice.integrations import hermes as hermes_integration


def _load(path):
    yaml = YAML()
    with open(path) as fh:
        return yaml.load(fh)


def test_install_creates_config_when_missing(tmp_path):
    config_path = tmp_path / "config.yaml"

    hermes_integration.install(hermes_config=config_path)

    assert config_path.exists()
    data = _load(config_path)
    assert data["tts"]["provider"] == "adjutantvoice"
    provider = data["tts"]["providers"]["adjutantvoice"]
    assert provider["type"] == "command"
    assert provider["output_format"] == "mp3"
    assert "adjutantvoice.clients.hermes" in provider["command"]


def test_install_preserves_unrelated_existing_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("agent:\n  name: my-agent\nother_setting: 42\n", encoding="utf-8")

    hermes_integration.install(hermes_config=config_path)

    data = _load(config_path)
    assert data["agent"]["name"] == "my-agent"
    assert data["other_setting"] == 42
    assert data["tts"]["provider"] == "adjutantvoice"


def test_install_replaces_existing_tts_provider(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "tts:\n  provider: some-other-provider\n  providers:\n    some-other-provider:\n      type: command\n",
        encoding="utf-8",
    )

    hermes_integration.install(hermes_config=config_path)

    data = _load(config_path)
    assert data["tts"]["provider"] == "adjutantvoice"
    # Prior provider entries aren't deleted, just no longer the active one.
    assert "some-other-provider" in data["tts"]["providers"]


def test_uninstall_removes_provider_and_active_selection(tmp_path):
    config_path = tmp_path / "config.yaml"
    hermes_integration.install(hermes_config=config_path)

    hermes_integration.uninstall(hermes_config=config_path)

    data = _load(config_path)
    assert "provider" not in data.get("tts", {})
    assert "adjutantvoice" not in data.get("tts", {}).get("providers", {})


def test_uninstall_on_missing_config_is_a_no_op(tmp_path, capsys):
    config_path = tmp_path / "does-not-exist.yaml"

    hermes_integration.uninstall(hermes_config=config_path)

    assert not config_path.exists()
    assert "nothing to do" in capsys.readouterr().out.lower()


def test_uninstall_leaves_other_providers_untouched(tmp_path):
    config_path = tmp_path / "config.yaml"
    hermes_integration.install(hermes_config=config_path)
    data = _load(config_path)
    data["tts"]["providers"]["some-other-provider"] = {"type": "command", "command": "echo hi"}
    yaml = YAML()
    with open(config_path, "w") as fh:
        yaml.dump(data, fh)

    hermes_integration.uninstall(hermes_config=config_path)

    data = _load(config_path)
    assert "some-other-provider" in data["tts"]["providers"]
