from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


class ConfigError(ValueError):
    """Raised when a configuration file cannot be loaded as a mapping."""


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    try:
        with config_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in configuration file: {config_path}") from exc

    if not isinstance(loaded, Mapping):
        raise ConfigError(f"Configuration file must contain a mapping: {config_path}")

    return dict(loaded)


def load_config(name: str, config_dir: str | Path = DEFAULT_CONFIG_DIR) -> dict[str, Any]:
    config_path = Path(config_dir) / f"{name}.yaml"
    return load_yaml_config(config_path)


def load_all_configs(config_dir: str | Path = DEFAULT_CONFIG_DIR) -> dict[str, dict[str, Any]]:
    return {
        name: load_config(name, config_dir=config_dir)
        for name in ("model", "dataset", "pipelines", "evaluation")
    }
