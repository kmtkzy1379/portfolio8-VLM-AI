"""Configuration loading and management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "vlm_default.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load YAML config, merging with defaults."""
    config = _load_yaml(_DEFAULT_CONFIG_PATH)
    if path is not None:
        override = _load_yaml(Path(path))
        config = _deep_merge(config, override)
    return config


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_nested(config: dict[str, Any], dotpath: str, default: Any = None) -> Any:
    """Get a nested config value using dot notation, e.g. 'capture.target_fps'."""
    keys = dotpath.split(".")
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
