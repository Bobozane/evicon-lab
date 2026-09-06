"""Strict local configuration loading for deterministic protocol runs."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import RunConfig


class ConfigError(ValueError):
    """Raised when a local JSON or TOML run configuration is invalid."""


def load_run_config(path: str | Path) -> RunConfig:
    """Load one strict TOML or JSON configuration without environment access."""
    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read configuration {config_path}: {exc.strerror or exc}") from exc

    try:
        if config_path.suffix == ".toml":
            payload: Any = tomllib.loads(raw.decode("utf-8"))
        elif config_path.suffix == ".json":
            payload = json.loads(raw.decode("utf-8"))
        else:
            raise ConfigError("configuration must use a .toml or .json extension")
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot parse configuration {config_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ConfigError("configuration root must be an object")
    if "seed" in payload and type(payload["seed"]) is not int:
        raise ConfigError("seed must be an integer")
    try:
        return RunConfig.model_validate(payload)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigError(f"invalid configuration {config_path}: {details}") from exc
