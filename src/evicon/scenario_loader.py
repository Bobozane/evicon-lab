"""Strict local loading for JSON and TOML ScenarioSpec files."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models.scenarios import ScenarioSpec


class ScenarioLoadError(ValueError):
    """Raised when a local scenario file cannot be parsed or validated."""


class ScenarioLoader:
    """Load a scenario without generating missing IDs or changing source data."""

    @staticmethod
    def load(path: str | Path) -> ScenarioSpec:
        scenario_path = Path(path)
        try:
            raw = scenario_path.read_bytes()
        except OSError as exc:
            raise ScenarioLoadError(
                f"cannot read scenario {scenario_path}: {exc.strerror or exc}"
            ) from exc

        try:
            if scenario_path.suffix == ".toml":
                payload: Any = tomllib.loads(raw.decode("utf-8"))
            elif scenario_path.suffix == ".json":
                payload = json.loads(raw.decode("utf-8"))
            else:
                raise ScenarioLoadError(
                    f"scenario {scenario_path} must use a .toml or .json extension"
                )
        except (UnicodeDecodeError, tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
            raise ScenarioLoadError(f"cannot parse scenario {scenario_path}: {exc}") from exc

        if not isinstance(payload, dict):
            raise ScenarioLoadError(f"invalid scenario {scenario_path}: root must be an object")
        try:
            return ScenarioSpec.model_validate(payload)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            )
            raise ScenarioLoadError(f"invalid scenario {scenario_path}: {details}") from exc
