"""Strict local loading for versioned ProbeSet files."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import ProbeSet


class ProbeLoadError(ValueError):
    """Raised when a local probe configuration cannot be parsed or validated."""


class ProbeSetLoader:
    """Load one probe set without generating items or repairing invalid source data."""

    @staticmethod
    def load(path: str | Path) -> ProbeSet:
        probe_path = Path(path)
        try:
            raw = probe_path.read_bytes()
        except OSError as exc:
            raise ProbeLoadError(f"cannot read probe set {probe_path}: {exc}") from exc

        try:
            if probe_path.suffix == ".toml":
                payload: Any = tomllib.loads(raw.decode("utf-8"))
            elif probe_path.suffix == ".json":
                payload = json.loads(raw.decode("utf-8"))
            else:
                raise ProbeLoadError(f"probe set {probe_path} must use a .toml or .json extension")
        except (UnicodeDecodeError, tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
            raise ProbeLoadError(f"cannot parse probe set {probe_path}: {exc}") from exc

        if not isinstance(payload, dict):
            raise ProbeLoadError(f"invalid probe set {probe_path}: root must be an object")
        try:
            return ProbeSet.model_validate(payload)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            )
            raise ProbeLoadError(f"invalid probe set {probe_path}: {details}") from exc
