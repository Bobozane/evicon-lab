"""Audit-safe manifest for a locally frozen English WVS 7 ProbeSet."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FrozenWVS7Manifest(BaseModel):
    """Metadata-only receipt; it never duplicates probe text or options."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: str = Field(min_length=1)
    material_status: Literal["frozen"]
    language_arm: Literal["english_core"]
    probe_set_id: str = Field(min_length=1)
    probe_set_filename: str = Field(min_length=1)
    probe_set_sha256: str = Field(min_length=64, max_length=64)
    item_count: int = Field(ge=1)
    variable_ids: list[str] = Field(min_length=1)
    source_sha256_prefixes: dict[str, str] = Field(min_length=1)
    acceptance_id: str = Field(min_length=1)
    review_id: str = Field(min_length=1)
    frozen_by: Literal["researcher_user_confirmed"]
    frozen_on: date
    wvs_official_factor_structure: Literal[False]
    q158_q163_secondary_analysis: Literal[True]
    special_code_summary: dict[str, str] = Field(min_length=1)

    @field_validator("probe_set_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        value = value.lower().strip()
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("probe_set_sha256 must be a SHA-256 digest")
        return value

    @field_validator("variable_ids")
    @classmethod
    def validate_variables(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("variable_ids must not contain duplicates")
        return value

    @field_validator("source_sha256_prefixes")
    @classmethod
    def validate_prefixes(cls, value: dict[str, str]) -> dict[str, str]:
        for filename, prefix in value.items():
            prefix = prefix.lower().strip()
            if len(prefix) != 12 or any(char not in "0123456789abcdef" for char in prefix):
                raise ValueError("source_sha256_prefixes must use 12-character hexadecimal prefixes")
            value[filename] = prefix
        return value


class FrozenWVS7ManifestError(ValueError):
    """Stable local manifest error."""


def sha256_file(path: str | Path) -> str:
    """Hash a local output file without putting its contents in a report."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FrozenWVS7ManifestError("cannot hash frozen ProbeSet") from exc
    return digest.hexdigest()


def write_frozen_manifest(manifest: FrozenWVS7Manifest, path: str | Path) -> None:
    destination = Path(path)
    if destination.exists():
        raise FrozenWVS7ManifestError("frozen manifest output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def load_frozen_manifest(path: str | Path) -> FrozenWVS7Manifest:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return FrozenWVS7Manifest.model_validate(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FrozenWVS7ManifestError("frozen manifest is invalid") from exc


__all__ = [
    "FrozenWVS7Manifest",
    "FrozenWVS7ManifestError",
    "load_frozen_manifest",
    "sha256_file",
    "write_frozen_manifest",
]
