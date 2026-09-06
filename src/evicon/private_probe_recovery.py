"""Private, content-minimized recovery checkpoints for interrupted probe phases."""

from __future__ import annotations

import hashlib
import json
import os
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import ProtocolCondition, ValueProbeResponse
from .models._validation import identifier_list, normalized_text


CHECKPOINT_VERSION = "private_probe_recovery.v1"


class ProbeRecoveryPhase(str, Enum):
    """The two private probe phases surrounding a public trajectory."""

    PRE = "pre"
    POST = "post"


class PrivateProbeRecoveryError(ValueError):
    """Stable recovery failure without exposing response or prompt content."""


class PrivateProbeRecoveryCheckpoint(BaseModel):
    """A local response checkpoint with identifiers and hashes, never prompt text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_version: str = CHECKPOINT_VERSION
    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    protocol: ProtocolCondition
    probe_set_id: str = Field(min_length=1)
    probe_model_name: str = Field(min_length=1)
    probe_seed: int
    phase: ProbeRecoveryPhase
    target_round: int = Field(ge=0)
    is_holdout: bool
    agent_ids: list[str] = Field(min_length=1)
    probe_item_ids: list[str] = Field(min_length=1)
    context_sha256: list[str] = Field(min_length=1)
    responses: list[ValueProbeResponse] = Field(default_factory=list)

    @field_validator("checkpoint_version", "run_id", "scenario_id", "probe_set_id", "probe_model_name")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("agent_ids", "probe_item_ids")
    @classmethod
    def validate_ids(cls, value: list[str], info: object) -> list[str]:
        return identifier_list(value, getattr(info, "field_name", "ids"))

    @field_validator("probe_seed", mode="before")
    @classmethod
    def validate_seed(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError("probe_seed must be an integer")
        return value

    @field_validator("context_sha256")
    @classmethod
    def validate_hashes(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("context_sha256 values must be unique")
        for digest in value:
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("context_sha256 must contain SHA-256 digests")
        return value

    @model_validator(mode="after")
    def validate_alignment(self) -> "PrivateProbeRecoveryCheckpoint":
        if self.checkpoint_version != CHECKPOINT_VERSION:
            raise ValueError("checkpoint version is not supported")
        if len(self.context_sha256) != len(self.agent_ids):
            raise ValueError("context_sha256 must align with agent_ids")
        seen: set[tuple[str, str]] = set()
        for response in self.responses:
            coordinate = (response.agent_id, response.probe_id)
            if coordinate in seen:
                raise ValueError("checkpoint responses must not repeat an agent/probe coordinate")
            seen.add(coordinate)
            if response.agent_id not in self.agent_ids:
                raise ValueError("checkpoint response names an unknown agent")
            if response.probe_id not in self.probe_item_ids:
                raise ValueError("checkpoint response names an unknown probe item")
            if response.round_id != self.target_round:
                raise ValueError("checkpoint response round does not match target_round")
        return self

    def response_for(self, *, agent_id: str, probe_id: str) -> ValueProbeResponse | None:
        return next(
            (
                response
                for response in self.responses
                if response.agent_id == agent_id and response.probe_id == probe_id
            ),
            None,
        )

    def with_response(self, response: ValueProbeResponse) -> "PrivateProbeRecoveryCheckpoint":
        if self.response_for(agent_id=response.agent_id, probe_id=response.probe_id) is not None:
            raise PrivateProbeRecoveryError("private_probe_recovery_duplicate_response")
        return self.model_copy(update={"responses": [*self.responses, response]})


class PrivateProbeRecoveryStore:
    """Append logical responses atomically while keeping raw data outside public artifacts."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def open(
        self,
        identity: PrivateProbeRecoveryCheckpoint,
        *,
        resume: bool,
        require_existing: bool,
    ) -> PrivateProbeRecoveryCheckpoint:
        path = self.path_for(identity.phase)
        if path.exists():
            if not resume:
                raise PrivateProbeRecoveryError("private_probe_recovery_checkpoint_already_exists")
            loaded = self._read(path)
            if _identity_facts(loaded) != _identity_facts(identity):
                raise PrivateProbeRecoveryError("private_probe_recovery_identity_mismatch")
            return loaded
        if resume and require_existing:
            raise PrivateProbeRecoveryError("private_probe_recovery_checkpoint_missing")
        self.directory.mkdir(parents=True, exist_ok=True)
        self._write(path, identity)
        return identity

    def append(
        self,
        checkpoint: PrivateProbeRecoveryCheckpoint,
        response: ValueProbeResponse,
    ) -> PrivateProbeRecoveryCheckpoint:
        updated = checkpoint.with_response(response)
        self._write(self.path_for(checkpoint.phase), updated)
        return updated

    def clear(self) -> None:
        if not self.directory.exists():
            return
        for phase in ProbeRecoveryPhase:
            path = self.path_for(phase)
            if path.exists():
                path.unlink()
        try:
            self.directory.rmdir()
        except OSError:
            pass
        try:
            self.directory.parent.rmdir()
        except OSError:
            pass

    def path_for(self, phase: ProbeRecoveryPhase) -> Path:
        return self.directory / f"{phase.value}_responses.json"

    @staticmethod
    def _read(path: Path) -> PrivateProbeRecoveryCheckpoint:
        try:
            return PrivateProbeRecoveryCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PrivateProbeRecoveryError("private_probe_recovery_checkpoint_invalid") from exc

    @staticmethod
    def _write(path: Path, checkpoint: PrivateProbeRecoveryCheckpoint) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(checkpoint.model_dump_json(indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, path)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise PrivateProbeRecoveryError("private_probe_recovery_checkpoint_write_failed") from exc


def context_sha256(context: BaseModel) -> str:
    """Hash public context for recovery validation without storing it in the checkpoint."""
    serialized = json.dumps(
        context.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _identity_facts(checkpoint: PrivateProbeRecoveryCheckpoint) -> dict[str, object]:
    payload = checkpoint.model_dump(mode="json")
    payload.pop("responses")
    return payload


__all__ = [
    "CHECKPOINT_VERSION",
    "PrivateProbeRecoveryCheckpoint",
    "PrivateProbeRecoveryError",
    "PrivateProbeRecoveryStore",
    "ProbeRecoveryPhase",
    "context_sha256",
]
