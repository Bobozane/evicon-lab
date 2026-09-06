"""Read-only replay validation for provenance-cascade exposure ledgers.

This is deliberately separate from :mod:`evicon.replay`; existing WVS replay
semantics and event formats are not changed.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .provenance_cascade import ProvenanceFixtureError, ProvenanceGraph, PublicScenarioLoader
from .provenance_cascade_exposure import ExposureLedger


class CascadeReplayStatus(str, Enum):
    PASSED = "passed"
    BLOCKED = "blocked"


class CascadeReplayReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CascadeReplayStatus
    error_code: str | None = None
    scenario_id: str | None = None
    agent_count: int = Field(ge=0)
    exposure_count: int = Field(ge=0)
    snapshot_count: int = Field(ge=0)
    root_relation_count: int = Field(ge=0)


class CascadeReplayValidationError(ValueError):
    """Stable non-sensitive replay validation failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ExposureLedgerLoader:
    """Read a local public ledger only; no evaluator fixture is accepted."""

    @staticmethod
    def load(path: str | Path) -> ExposureLedger:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            return ExposureLedger.model_validate(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise CascadeReplayValidationError("exposure_ledger_invalid") from exc


class CascadeReplayValidator:
    """Validate public exposure timing and snapshots without evaluator truth."""

    @staticmethod
    def validate(graph: ProvenanceGraph, ledger: ExposureLedger) -> CascadeReplayReport:
        try:
            ledger.validate_against_graph(graph)
        except ValueError as exc:
            raise CascadeReplayValidationError("cascade_replay_invalid") from exc
        return CascadeReplayReport(
            status=CascadeReplayStatus.PASSED,
            scenario_id=graph.scenario_id,
            agent_count=len(ledger.agent_ids),
            exposure_count=len(ledger.events),
            snapshot_count=len(ledger.snapshots),
            root_relation_count=sum(len(snapshot.visible_root_relations) for snapshot in ledger.snapshots),
        )

    @classmethod
    def validate_paths(cls, public_path: str | Path, ledger_path: str | Path) -> CascadeReplayReport:
        try:
            graph = PublicScenarioLoader.load(public_path)
            ledger = ExposureLedgerLoader.load(ledger_path)
        except (ProvenanceFixtureError, CascadeReplayValidationError) as exc:
            raise CascadeReplayValidationError("cascade_replay_input_invalid") from exc
        return cls.validate(graph, ledger)


def safe_file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "CascadeReplayReport", "CascadeReplayStatus", "CascadeReplayValidationError", "CascadeReplayValidator",
    "ExposureLedgerLoader", "safe_file_sha256",
]
