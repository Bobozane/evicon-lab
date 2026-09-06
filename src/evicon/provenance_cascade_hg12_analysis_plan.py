"""Hash-bound, evaluator-only analysis plan for the completed H-G.1.2 Pilot."""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .provenance_cascade_pilot_analysis_plan import EXPECTED_CONDITIONS, EXPECTED_METRICS

HG12_ANALYSIS_VERSION = "provenance_cascade_hg12_pilot_analysis.v1"
EXPECTED_HG12_SCENARIOS = (
    "cascade-hg1-false-majority",
    "cascade-hg1-true-minority-correction",
    "cascade-hg1-independent-true-consensus",
    "cascade-hg1-unresolved-disagreement",
)
EXPECTED_HG12_SEEDS = (20261011, 20261012, 20261013)


class HG12AnalysisPlanError(ValueError):
    """Stable, content-free plan error."""


class HG12TruthFixtureBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def digest(cls, value: str) -> str:
        value = value.lower()
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("truth fixture hash must be SHA-256")
        return value


class HG12AnalysisPlan(BaseModel):
    """Coordinates and hashes only; never evaluator-private values."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    analysis_id: Literal["provenance-cascade-hg12-pilot-analysis"]
    analysis_version: Literal["provenance_cascade_hg12_pilot_analysis.v1"]
    status: Literal["offline_evaluator_only_analysis"]
    development_only: Literal[True]
    pilot_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    config_path: str
    config_sha256: str
    protocol_path: str
    protocol_sha256: str
    controller_path: str
    controller_sha256: str
    replay_path: str
    replay_sha256: str
    amendment_path: str
    amendment_sha256: str
    approval_path: str
    approval_sha256: str
    compatibility_receipt_path: str
    compatibility_receipt_sha256: str
    amendment_receipt_path: str
    amendment_receipt_sha256: str
    preregistration_path: str
    preregistration_sha256: str
    pilot_receipt_path: str
    pilot_receipt_sha256: str
    batch_record_path: str
    batch_record_sha256: str
    pilot_root: str
    output_dir: str
    scenario_ids: tuple[str, ...]
    conditions: tuple[str, ...]
    seeds: tuple[int, ...]
    run_count: Literal[48]
    matched_group_count: Literal[12]
    logical_request_count: Literal[864]
    completion_reservation_cap: Literal[884736]
    primary_metrics: tuple[str, ...]
    thresholds_pre_registered: Literal[False]
    excluded_input_markers: tuple[str, ...]
    truth_fixtures: tuple[HG12TruthFixtureBinding, ...]

    @field_validator(
        "config_sha256", "protocol_sha256", "controller_sha256", "replay_sha256",
        "amendment_sha256", "approval_sha256", "compatibility_receipt_sha256",
        "amendment_receipt_sha256", "preregistration_sha256", "pilot_receipt_sha256",
        "batch_record_sha256",
    )
    @classmethod
    def hashes(cls, value: str) -> str:
        value = value.lower()
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("invalid SHA-256")
        return value

    @model_validator(mode="after")
    def fixed_design(self) -> "HG12AnalysisPlan":
        if self.scenario_ids != EXPECTED_HG12_SCENARIOS:
            raise ValueError("scenario set mismatch")
        if self.conditions != tuple(item.value for item in EXPECTED_CONDITIONS):
            raise ValueError("condition set mismatch")
        if self.seeds != EXPECTED_HG12_SEEDS:
            raise ValueError("seed set mismatch")
        if self.primary_metrics != EXPECTED_METRICS:
            raise ValueError("metric suite mismatch")
        if len(self.truth_fixtures) != 4 or {item.scenario_id for item in self.truth_fixtures} != set(self.scenario_ids):
            raise ValueError("truth fixture set mismatch")
        required_exclusions = ("hg11", "hg1-v1", "hd21", "wvs", "002", "003", "calibration")
        lowered = " ".join(self.excluded_input_markers).lower()
        if any(marker not in lowered for marker in required_exclusions):
            raise ValueError("legacy exclusion incomplete")
        if self.pilot_root != "results/provenance-cascade-pilot-hg12-v1":
            raise ValueError("pilot root mismatch")
        if self.output_dir.startswith(self.pilot_root):
            raise ValueError("analysis output overlaps pilot")
        return self


def load_hg12_analysis_plan(path: str | Path) -> HG12AnalysisPlan:
    try:
        return HG12AnalysisPlan.model_validate(tomllib.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise HG12AnalysisPlanError("hg12_analysis_plan_invalid") from exc


__all__ = [
    "EXPECTED_HG12_SCENARIOS", "EXPECTED_HG12_SEEDS", "HG12_ANALYSIS_VERSION",
    "HG12AnalysisPlan", "HG12AnalysisPlanError", "HG12TruthFixtureBinding",
    "load_hg12_analysis_plan",
]
