"""Strict, hash-bound plan for offline H-G.1.1 Pilot analysis."""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .provenance_cascade_pilot_analysis_plan import EXPECTED_CONDITIONS, EXPECTED_METRICS

HG11_ANALYSIS_VERSION = "provenance_cascade_hg11_pilot_analysis.v1"
EXPECTED_HG11_SCENARIOS = (
    "cascade-hg1-false-majority",
    "cascade-hg1-true-minority-correction",
    "cascade-hg1-independent-true-consensus",
    "cascade-hg1-unresolved-disagreement",
)
EXPECTED_HG11_SEEDS = (20261001, 20261002, 20261003)


class HG11AnalysisPlanError(ValueError):
    """Stable, content-free plan error."""


class HG11TruthFixtureBinding(BaseModel):
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


class HG11AnalysisPlan(BaseModel):
    """Coordinates and hashes only; never evaluator-private values."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    analysis_id: Literal["provenance-cascade-hg11-pilot-analysis"]
    analysis_version: Literal["provenance_cascade_hg11_pilot_analysis.v1"]
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
    pilot_root: Literal["results/provenance-cascade-pilot-hg11-v1"]
    output_dir: Literal["results/analyses/provenance-cascade-hg11-pilot-evaluator-v1"]
    scenario_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    conditions: tuple[str, ...] = Field(min_length=4, max_length=4)
    seeds: tuple[int, ...] = Field(min_length=3, max_length=3)
    run_count: Literal[48]
    matched_group_count: Literal[12]
    logical_request_count: Literal[864]
    completion_reservation_cap: Literal[884736]
    primary_metrics: tuple[str, ...] = Field(min_length=9, max_length=9)
    thresholds_pre_registered: Literal[False]
    excluded_input_markers: tuple[str, ...]
    truth_fixtures: tuple[HG11TruthFixtureBinding, ...] = Field(min_length=4, max_length=4)

    @field_validator(
        "config_sha256", "protocol_sha256", "controller_sha256", "replay_sha256",
        "amendment_sha256", "approval_sha256", "compatibility_receipt_sha256",
        "amendment_receipt_sha256", "preregistration_sha256", "pilot_receipt_sha256",
        "batch_record_sha256",
    )
    @classmethod
    def digest(cls, value: str) -> str:
        value = value.lower()
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("analysis binding must be SHA-256")
        return value

    @model_validator(mode="after")
    def fixed_contract(self) -> "HG11AnalysisPlan":
        if self.scenario_ids != EXPECTED_HG11_SCENARIOS:
            raise ValueError("HG11 scenarios must use the locked order")
        if self.conditions != tuple(item.value for item in EXPECTED_CONDITIONS):
            raise ValueError("HG11 conditions must use the preregistered order")
        if self.seeds != EXPECTED_HG11_SEEDS:
            raise ValueError("HG11 seeds must use the completed Pilot seeds")
        if self.primary_metrics != EXPECTED_METRICS:
            raise ValueError("HG11 metrics must match the 24A preregistration")
        if {item.scenario_id for item in self.truth_fixtures} != set(self.scenario_ids):
            raise ValueError("truth bindings must cover every scenario exactly once")
        required = {"provenance-cascade-pilot-hg1-v1", "provenance-cascade-pilot-hd21-v1", "wvs", "002", "003", "calibration"}
        if not required.issubset(set(self.excluded_input_markers)):
            raise ValueError("legacy, WVS, and calibration exclusions must be explicit")
        return self


def load_hg11_analysis_plan(path: str | Path) -> HG11AnalysisPlan:
    try:
        return HG11AnalysisPlan.model_validate(tomllib.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise HG11AnalysisPlanError("hg11_analysis_plan_invalid") from exc


__all__ = [
    "EXPECTED_HG11_SCENARIOS", "EXPECTED_HG11_SEEDS", "HG11_ANALYSIS_VERSION",
    "HG11AnalysisPlan", "HG11AnalysisPlanError", "HG11TruthFixtureBinding", "load_hg11_analysis_plan",
]
