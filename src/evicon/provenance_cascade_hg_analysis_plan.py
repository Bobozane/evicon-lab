"""Strict, hash-bound plan for the evaluator-only H-G Pilot analysis."""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .provenance_cascade_pilot_analysis_plan import EXPECTED_CONDITIONS, EXPECTED_METRICS

HG_ANALYSIS_VERSION = "provenance_cascade_hg_pilot_analysis.v1"
EXPECTED_HG_SCENARIOS = (
    "cascade-hg-false-majority",
    "cascade-hg-true-minority-correction",
    "cascade-hg-independent-true-consensus",
    "cascade-hg-unresolved-disagreement",
)
EXPECTED_HG_SEEDS = (20260911, 20260912, 20260913)
EXPECTED_HG_REPLAY_CONTRACT = "cascade_hg_outcome_replay.v2.visible_cross_claim_context"


class HGAnalysisPlanError(ValueError):
    """Stable, content-free plan error."""


class HGTruthFixtureBinding(BaseModel):
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


class HGAnalysisPlan(BaseModel):
    """The plan contains hashes and coordinates, never evaluator-private values."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    analysis_id: Literal["provenance-cascade-hg-pilot-analysis"]
    analysis_version: Literal["provenance_cascade_hg_pilot_analysis.v1"]
    status: Literal["offline_evaluator_only_analysis"]
    development_only: Literal[True]
    pilot_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    study_id: Literal["evicon-provenance-cascade-pilot-hg"]
    config_path: str
    config_sha256: str
    protocol_path: str
    protocol_sha256: str
    policy_path: str
    policy_sha256: str
    amendment_path: str
    amendment_sha256: str
    amendment_receipt_path: str
    amendment_receipt_sha256: str
    approval_path: str
    approval_sha256: str
    compatibility_receipt_path: str
    compatibility_receipt_sha256: str
    technical_replay_amendment_path: str
    technical_replay_amendment_sha256: str
    old_validator_path: str
    old_validator_sha256: str
    new_validator_path: str
    new_validator_sha256: str
    outcome_replay_contract_version: Literal["cascade_hg_outcome_replay.v2.visible_cross_claim_context"]
    pilot_receipt_path: str
    pilot_receipt_sha256: str
    batch_record_path: str
    batch_record_sha256: str
    pilot_root: Literal["results/provenance-cascade-pilot-hg-v1"]
    preregistration_path: str
    preregistration_sha256: str
    truth_fixtures: tuple[HGTruthFixtureBinding, ...] = Field(min_length=4, max_length=4)
    scenario_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    conditions: tuple[str, ...] = Field(min_length=4, max_length=4)
    seeds: tuple[int, ...] = Field(min_length=3, max_length=3)
    run_count: Literal[48]
    matched_group_count: Literal[12]
    logical_request_count: Literal[864]
    transport_attempt_count: Literal[864]
    completion_reservation_cap: Literal[442368]
    primary_metrics: tuple[str, ...] = Field(min_length=9, max_length=9)
    thresholds_pre_registered: Literal[False]
    output_dir: Literal["results/analyses/provenance-cascade-hg-pilot-evaluator-v1"]
    excluded_input_markers: tuple[str, ...]

    @field_validator(
        "config_sha256", "protocol_sha256", "policy_sha256", "amendment_sha256",
        "amendment_receipt_sha256", "approval_sha256", "compatibility_receipt_sha256",
        "technical_replay_amendment_sha256", "old_validator_sha256", "new_validator_sha256",
        "pilot_receipt_sha256", "batch_record_sha256", "preregistration_sha256",
    )
    @classmethod
    def digest(cls, value: str) -> str:
        value = value.lower()
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("analysis binding must be SHA-256")
        return value

    @model_validator(mode="after")
    def fixed_contract(self) -> "HGAnalysisPlan":
        if self.scenario_ids != EXPECTED_HG_SCENARIOS:
            raise ValueError("H-G analysis scenarios must use the registered order")
        if self.conditions != tuple(item.value for item in EXPECTED_CONDITIONS):
            raise ValueError("H-G analysis conditions must use the fixed order")
        if self.seeds != EXPECTED_HG_SEEDS:
            raise ValueError("H-G analysis seeds must use the completed Pilot seeds")
        if self.primary_metrics != EXPECTED_METRICS:
            raise ValueError("H-G metrics must match the 24A preregistration")
        if {item.scenario_id for item in self.truth_fixtures} != set(self.scenario_ids):
            raise ValueError("H-G truth bindings must match all scenarios")
        required = {
            "provenance-cascade-pilot-hd21-v1",
            "provenance-cascade-hd21-pilot-hf-v1",
            "wvs", "002", "003", "calibration",
        }
        if not required.issubset(set(self.excluded_input_markers)):
            raise ValueError("legacy, WVS, and calibration exclusions must be explicit")
        return self


def load_hg_analysis_plan(path: str | Path) -> HGAnalysisPlan:
    try:
        payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        return HGAnalysisPlan.model_validate(payload)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise HGAnalysisPlanError("hg_analysis_plan_invalid") from exc


__all__ = [
    "EXPECTED_HG_REPLAY_CONTRACT", "EXPECTED_HG_SCENARIOS", "EXPECTED_HG_SEEDS",
    "HG_ANALYSIS_VERSION", "HGAnalysisPlan", "HGAnalysisPlanError",
    "HGTruthFixtureBinding", "load_hg_analysis_plan",
]
