"""Strict offline plan for the H-D.2.1 provenance-cascade Pilot analysis."""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .provenance_cascade_preregistration import CascadeCondition

ANALYSIS_VERSION = "provenance_cascade_pilot_analysis.v1"
EXPECTED_CONDITIONS = tuple(CascadeCondition)
EXPECTED_SEEDS = (20260911, 20260912, 20260913)
EXPECTED_METRICS = (
    "false_cascade_adoption_rate", "supported_correction_retention",
    "beneficial_receptivity", "harmful_conformity", "intervention_false_positive_rate",
    "provenance_diversity", "replay_audit_pass_rate", "token_cost", "latency_ms",
)


class ProvenanceCascadePilotAnalysisPlanError(ValueError):
    """Stable failure for local plan parsing."""


class TruthFixtureBinding(BaseModel):
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


class ProvenanceCascadePilotAnalysisPlan(BaseModel):
    """Hash-bound analysis plan; it never contains private truth values."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    analysis_id: str = Field(min_length=1)
    analysis_version: Literal["provenance_cascade_pilot_analysis.v1"]
    status: Literal["offline_pilot_analysis"]
    development_only: Literal[True]
    pilot_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    study_id: Literal["evicon-provenance-cascade-pilot-hd21"]
    config_path: str
    config_sha256: str
    protocol_path: str
    protocol_sha256: str
    template_path: str
    template_sha256: str
    pilot_receipt_path: str
    pilot_receipt_sha256: str
    batch_record_path: str
    batch_record_sha256: str
    pilot_root: str
    preregistration_path: str
    preregistration_sha256: str
    truth_fixtures: tuple[TruthFixtureBinding, ...] = Field(min_length=4, max_length=4)
    scenario_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    conditions: tuple[CascadeCondition, ...] = Field(min_length=4, max_length=4)
    seeds: tuple[int, ...] = Field(min_length=3, max_length=3)
    run_count: Literal[48]
    matched_group_count: Literal[12]
    primary_metrics: tuple[str, ...] = Field(min_length=9, max_length=9)
    thresholds_pre_registered: Literal[False]
    output_dir: str
    excluded_input_markers: tuple[str, ...]

    @field_validator(
        "config_sha256", "protocol_sha256", "template_sha256",
        "pilot_receipt_sha256", "batch_record_sha256", "preregistration_sha256",
    )
    @classmethod
    def digest(cls, value: str) -> str:
        value = value.lower()
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("analysis binding must be SHA-256")
        return value

    @model_validator(mode="after")
    def fixed_contract(self) -> "ProvenanceCascadePilotAnalysisPlan":
        if self.conditions != EXPECTED_CONDITIONS:
            raise ValueError("analysis conditions must use the fixed order")
        if self.seeds != EXPECTED_SEEDS:
            raise ValueError("analysis seeds must match H-D.2.1")
        if self.primary_metrics != EXPECTED_METRICS:
            raise ValueError("analysis metrics must match the 24A preregistration")
        if len(set(self.scenario_ids)) != 4 or len({item.scenario_id for item in self.truth_fixtures}) != 4:
            raise ValueError("analysis requires four unique scenario and truth bindings")
        if set(self.scenario_ids) != {item.scenario_id for item in self.truth_fixtures}:
            raise ValueError("truth fixture scenarios must match the analysis scenarios")
        if self.pilot_root != "results/provenance-cascade-pilot-hd21-v1":
            raise ValueError("analysis may only consume the H-D.2.1 Pilot root")
        if not self.output_dir.startswith("results/analyses/"):
            raise ValueError("analysis output must be isolated below results/analyses")
        required = {"provenance-cascade-pilot-hd2-v1", "wvs", "002", "003", "calibration"}
        if not required.issubset(set(self.excluded_input_markers)):
            raise ValueError("legacy and calibration exclusions must be explicit")
        return self


def load_provenance_cascade_pilot_analysis_plan(path: str | Path) -> ProvenanceCascadePilotAnalysisPlan:
    try:
        payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        return ProvenanceCascadePilotAnalysisPlan.model_validate(payload)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise ProvenanceCascadePilotAnalysisPlanError("pilot_analysis_plan_invalid") from exc


__all__ = [
    "ANALYSIS_VERSION", "EXPECTED_CONDITIONS", "EXPECTED_METRICS", "EXPECTED_SEEDS",
    "ProvenanceCascadePilotAnalysisPlan", "ProvenanceCascadePilotAnalysisPlanError",
    "TruthFixtureBinding", "load_provenance_cascade_pilot_analysis_plan",
]
