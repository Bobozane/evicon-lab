"""Strict, content-free plan contract for confirmatory v2 analysis."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .models import ProtocolCondition


ANALYSIS_VERSION = "wvs7_protocol_blind_v2_confirmatory_analysis.v1"
EXPECTED_SEEDS = (20260820, 20260821, 20260822)
EXPECTED_CONDITIONS = (
    ProtocolCondition.INDEPENDENT,
    ProtocolCondition.SOCIAL_ONLY,
    ProtocolCondition.EVIDENCE_ONLY,
    ProtocolCondition.EVIDENCE_SOCIAL,
)
EXPECTED_PRIMARY_METRICS = (
    "pairwise_diversity",
    "social_influence_loss",
    "profile_drift",
)
EXPECTED_FROZEN_PROBE_SHA256 = "181fc27816565e22d23689bb2fd3da5fe693db7943c734a0ccfc355aa40190f2"
EXPECTED_EXCLUDED_RUN_IDS = (
    "wvs7-real-baseline-pilot-seed-002",
    "wvs7-real-baseline-pilot-seed-003",
)


class ConfirmatoryAnalysisPlanError(ValueError):
    """Stable, safe failure for analysis-plan loading or validation."""


class ConfirmatoryAnalysisInputs(BaseModel):
    """The four batch-level inputs permitted before safe run-record metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_receipt_path: str = Field(min_length=1)
    completeness_audit_path: str = Field(min_length=1)
    offline_evaluation_report_path: str = Field(min_length=1)
    batch_manifest_path: str = Field(min_length=1)


class ConfirmatoryAnalysisPlan(BaseModel):
    """Pre-registered offline analysis scope. It contains no model content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: str = Field(min_length=1)
    analysis_id: str = Field(min_length=1)
    analysis_version: Literal["wvs7_protocol_blind_v2_confirmatory_analysis.v1"]
    status: Literal["confirmatory_analysis"]
    confirmatory: Literal[True]
    statistical_analysis: Literal[True]
    not_causal_conclusion: Literal[True]
    v1_002_excluded: Literal[True]
    v2_003_excluded: Literal[True]
    batch_id: str = Field(min_length=1)
    inputs: ConfirmatoryAnalysisInputs
    output_dir: str = Field(min_length=1)
    expected_plan_sha256: str = Field(min_length=64, max_length=64)
    expected_plan_lock_sha256: str = Field(min_length=64, max_length=64)
    expected_frozen_probe_set_sha256: str = Field(min_length=64, max_length=64)
    expected_test_scenario_sha256: str = Field(min_length=64, max_length=64)
    seeds: list[int] = Field(min_length=1)
    conditions: list[ProtocolCondition] = Field(min_length=1)
    primary_metrics: list[str] = Field(min_length=1)
    excluded_run_ids: list[str] = Field(min_length=1)

    @field_validator(
        "study_id", "analysis_id", "batch_id", "output_dir",
        "expected_plan_sha256", "expected_plan_lock_sha256",
        "expected_frozen_probe_set_sha256", "expected_test_scenario_sha256",
    )
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text values must not be blank")
        return value

    @field_validator(
        "expected_plan_sha256", "expected_plan_lock_sha256",
        "expected_frozen_probe_set_sha256", "expected_test_scenario_sha256",
    )
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.lower()
        if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError("expected hash values must be SHA-256 digests")
        return normalized

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, value: list[int]) -> list[int]:
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in value):
            raise ValueError("seeds must contain integers")
        if len(value) != len(set(value)):
            raise ValueError("seeds must not contain duplicates")
        return value

    @field_validator("primary_metrics", "excluded_run_ids")
    @classmethod
    def validate_unique_text(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("listed values must be unique non-empty strings")
        return normalized

    @model_validator(mode="after")
    def validate_fixed_confirmatory_contract(self) -> "ConfirmatoryAnalysisPlan":
        if self.analysis_version != ANALYSIS_VERSION:
            raise ValueError("analysis_version is not the preregistered confirmatory version")
        if tuple(self.seeds) != EXPECTED_SEEDS:
            raise ValueError("analysis must use exactly the three preregistered confirmatory seeds")
        if tuple(self.conditions) != EXPECTED_CONDITIONS:
            raise ValueError("analysis must use the four confirmatory conditions in fixed order")
        if tuple(self.primary_metrics) != EXPECTED_PRIMARY_METRICS:
            raise ValueError("analysis primary metrics are fixed")
        if self.expected_frozen_probe_set_sha256 != EXPECTED_FROZEN_PROBE_SHA256:
            raise ValueError("analysis frozen ProbeSet hash is not the audited v2 hash")
        if not set(EXPECTED_EXCLUDED_RUN_IDS).issubset(self.excluded_run_ids):
            raise ValueError("analysis must explicitly exclude v1 002 and v2 calibration 003")
        return self


def load_confirmatory_analysis_plan(path: str | Path) -> ConfirmatoryAnalysisPlan:
    """Load only local TOML plan metadata; it never accesses providers or results."""
    try:
        payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        return ConfirmatoryAnalysisPlan.model_validate(payload)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise ConfirmatoryAnalysisPlanError("confirmatory_analysis_plan_invalid") from exc


__all__ = [
    "ANALYSIS_VERSION",
    "EXPECTED_CONDITIONS",
    "EXPECTED_EXCLUDED_RUN_IDS",
    "EXPECTED_PRIMARY_METRICS",
    "EXPECTED_SEEDS",
    "ConfirmatoryAnalysisInputs",
    "ConfirmatoryAnalysisPlan",
    "ConfirmatoryAnalysisPlanError",
    "load_confirmatory_analysis_plan",
]
