"""Strict, JSON-serializable contracts for offline evaluation inputs and reports."""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..metrics import MetricResult
from ..models import ProtocolCondition
from ..models._validation import Metadata, identifier_list, normalized_text


class EvaluationCondition(str, Enum):
    """Named experimental arms accepted by an evaluation manifest."""

    INDEPENDENT = "independent"
    SOCIAL_ONLY = "social_only"
    EVIDENCE_ONLY = "evidence_only"
    EVIDENCE_SOCIAL = "evidence_social"
    GENERIC_MEDIATOR = "generic_mediator"
    FIXED_PROMPT = "fixed_prompt"
    RANDOM_INTERVENTION = "random_intervention"
    EVICON = "evicon"


class MetricSuite(str, Enum):
    """Explicit collections of local-only metrics available to the evaluator."""

    BASELINE_DIVERSITY = "baseline_diversity"
    REPRESENTATION = "representation"
    HOLDOUT = "holdout"
    COUNTERFACTUAL = "counterfactual"
    AUDIT = "audit"
    ALL = "all"


class EvaluationStatus(str, Enum):
    """Lifecycle state for a persisted offline evaluation report."""

    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"


class EvaluationMetricOptions(BaseModel):
    """Optional, explicit inputs required by threshold-based metrics."""

    model_config = ConfigDict(extra="forbid")

    coverage_threshold: float | None = None
    minority_dimensions: list[str] = Field(default_factory=list)
    strict: bool = True

    @field_validator("coverage_threshold")
    @classmethod
    def validate_threshold(cls, value: float | None) -> float | None:
        if value is None:
            return None
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < 0.0 or normalized > 1.0:
            raise ValueError("coverage_threshold must be a finite float in [0, 1]")
        return normalized

    @field_validator("minority_dimensions")
    @classmethod
    def validate_minorities(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "minority_dimensions")


class EvaluationRunEntry(BaseModel):
    """One declared input run; paths are resolved relative to the manifest file."""

    model_config = ConfigDict(extra="forbid")

    entry_id: str | None = None
    run_id: str = Field(min_length=1)
    condition: EvaluationCondition
    protocol: ProtocolCondition
    run_record_path: str = Field(min_length=1)
    events_path: str = Field(min_length=1)
    probe_result_path: str | None = None
    seed: int
    model_name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    counterfactual_group_id: str = Field(min_length=1)
    allow_failed_for_audit: bool = False

    @field_validator(
        "run_id",
        "entry_id",
        "run_record_path",
        "events_path",
        "probe_result_path",
        "model_name",
        "role",
        "counterfactual_group_id",
    )
    @classmethod
    def validate_text(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return normalized_text(value, getattr(info, "field_name", "text"))

    @property
    def evaluation_entry_id(self) -> str:
        """Stable manifest-local identifier; role distinguishes one run's phases."""
        return self.entry_id or f"{self.run_id}::{self.role}"


class EvaluationManifest(BaseModel):
    """The sole source of pairing intent for an offline metric computation."""

    model_config = ConfigDict(extra="forbid")

    evaluation_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    metric_suite: list[MetricSuite] = Field(min_length=1)
    runs: list[EvaluationRunEntry] = Field(min_length=1)
    probe_sets: list[str] = Field(default_factory=list)
    metric_options: EvaluationMetricOptions = Field(default_factory=EvaluationMetricOptions)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("evaluation_id", "scenario_id")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("probe_sets")
    @classmethod
    def validate_probe_sets(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "probe_sets")

    @model_validator(mode="after")
    def validate_unique_entries(self) -> "EvaluationManifest":
        entry_ids = [entry.evaluation_entry_id for entry in self.runs]
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("runs must not contain duplicate evaluation entry identifiers")
        if len(self.metric_suite) != len(set(self.metric_suite)):
            raise ValueError("metric_suite must not contain duplicate values")
        return self


class InputFileAudit(BaseModel):
    """A content hash and byte count proving exactly which local input was read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(ge=0)

    @field_validator("path", "kind")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))


class MatchedRunPair(BaseModel):
    """An explicit, validated comparison pair; no file-name matching is used."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    comparison: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    left_run_id: str = Field(min_length=1)
    right_run_id: str = Field(min_length=1)
    probe_set_id: str | None = None
    matching_key: Metadata = Field(default_factory=dict)

    @field_validator("comparison", "scope", "left_run_id", "right_run_id", "probe_set_id")
    @classmethod
    def validate_text(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return normalized_text(value, getattr(info, "field_name", "text"))


class UnmatchedEntry(BaseModel):
    """A declared run that could not be used for one named comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    comparison: str = Field(min_length=1)
    reason: str = Field(min_length=1)

    @field_validator("run_id", "comparison", "reason")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))


class EvaluationReport(BaseModel):
    """Persisted, audit-oriented result of a no-network evaluation pass."""

    model_config = ConfigDict(extra="forbid")

    evaluation_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    metric_suite: list[MetricSuite]
    status: EvaluationStatus
    metrics: list[MetricResult] = Field(default_factory=list)
    matched_pairs: list[MatchedRunPair] = Field(default_factory=list)
    unmatched_entries: list[UnmatchedEntry] = Field(default_factory=list)
    input_audit: list[InputFileAudit] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("evaluation_id", "scenario_id")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))
