"""Read-only, JSON-serializable result contracts for pure metrics."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..models._validation import Metadata, identifier_list, normalized_text


class MetricResult(BaseModel):
    """Generic provenance envelope that intentionally excludes prompts and raw text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_name: str = Field(min_length=1)
    value: float | None = None
    unit: str = Field(min_length=1)
    input_ids: list[str] = Field(default_factory=list)
    configuration: Metadata = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    valid: bool

    @field_validator("metric_name", "unit")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("input_ids")
    @classmethod
    def validate_input_ids(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "input_ids")


class AnswerDistanceResult(BaseModel):
    """Distance across overlapping answer items and the number used."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    distance: float = Field(ge=0.0, le=1.0)
    valid_question_count: int = Field(gt=0)


class PairDistance(BaseModel):
    """One valid normalized distance between two unique agents."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_a: str = Field(min_length=1)
    agent_b: str = Field(min_length=1)
    distance: float = Field(ge=0.0, le=1.0)

    @field_validator("agent_a", "agent_b")
    @classmethod
    def validate_agent_id(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "agent_id"))


class SkippedPair(BaseModel):
    """A pair omitted only under an explicit non-strict request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_a: str = Field(min_length=1)
    agent_b: str = Field(min_length=1)
    reason: str = Field(min_length=1)

    @field_validator("agent_a", "agent_b", "reason")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))


class PairwiseDiversityResult(BaseModel):
    """Mean over valid profile pairs, plus explicit skipped-pair provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean_distance: float = Field(ge=0.0, le=1.0)
    pair_distances: list[PairDistance]
    agent_ids: list[str]
    valid_pair_count: int = Field(ge=0)
    skipped_pair_count: int = Field(ge=0)
    skipped_pairs: list[SkippedPair] = Field(default_factory=list)


class MstEdge(BaseModel):
    """One canonical undirected edge selected by Kruskal's algorithm."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_a: str = Field(min_length=1)
    agent_b: str = Field(min_length=1)
    distance: float = Field(ge=0.0, le=1.0)


class StructuralDiversityResult(BaseModel):
    """MST span result, optionally invalid in non-strict disconnected graphs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mst_span: float | None = Field(default=None, ge=0.0, le=1.0)
    mst_total_length: float | None = Field(default=None, ge=0.0)
    mst_edges: list[MstEdge] = Field(default_factory=list)
    agent_ids: list[str]
    warnings: list[str] = Field(default_factory=list)
    valid: bool


class DimensionSummary(BaseModel):
    """Group-level descriptive summary for one declared value dimension."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum: float = Field(ge=0.0, le=1.0)
    maximum: float = Field(ge=0.0, le=1.0)
    mean: float = Field(ge=0.0, le=1.0)


class ValueDimensionCoverageResult(BaseModel):
    """Coverage is a thresholded expression summary, not a diversity score."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    expressed_dimensions: list[str]
    coverage_rate: float = Field(ge=0.0, le=1.0)
    dimension_summaries: dict[str, DimensionSummary]
    threshold: float = Field(ge=0.0, le=1.0)


class MinorityRetentionResult(BaseModel):
    """Retention of explicitly supplied minority dimensions only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    initial_minority_dimensions: list[str]
    retained_minority_dimensions: list[str]
    retention_rate: float = Field(ge=0.0, le=1.0)
    lost_dimensions: list[str]
    threshold: float = Field(ge=0.0, le=1.0)


class HoldoutProfileDriftResult(BaseModel):
    """Profile change per agent without an evaluative judgment about the change."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_drifts: dict[str, float]
    mean_drift: float = Field(ge=0.0, le=1.0)
    max_drift: float = Field(ge=0.0, le=1.0)


class SocialInfluenceLossResult(BaseModel):
    """Controlled matched-condition diversity contrast, not a causal proof."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_only_diversity: float = Field(ge=0.0, le=1.0)
    evidence_social_diversity: float = Field(ge=0.0, le=1.0)
    social_influence_loss: float = Field(ge=-1.0, le=1.0)
    interpretation: Literal[
        "social_diversity_lower",
        "no_diversity_difference",
        "social_diversity_higher",
    ]
    scope: Literal["exposed", "holdout"]
    warnings: list[str] = Field(default_factory=list)
