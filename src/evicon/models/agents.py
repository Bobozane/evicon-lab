"""Agent specifications and value-profile contracts."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._validation import Metadata, identifier_list, normalized_text


class AgentSpec(BaseModel):
    """Static information that identifies an agent within one experiment run."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    initial_value_labels: list[str] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("agent_id", "role")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("initial_value_labels")
    @classmethod
    def validate_initial_value_labels(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "initial_value_labels")


class ValueProfile(BaseModel):
    """An ordered, normalized snapshot of one agent's value dimensions."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    dimensions: list[str] = Field(min_length=1)
    scores: list[float] = Field(min_length=1)
    source: str = Field(min_length=1)
    probe_id: str | None = None

    @field_validator("agent_id", "source")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("probe_id")
    @classmethod
    def validate_optional_probe_id(cls, value: str | None) -> str | None:
        return None if value is None else normalized_text(value, "probe_id")

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "dimensions")

    @field_validator("scores")
    @classmethod
    def validate_scores(cls, value: list[float]) -> list[float]:
        normalized = [float(score) for score in value]
        if any(not math.isfinite(score) or score < 0.0 or score > 1.0 for score in normalized):
            raise ValueError("scores must be finite normalized floats in [0, 1]")
        return normalized

    @model_validator(mode="after")
    def validate_dimension_score_alignment(self) -> "ValueProfile":
        if len(self.dimensions) != len(self.scores):
            raise ValueError("dimensions and scores must contain the same number of items")
        return self

    @property
    def dimension_scores(self) -> dict[str, float]:
        """Return the dimension-to-score mapping without losing order on disk."""
        return dict(zip(self.dimensions, self.scores, strict=True))
