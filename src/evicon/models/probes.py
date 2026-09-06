"""Value-probe contracts kept separate from dialogue state."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._validation import Metadata, identifier_list, normalized_text
from .agents import ValueProfile


class ValueProbeItem(BaseModel):
    """A structured item used to elicit one value dimension."""

    model_config = ConfigDict(extra="forbid")

    probe_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    dimension: str = Field(min_length=1)
    response_scale: list[str] = Field(min_length=2)
    is_holdout: bool
    reverse_scored: bool = False

    @field_validator("probe_id", "text", "dimension")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("response_scale")
    @classmethod
    def validate_response_scale(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "response_scale")

    def normalized_score_for(self, raw_response: str) -> float:
        """Map one declared scale label into the closed normalized range."""
        try:
            index = self.response_scale.index(raw_response)
        except ValueError as exc:
            raise ValueError(
                f"raw_response must be one of the declared response_scale values for {self.probe_id}"
            ) from exc
        return index / (len(self.response_scale) - 1)


class ValueProbeResponse(BaseModel):
    """A private response record; it is not a field of ``DialogueState``."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(min_length=1)
    probe_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    raw_response: str = Field(min_length=1)
    normalized_score: float

    @field_validator("agent_id", "probe_id", "raw_response")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("normalized_score")
    @classmethod
    def validate_normalized_score(cls, value: float) -> float:
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < 0.0 or normalized > 1.0:
            raise ValueError("normalized_score must be a finite float in [0, 1]")
        return normalized


class ProbeSet(BaseModel):
    """A strict, versioned collection of probe items with stable dimensions."""

    model_config = ConfigDict(extra="forbid")

    probe_set_id: str = Field(min_length=1)
    dimensions: list[str] = Field(min_length=1)
    items: list[ValueProbeItem] = Field(min_length=1)
    version: str = Field(min_length=1)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("probe_set_id", "version")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "dimensions")

    @model_validator(mode="after")
    def validate_item_coverage(self) -> "ProbeSet":
        item_ids = [item.probe_id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("items must not contain duplicate probe_id values")
        item_dimensions = {item.dimension for item in self.items}
        unknown_dimensions = item_dimensions - set(self.dimensions)
        if unknown_dimensions:
            raise ValueError(f"items name dimensions absent from dimensions: {sorted(unknown_dimensions)}")
        uncovered = set(self.dimensions) - item_dimensions
        if uncovered:
            raise ValueError(f"every dimension needs at least one probe item: {sorted(uncovered)}")
        return self

    def items_for_holdout(self, is_holdout: bool) -> list[ValueProbeItem]:
        """Return an ordered subset without exposing it to dialogue components."""
        return [item for item in self.items if item.is_holdout is is_holdout]


class ProbeRunConfig(BaseModel):
    """Serializable input for one isolated probe execution pass."""

    model_config = ConfigDict(extra="forbid")

    probe_set_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    agent_ids: list[str] = Field(min_length=1)
    model_name: str = Field(min_length=1)
    seed: int
    is_holdout: bool

    @field_validator("probe_set_id", "model_name")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("agent_ids")
    @classmethod
    def validate_agent_ids(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "agent_ids")


class ProbeResult(BaseModel):
    """Offline-only probe responses and their aggregated value profile."""

    model_config = ConfigDict(extra="forbid")

    probe_set_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    responses: list[ValueProbeResponse] = Field(min_length=1)
    value_profile: ValueProfile
    completed: bool
    is_holdout: bool

    @field_validator("probe_set_id", "agent_id")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @model_validator(mode="after")
    def validate_response_alignment(self) -> "ProbeResult":
        response_ids = [response.probe_id for response in self.responses]
        if len(response_ids) != len(set(response_ids)):
            raise ValueError("responses must not contain duplicate probe_id values")
        if any(
            response.agent_id != self.agent_id or response.round_id != self.round_id
            for response in self.responses
        ):
            raise ValueError("responses must match ProbeResult agent_id and round_id")
        if (
            self.value_profile.agent_id != self.agent_id
            or self.value_profile.round_id != self.round_id
        ):
            raise ValueError("value_profile must match ProbeResult agent_id and round_id")
        return self
