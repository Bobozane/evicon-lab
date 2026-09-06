"""Evidence provenance, visibility, and exposure contracts."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._validation import can_view, identifier_list, normalized_text


class EvidenceCard(BaseModel):
    """A factual claim with explicit provenance and visibility constraints."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    source: str = Field(min_length=1)
    supports: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    introduced_round: int = Field(ge=0)
    visible_to: list[str] = Field(min_length=1)
    reliability: float

    @field_validator("evidence_id", "claim", "source")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("supports", "contradicts")
    @classmethod
    def validate_claim_links(cls, value: list[str], info: object) -> list[str]:
        return identifier_list(value, getattr(info, "field_name", "claim_links"))

    @field_validator("visible_to")
    @classmethod
    def validate_visibility(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "visible_to", allow_public_audience=True)

    @field_validator("reliability")
    @classmethod
    def validate_reliability(cls, value: float) -> float:
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < 0.0 or normalized > 1.0:
            raise ValueError("reliability must be a finite float in [0, 1]")
        return normalized

    @model_validator(mode="after")
    def validate_non_conflicting_links(self) -> "EvidenceCard":
        if set(self.supports) & set(self.contradicts):
            raise ValueError("supports and contradicts must not overlap")
        return self

    def is_available_at(self, round_id: int) -> bool:
        """Return whether this evidence had been introduced by ``round_id``."""
        if round_id < 0:
            raise ValueError("round_id must be non-negative")
        return self.introduced_round <= round_id

    def is_visible_to(self, agent_id: str) -> bool:
        """Return whether the card grants this agent access."""
        return can_view(self.visible_to, agent_id)


class EvidenceExposure(BaseModel):
    """A record that a named evidence card was shown in a particular round."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    exposed_to: list[str] = Field(min_length=1)
    exposure_reason: str = Field(min_length=1)

    @field_validator("evidence_id", "exposure_reason")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("exposed_to")
    @classmethod
    def validate_exposed_to(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "exposed_to", allow_public_audience=True)

    def is_exposed_to(self, agent_id: str) -> bool:
        """Return whether this record includes the supplied agent."""
        return can_view(self.exposed_to, agent_id)
