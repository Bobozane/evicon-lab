"""Protocol conditions and recorded intervention decisions."""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._validation import identifier_list, normalized_text


class ProtocolCondition(str, Enum):
    """Controlled visibility conditions for one experimental arm."""

    INDEPENDENT = "independent"
    SOCIAL_ONLY = "social_only"
    EVIDENCE_ONLY = "evidence_only"
    EVIDENCE_SOCIAL = "evidence_social"


class InterventionAction(str, Enum):
    """Actions a future programmatic mediator may select."""

    NO_OP = "no_op"
    REQUEST_EVIDENCE = "request_evidence"
    BLIND_EVIDENCE_REFLECTION = "blind_evidence_reflection"
    SOLICIT_DISSENT = "solicit_dissent"
    ADAPTIVE_EXPOSURE = "adaptive_exposure"
    MINORITY_REPORT = "minority_report"
    RESTRUCTURE = "restructure"


class InterventionDecision(BaseModel):
    """An auditable decision record, without any mediator policy implementation."""

    model_config = ConfigDict(extra="forbid")

    action: InterventionAction
    target_agent_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)
    risk_score: float
    estimated_cost: float = Field(ge=0.0)
    round_id: int = Field(ge=0)

    @field_validator("target_agent_ids")
    @classmethod
    def validate_target_agents(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "target_agent_ids")

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return normalized_text(value, "reason")

    @field_validator("risk_score")
    @classmethod
    def validate_risk_score(cls, value: float) -> float:
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < 0.0 or normalized > 1.0:
            raise ValueError("risk_score must be a finite float in [0, 1]")
        return normalized

    @model_validator(mode="after")
    def validate_action_targets(self) -> "InterventionDecision":
        if self.action is InterventionAction.NO_OP and self.target_agent_ids:
            raise ValueError("no_op decisions must not name target agents")
        if self.action is not InterventionAction.NO_OP and not self.target_agent_ids:
            raise ValueError("non-no_op decisions must name at least one target agent")
        return self
