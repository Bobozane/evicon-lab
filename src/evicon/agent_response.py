"""Strict, side-effect-free parsing for one agent's JSON turn response."""

from __future__ import annotations

import json
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .agent_prompts import AgentPromptContext
from .models._validation import identifier_list


class AgentResponseError(str, Enum):
    """Stable parser failures that contain no raw provider output."""

    MALFORMED_JSON = "malformed_json"
    INVALID_SCHEMA = "invalid_schema"
    EMPTY_MESSAGE = "empty_message"
    FORBIDDEN_CONTROL_FIELD = "forbidden_control_field"
    UNAVAILABLE_EVIDENCE = "unavailable_evidence"
    PRIVATE_CONTENT = "private_content"


class AgentResponse(BaseModel):
    """Validated in-memory content for one agent turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message: str = ""
    evidence_ids_used: list[str] = Field(default_factory=list)
    valid: bool
    validation_errors: list[AgentResponseError] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        return value.strip()

    @field_validator("evidence_ids_used")
    @classmethod
    def validate_evidence_ids(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "evidence_ids_used")

    @model_validator(mode="after")
    def validate_outcome(self) -> "AgentResponse":
        if self.valid and self.validation_errors:
            raise ValueError("valid AgentResponse must not include validation_errors")
        if not self.valid and not self.validation_errors:
            raise ValueError("invalid AgentResponse must include validation_errors")
        if self.valid and not self.message:
            raise ValueError("valid AgentResponse must include a non-empty message")
        return self


class _RawAgentPayload(BaseModel):
    """The sole successful model-output shape accepted by the parser."""

    model_config = ConfigDict(extra="forbid")

    message: str
    evidence_ids_used: list[str] = Field(default_factory=list)


_CONTROL_FIELDS = frozenset(
    {
        "action",
        "target_agent_ids",
        "visibility_override",
        "visible_evidence_ids",
        "turn_order",
        "turn_order_override",
        "budget",
        "cooldown",
        "policy",
        "executor",
        "intervention_plan",
    }
)
_PRIVATE_TOKENS = (
    "hidden_probe",
    "hidden probe",
    "hidden_profile",
    "hidden profile",
    "valueprofile",
    "evaluationreport",
    "evaluation_report",
    "proberesult",
    "probe result",
)


def parse_agent_response(content: str, public_context: AgentPromptContext) -> AgentResponse:
    """Parse one JSON response without calling a provider or changing any state."""
    if _contains_private_content(content):
        return _invalid(AgentResponseError.PRIVATE_CONTENT)
    try:
        decoded = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return _invalid(AgentResponseError.MALFORMED_JSON)
    if not isinstance(decoded, dict):
        return _invalid(AgentResponseError.INVALID_SCHEMA)
    if _CONTROL_FIELDS.intersection(decoded):
        return _invalid(AgentResponseError.FORBIDDEN_CONTROL_FIELD)
    try:
        raw = _RawAgentPayload.model_validate(decoded)
    except ValidationError:
        return _invalid(AgentResponseError.INVALID_SCHEMA)

    message = raw.message.strip()
    if not message:
        return _invalid(AgentResponseError.EMPTY_MESSAGE)
    if _contains_private_content(message):
        return _invalid(AgentResponseError.PRIVATE_CONTENT)
    try:
        evidence_ids = identifier_list(raw.evidence_ids_used, "evidence_ids_used")
    except ValueError:
        return _invalid(AgentResponseError.INVALID_SCHEMA)
    if any(evidence_id not in public_context.exposure_snapshot.visible_evidence_ids for evidence_id in evidence_ids):
        return _invalid(AgentResponseError.UNAVAILABLE_EVIDENCE)
    return AgentResponse(
        message=message,
        evidence_ids_used=evidence_ids,
        valid=True,
        validation_errors=[],
    )


def _contains_private_content(value: object) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return any(token in lowered for token in _PRIVATE_TOKENS)


def _invalid(error: AgentResponseError) -> AgentResponse:
    return AgentResponse(valid=False, validation_errors=[error])


__all__ = ["AgentResponse", "AgentResponseError", "parse_agent_response"]
