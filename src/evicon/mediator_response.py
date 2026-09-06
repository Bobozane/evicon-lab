"""Strict, side-effect-free parsing for bounded mediator JSON responses."""

from __future__ import annotations

import json
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .executor import InterventionPlan
from .mediator_prompts import PromptContext
from .models._validation import identifier_list, normalized_text


class MediatorResponseError(str, Enum):
    """Stable parser error codes; none contains provider or prompt content."""

    MALFORMED_JSON = "malformed_json"
    INVALID_SCHEMA = "invalid_schema"
    EMPTY_MESSAGE = "empty_message"
    FORBIDDEN_DECISION_FIELD = "forbidden_decision_field"
    UNAVAILABLE_EVIDENCE = "unavailable_evidence"
    PRIVATE_CONTENT = "private_content"
    CONTEXT_MISMATCH = "context_mismatch"


class MediatorResponse(BaseModel):
    """Validated language output; policy action and targets are intentionally absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message: str = ""
    evidence_ids_requested: list[str] = Field(default_factory=list)
    dissent_preserved: bool = False
    factual_claims: list[str] = Field(default_factory=list)
    valid: bool
    validation_errors: list[MediatorResponseError] = Field(default_factory=list)

    @field_validator("evidence_ids_requested")
    @classmethod
    def validate_evidence_ids(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "evidence_ids_requested")

    @field_validator("factual_claims")
    @classmethod
    def validate_claims(cls, value: list[str]) -> list[str]:
        return [normalized_text(claim, "factual_claims") for claim in value]

    @model_validator(mode="after")
    def validate_outcome(self) -> "MediatorResponse":
        if self.valid and self.validation_errors:
            raise ValueError("valid MediatorResponse must not include validation_errors")
        if not self.valid and not self.validation_errors:
            raise ValueError("invalid MediatorResponse must include validation_errors")
        return self


class _RawMediatorPayload(BaseModel):
    """Exact JSON schema accepted before contextual validation."""

    model_config = ConfigDict(extra="forbid")

    message: str
    evidence_ids_requested: list[str] = Field(default_factory=list)
    dissent_preserved: bool = False
    factual_claims: list[str] = Field(default_factory=list)


def parse_mediator_response(
    content: str,
    plan: InterventionPlan,
    public_context: PromptContext,
) -> MediatorResponse:
    """Parse JSON without calling a model or changing the fixed plan/context."""
    if _contains_private_content(content):
        return _invalid(MediatorResponseError.PRIVATE_CONTENT)
    if not _context_matches(plan, public_context):
        return _invalid(MediatorResponseError.CONTEXT_MISMATCH)
    try:
        decoded = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return _invalid(MediatorResponseError.MALFORMED_JSON)
    if not isinstance(decoded, dict):
        return _invalid(MediatorResponseError.INVALID_SCHEMA)
    if "action" in decoded or "target_agent_ids" in decoded:
        return _invalid(MediatorResponseError.FORBIDDEN_DECISION_FIELD)
    try:
        raw = _RawMediatorPayload.model_validate(decoded)
    except ValidationError:
        return _invalid(MediatorResponseError.INVALID_SCHEMA)

    message = raw.message.strip()
    if not message:
        return _invalid(MediatorResponseError.EMPTY_MESSAGE)
    if _contains_private_content(message) or any(
        _contains_private_content(claim) for claim in raw.factual_claims
    ):
        return _invalid(MediatorResponseError.PRIVATE_CONTENT)
    try:
        evidence_ids = identifier_list(raw.evidence_ids_requested, "evidence_ids_requested")
        claims = [normalized_text(claim, "factual_claims") for claim in raw.factual_claims]
    except ValueError:
        return _invalid(MediatorResponseError.INVALID_SCHEMA)
    allowed_evidence_ids = set(public_context.visible_evidence_ids) | set(
        public_context.requested_evidence_ids
    )
    if any(evidence_id not in allowed_evidence_ids for evidence_id in evidence_ids):
        return _invalid(MediatorResponseError.UNAVAILABLE_EVIDENCE)
    return MediatorResponse(
        message=message,
        evidence_ids_requested=evidence_ids,
        dissent_preserved=raw.dissent_preserved,
        factual_claims=claims,
        valid=True,
        validation_errors=[],
    )


def _context_matches(plan: InterventionPlan, context: PromptContext) -> bool:
    return (
        plan.valid
        and plan.run_id == context.run_id
        and plan.scenario_id == context.scenario_id
        and plan.round_id == context.round_id
        and plan.action is context.action
        and plan.target_agent_ids == context.target_agent_ids
    )


def _contains_private_content(value: str) -> bool:
    lowered = value.lower()
    return any(
        token in lowered
        for token in (
            "hidden_probe",
            "hidden probe",
            "hidden_profile",
            "hidden profile",
            "valueprofile",
            "evaluationreport",
            "evaluation_report",
        )
    )


def _invalid(error: MediatorResponseError) -> MediatorResponse:
    return MediatorResponse(
        valid=False,
        validation_errors=[error],
    )


__all__ = ["MediatorResponse", "MediatorResponseError", "parse_mediator_response"]
