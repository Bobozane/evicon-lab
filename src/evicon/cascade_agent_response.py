"""Strict parser for one public provenance-cascade Agent response."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .cascade_agent_prompts import CascadeAgentPromptContext
from .cascade_outcomes import ClaimStance
from .models._validation import identifier_list


class CascadeAgentResponseError(str, Enum):
    MALFORMED_JSON = "malformed_json"
    DUPLICATE_FIELD = "duplicate_field"
    INVALID_SCHEMA = "invalid_schema"
    FORBIDDEN_FIELD = "forbidden_field"
    UNAVAILABLE_CONTENT = "unavailable_content"
    UNAVAILABLE_EVIDENCE = "unavailable_evidence"
    PRIVATE_CONTENT = "private_content"


class CascadeAgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stance: ClaimStance
    content_ids_used: tuple[str, ...] = ()
    evidence_ids_used: tuple[str, ...] = ()
    share_content_id: str | None = None
    valid: bool = True
    validation_errors: tuple[CascadeAgentResponseError, ...] = ()

    @field_validator("content_ids_used", "evidence_ids_used")
    @classmethod
    def unique_ids(cls, value: tuple[str, ...], info: object) -> tuple[str, ...]:
        return tuple(identifier_list(list(value), getattr(info, "field_name", "ids")))

    @model_validator(mode="after")
    def valid_shape(self) -> "CascadeAgentResponse":
        if self.valid and self.validation_errors:
            raise ValueError("valid response cannot contain validation_errors")
        if not self.valid and not self.validation_errors:
            raise ValueError("invalid response must contain validation_errors")
        if self.share_content_id is not None and self.share_content_id not in self.content_ids_used:
            raise ValueError("share_content_id must be one of content_ids_used")
        return self



class _RawCascadeAgentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stance: ClaimStance
    content_ids_used: tuple[str, ...] = ()
    evidence_ids_used: tuple[str, ...] = ()
    share_content_id: str | None = None

def parse_cascade_agent_response(content: str, context: CascadeAgentPromptContext) -> CascadeAgentResponse:
    try:
        decoded = json.loads(content, object_pairs_hook=_pairs_without_duplicates)
    except _DuplicateField:
        return _invalid(CascadeAgentResponseError.DUPLICATE_FIELD)
    except (TypeError, json.JSONDecodeError):
        return _invalid(CascadeAgentResponseError.MALFORMED_JSON)
    if not isinstance(decoded, dict):
        return _invalid(CascadeAgentResponseError.INVALID_SCHEMA)
    forbidden = {
        "claim_id", "agent_id", "round_id", "action", "target_agent_id", "source_root_id",
        "verification_status", "directive", "budget", "cooldown", "controller_verdict",
        "ground_truth_label", "source_independence_label", "independence_group",
    }
    if forbidden.intersection(decoded):
        return _invalid(CascadeAgentResponseError.FORBIDDEN_FIELD)
    if _contains_private(decoded):
        return _invalid(CascadeAgentResponseError.PRIVATE_CONTENT)
    try:
        raw = _RawCascadeAgentPayload.model_validate(decoded)
    except (ValidationError, ValueError, TypeError):
        return _invalid(CascadeAgentResponseError.INVALID_SCHEMA)
    visible_content = {item.content_id for item in context.contents}
    visible_evidence = {item.evidence_id for item in context.evidence}
    if not set(raw.content_ids_used).issubset(visible_content) or (
        raw.share_content_id is not None and raw.share_content_id not in visible_content
    ):
        return _invalid(CascadeAgentResponseError.UNAVAILABLE_CONTENT)
    if not set(raw.evidence_ids_used).issubset(visible_evidence):
        return _invalid(CascadeAgentResponseError.UNAVAILABLE_EVIDENCE)
    try:
        return CascadeAgentResponse(
            stance=raw.stance,
            content_ids_used=raw.content_ids_used,
            evidence_ids_used=raw.evidence_ids_used,
            share_content_id=raw.share_content_id,
            valid=True,
            validation_errors=(),
        )
    except (ValidationError, ValueError, TypeError):
        return _invalid(CascadeAgentResponseError.INVALID_SCHEMA)


class _DuplicateField(ValueError):
    pass


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateField(key)
        result[key] = value
    return result


def _contains_private(value: object) -> bool:
    tokens = (
        "ground_truth_label", "source_independence_label", "hidden_probe", "hidden probe",
        "hidden_profile", "hidden profile", "evaluationreport", "private fixture", "api_key", "authorization:",
    )
    if isinstance(value, str):
        lowered = value.lower()
        return any(token in lowered for token in tokens)
    if isinstance(value, dict):
        return any(_contains_private(key) or _contains_private(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_private(item) for item in value)
    return False


def _invalid(error: CascadeAgentResponseError) -> CascadeAgentResponse:
    return CascadeAgentResponse(
        stance=ClaimStance.NO_POSITION,
        valid=False,
        validation_errors=(error,),
    )


__all__ = ["CascadeAgentResponse", "CascadeAgentResponseError", "parse_cascade_agent_response"]
