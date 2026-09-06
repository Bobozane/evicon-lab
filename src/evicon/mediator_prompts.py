"""Deterministic rendering of fixed intervention plans into offline LLM requests."""

from __future__ import annotations

import hashlib
import json
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from .executor import InterventionPlan
from .llm_contract import LLMRequest
from .models._validation import Metadata, identifier_list, normalized_text
from .models.protocols import InterventionAction


TEMPLATE_VERSIONS: Final[dict[InterventionAction, str]] = {
    InterventionAction.REQUEST_EVIDENCE: "request_evidence.v1",
    InterventionAction.BLIND_EVIDENCE_REFLECTION: "blind_evidence_reflection.v1",
    InterventionAction.SOLICIT_DISSENT: "solicit_dissent.v1",
    InterventionAction.ADAPTIVE_EXPOSURE: "adaptive_exposure.v1",
    InterventionAction.MINORITY_REPORT: "minority_report.v1",
    InterventionAction.RESTRUCTURE: "restructure.v1",
}

_ACTION_INSTRUCTIONS: Final[dict[InterventionAction, str]] = {
    InterventionAction.REQUEST_EVIDENCE: "Identify which supplied evidence IDs would help clarify the public factual question.",
    InterventionAction.BLIND_EVIDENCE_REFLECTION: "Reflect on the supplied evidence without using peer-position claims as evidence.",
    InterventionAction.SOLICIT_DISSENT: "State any genuine remaining reservation without inventing a contrary position.",
    InterventionAction.ADAPTIVE_EXPOSURE: "Compare only the supplied public material and state which distinctions remain relevant.",
    InterventionAction.MINORITY_REPORT: "Record a concise minority perspective when one is genuinely retained.",
    InterventionAction.RESTRUCTURE: "Restate the public issues in the fixed deliberation structure without choosing a new procedure.",
}

_SYSTEM_PROMPT: Final[str] = (
    "Return only one compact JSON object with exactly these fields: message, evidence_ids_requested, "
    "dissent_preserved, factual_claims. Use short text and empty lists when no evidence request or "
    "factual claim is needed. For example: {\"message\":\"Need public evidence.\","
    "\"evidence_ids_requested\":[],\"dissent_preserved\":true,\"factual_claims\":[]}. "
    "Use only the supplied public context and evidence IDs. "
    "Do not fabricate evidence, attack participants, force disagreement, or treat a value preference "
    "as a factual conclusion. The action and target agents are fixed program inputs, not choices for you."
)


class PromptContext(BaseModel):
    """Public-only, compact material a renderer may place in a mediator request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    action: InterventionAction
    target_agent_ids: list[str] = Field(default_factory=list)
    public_dialogue_summary: str = Field(min_length=1, max_length=4000)
    visible_evidence_ids: list[str] = Field(default_factory=list)
    requested_evidence_ids: list[str] = Field(default_factory=list)
    preservation_constraints: list[str] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("run_id", "scenario_id", "public_dialogue_summary")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        normalized = normalized_text(value, getattr(info, "field_name", "text"))
        _reject_private_text(normalized, getattr(info, "field_name", "text"))
        return normalized

    @field_validator("target_agent_ids", "visible_evidence_ids", "requested_evidence_ids")
    @classmethod
    def validate_identifiers(cls, value: list[str], info: object) -> list[str]:
        return identifier_list(value, getattr(info, "field_name", "identifiers"))

    @field_validator("preservation_constraints")
    @classmethod
    def validate_constraints(cls, value: list[str]) -> list[str]:
        normalized = [normalized_text(item, "preservation_constraints") for item in value]
        if len(set(normalized)) != len(normalized):
            raise ValueError("preservation_constraints must not contain duplicates")
        for item in normalized:
            _reject_private_text(item, "preservation_constraints")
        return normalized

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: Metadata) -> Metadata:
        _reject_private_value(value, "PromptContext.metadata")
        return value

    @model_validator(mode="after")
    def validate_requested_evidence(self) -> "PromptContext":
        if not set(self.requested_evidence_ids).issubset(self.visible_evidence_ids):
            raise ValueError("requested_evidence_ids must be drawn from visible_evidence_ids")
        if self.action is InterventionAction.NO_OP and self.target_agent_ids:
            raise ValueError("no_op PromptContext must not include target_agent_ids")
        return self


def render_action_instruction(
    plan: InterventionPlan,
    public_context: PromptContext,
    *,
    model_name: str = "fake-mediator-v1",
    temperature: float = 0.0,
    max_tokens: int = 400,
    seed: int = 0,
) -> LLMRequest | None:
    """Render a fixed plan deterministically; this function never selects an action."""
    if plan.action is InterventionAction.NO_OP:
        return None
    if not plan.valid:
        raise ValueError("cannot render an invalid intervention plan")
    if public_context.run_id != plan.run_id or public_context.scenario_id != plan.scenario_id:
        raise ValueError("PromptContext identifiers must match the intervention plan")
    if public_context.round_id != plan.round_id:
        raise ValueError("PromptContext round_id must match the intervention plan")
    if public_context.action is not plan.action:
        raise ValueError("PromptContext action must match the fixed intervention plan")
    if public_context.target_agent_ids != plan.target_agent_ids:
        raise ValueError("PromptContext target_agent_ids must match the fixed intervention plan")
    if not set(plan.requested_evidence_ids).issubset(public_context.visible_evidence_ids):
        raise ValueError("plan requested_evidence_ids must be visible in PromptContext")

    template_version = TEMPLATE_VERSIONS[plan.action]
    payload = {
        "template_version": template_version,
        "action": plan.action.value,
        "target_agent_ids": public_context.target_agent_ids,
        "public_dialogue_summary": public_context.public_dialogue_summary,
        "visible_evidence_ids": public_context.visible_evidence_ids,
        "requested_evidence_ids": public_context.requested_evidence_ids,
        "preservation_constraints": public_context.preservation_constraints,
        "action_instruction": _ACTION_INSTRUCTIONS[plan.action],
    }
    user_prompt = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    request_id = _request_id(plan, public_context, model_name, temperature, max_tokens, seed)
    return LLMRequest(
        request_id=request_id,
        model_name=model_name,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        metadata={
            "contract_version": "mediator_request.v1",
            "template_version": template_version,
            "action": plan.action.value,
            "target_agent_ids": plan.target_agent_ids,
            "visible_evidence_ids": public_context.visible_evidence_ids,
            "requested_evidence_ids": public_context.requested_evidence_ids,
        },
    )


def _request_id(
    plan: InterventionPlan,
    context: PromptContext,
    model_name: str,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> str:
    canonical = json.dumps(
        {
            "plan": plan.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"mediator-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:20]}"


def _reject_private_text(value: str, field_name: str) -> None:
    lowered = value.lower()
    forbidden = ("hidden_probe", "hidden probe", "hidden_profile", "hidden profile", "valueprofile", "evaluationreport")
    if any(token in lowered for token in forbidden):
        raise ValueError(f"{field_name} must not contain private probe, profile, or evaluation content")


def _reject_private_value(value: JsonValue, field_name: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _reject_private_text(key, field_name)
            _reject_private_value(nested, f"{field_name}.{key}")
    elif isinstance(value, list):
        for nested in value:
            _reject_private_value(nested, field_name)
    elif isinstance(value, str):
        _reject_private_text(value, field_name)


__all__ = ["PromptContext", "TEMPLATE_VERSIONS", "render_action_instruction"]
