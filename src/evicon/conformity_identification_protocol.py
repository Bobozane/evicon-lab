"""Strict public-response protocol for Conformity Identification Study v1."""
from __future__ import annotations

import hashlib
import json
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from .conformity_identification import (
    IdentificationStage,
    OrdinalJudgment,
    PublicAction,
    ShareDecision,
)
from .llm_contract import LLMRequest

PROTOCOL_VERSION: Final[str] = "conformity_identification_protocol.v1"
TEMPLATE_VERSION: Final[str] = "conformity_identification_turn.v1"
SCHEMA_NAME: Final[str] = "conformity_identification_response_v1"


class IdentificationPromptContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_id: str
    agent_id: str
    seed: int
    stage: IdentificationStage
    target_claim_id: str
    decision_task: str
    visible_stimulus_ids: tuple[str, ...]
    visible_public_summaries: tuple[str, ...]
    visible_source_root_ids: tuple[str, ...]
    visible_evidence_ids: tuple[str, ...]
    reflection_only: bool
    model_name: str = "fake-identification-model"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0)


class IdentificationAgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ordinal_judgment: OrdinalJudgment
    public_action: PublicAction
    share_decision: ShareDecision
    confidence: int = Field(ge=1, le=7)
    content_ids_used: tuple[str, ...]
    evidence_ids_used: tuple[str, ...]


def parse_identification_response(raw: str, context: IdentificationPromptContext) -> IdentificationAgentResponse:
    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_field")
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=unique_pairs)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("malformed_json") from exc
    if not isinstance(payload, dict):
        raise ValueError("top_level_type")
    try:
        response = IdentificationAgentResponse.model_validate(payload)
    except Exception as exc:
        raise ValueError("invalid_schema") from exc
    if len(set(response.content_ids_used)) != len(response.content_ids_used):
        raise ValueError("duplicate_content_id")
    if len(set(response.evidence_ids_used)) != len(response.evidence_ids_used):
        raise ValueError("duplicate_evidence_id")
    if not set(response.content_ids_used).issubset(context.visible_stimulus_ids):
        raise ValueError("unavailable_content_id")
    if not set(response.evidence_ids_used).issubset(context.visible_evidence_ids):
        raise ValueError("unavailable_evidence_id")
    return response


def render_identification_turn(context: IdentificationPromptContext) -> LLMRequest:
    payload = {
        "template_version": TEMPLATE_VERSION,
        "scenario_id": context.scenario_id,
        "agent_id": context.agent_id,
        "stage": context.stage.value,
        "target_claim_id": context.target_claim_id,
        "decision_task": context.decision_task,
        "visible_content": [
            {"content_id": identifier, "public_summary": summary}
            for identifier, summary in zip(context.visible_stimulus_ids, context.visible_public_summaries, strict=True)
        ],
        "visible_source_root_ids": list(context.visible_source_root_ids),
        "visible_evidence_ids": list(context.visible_evidence_ids),
        "reflection_only": context.reflection_only,
    }
    user_prompt = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(
        f"{context.scenario_id}|{context.agent_id}|{context.seed}|{context.stage.value}|{user_prompt}".encode()
    ).hexdigest()[:24]
    return LLMRequest(
        request_id=f"ident-v1-{digest}",
        model_name=context.model_name,
        system_prompt=(
            "Return exactly one JSON object with ordinal_judgment, public_action, share_decision, confidence, "
            "content_ids_used, and evidence_ids_used. The judgment scale runs from strongly_reject through "
            "strongly_adopt. Treat repetition, social source information, and evidence as distinct observations. "
            "Use only visible IDs. Do not invent truth labels, sources, evidence, controller actions, or private data."
        ),
        user_prompt=user_prompt,
        temperature=context.temperature,
        max_tokens=context.max_tokens,
        seed=context.seed,
        metadata={"protocol_version": PROTOCOL_VERSION, "template_version": TEMPLATE_VERSION},
    )


__all__ = [
    "IdentificationAgentResponse", "IdentificationPromptContext", "PROTOCOL_VERSION", "SCHEMA_NAME",
    "TEMPLATE_VERSION", "parse_identification_response", "render_identification_turn",
]
