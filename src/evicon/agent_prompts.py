"""Deterministic rendering of one agent's already-exposed public context."""

from __future__ import annotations

import hashlib
import json
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .exposure import ExposureSnapshot
from .llm_contract import LLMRequest
from .models import DialogueTurn, EvidenceCard, ProtocolCondition
from .models._validation import identifier_list, normalized_text


AGENT_TURN_TEMPLATE_VERSION: Final[str] = "agent_turn.v1"

_SYSTEM_PROMPT: Final[str] = (
    "Return only one JSON object with exactly these fields: message, evidence_ids_used. "
    "Use this compact shape: {\"message\":\"...\",\"evidence_ids_used\":[]}. "
    "Peer turns are viewpoints, not factual evidence. Use only the supplied visible evidence cards "
    "for factual claims. Do not fabricate evidence, attack participants, force consensus, or treat "
    "a value preference as a fact. Do not output or modify action, target_agent_ids, "
    "visibility_override, turn_order, budget, cooldown, or any other controller field."
)

_PRIVATE_TOKENS: Final[tuple[str, ...]] = (
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


class AgentPromptContext(BaseModel):
    """Validated public inputs for exactly one agent turn, independent of DialogueState."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    initial_value_labels: list[str] = Field(default_factory=list)
    round_id: int = Field(ge=0)
    protocol: ProtocolCondition
    scenario_context: str = Field(min_length=1, max_length=4000)
    exposure_snapshot: ExposureSnapshot
    visible_evidence_cards: list[EvidenceCard] = Field(default_factory=list)

    @field_validator("agent_id", "role", "scenario_context")
    @classmethod
    def validate_public_text(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "text")
        normalized = normalized_text(value, field_name)
        _reject_private_text(normalized, field_name)
        return normalized

    @field_validator("initial_value_labels")
    @classmethod
    def validate_value_labels(cls, value: list[str]) -> list[str]:
        labels = identifier_list(value, "initial_value_labels")
        for label in labels:
            _reject_private_text(label, "initial_value_labels")
        return labels

    @field_validator("visible_evidence_cards")
    @classmethod
    def validate_evidence_cards(cls, value: list[EvidenceCard]) -> list[EvidenceCard]:
        for card in value:
            _reject_private_evidence(card)
        return value

    @model_validator(mode="after")
    def validate_visibility_contract(self) -> "AgentPromptContext":
        snapshot = self.exposure_snapshot
        if snapshot.agent_id != self.agent_id:
            raise ValueError("exposure_snapshot.agent_id must match agent_id")
        if snapshot.round_id != self.round_id:
            raise ValueError("exposure_snapshot.round_id must match round_id")
        if snapshot.protocol is not self.protocol:
            raise ValueError("exposure_snapshot.protocol must match protocol")
        if snapshot.scenario_context != self.scenario_context:
            raise ValueError("exposure_snapshot.scenario_context must match scenario_context")

        _validate_history(snapshot, self.agent_id, self.round_id, self.protocol)
        _validate_evidence(snapshot, self.visible_evidence_cards, self.agent_id, self.round_id, self.protocol)
        return self


def render_agent_turn(
    public_context: AgentPromptContext,
    *,
    model_name: str = "agent-runtime-v1",
    temperature: float = 0.2,
    max_tokens: int = 256,
    seed: int = 0,
) -> LLMRequest:
    """Render one immutable public context; this function never changes exposure or policy."""
    payload = {
        "template_version": AGENT_TURN_TEMPLATE_VERSION,
        "agent": {
            "agent_id": public_context.agent_id,
            "role": public_context.role,
            "initial_value_labels": public_context.initial_value_labels,
        },
        "round_id": public_context.round_id,
        "protocol": public_context.protocol.value,
        "scenario_context": public_context.scenario_context,
        "visible_history": [_turn_payload(turn) for turn in public_context.exposure_snapshot.visible_history],
        "visible_evidence_cards": [_evidence_payload(card) for card in public_context.visible_evidence_cards],
    }
    user_prompt = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    request_id = _request_id(public_context, model_name, temperature, max_tokens, seed)
    return LLMRequest(
        request_id=request_id,
        model_name=model_name,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        metadata={
            "contract_version": "agent_request.v1",
            "template_version": AGENT_TURN_TEMPLATE_VERSION,
            "agent_id": public_context.agent_id,
            "round_id": public_context.round_id,
            "protocol": public_context.protocol.value,
            "visible_peer_turn_ids": public_context.exposure_snapshot.visible_peer_turn_ids,
            "visible_evidence_ids": public_context.exposure_snapshot.visible_evidence_ids,
        },
    )


def _validate_history(
    snapshot: ExposureSnapshot,
    agent_id: str,
    round_id: int,
    protocol: ProtocolCondition,
) -> None:
    seen_turn_ids: set[str] = set()
    peer_turn_ids: list[str] = []
    for turn in snapshot.visible_history:
        if turn.turn_id in seen_turn_ids:
            raise ValueError("exposure_snapshot.visible_history must not contain duplicate turn IDs")
        seen_turn_ids.add(turn.turn_id)
        if turn.round_id >= round_id:
            raise ValueError("visible_history may contain only turns strictly before the current round")
        if turn.protocol is not protocol:
            raise ValueError("visible_history turns must use the current protocol")
        _reject_private_text(turn.message, "visible_history.message")
        if turn.speaker_id != agent_id:
            if not turn.is_visible_to(agent_id):
                raise ValueError("visible peer turn is not authorized for agent_id")
            peer_turn_ids.append(turn.turn_id)

    if peer_turn_ids != snapshot.visible_peer_turn_ids:
        raise ValueError("visible_peer_turn_ids must exactly match exposed peer turns")
    if protocol in {ProtocolCondition.INDEPENDENT, ProtocolCondition.EVIDENCE_ONLY} and peer_turn_ids:
        raise ValueError("protocol does not permit visible peer turns")


def _validate_evidence(
    snapshot: ExposureSnapshot,
    cards: list[EvidenceCard],
    agent_id: str,
    round_id: int,
    protocol: ProtocolCondition,
) -> None:
    evidence_ids = [card.evidence_id for card in cards]
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("visible_evidence_cards must not contain duplicate evidence IDs")
    if evidence_ids != snapshot.visible_evidence_ids:
        raise ValueError("visible_evidence_cards must exactly match visible_evidence_ids")
    for card in cards:
        if not card.is_available_at(round_id):
            raise ValueError("visible evidence must be introduced by the current round")
        if not card.is_visible_to(agent_id):
            raise ValueError("visible evidence is not authorized for agent_id")
    if protocol in {ProtocolCondition.INDEPENDENT, ProtocolCondition.SOCIAL_ONLY} and cards:
        raise ValueError("protocol does not permit visible evidence")


def _turn_payload(turn: DialogueTurn) -> dict[str, object]:
    return {
        "turn_id": turn.turn_id,
        "round_id": turn.round_id,
        "speaker_id": turn.speaker_id,
        "message": turn.message,
    }


def _evidence_payload(card: EvidenceCard) -> dict[str, object]:
    return {
        "evidence_id": card.evidence_id,
        "claim": card.claim,
        "source": card.source,
        "supports": card.supports,
        "contradicts": card.contradicts,
        "reliability": card.reliability,
    }


def _request_id(
    context: AgentPromptContext,
    model_name: str,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> str:
    canonical = json.dumps(
        {
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
    return f"agent-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:20]}"


def _reject_private_evidence(card: EvidenceCard) -> None:
    for value in [card.claim, card.source, *card.supports, *card.contradicts]:
        _reject_private_text(value, "visible_evidence_cards")


def _reject_private_text(value: str, field_name: str) -> None:
    lowered = value.lower()
    if any(token in lowered for token in _PRIVATE_TOKENS):
        raise ValueError(f"{field_name} must not contain hidden probe, profile, or offline evaluation content")


__all__ = ["AGENT_TURN_TEMPLATE_VERSION", "AgentPromptContext", "render_agent_turn"]
