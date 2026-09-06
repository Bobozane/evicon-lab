"""Public-only fixtures and local providers for isolated Agent runtime tests."""

from __future__ import annotations

import json

from evicon.agent_prompts import AgentPromptContext
from evicon.exposure import ExposureSnapshot
from evicon.llm_contract import LLMProviderError, LLMRequest, LLMResponse
from evicon.models import DialogueTurn, EvidenceCard, ProtocolCondition


def public_context(*, protocol: ProtocolCondition = ProtocolCondition.EVIDENCE_SOCIAL) -> AgentPromptContext:
    """Return one round-one context containing only exposed public material."""
    scenario_context = "A public scenario asks agents to weigh safety and fairness."
    own_turn = DialogueTurn(
        turn_id="turn-own-r0",
        round_id=0,
        speaker_id="agent-a",
        message="My earlier public view noted a tradeoff.",
        visible_to=["agent-a"],
        visible_peer_turn_ids=[],
        visible_evidence_ids=[],
        protocol=protocol,
    )
    peer_turn = DialogueTurn(
        turn_id="turn-peer-r0",
        round_id=0,
        speaker_id="agent-b",
        message="A peer expressed a public preference.",
        visible_to=["agent-a", "agent-b"],
        visible_peer_turn_ids=[],
        visible_evidence_ids=[],
        protocol=protocol,
    )
    evidence = EvidenceCard(
        evidence_id="evidence-0",
        claim="The public source reports a verified constraint.",
        source="public-source",
        supports=["constraint"],
        contradicts=[],
        introduced_round=0,
        visible_to=["agent-a"],
        reliability=0.8,
    )

    if protocol is ProtocolCondition.INDEPENDENT:
        history = [own_turn]
        peer_ids: list[str] = []
        evidence_ids: list[str] = []
        evidence_cards: list[EvidenceCard] = []
    elif protocol is ProtocolCondition.SOCIAL_ONLY:
        history = [own_turn, peer_turn]
        peer_ids = [peer_turn.turn_id]
        evidence_ids = []
        evidence_cards = []
    elif protocol is ProtocolCondition.EVIDENCE_ONLY:
        history = [own_turn]
        peer_ids = []
        evidence_ids = [evidence.evidence_id]
        evidence_cards = [evidence]
    else:
        history = [own_turn, peer_turn]
        peer_ids = [peer_turn.turn_id]
        evidence_ids = [evidence.evidence_id]
        evidence_cards = [evidence]

    return AgentPromptContext(
        agent_id="agent-a",
        role="public deliberation participant",
        initial_value_labels=["fairness", "safety"],
        round_id=1,
        protocol=protocol,
        scenario_context=scenario_context,
        exposure_snapshot=ExposureSnapshot(
            agent_id="agent-a",
            round_id=1,
            protocol=protocol,
            scenario_context=scenario_context,
            visible_history=history,
            visible_peer_turn_ids=peer_ids,
            visible_evidence_ids=evidence_ids,
        ),
        visible_evidence_cards=evidence_cards,
    )


def context_data(context: AgentPromptContext) -> dict[str, object]:
    """Return independently validated input data for negative-context tests."""
    return context.model_dump(mode="python")


def agent_response_content(
    *,
    message: str = "A concise public agent response.",
    evidence_ids_used: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "message": message,
            "evidence_ids_used": list(evidence_ids_used or []),
        },
        ensure_ascii=True,
        sort_keys=True,
    )


class LocalAgentProvider:
    """A local deterministic provider test double with no network boundary."""

    def __init__(self, *, content: str | None = None, error: LLMProviderError | None = None) -> None:
        self.content = content
        self.error = error
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        response_content = self.content or agent_response_content()
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content=response_content,
            finish_reason="stop",
            prompt_tokens=9,
            completion_tokens=11,
            total_tokens=20,
            latency_ms=1.0,
            provider_metadata={"provider_name": "local_agent_test"},
        )
