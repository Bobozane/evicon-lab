"""In-memory public trajectory fixtures for contextual probe tests."""

from __future__ import annotations

import json

from evicon.llm_contract import LLMProviderError, LLMRequest, LLMResponse
from evicon.models import (
    AgentSpec,
    DialogueTurn,
    EvidenceCard,
    EvidenceExposure,
    ProtocolCondition,
    RunConfig,
    RunRecord,
    RunStatus,
    ScenarioSpec,
    ValueProbeItem,
)


def probe_item() -> ValueProbeItem:
    return ValueProbeItem(
        probe_id="contextual-fairness-item",
        text="Select one declared option for this private value probe.",
        dimension="fairness",
        response_scale=["1", "2", "3", "4", "5", "6", "7"],
        is_holdout=True,
        reverse_scored=False,
    )


def completed_record(protocol: ProtocolCondition = ProtocolCondition.EVIDENCE_SOCIAL) -> RunRecord:
    """Return a completed two-agent trajectory with protocol-consistent inputs."""
    social = protocol in {ProtocolCondition.SOCIAL_ONLY, ProtocolCondition.EVIDENCE_SOCIAL}
    evidence = protocol in {ProtocolCondition.EVIDENCE_ONLY, ProtocolCondition.EVIDENCE_SOCIAL}
    scenario = ScenarioSpec(
        scenario_id=f"measurement-{protocol.value}",
        title="Public measurement context fixture",
        description="A local record used only for contextual probe tests.",
        initial_context="A public scenario asks two participants to consider a tradeoff.",
        agents=[
            AgentSpec(agent_id="agent-a", role="participant-a", initial_value_labels=["fairness"], metadata={}),
            AgentSpec(agent_id="agent-b", role="participant-b", initial_value_labels=["safety"], metadata={}),
        ],
        evidence_cards=[
            EvidenceCard(
                evidence_id="evidence-public-0",
                claim="A public source reports a factual constraint.",
                source="local-public-source",
                supports=["constraint"],
                contradicts=[],
                introduced_round=0,
                visible_to=["agent-a", "agent-b"],
                reliability=0.8,
            )
        ],
        max_rounds=2,
        metadata={},
    )
    audience_a = ["agent-a", "agent-b"] if social else ["agent-a"]
    audience_b = ["agent-a", "agent-b"] if social else ["agent-b"]
    turns = [
        DialogueTurn(
            turn_id="turn-a-r0",
            round_id=0,
            speaker_id="agent-a",
            message="Agent A stated an initial public position.",
            visible_to=audience_a,
            visible_peer_turn_ids=[],
            visible_evidence_ids=["evidence-public-0"] if evidence else [],
            protocol=protocol,
        ),
        DialogueTurn(
            turn_id="turn-b-r0",
            round_id=0,
            speaker_id="agent-b",
            message="Agent B stated an initial public position.",
            visible_to=audience_b,
            visible_peer_turn_ids=[],
            visible_evidence_ids=["evidence-public-0"] if evidence else [],
            protocol=protocol,
        ),
        DialogueTurn(
            turn_id="turn-a-r1",
            round_id=1,
            speaker_id="agent-a",
            message="Agent A stated a later public position.",
            visible_to=audience_a,
            visible_peer_turn_ids=["turn-b-r0"] if social else [],
            visible_evidence_ids=["evidence-public-0"] if evidence else [],
            protocol=protocol,
        ),
        DialogueTurn(
            turn_id="turn-b-r1",
            round_id=1,
            speaker_id="agent-b",
            message="Agent B stated a later public position.",
            visible_to=audience_b,
            visible_peer_turn_ids=["turn-a-r0"] if social else [],
            visible_evidence_ids=["evidence-public-0"] if evidence else [],
            protocol=protocol,
        ),
    ]
    return RunRecord(
        config=RunConfig(
            run_id=f"measurement-run-{protocol.value}",
            scenario_id=scenario.scenario_id,
            model_name="local-measurement-model",
            protocol=protocol,
            agent_count=2,
            max_rounds=2,
            seed=11,
            intervention_budget=0.0,
            output_dir="results",
        ),
        scenario=scenario,
        turns=turns,
        value_profiles=[],
        evidence_exposures=[
            EvidenceExposure(
                evidence_id="evidence-public-0",
                round_id=0,
                exposed_to=["agent-a", "agent-b"],
                exposure_reason="public scenario evidence",
            )
        ] if evidence else [],
        intervention_decisions=[],
        status=RunStatus.COMPLETED,
        error_message=None,
    )


def probe_response_content(choice: str = "4", **extra: object) -> str:
    return json.dumps({"choice": choice, **extra}, ensure_ascii=True, sort_keys=True)


class LocalProbeProvider:
    """A deterministic local provider test double with no transport."""

    def __init__(self, *, content: str | None = None, error: LLMProviderError | None = None) -> None:
        self.content = content
        self.error = error
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content=self.content or probe_response_content(),
            finish_reason="stop",
            prompt_tokens=8,
            completion_tokens=4,
            total_tokens=12,
            latency_ms=1.0,
            provider_metadata={"provider_name": "local_contextual_probe_test"},
        )
