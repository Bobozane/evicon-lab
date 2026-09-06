"""Local providers and fixtures for measured protocol orchestration tests."""

from __future__ import annotations

import json
from pathlib import Path

from evicon.agent_runtime import AgentRequestSettings, AgentRuntime
from evicon.contextual_probe_runtime import ContextualProbeRequestSettings, ContextualProbeRuntime
from evicon.llm_contract import LLMProviderError, LLMRequest, LLMResponse
from evicon.measured_protocol_experiment import MeasuredProtocolExperimentRunner
from evicon.models import (
    AgentSpec,
    EvidenceCard,
    ProbeSet,
    ProtocolCondition,
    RunConfig,
    ScenarioSpec,
    ValueProbeItem,
)


PROBE_TEXT = "private-probe-text-marker"
RAW_ANSWER = "raw-answer-marker"
AGENT_MESSAGE = "public-agent-message-marker"


class RecordedProvider:
    """A local deterministic LLMProvider with an observable call order."""

    def __init__(
        self,
        kind: str,
        call_order: list[str],
        *,
        content: str,
        error_at: int | None = None,
        error: LLMProviderError | None = None,
    ) -> None:
        self.kind = kind
        self.call_order = call_order
        self.content = content
        self.error_at = error_at
        self.error = error
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        self.call_order.append(self.kind)
        if self.error_at == len(self.requests):
            assert self.error is not None
            raise self.error
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content=self.content,
            finish_reason="stop",
            prompt_tokens=7,
            completion_tokens=5,
            total_tokens=12,
            latency_ms=1.0,
            provider_metadata={"provider_name": f"local_{self.kind}_provider"},
        )


def scenario(protocol: ProtocolCondition) -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id=f"measured-{protocol.value}",
        title="Measured protocol test scenario",
        description="A local public scenario fixture for orchestration tests.",
        initial_context="A public scenario asks two participants to state concise positions.",
        agents=[
            AgentSpec(agent_id="agent-a", role="participant-a", initial_value_labels=["fairness"], metadata={}),
            AgentSpec(agent_id="agent-b", role="participant-b", initial_value_labels=["safety"], metadata={}),
        ],
        evidence_cards=[
            EvidenceCard(
                evidence_id="evidence-public-marker",
                claim="A public source reports a factual constraint.",
                source="local-source",
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


def probe_set() -> ProbeSet:
    return ProbeSet(
        probe_set_id="measured-probe-set",
        dimensions=["fairness"],
        items=[
            ValueProbeItem(
                probe_id="private-probe-id-marker",
                text=PROBE_TEXT,
                dimension="fairness",
                response_scale=["scale-low", RAW_ANSWER],
                is_holdout=True,
                reverse_scored=False,
            )
        ],
        version="v1",
        metadata={},
    )


def experiment(
    tmp_path: Path,
    *,
    protocol: ProtocolCondition = ProtocolCondition.SOCIAL_ONLY,
    run_id: str = "measured-run",
    agent_provider: RecordedProvider | None = None,
    probe_provider: RecordedProvider | None = None,
    call_order: list[str] | None = None,
    probe_seed: int = 29,
    recovery_checkpoint_dir: Path | None = None,
) -> MeasuredProtocolExperimentRunner:
    order = call_order if call_order is not None else []
    active_scenario = scenario(protocol)
    active_agent_provider = agent_provider or RecordedProvider(
        "agent",
        order,
        content=json.dumps({"message": AGENT_MESSAGE, "evidence_ids_used": []}),
    )
    active_probe_provider = probe_provider or RecordedProvider(
        "probe",
        order,
        content=json.dumps({"choice": RAW_ANSWER}),
    )
    config = RunConfig(
        run_id=run_id,
        scenario_id=active_scenario.scenario_id,
        model_name="measured-agent-model",
        protocol=protocol,
        agent_count=2,
        max_rounds=2,
        seed=17,
        intervention_budget=0.0,
        output_dir=str(tmp_path / "results"),
    )
    return MeasuredProtocolExperimentRunner(
        config,
        scenario=active_scenario,
        probe_set=probe_set(),
        agent_runtime=AgentRuntime(
            AgentRequestSettings(model_name="measured-agent-model", temperature=0.2, max_tokens=256, seed=17)
        ),
        contextual_probe_runtime=ContextualProbeRuntime(
            ContextualProbeRequestSettings(model_name="measured-probe-model", temperature=0.0, max_tokens=128, seed=29)
        ),
        agent_provider=active_agent_provider,
        probe_provider=active_probe_provider,
        probe_seed=probe_seed,
        pre_target_round=0,
        post_target_round=1,
        execute_holdout_probe=True,
        recovery_checkpoint_dir=recovery_checkpoint_dir,
    )
