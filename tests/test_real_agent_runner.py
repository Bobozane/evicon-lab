"""Offline coverage for the isolated real Agent baseline protocol runner."""

from __future__ import annotations

import json

import pytest

from agent_cases import LocalAgentProvider, agent_response_content

from evicon.agent_runtime import AgentRequestSettings, AgentRuntime
from evicon.llm_contract import LLMProviderError, ProviderErrorCode
from evicon.models import AgentSpec, EvidenceCard, ProtocolCondition, RunConfig, RunRecord, RunStatus, ScenarioSpec
from evicon.real_agent_runner import RealAgentProtocolRunner, RealAgentRunError
from evicon.replay import validate_replay


_MODEL_NAME = "real-agent-runner-test-model"


def _scenario(protocol: ProtocolCondition, *, evidence: bool = True) -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id=f"real-agent-{protocol.value}-scenario",
        title="Public protocol runner test",
        description="A local scenario for Agent runtime protocol runner tests.",
        initial_context="A public scenario asks agents to state short positions.",
        agents=[
            AgentSpec(agent_id="agent-a", role="participant-a", initial_value_labels=["fairness"], metadata={}),
            AgentSpec(agent_id="agent-b", role="participant-b", initial_value_labels=["safety"], metadata={}),
        ],
        evidence_cards=(
            [
                EvidenceCard(
                    evidence_id="evidence-0",
                    claim="A public source reports a constraint.",
                    source="public-source",
                    supports=["constraint"],
                    contradicts=[],
                    introduced_round=0,
                    visible_to=["*"],
                    reliability=0.9,
                )
            ]
            if evidence
            else []
        ),
        max_rounds=2,
        metadata={},
    )


def _runner(
    tmp_path: object,
    protocol: ProtocolCondition,
    provider: LocalAgentProvider,
    *,
    run_id: str | None = None,
) -> RealAgentProtocolRunner:
    scenario = _scenario(protocol)
    config = RunConfig(
        run_id=run_id or f"real-agent-{protocol.value}",
        scenario_id=scenario.scenario_id,
        model_name=_MODEL_NAME,
        protocol=protocol,
        agent_count=2,
        max_rounds=2,
        seed=31,
        intervention_budget=0.0,
        output_dir=str(tmp_path),
    )
    return RealAgentProtocolRunner(
        config,
        scenario=scenario,
        agent_runtime=AgentRuntime(AgentRequestSettings(model_name=_MODEL_NAME, seed=31)),
        provider=provider,
    )


@pytest.mark.parametrize(
    "protocol,expected_peer_count,expected_evidence_count",
    [
        (ProtocolCondition.INDEPENDENT, 0, 0),
        (ProtocolCondition.SOCIAL_ONLY, 1, 0),
        (ProtocolCondition.EVIDENCE_ONLY, 0, 1),
        (ProtocolCondition.EVIDENCE_SOCIAL, 1, 1),
    ],
)
def test_four_protocols_preserve_existing_visibility_rules(
    tmp_path: object,
    protocol: ProtocolCondition,
    expected_peer_count: int,
    expected_evidence_count: int,
) -> None:
    provider = LocalAgentProvider(content=agent_response_content())
    runner = _runner(tmp_path, protocol, provider)

    record = runner.run()

    assert record.status is RunStatus.COMPLETED
    assert len(record.turns) == 4
    for agent_id in ("agent-a", "agent-b"):
        turn = next(turn for turn in record.turns if turn.round_id == 1 and turn.speaker_id == agent_id)
        assert len(turn.visible_peer_turn_ids) == expected_peer_count
        assert len(turn.visible_evidence_ids) == expected_evidence_count


def test_same_round_snapshot_isolation_hides_first_agent_turn_from_second_agent(tmp_path: object) -> None:
    provider = LocalAgentProvider(content=agent_response_content())
    record = _runner(tmp_path, ProtocolCondition.SOCIAL_ONLY, provider).run()
    requests = [json.loads(request.user_prompt) for request in provider.requests]

    assert len(record.turns) == 4
    assert len(requests) == 4
    assert requests[0]["visible_history"] == []
    assert requests[1]["visible_history"] == []
    assert {turn["turn_id"] for turn in requests[2]["visible_history"]} == {
        "turn-r0-agent-a",
        "turn-r0-agent-b",
    }
    assert {turn["turn_id"] for turn in requests[3]["visible_history"]} == {
        "turn-r0-agent-a",
        "turn-r0-agent-b",
    }


def test_model_mismatch_is_rejected_before_an_output_directory_is_created(tmp_path: object) -> None:
    scenario = _scenario(ProtocolCondition.INDEPENDENT)
    config = RunConfig(
        run_id="mismatched-model",
        scenario_id=scenario.scenario_id,
        model_name="configured-model",
        protocol=ProtocolCondition.INDEPENDENT,
        agent_count=2,
        max_rounds=2,
        seed=0,
        intervention_budget=0.0,
        output_dir=str(tmp_path),
    )

    with pytest.raises(ValueError, match="AgentRuntime request model_name"):
        RealAgentProtocolRunner(
            config,
            scenario=scenario,
            agent_runtime=AgentRuntime(AgentRequestSettings(model_name="runtime-model")),
            provider=LocalAgentProvider(),
        )

    assert not (tmp_path / "mismatched-model").exists()


def test_public_parsed_turns_are_saved_but_logs_exclude_private_provider_material(tmp_path: object) -> None:
    message = "A public parsed agent message."
    runner = _runner(
        tmp_path,
        ProtocolCondition.EVIDENCE_ONLY,
        LocalAgentProvider(content=agent_response_content(message=message, evidence_ids_used=["evidence-0"])),
    )

    record = runner.run()
    event_text = (runner.output_directory / "events.jsonl").read_text(encoding="utf-8")

    assert all(turn.message == message for turn in record.turns)
    assert '"message": "A public parsed agent message."' in event_text
    for forbidden in ("system_prompt", "user_prompt", "api_key", "hidden_probe", "hidden_profile", "provider_metadata"):
        assert forbidden not in event_text.lower()
    assert validate_replay(
        runner.output_directory / "events.jsonl",
        runner.output_directory / "run_record.json",
    ).validation == "passed"


@pytest.mark.parametrize(
    "provider",
    [
        LocalAgentProvider(error=LLMProviderError(ProviderErrorCode.HTTP_SERVER_ERROR, "secret prompt")),
        LocalAgentProvider(content="{not-json"),
    ],
)
def test_provider_or_parser_failure_writes_failed_record_without_fabricated_turn(
    tmp_path: object,
    provider: LocalAgentProvider,
) -> None:
    runner = _runner(tmp_path, ProtocolCondition.INDEPENDENT, provider)

    with pytest.raises(RealAgentRunError):
        runner.run()

    record = RunRecord.model_validate_json((runner.output_directory / "run_record.json").read_text(encoding="utf-8"))
    event_text = (runner.output_directory / "events.jsonl").read_text(encoding="utf-8")
    assert record.status is RunStatus.FAILED
    assert record.turns == []
    assert '"event_type": "run_failed"' in event_text
    assert "secret prompt" not in event_text
    assert validate_replay(
        runner.output_directory / "events.jsonl",
        runner.output_directory / "run_record.json",
    ).validation == "passed_with_failure"
