"""No-network tests for deterministic base-protocol execution."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from evicon.config import ConfigError, load_run_config
from evicon.events import EventType
from evicon.fake_llm import FakeLLM, FakeLLMRequest
from evicon.models import (
    AgentSpec,
    DialogueTurn,
    EvidenceCard,
    ProtocolCondition,
    RunConfig,
    ScenarioSpec,
)
from evicon.runner import ProtocolRunner


def make_config(tmp_path: Path, protocol: ProtocolCondition, *, run_id: str = "run") -> RunConfig:
    return RunConfig(
        run_id=run_id,
        scenario_id="local-test-scenario",
        model_name="fake-llm",
        protocol=protocol,
        agent_count=2,
        max_rounds=2,
        seed=42,
        intervention_budget=0.0,
        output_dir=str(tmp_path / "results"),
    )


def public_evidence() -> list[EvidenceCard]:
    return [
        EvidenceCard(
            evidence_id="evidence-public",
            claim="Synthetic evidence for local tests.",
            source="test-fixture",
            supports=[],
            contradicts=[],
            introduced_round=0,
            visible_to=["*"],
            reliability=1.0,
        )
    ]


def public_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="local-test-scenario",
        title="Local protocol test",
        description="A deterministic ScenarioSpec fixture for protocol tests.",
        initial_context="Synthetic test context.",
        agents=[
            AgentSpec(agent_id="agent-1", role="participant"),
            AgentSpec(agent_id="agent-2", role="participant"),
        ],
        evidence_cards=public_evidence(),
        max_rounds=2,
        metadata={},
    )


def read_events(output_directory: Path) -> list[dict[str, object]]:
    with (output_directory / "events.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def request_event(events: list[dict[str, object]], round_id: int, agent_id: str) -> dict[str, object]:
    for event in events:
        if event["event_type"] != EventType.LLM_REQUEST.value or event["round_id"] != round_id:
            continue
        payload = event["payload"]
        assert isinstance(payload, dict)
        if payload["agent_id"] == agent_id:
            return event
    raise AssertionError(f"missing request event for {agent_id} at round {round_id}")


def test_fake_llm_is_deterministic_for_same_request() -> None:
    history = [
        DialogueTurn(
            turn_id="turn-1",
            round_id=0,
            speaker_id="agent-a",
            message="Earlier local message.",
            visible_to=["agent-a"],
            visible_peer_turn_ids=[],
            visible_evidence_ids=[],
            protocol=ProtocolCondition.INDEPENDENT,
        )
    ]
    request = FakeLLMRequest(
        agent_id="agent-a",
        round_id=1,
        protocol=ProtocolCondition.EVIDENCE_SOCIAL,
        visible_history=history,
        visible_peer_turn_ids=["turn-peer"],
        visible_evidence_ids=["evidence-1"],
        seed=7,
    )

    provider = FakeLLM()

    assert provider.complete(request) == provider.complete(request)


def test_fake_llm_does_not_open_network_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("FakeLLM attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", reject_connection)
    response = FakeLLM().complete(
        FakeLLMRequest(
            agent_id="agent-a",
            round_id=0,
            protocol=ProtocolCondition.INDEPENDENT,
            visible_history=[],
            visible_peer_turn_ids=[],
            visible_evidence_ids=[],
            seed=1,
        )
    )

    assert response.message.startswith("agent=agent-a")


@pytest.mark.parametrize(
    ("protocol", "expected_peers", "expected_evidence"),
    [
        (ProtocolCondition.INDEPENDENT, [], []),
        (ProtocolCondition.SOCIAL_ONLY, ["turn-r0-agent-2"], []),
        (ProtocolCondition.EVIDENCE_ONLY, [], ["evidence-public"]),
        (
            ProtocolCondition.EVIDENCE_SOCIAL,
            ["turn-r0-agent-2"],
            ["evidence-public"],
        ),
    ],
)
def test_protocol_visibility_at_second_round(
    tmp_path: Path,
    protocol: ProtocolCondition,
    expected_peers: list[str],
    expected_evidence: list[str],
) -> None:
    config = make_config(tmp_path, protocol, run_id=f"visibility-{protocol.value}")
    record = ProtocolRunner(config, scenario=public_scenario()).run()
    event = request_event(read_events(Path(config.output_dir) / config.run_id), 1, "agent-1")
    payload = event["payload"]
    assert isinstance(payload, dict)

    assert payload["visible_peer_turn_ids"] == expected_peers
    assert payload["visible_evidence_ids"] == expected_evidence
    assert len(record.turns) == 4


def test_same_round_has_no_order_leakage(tmp_path: Path) -> None:
    config = make_config(tmp_path, ProtocolCondition.EVIDENCE_SOCIAL, run_id="no-leakage")
    ProtocolRunner(config, scenario=public_scenario()).run()
    events = read_events(Path(config.output_dir) / config.run_id)

    round_zero_requests = [
        event
        for event in events
        if event["event_type"] == EventType.LLM_REQUEST.value and event["round_id"] == 0
    ]
    assert len(round_zero_requests) == config.agent_count
    for event in round_zero_requests:
        payload = event["payload"]
        assert isinstance(payload, dict)
        assert payload["visible_peer_turn_ids"] == []


def test_jsonl_is_parseable_and_excludes_holdout_probe_data(tmp_path: Path) -> None:
    config = make_config(tmp_path, ProtocolCondition.EVIDENCE_SOCIAL, run_id="jsonl")
    runner = ProtocolRunner(config, scenario=public_scenario())
    runner.run()
    events_path = Path(config.output_dir) / config.run_id / "events.jsonl"
    events = read_events(events_path.parent)

    assert events[0]["event_type"] == EventType.RUN_STARTED.value
    assert events[-1]["event_type"] == EventType.RUN_COMPLETED.value
    assert "probe" not in events_path.read_text(encoding="utf-8").lower()
    assert runner.last_state is not None
    assert "ValueProbeResponse" not in type(runner.last_state).model_fields


def test_same_seed_produces_identical_core_events(tmp_path: Path) -> None:
    first = make_config(tmp_path / "first", ProtocolCondition.EVIDENCE_SOCIAL, run_id="same-seed")
    second = make_config(tmp_path / "second", ProtocolCondition.EVIDENCE_SOCIAL, run_id="same-seed")

    ProtocolRunner(first, scenario=public_scenario()).run()
    ProtocolRunner(second, scenario=public_scenario()).run()

    first_events = read_events(Path(first.output_dir) / first.run_id)
    second_events = read_events(Path(second.output_dir) / second.run_id)
    assert first_events == second_events


def test_different_protocols_produce_different_visible_contexts(tmp_path: Path) -> None:
    social = make_config(tmp_path / "social", ProtocolCondition.SOCIAL_ONLY, run_id="comparison")
    evidence = make_config(tmp_path / "evidence", ProtocolCondition.EVIDENCE_ONLY, run_id="comparison")

    ProtocolRunner(social, scenario=public_scenario()).run()
    ProtocolRunner(evidence, scenario=public_scenario()).run()

    social_payload = request_event(
        read_events(Path(social.output_dir) / social.run_id), 1, "agent-1"
    )["payload"]
    evidence_payload = request_event(
        read_events(Path(evidence.output_dir) / evidence.run_id), 1, "agent-1"
    )["payload"]
    assert isinstance(social_payload, dict)
    assert isinstance(evidence_payload, dict)
    assert social_payload["visible_peer_turn_ids"] != evidence_payload["visible_peer_turn_ids"]
    assert social_payload["visible_evidence_ids"] != evidence_payload["visible_evidence_ids"]


@pytest.mark.parametrize(
    ("name", "contents", "expected_message"),
    [
        (
            "unknown_protocol.toml",
            """run_id = \"bad\"\nscenario_id = \"s\"\nmodel_name = \"fake\"\nprotocol = \"unknown\"\nagent_count = 2\nmax_rounds = 1\nseed = 1\nintervention_budget = 0.0\noutput_dir = \"results\"\n""",
            "protocol",
        ),
        (
            "bad_agent_count.toml",
            """run_id = \"bad\"\nscenario_id = \"s\"\nmodel_name = \"fake\"\nprotocol = \"independent\"\nagent_count = 0\nmax_rounds = 1\nseed = 1\nintervention_budget = 0.0\noutput_dir = \"results\"\n""",
            "agent_count",
        ),
        (
            "bad_seed.toml",
            """run_id = \"bad\"\nscenario_id = \"s\"\nmodel_name = \"fake\"\nprotocol = \"independent\"\nagent_count = 2\nmax_rounds = 1\nseed = \"one\"\nintervention_budget = 0.0\noutput_dir = \"results\"\n""",
            "seed must be an integer",
        ),
        (
            "extra_field.toml",
            """run_id = \"bad\"\nscenario_id = \"s\"\nmodel_name = \"fake\"\nprotocol = \"independent\"\nagent_count = 2\nmax_rounds = 1\nseed = 1\nintervention_budget = 0.0\noutput_dir = \"results\"\nunexpected = true\n""",
            "unexpected",
        ),
    ],
)
def test_invalid_config_is_rejected(
    tmp_path: Path,
    name: str,
    contents: str,
    expected_message: str,
) -> None:
    config_path = tmp_path / name
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError, match=expected_message):
        load_run_config(config_path)


def test_failed_run_appends_run_failed_and_preserves_prior_events(tmp_path: Path) -> None:
    class FailingProvider:
        def complete(self, request: FakeLLMRequest) -> object:
            raise RuntimeError("intentional local failure")

    config = make_config(tmp_path, ProtocolCondition.EVIDENCE_ONLY, run_id="failed-run")
    with pytest.raises(RuntimeError, match="intentional local failure"):
        ProtocolRunner(config, provider=FailingProvider(), scenario=public_scenario()).run()

    output_directory = Path(config.output_dir) / config.run_id
    events = read_events(output_directory)
    record = json.loads((output_directory / "run_record.json").read_text(encoding="utf-8"))
    assert any(event["event_type"] == EventType.RUN_STARTED.value for event in events)
    assert events[-1]["event_type"] == EventType.RUN_FAILED.value
    assert record["status"] == "failed"
    assert "intentional local failure" not in (output_directory / "events.jsonl").read_text(encoding="utf-8")


def test_existing_result_directory_is_not_overwritten(tmp_path: Path) -> None:
    config = make_config(tmp_path, ProtocolCondition.INDEPENDENT, run_id="no-overwrite")
    ProtocolRunner(config, scenario=public_scenario()).run()
    events_path = Path(config.output_dir) / config.run_id / "events.jsonl"
    original_contents = events_path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ProtocolRunner(config, scenario=public_scenario()).run()

    assert events_path.read_text(encoding="utf-8") == original_contents
