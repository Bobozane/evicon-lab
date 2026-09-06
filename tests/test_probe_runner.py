"""No-network execution, isolation, profile, and persistence tests for probes."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from evicon.fake_llm import FakeProbeRequest
from evicon.models import (
    AgentSpec,
    DialogueState,
    ProbeRunConfig,
    ProbeSet,
    ProtocolCondition,
    RunConfig,
    ScenarioSpec,
    ValueProbeItem,
    ValueProbeResponse,
)
from evicon.probe_loader import ProbeSetLoader
from evicon.probe_profiles import ProbeProfileError, build_value_profile
from evicon.probe_results import ProbeResultStoreError, read_probe_results, write_probe_results
from evicon.probe_runner import ProbeResponseParseError, ProbeRunner, parse_probe_response
from evicon.runner import ProtocolRunner


def smoke_probe_set() -> ProbeSet:
    project_root = Path(__file__).resolve().parents[1]
    return ProbeSetLoader.load(project_root / "configs/probes/smoke_probe.toml")


def standard_config(agent_ids: list[str] | None = None) -> ProbeRunConfig:
    return ProbeRunConfig(
        probe_set_id="smoke-value-probes",
        round_id=1,
        agent_ids=agent_ids or ["agent-1", "agent-2"],
        model_name="fake-llm",
        seed=9,
        is_holdout=False,
    )


def item(probe_id: str, dimension: str, *, reverse_scored: bool = False) -> ValueProbeItem:
    return ValueProbeItem(
        probe_id=probe_id,
        text=f"Question for {probe_id}.",
        dimension=dimension,
        response_scale=["1", "2", "3", "4", "5", "6", "7"],
        is_holdout=False,
        reverse_scored=reverse_scored,
    )


def response(agent_id: str, probe_id: str, score_label: str) -> ValueProbeResponse:
    return ValueProbeResponse(
        agent_id=agent_id,
        probe_id=probe_id,
        round_id=2,
        raw_response=score_label,
        normalized_score=(int(score_label) - 1) / 6,
    )


def test_probe_runner_is_deterministic_and_isolates_agents() -> None:
    probe_set = smoke_probe_set()
    config = standard_config()
    both = ProbeRunner(probe_set, config).run()
    repeated = ProbeRunner(probe_set, config).run()
    single = ProbeRunner(probe_set, standard_config(["agent-1"])).run()

    assert type(config).model_validate_json(config.model_dump_json()) == config
    assert both == repeated
    assert both[0] == single[0]
    assert both[0].agent_id == "agent-1"
    assert both[1].agent_id == "agent-2"
    assert all(len(result.responses) == 2 for result in both)


def test_probe_runner_does_not_open_network_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("ProbeRunner attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", reject_connection)

    assert ProbeRunner(smoke_probe_set(), standard_config(["agent-1"])).run()[0].completed


def test_hidden_probe_is_excluded_from_dialogue_state_and_protocol_events(tmp_path: Path) -> None:
    probe_set = smoke_probe_set()
    holdout_config = ProbeRunConfig(
        probe_set_id=probe_set.probe_set_id,
        round_id=0,
        agent_ids=["agent-1"],
        model_name="fake-llm",
        seed=3,
        is_holdout=True,
    )
    result = ProbeRunner(probe_set, holdout_config).run()[0]
    config = RunConfig(
        run_id="probe-isolation-dialogue",
        scenario_id="probe-isolation-scenario",
        model_name="fake-llm",
        protocol=ProtocolCondition.INDEPENDENT,
        agent_count=1,
        max_rounds=1,
        seed=3,
        intervention_budget=0.0,
        output_dir=str(tmp_path / "results"),
    )
    runner = ProtocolRunner(
        config,
        scenario=ScenarioSpec(
            scenario_id="probe-isolation-scenario",
            title="Probe isolation fixture",
            description="A dialogue run with no probe input.",
            agents=[AgentSpec(agent_id="agent-1", role="participant")],
            evidence_cards=[],
            max_rounds=1,
            metadata={},
        ),
    )
    runner.run()
    assert runner.last_state is not None
    ordinary_events = Path(config.output_dir) / config.run_id / "events.jsonl"
    event_text = ordinary_events.read_text(encoding="utf-8")

    assert result.is_holdout
    assert "probe_responses" not in runner.last_state.model_dump()
    assert "ValueProbeResponse" not in type(runner.last_state).model_fields
    assert '"raw_response"' not in event_text
    assert '"normalized_score"' not in event_text
    assert result.responses[0].probe_id not in event_text
    assert "Avoid unjustified unequal treatment" not in event_text


def test_profile_builder_averages_reverses_and_preserves_dimension_order() -> None:
    probe_set = ProbeSet(
        probe_set_id="manual-probes",
        dimensions=["transparency", "fairness"],
        items=[
            item("transparency-one", "transparency", reverse_scored=True),
            item("fairness-one", "fairness"),
            item("transparency-two", "transparency"),
            item("fairness-two", "fairness"),
        ],
        version="1.0",
        metadata={},
    )
    profile = build_value_profile(
        probe_set,
        [
            response("agent-1", "fairness-two", "7"),
            response("agent-1", "transparency-two", "4"),
            response("agent-1", "fairness-one", "1"),
            response("agent-1", "transparency-one", "2"),
        ],
        agent_id="agent-1",
        round_id=2,
        is_holdout=False,
    )

    assert profile.dimensions == ["transparency", "fairness"]
    assert profile.scores == pytest.approx([2 / 3, 0.5])


def test_profile_builder_rejects_missing_responses() -> None:
    probe_set = ProbeSet(
        probe_set_id="missing-probes",
        dimensions=["fairness"],
        items=[item("fairness-one", "fairness"), item("fairness-two", "fairness")],
        version="1.0",
        metadata={},
    )

    with pytest.raises(ProbeProfileError, match="missing"):
        build_value_profile(
            probe_set,
            [response("agent-1", "fairness-one", "3")],
            agent_id="agent-1",
            round_id=2,
            is_holdout=False,
        )


def test_invalid_structured_probe_response_is_rejected() -> None:
    probe = item("fairness-one", "fairness")
    request = FakeProbeRequest(agent_id="agent-1", round_id=0, probe=probe, seed=1)

    with pytest.raises(ProbeResponseParseError, match="response_scale"):
        parse_probe_response(
            {
                "agent_id": "agent-1",
                "probe_id": "fairness-one",
                "round_id": 0,
                "raw_response": "8",
                "normalized_score": 1.0,
                "request_fingerprint": "fixture",
            },
            request,
        )


def test_probe_results_are_parseable_offline_and_not_overwritten(tmp_path: Path) -> None:
    results = ProbeRunner(smoke_probe_set(), standard_config()).run()
    result_path = write_probe_results(tmp_path / "results", "probe-output", results)

    assert [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
    assert read_probe_results(result_path) == results
    with pytest.raises(ProbeResultStoreError, match="refusing to overwrite"):
        write_probe_results(tmp_path / "results", "probe-output", results)
