"""Read-only replay validation tests for local deterministic runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evicon.models import AgentSpec, EvidenceCard, ProtocolCondition, RunConfig, ScenarioSpec
from evicon.fake_llm import FakeLLMRequest
from evicon.replay import ReplayValidationError, validate_replay
from evicon.runner import ProtocolRunner


def replay_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="replay-scenario",
        title="Replay validator fixture",
        description="A local scenario for strict event-log validation tests.",
        initial_context="Local replay fixture context.",
        agents=[
            AgentSpec(agent_id="agent-1", role="participant"),
            AgentSpec(agent_id="agent-2", role="participant"),
        ],
        evidence_cards=[
            EvidenceCard(
                evidence_id="evidence-public",
                claim="Public local fixture evidence.",
                source="local-fixture",
                supports=["claim-1"],
                contradicts=[],
                introduced_round=0,
                visible_to=["*"],
                reliability=1.0,
            ),
            EvidenceCard(
                evidence_id="evidence-restricted",
                claim="Restricted local fixture evidence.",
                source="local-fixture",
                supports=[],
                contradicts=["claim-1"],
                introduced_round=1,
                visible_to=["agent-1"],
                reliability=0.8,
            ),
        ],
        max_rounds=2,
        metadata={},
    )


def make_config(tmp_path: Path, run_id: str = "replay-run") -> RunConfig:
    return RunConfig(
        run_id=run_id,
        scenario_id="replay-scenario",
        model_name="fake-llm",
        protocol=ProtocolCondition.EVIDENCE_SOCIAL,
        agent_count=2,
        max_rounds=2,
        seed=17,
        intervention_budget=0.0,
        output_dir=str(tmp_path / "results"),
    )


def run_paths(tmp_path: Path, run_id: str = "replay-run") -> tuple[Path, Path, RunConfig]:
    config = make_config(tmp_path, run_id)
    ProtocolRunner(config, scenario=replay_scenario()).run()
    output = Path(config.output_dir) / config.run_id
    return output / "events.jsonl", output / "run_record.json", config


def load_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_events(path: Path, events: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in events), encoding="utf-8")


def mutate_visibility(
    events: list[dict[str, Any]],
    record: dict[str, Any],
    *,
    round_id: int,
    agent_id: str,
    peer_ids: list[str] | None = None,
    evidence_ids: list[str] | None = None,
) -> None:
    for turn in record["turns"]:
        if turn["round_id"] == round_id and turn["speaker_id"] == agent_id:
            if peer_ids is not None:
                turn["visible_peer_turn_ids"] = peer_ids
            if evidence_ids is not None:
                turn["visible_evidence_ids"] = evidence_ids
    for event in events:
        payload = event["payload"]
        matches_agent = payload.get("agent_id") == agent_id or payload.get("speaker_id") == agent_id
        if event["round_id"] == round_id and matches_agent:
            if peer_ids is not None and "visible_peer_turn_ids" in payload:
                payload["visible_peer_turn_ids"] = peer_ids
            if evidence_ids is not None and "visible_evidence_ids" in payload:
                payload["visible_evidence_ids"] = evidence_ids


def test_valid_run_passes_replay_validation(tmp_path: Path) -> None:
    events_path, record_path, _ = run_paths(tmp_path)

    report = validate_replay(events_path, record_path)

    assert report.run_id == "replay-run"
    assert report.event_count > 0
    assert report.round_count == 2
    assert report.turn_count == 4
    assert report.validation == "passed"


def failed_paths(tmp_path: Path) -> tuple[Path, Path]:
    class FailingProvider:
        def complete(self, request: FakeLLMRequest) -> object:
            raise RuntimeError("local provider failure")

    config = make_config(tmp_path, run_id="failed-replay-run")
    with pytest.raises(RuntimeError, match="local provider failure"):
        ProtocolRunner(config, scenario=replay_scenario(), provider=FailingProvider()).run()
    output = Path(config.output_dir) / config.run_id
    return output / "events.jsonl", output / "run_record.json"


def test_failed_run_passes_replay_with_one_unclosed_final_round(tmp_path: Path) -> None:
    events_path, record_path = failed_paths(tmp_path)

    report = validate_replay(events_path, record_path)

    assert report.status.value == "failed"
    assert report.validation == "passed_with_failure"


def test_unclosed_run_without_run_failed_is_rejected(tmp_path: Path) -> None:
    events_path, record_path = failed_paths(tmp_path)
    events = load_events(events_path)
    del events[-1]
    write_events(events_path, events)

    with pytest.raises(ReplayValidationError, match="run_completed or run_failed"):
        validate_replay(events_path, record_path)


def test_business_event_after_run_failed_is_rejected(tmp_path: Path) -> None:
    events_path, record_path = failed_paths(tmp_path)
    events = load_events(events_path)
    sequence = len(events) + 1
    events.append(
        {
            "event_id": f"failed-replay-run:{sequence:06d}",
            "event_type": "round_started",
            "run_id": "failed-replay-run",
            "round_id": 1,
            "sequence": sequence,
            "payload": {},
        }
    )
    write_events(events_path, events)

    with pytest.raises(ReplayValidationError, match="run_failed must be final"):
        validate_replay(events_path, record_path)


def test_deleted_event_is_rejected(tmp_path: Path) -> None:
    events_path, record_path, _ = run_paths(tmp_path)
    events = load_events(events_path)
    del events[2]
    write_events(events_path, events)

    with pytest.raises(ReplayValidationError, match="sequence"):
        validate_replay(events_path, record_path)


def test_modified_sequence_is_rejected(tmp_path: Path) -> None:
    events_path, record_path, _ = run_paths(tmp_path)
    events = load_events(events_path)
    events[1]["sequence"] = 9
    write_events(events_path, events)

    with pytest.raises(ReplayValidationError, match="sequence"):
        validate_replay(events_path, record_path)


def test_same_round_peer_reference_is_rejected(tmp_path: Path) -> None:
    events_path, record_path, _ = run_paths(tmp_path)
    events = load_events(events_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    mutate_visibility(
        events,
        record,
        round_id=0,
        agent_id="agent-1",
        peer_ids=["turn-r0-agent-2"],
    )
    write_events(events_path, events)
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ReplayValidationError, match="non-prior peer"):
        validate_replay(events_path, record_path)


def test_early_evidence_reference_is_rejected(tmp_path: Path) -> None:
    events_path, record_path, _ = run_paths(tmp_path)
    events = load_events(events_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    mutate_visibility(
        events,
        record,
        round_id=0,
        agent_id="agent-1",
        evidence_ids=["evidence-restricted"],
    )
    write_events(events_path, events)
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ReplayValidationError, match="before introduction"):
        validate_replay(events_path, record_path)


def test_invisible_evidence_reference_is_rejected(tmp_path: Path) -> None:
    events_path, record_path, _ = run_paths(tmp_path)
    events = load_events(events_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    mutate_visibility(
        events,
        record,
        round_id=1,
        agent_id="agent-2",
        evidence_ids=["evidence-restricted"],
    )
    write_events(events_path, events)
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ReplayValidationError, match="not visible"):
        validate_replay(events_path, record_path)


def test_holdout_probe_material_is_rejected(tmp_path: Path) -> None:
    events_path, record_path, _ = run_paths(tmp_path)
    events = load_events(events_path)
    events[1]["payload"]["holdout_probe_response"] = "private response"
    write_events(events_path, events)

    with pytest.raises(ReplayValidationError, match="holdout probe"):
        validate_replay(events_path, record_path)


def test_existing_result_directory_is_not_overwritten(tmp_path: Path) -> None:
    events_path, _, config = run_paths(tmp_path, run_id="no-overwrite")
    original_events = events_path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ProtocolRunner(config, scenario=replay_scenario()).run()

    assert events_path.read_text(encoding="utf-8") == original_events
