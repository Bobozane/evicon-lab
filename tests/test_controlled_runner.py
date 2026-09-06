"""End-to-end local controlled-runner coverage with explicit schedules only."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from controlled_cases import config, plan, scenario

from evicon.controlled_runner import ControlledProtocolRunner, InterventionSchedule
from evicon.events import EventType
from evicon.models import InterventionAction
from evicon.replay import validate_replay
from evicon.runner import ProtocolRunner


def _schedule(*plans: object) -> InterventionSchedule:
    return InterventionSchedule(
        schedule_id="controlled-schedule",
        run_id="controlled-run",
        scenario_id="controlled-scenario",
        plans=list(plans),
        metadata={},
    )


def _events(output_directory: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (output_directory / "controlled-run" / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]


def _turn(record: object, round_id: int, agent_id: str) -> object:
    matches = [turn for turn in record.turns if turn.round_id == round_id and turn.speaker_id == agent_id]
    assert len(matches) == 1
    return matches[0]


def test_no_schedule_uses_the_original_runner_behavior(tmp_path: Path) -> None:
    base = ProtocolRunner(config(tmp_path / "base", "controlled-run"), scenario=scenario())
    controlled = ControlledProtocolRunner(
        config(tmp_path / "controlled", "controlled-run"),
        scenario=scenario(),
        intervention_schedule=None,
    )

    base_record = base.run()
    controlled_record = controlled.run()

    assert base_record.turns == controlled_record.turns
    assert base_record.evidence_exposures == controlled_record.evidence_exposures
    assert [event["event_type"] for event in _events(tmp_path / "controlled")] == [
        event["event_type"] for event in _events(tmp_path / "base")
    ]


def test_empty_schedule_produces_no_extra_intervention_events(tmp_path: Path) -> None:
    runner = ControlledProtocolRunner(
        config(tmp_path, "controlled-run"),
        scenario=scenario(),
        intervention_schedule=_schedule(),
    )
    runner.run()

    event_types = [event["event_type"] for event in _events(tmp_path)]
    assert not any(event_type.startswith("intervention_") for event_type in event_types)


def test_blind_reflection_changes_only_the_target_next_round_visibility(tmp_path: Path) -> None:
    runner = ControlledProtocolRunner(
        config(tmp_path, "controlled-run"),
        scenario=scenario(),
        intervention_schedule=_schedule(
            plan(InterventionAction.BLIND_EVIDENCE_REFLECTION, round_id=0, targets=["agent-a"])
        ),
    )
    record = runner.run()

    assert _turn(record, 1, "agent-a").visible_peer_turn_ids == []
    assert _turn(record, 1, "agent-b").visible_peer_turn_ids
    events = _events(tmp_path)
    assert EventType.VISIBILITY_OVERRIDE_APPLIED.value in [event["event_type"] for event in events]


def test_solicit_dissent_and_restructure_change_only_next_round_order(tmp_path: Path) -> None:
    runner = ControlledProtocolRunner(
        config(tmp_path, "controlled-run"),
        scenario=scenario(),
        intervention_schedule=_schedule(
            plan(InterventionAction.SOLICIT_DISSENT, round_id=0, targets=["agent-c"]),
            plan(InterventionAction.RESTRUCTURE, round_id=1, targets=["agent-b"]),
        ),
    )
    record = runner.run()

    assert [turn.speaker_id for turn in record.turns if turn.round_id == 0] == [
        "agent-a",
        "agent-b",
        "agent-c",
    ]
    assert [turn.speaker_id for turn in record.turns if turn.round_id == 1] == [
        "agent-c",
        "agent-a",
        "agent-b",
    ]
    assert [turn.speaker_id for turn in record.turns if turn.round_id == 2] == [
        "agent-b",
        "agent-a",
        "agent-c",
    ]


def test_request_and_adaptive_exposure_are_structured_and_prior_only(tmp_path: Path) -> None:
    runner = ControlledProtocolRunner(
        config(tmp_path, "controlled-run"),
        scenario=scenario(),
        intervention_schedule=_schedule(
            plan(InterventionAction.REQUEST_EVIDENCE, round_id=0, targets=["agent-a"]),
            plan(
                InterventionAction.ADAPTIVE_EXPOSURE,
                round_id=1,
                targets=["agent-a"],
                peer_turn_ids=["turn-r0-agent-b"],
            ),
        ),
    )
    record = runner.run()

    assert "evidence_request=true" in _turn(record, 1, "agent-a").message
    assert _turn(record, 2, "agent-a").visible_peer_turn_ids == ["turn-r0-agent-b"]
    assert all(
        turn_id.startswith("turn-r0-")
        for turn_id in _turn(record, 2, "agent-a").visible_peer_turn_ids
    )


def test_same_round_snapshot_prevents_order_leakage_and_log_replays(tmp_path: Path) -> None:
    runner = ControlledProtocolRunner(
        config(tmp_path, "controlled-run"),
        scenario=scenario(),
        intervention_schedule=_schedule(
            plan(InterventionAction.BLIND_EVIDENCE_REFLECTION, round_id=0, targets=["agent-a"])
        ),
    )
    runner.run()
    events = _events(tmp_path)
    record_path = tmp_path / "controlled-run" / "run_record.json"

    for event in events:
        if event["event_type"] != EventType.LLM_REQUEST.value:
            continue
        payload = event["payload"]
        assert isinstance(payload, dict)
        for peer_turn_id in payload["visible_peer_turn_ids"]:
            peer_round = int(peer_turn_id.split("-")[1][1:])
            assert peer_round < event["round_id"]
    assert validate_replay(tmp_path / "controlled-run" / "events.jsonl", record_path).validation == "passed"


def test_invalid_target_or_future_turn_plan_is_rejected_and_fails_the_run(tmp_path: Path) -> None:
    invalid_target = plan(
        InterventionAction.SOLICIT_DISSENT, round_id=0, targets=["agent-a"]
    ).model_copy(update={"target_agent_ids": ["unknown"]})
    runner = ControlledProtocolRunner(
        config(tmp_path / "target", "controlled-run"),
        scenario=scenario(),
        intervention_schedule=_schedule(invalid_target),
    )
    with pytest.raises(ValueError, match="rejected"):
        runner.run()
    assert EventType.INTERVENTION_PLAN_REJECTED.value in [
        event["event_type"] for event in _events(tmp_path / "target")
    ]

    future_turn = plan(
        InterventionAction.ADAPTIVE_EXPOSURE,
        round_id=1,
        targets=["agent-a"],
        peer_turn_ids=["turn-r0-agent-b"],
    ).model_copy(update={"round_id": 0})
    future_runner = ControlledProtocolRunner(
        config(tmp_path / "future", "controlled-run"),
        scenario=scenario(),
        intervention_schedule=_schedule(future_turn),
    )
    with pytest.raises(ValueError, match="rejected"):
        future_runner.run()


def test_controlled_runner_opens_no_network_connections(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def reject_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("ControlledProtocolRunner attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", reject_connection)
    record = ControlledProtocolRunner(
        config(tmp_path, "controlled-run"),
        scenario=scenario(),
        intervention_schedule=_schedule(plan(InterventionAction.NO_OP, round_id=0)),
    ).run()

    assert record.status.value == "completed"
