"""Read-only replay checks specific to logged controlled intervention plans."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from controlled_cases import config, plan, scenario

from evicon.controlled_runner import ControlledProtocolRunner, InterventionSchedule
from evicon.models import InterventionAction
from evicon.replay import ReplayValidationError, validate_replay


def _schedule() -> InterventionSchedule:
    return InterventionSchedule(
        schedule_id="replay-schedule",
        run_id="controlled-run",
        scenario_id="controlled-scenario",
        plans=[
            plan(
                InterventionAction.ADAPTIVE_EXPOSURE,
                round_id=1,
                targets=["agent-a"],
                peer_turn_ids=["turn-r0-agent-b"],
            )
        ],
        metadata={},
    )


def _run(tmp_path: Path) -> tuple[Path, Path]:
    ControlledProtocolRunner(
        config(tmp_path, "controlled-run"),
        scenario=scenario(),
        intervention_schedule=_schedule(),
    ).run()
    directory = tmp_path / "controlled-run"
    return directory / "events.jsonl", directory / "run_record.json"


def _write_events(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )


def test_controlled_log_replays_when_seen_and_applied_plan_match(tmp_path: Path) -> None:
    events_path, record_path = _run(tmp_path)

    assert validate_replay(events_path, record_path).validation == "passed"


def test_replay_rejects_applied_event_without_a_matching_seen_plan(tmp_path: Path) -> None:
    events_path, record_path = _run(tmp_path)
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    events = [event for event in events if event["event_type"] != "intervention_plan_seen"]
    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence
        event["event_id"] = f"controlled-run:{sequence:06d}"
    _write_events(events_path, events)

    with pytest.raises(ReplayValidationError, match="earlier intervention_plan_seen"):
        validate_replay(events_path, record_path)


def test_replay_rejects_adaptive_exposure_of_a_future_turn(tmp_path: Path) -> None:
    events_path, record_path = _run(tmp_path)
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    for event in events:
        if event["event_type"] not in {
            "intervention_plan_seen",
            "intervention_plan_applied",
            "visibility_override_applied",
        }:
            continue
        overrides = event["payload"]["visibility_overrides"]
        overrides[0]["visible_peer_turn_ids"] = ["turn-r2-agent-b"]
    _write_events(events_path, events)

    with pytest.raises(ReplayValidationError, match="strictly earlier peer turn"):
        validate_replay(events_path, record_path)
