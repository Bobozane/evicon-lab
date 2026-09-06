"""Adaptive replay tamper detection coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_cases import config, high_risk_signal, monitor_config, policy_config, providers, scenario, stable_signal

from evicon.adaptive_runner import AdaptiveProtocolRunner
from evicon.replay import ReplayValidationError, validate_replay


def _run(tmp_path: Path) -> tuple[Path, Path]:
    online, candidates = providers({0: stable_signal(), 1: high_risk_signal(), 2: high_risk_signal(), 3: high_risk_signal()})
    AdaptiveProtocolRunner(
        config(tmp_path, "adaptive-run"),
        scenario=scenario(),
        online_state_provider=online,
        target_candidate_provider=candidates,
        monitor_config=monitor_config(),
        policy_config=policy_config(),
    ).run()
    directory = tmp_path / "adaptive-run"
    return directory / "events.jsonl", directory / "run_record.json"


def _write(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )


def test_adaptive_replay_passes_a_valid_run(tmp_path: Path) -> None:
    events, record = _run(tmp_path)

    assert validate_replay(events, record).validation == "passed"


def test_replay_rejects_budget_spent_without_applied_plan(tmp_path: Path) -> None:
    events_path, record_path = _run(tmp_path)
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    events = [event for event in events if event["event_type"] != "intervention_plan_applied"]
    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence
        event["event_id"] = f"adaptive-run:{sequence:06d}"
    _write(events_path, events)

    with pytest.raises(ReplayValidationError, match="plan_applied|budget_spent|scheduled"):
        validate_replay(events_path, record_path)


def test_replay_rejects_monitor_event_without_observation(tmp_path: Path) -> None:
    events_path, record_path = _run(tmp_path)
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    events = [event for event in events if event["event_type"] != "online_state_observed"]
    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence
        event["event_id"] = f"adaptive-run:{sequence:06d}"
    _write(events_path, events)

    with pytest.raises(ReplayValidationError, match="online_state_observed"):
        validate_replay(events_path, record_path)
