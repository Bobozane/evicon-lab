"""End-to-end adaptive control-loop coverage with FakeLLM only."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from adaptive_cases import config, high_risk_signal, monitor_config, policy_config, providers, scenario, stable_signal

from evicon.adaptive_runner import AdaptiveProtocolRunner
from evicon.events import EventType
from evicon.models import ProtocolCondition
from evicon.replay import validate_replay


def _run(tmp_path: Path, signals: dict[int, dict[str, float]], *, budget: float = 0.5):
    online, candidates = providers(signals)
    runner = AdaptiveProtocolRunner(
        config(tmp_path, "adaptive-run", budget=budget),
        scenario=scenario(),
        online_state_provider=online,
        target_candidate_provider=candidates,
        monitor_config=monitor_config(),
        policy_config=policy_config(),
    )
    record = runner.run()
    directory = tmp_path / "adaptive-run"
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    return runner, record, events, directory


def test_stable_signals_create_no_plan_and_do_not_spend_budget(tmp_path: Path) -> None:
    runner, record, events, _ = _run(tmp_path, {round_id: stable_signal() for round_id in range(4)})

    assert record.status.value == "completed"
    assert not [event for event in events if event["event_type"] == EventType.INTERVENTION_PLAN_CREATED.value]
    assert runner.controller_state.spent_budget == 0.0
    assert runner.controller_state.remaining_budget == 0.5


def test_high_risk_plan_is_created_after_round_and_applied_next_round(tmp_path: Path) -> None:
    runner, record, events, directory = _run(
        tmp_path,
        {0: stable_signal(), 1: high_risk_signal(), 2: high_risk_signal(), 3: high_risk_signal()},
    )

    created = [event for event in events if event["event_type"] == EventType.INTERVENTION_PLAN_CREATED.value]
    scheduled = [event for event in events if event["event_type"] == EventType.INTERVENTION_PLAN_SCHEDULED.value]
    applied = [event for event in events if event["event_type"] == EventType.INTERVENTION_PLAN_APPLIED.value]
    assert len(created) == 1
    assert created[0]["round_id"] == 1
    assert scheduled[0]["payload"]["scheduled_round"] == 2
    assert applied[0]["round_id"] == 2
    assert runner.controller_state.spent_budget == pytest.approx(0.3)
    assert runner.controller_state.remaining_budget == pytest.approx(0.2)
    assert runner.controller_state.cooldown_until == 4
    assert "current_action" not in record.turns[0].message
    assert validate_replay(directory / "events.jsonl", directory / "run_record.json").validation == "passed"


def test_cooldown_suppresses_later_high_risk_plans(tmp_path: Path) -> None:
    _, _, events, _ = _run(tmp_path, {round_id: high_risk_signal() for round_id in range(4)})

    decisions = [event for event in events if event["event_type"] == EventType.POLICY_DECIDED.value]
    created = [event for event in events if event["event_type"] == EventType.INTERVENTION_PLAN_CREATED.value]
    assert len(decisions) == 4
    assert len(created) == 1
    assert decisions[2]["payload"]["action"] == "no_op"
    assert decisions[3]["payload"]["action"] == "no_op"


def test_insufficient_budget_produces_no_reservation_or_spend(tmp_path: Path) -> None:
    runner, _, events, _ = _run(
        tmp_path,
        {round_id: high_risk_signal() for round_id in range(4)},
        budget=0.1,
    )

    assert runner.controller_state.spent_budget == 0.0
    assert runner.controller_state.reserved_budget == 0.0
    assert not [event for event in events if event["event_type"] == EventType.BUDGET_RESERVED.value]
    assert not [event for event in events if event["event_type"] == EventType.BUDGET_SPENT.value]


def test_scripted_seed_and_signals_are_replay_deterministic(tmp_path: Path) -> None:
    first = _run(tmp_path / "first", {0: stable_signal(), 1: high_risk_signal(), 2: high_risk_signal(), 3: high_risk_signal()})
    second = _run(tmp_path / "second", {0: stable_signal(), 1: high_risk_signal(), 2: high_risk_signal(), 3: high_risk_signal()})

    first_events = [event for event in first[2] if event["event_type"] not in {"run_started"}]
    second_events = [event for event in second[2] if event["event_type"] not in {"run_started"}]
    assert first_events == second_events
    assert first[0].controller_state == second[0].controller_state


def test_adaptive_runner_opens_no_network_and_generates_no_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def reject_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("adaptive runner attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", reject_connection)
    _, _, events, _ = _run(tmp_path, {round_id: stable_signal() for round_id in range(4)})

    serialized = json.dumps(events, ensure_ascii=True).lower()
    assert "prompt" not in serialized
    assert "hidden_probe" not in serialized
