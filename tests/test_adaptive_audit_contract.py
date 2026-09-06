"""Adaptive audit-envelope and plan-rejection accounting coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_cases import config, high_risk_signal, monitor_config, policy_config, providers, scenario, stable_signal

from evicon.adaptive_runner import AdaptiveProtocolRunner
from evicon.controlled_runner import ControlledProtocolRunner
from evicon.events import EventType
from evicon.replay import ReplayValidationError, validate_replay


_ADAPTIVE_EVENTS = {
    EventType.ONLINE_STATE_OBSERVED.value,
    EventType.MONITOR_EVALUATED.value,
    EventType.TARGET_CANDIDATES_BUILT.value,
    EventType.POLICY_DECIDED.value,
    EventType.INTERVENTION_PLAN_CREATED.value,
    EventType.INTERVENTION_PLAN_SCHEDULED.value,
    EventType.BUDGET_RESERVED.value,
    EventType.BUDGET_SPENT.value,
    EventType.BUDGET_RELEASED.value,
    EventType.COOLDOWN_UPDATED.value,
}


def _events(directory: Path, run_id: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (directory / run_id / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _adaptive_runner(tmp_path: Path, run_id: str, signals: dict[int, dict[str, float]]) -> AdaptiveProtocolRunner:
    online, candidates = providers(signals)
    return AdaptiveProtocolRunner(
        config(tmp_path, run_id),
        scenario=scenario(),
        online_state_provider=online,
        target_candidate_provider=candidates,
        monitor_config=monitor_config(),
        policy_config=policy_config(),
    )


def test_adaptive_audit_events_have_the_required_structured_envelope(tmp_path: Path) -> None:
    run_id = "adaptive-audit"
    _adaptive_runner(
        tmp_path,
        run_id,
        {0: stable_signal(), 1: high_risk_signal(), 2: high_risk_signal(), 3: high_risk_signal()},
    ).run()
    events = _events(tmp_path, run_id)

    for event in events:
        if event["event_type"] not in _ADAPTIVE_EVENTS:
            continue
        assert event["scenario_id"] == "adaptive-scenario"
        payload = event["payload"]
        assert isinstance(payload, dict)
        assert payload["scenario_id"] == "adaptive-scenario"
        assert "action" in payload
        assert isinstance(payload["reason_codes"], list)
        assert "budget_before" in payload
        assert "budget_after" in payload
        assert "cooldown_until" in payload
        assert isinstance(payload["warnings"], list)
        assert isinstance(payload["errors"], list)


def test_replay_rejects_an_adaptive_event_missing_the_audit_envelope(tmp_path: Path) -> None:
    run_id = "adaptive-envelope-tamper"
    _adaptive_runner(tmp_path, run_id, {round_id: stable_signal() for round_id in range(4)}).run()
    events_path = tmp_path / run_id / "events.jsonl"
    record_path = tmp_path / run_id / "run_record.json"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    next(event for event in events if event["event_type"] == EventType.ONLINE_STATE_OBSERVED.value)["payload"].pop("action")
    events_path.write_text(
        "".join(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )

    with pytest.raises(ReplayValidationError, match="audit fields"):
        validate_replay(events_path, record_path)


class _RejectingAdaptiveRunner(AdaptiveProtocolRunner):
    """Test double that invalidates an already-reserved plan at the next boundary."""

    def _apply_pending_plan(self, state: object, round_id: int, logger: object, applied_plan_ids: set[str]) -> object:
        if self._pending_plan is not None:
            self._pending_plan = self._pending_plan.model_copy(
                update={"target_agent_ids": ["unknown-public-agent"]}
            )
        return super()._apply_pending_plan(state, round_id, logger, applied_plan_ids)


def test_rejected_reserved_plan_releases_budget_without_spending(tmp_path: Path) -> None:
    run_id = "adaptive-rejected"
    online, candidates = providers({round_id: high_risk_signal() for round_id in range(4)})
    runner = _RejectingAdaptiveRunner(
        config(tmp_path, run_id),
        scenario=scenario(),
        online_state_provider=online,
        target_candidate_provider=candidates,
        monitor_config=monitor_config(),
        policy_config=policy_config(),
    )
    record = runner.run()
    event_types = [event["event_type"] for event in _events(tmp_path, run_id)]

    assert record.status.value == "completed"
    assert EventType.INTERVENTION_PLAN_REJECTED.value in event_types
    assert EventType.BUDGET_RESERVED.value in event_types
    assert EventType.BUDGET_RELEASED.value in event_types
    assert EventType.BUDGET_SPENT.value not in event_types
    assert runner.controller_state.reserved_budget == 0.0
    assert runner.controller_state.spent_budget == 0.0
    assert runner.controller_state.remaining_budget == pytest.approx(0.5)
    assert validate_replay(tmp_path / run_id / "events.jsonl", tmp_path / run_id / "run_record.json").validation == "passed"


def test_stable_adaptive_dialogue_matches_controlled_runner_without_a_schedule(tmp_path: Path) -> None:
    run_id = "adaptive-no-op"
    adaptive = _adaptive_runner(
        tmp_path / "adaptive",
        run_id,
        {round_id: stable_signal() for round_id in range(4)},
    ).run()
    controlled = ControlledProtocolRunner(
        config(tmp_path / "controlled", run_id),
        scenario=scenario(),
        intervention_schedule=None,
    ).run()

    assert adaptive.turns == controlled.turns
    assert adaptive.evidence_exposures == controlled.evidence_exposures
