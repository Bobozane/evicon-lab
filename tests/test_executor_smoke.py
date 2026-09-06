"""No-network smoke coverage for the executor demonstration command."""

from __future__ import annotations

import json

from evicon.executor_smoke import main, smoke_cases


def test_executor_smoke_contains_all_required_action_and_rejection_cases() -> None:
    labels = [label for label, _ in smoke_cases()]

    assert labels == [
        "no_op",
        "request_evidence",
        "blind_evidence_reflection",
        "solicit_dissent",
        "adaptive_exposure",
        "minority_report",
        "restructure",
        "invalid_target_agent",
        "over_budget_action",
        "invisible_evidence",
    ]
    plans = dict(smoke_cases())
    assert plans["no_op"].valid is True
    assert plans["invalid_target_agent"].valid is False
    assert plans["over_budget_action"].valid is False
    assert plans["invisible_evidence"].valid is False


def test_executor_smoke_prints_only_structured_plan_fields(capsys: object) -> None:
    assert main() == 0
    output = capsys.readouterr().out.strip().splitlines()  # type: ignore[attr-defined]
    rows = [json.loads(line) for line in output]

    assert len(rows) == 10
    assert set(rows[0]) == {
        "case",
        "action",
        "target_agent_ids",
        "mediator_event_type",
        "visibility_overrides",
        "speaking_priority_agents",
        "turn_order_override",
        "estimated_cost",
        "valid",
        "validation_errors",
    }
    assert all("prompt" not in json.dumps(row) for row in rows)
