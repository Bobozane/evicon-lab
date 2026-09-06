"""Pure action-plan compilation coverage for InterventionExecutor."""

from __future__ import annotations

import copy
import json
import socket
from pathlib import Path

import pytest
from pydantic import ValidationError

from evicon.executor import (
    ExecutionContext,
    InterventionExecutor,
    InterventionPlan,
    MediatorEventType,
    PlanValidationError,
)
from evicon.models import AgentSpec, DialogueState, InterventionAction
from evicon.policy import InterventionDecision, PolicyReason


def context(**overrides: object) -> ExecutionContext:
    values: dict[str, object] = {
        "run_id": "executor-run",
        "scenario_id": "executor-scenario",
        "round_id": 3,
        "known_agent_ids": ["agent-a", "agent-b", "agent-c"],
        "visible_evidence_ids": ["evidence-0"],
        "visible_evidence_introduced_rounds": {"evidence-0": 0},
        "candidate_peer_turn_ids": ["turn-r2-agent-b"],
        "candidate_peer_turn_rounds": {"turn-r2-agent-b": 2},
        "current_budget": 1.0,
        "max_target_count": 2,
        "intervention_history": [],
        "last_action": None,
        "metadata": {"fixture": "executor"},
    }
    values.update(overrides)
    return ExecutionContext.model_validate(values)


def decision(
    action: InterventionAction,
    *,
    targets: list[str] | None = None,
    cost: float | None = None,
) -> InterventionDecision:
    return InterventionDecision(
        action=action,
        target_agent_ids=targets or [],
        reason_codes=[PolicyReason.HIGH_RISK],
        risk_score=0.8,
        estimated_cost=0.0 if action is InterventionAction.NO_OP else (0.2 if cost is None else cost),
        round_id=3,
        policy_version="executor-test-v1",
        cooldown_until=5,
        valid=True,
    )


def test_no_op_generates_a_zero_cost_empty_plan() -> None:
    plan = InterventionExecutor.plan(decision(InterventionAction.NO_OP), context())

    assert plan.valid is True
    assert plan.mediator_event_type is MediatorEventType.NONE
    assert plan.target_agent_ids == []
    assert plan.estimated_cost == 0.0
    assert plan.visibility_overrides == []
    assert plan.requires_target_response is False


def test_request_evidence_never_creates_evidence() -> None:
    plan = InterventionExecutor.plan(decision(InterventionAction.REQUEST_EVIDENCE), context())

    assert plan.valid is True
    assert plan.mediator_event_type is MediatorEventType.REQUEST_EVIDENCE
    assert plan.requested_evidence_ids == []
    assert plan.requires_target_response is True


def test_blind_reflection_removes_peer_turns_from_the_next_context() -> None:
    plan = InterventionExecutor.plan(
        decision(InterventionAction.BLIND_EVIDENCE_REFLECTION, targets=["agent-a"]),
        context(),
    )

    assert plan.valid is True
    assert plan.visibility_overrides[0].visible_evidence_ids == ["evidence-0"]
    assert plan.visibility_overrides[0].visible_peer_turn_ids == []
    assert plan.visibility_overrides[0].preserve_own_history is True


def test_solicit_dissent_sets_target_speaking_priority_without_a_position() -> None:
    plan = InterventionExecutor.plan(
        decision(InterventionAction.SOLICIT_DISSENT, targets=["agent-b"]),
        context(),
    )

    assert plan.valid is True
    assert plan.mediator_event_type is MediatorEventType.SOLICIT_DISSENT
    assert plan.speaking_priority_agents == ["agent-b"]
    assert plan.requires_target_response is True
    assert "stance" not in plan.model_dump_json()


def test_adaptive_exposure_only_uses_visible_earlier_peer_turns() -> None:
    plan = InterventionExecutor.plan(
        decision(InterventionAction.ADAPTIVE_EXPOSURE, targets=["agent-a"]),
        context(),
    )

    assert plan.valid is True
    assert plan.mediator_event_type is MediatorEventType.ADAPTIVE_EXPOSURE
    assert plan.visibility_overrides[0].visible_peer_turn_ids == ["turn-r2-agent-b"]


def test_minority_report_requires_final_retention_without_changing_position() -> None:
    plan = InterventionExecutor.plan(
        decision(InterventionAction.MINORITY_REPORT, targets=["agent-c"]),
        context(),
    )

    assert plan.valid is True
    assert plan.mediator_event_type is MediatorEventType.MINORITY_REPORT
    assert plan.requires_final_minority_report is True
    assert "position" not in plan.model_dump_json()


def test_restructure_produces_a_future_target_first_turn_order() -> None:
    plan = InterventionExecutor.plan(
        decision(InterventionAction.RESTRUCTURE, targets=["agent-c"]),
        context(),
    )

    assert plan.valid is True
    assert plan.mediator_event_type is MediatorEventType.RESTRUCTURE
    assert plan.turn_order_override == ["agent-c", "agent-a", "agent-b"]


def test_unknown_target_and_over_budget_return_inert_invalid_plans() -> None:
    unknown = InterventionExecutor.plan(
        decision(InterventionAction.SOLICIT_DISSENT, targets=["unknown"]), context()
    )
    over_budget = InterventionExecutor.plan(
        decision(InterventionAction.REQUEST_EVIDENCE, cost=1.1), context()
    )

    assert unknown.valid is False
    assert PlanValidationError.UNKNOWN_TARGET_AGENT in unknown.validation_errors
    assert unknown.mediator_event_type is MediatorEventType.INVALID
    assert over_budget.valid is False
    assert PlanValidationError.COST_EXCEEDS_BUDGET in over_budget.validation_errors


def test_duplicate_targets_and_hidden_prompt_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="target_agent_ids"):
        InterventionDecision.model_validate(
            decision(InterventionAction.SOLICIT_DISSENT, targets=["agent-a"]).model_dump()
            | {"target_agent_ids": ["agent-a", "agent-a"]}
        )
    with pytest.raises(ValidationError, match="prompt"):
        InterventionPlan.model_validate(
            {
                "plan_id": "plan-test",
                "run_id": "executor-run",
                "scenario_id": "executor-scenario",
                "round_id": 3,
                "action": "no_op",
                "target_agent_ids": [],
                "mediator_event_type": "none",
                "estimated_cost": 0.0,
                "policy_version": "executor-test-v1",
                "valid": True,
                "validation_errors": [],
                "prompt": "must not enter plan",
            }
        )


def test_executor_is_deterministic_json_serializable_and_has_no_prompt() -> None:
    item = decision(InterventionAction.SOLICIT_DISSENT, targets=["agent-a"])
    first = InterventionExecutor.plan(item, context())
    second = InterventionExecutor.plan(item, context())

    assert first == second
    assert InterventionPlan.model_validate_json(first.model_dump_json()) == first
    assert "prompt" not in InterventionPlan.model_fields
    assert "hidden_probe" not in first.model_dump_json()


def test_executor_does_not_modify_dialogue_state_or_create_files(tmp_path: Path) -> None:
    state = DialogueState(
        run_id="state-run",
        scenario_id="state-scenario",
        current_round=0,
        agents=[AgentSpec(agent_id="state-agent", role="participant")],
        turns=[],
        evidence_cards=[],
        value_profiles=[],
        intervention_budget=1.0,
        metadata={},
    )
    before_state = copy.deepcopy(state.model_dump())
    before_files = list(tmp_path.iterdir())

    InterventionExecutor.plan(decision(InterventionAction.REQUEST_EVIDENCE), context())

    assert state.model_dump() == before_state
    assert list(tmp_path.iterdir()) == before_files


def test_executor_opens_no_network_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("InterventionExecutor attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", reject_connection)
    plan = InterventionExecutor.plan(decision(InterventionAction.REQUEST_EVIDENCE), context())

    assert plan.valid is True
    assert json.loads(plan.model_dump_json())["mediator_event_type"] == "request_evidence"
