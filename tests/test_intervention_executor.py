"""Collected executor tests, including the valid policy-cost budget boundary."""

from __future__ import annotations

from executor_cases import (
    context,
    decision,
    test_adaptive_exposure_only_uses_visible_earlier_peer_turns,
    test_blind_reflection_removes_peer_turns_from_the_next_context,
    test_duplicate_targets_and_hidden_prompt_fields_are_rejected,
    test_executor_does_not_modify_dialogue_state_or_create_files,
    test_executor_is_deterministic_json_serializable_and_has_no_prompt,
    test_executor_opens_no_network_connections,
    test_minority_report_requires_final_retention_without_changing_position,
    test_no_op_generates_a_zero_cost_empty_plan,
    test_request_evidence_never_creates_evidence,
    test_restructure_produces_a_future_target_first_turn_order,
    test_solicit_dissent_sets_target_speaking_priority_without_a_position,
)

from evicon.executor import InterventionExecutor, MediatorEventType, PlanValidationError
from evicon.models import InterventionAction


def test_unknown_target_and_over_budget_return_inert_invalid_plans() -> None:
    unknown = InterventionExecutor.plan(
        decision(InterventionAction.SOLICIT_DISSENT, targets=["unknown"]), context()
    )
    over_budget = InterventionExecutor.plan(
        decision(InterventionAction.REQUEST_EVIDENCE, cost=0.2), context(current_budget=0.1)
    )

    assert unknown.valid is False
    assert PlanValidationError.UNKNOWN_TARGET_AGENT in unknown.validation_errors
    assert unknown.mediator_event_type is MediatorEventType.INVALID
    assert over_budget.valid is False
    assert PlanValidationError.COST_EXCEEDS_BUDGET in over_budget.validation_errors
