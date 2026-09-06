"""Validation-only coverage for executable-description InterventionPlans."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evicon.executor import (
    ExecutionContext,
    InterventionExecutor,
    PlanValidationError,
    validate_intervention_plan,
)
from evicon.models import InterventionAction
from evicon.policy import InterventionDecision, PolicyReason


def _context(**overrides: object) -> ExecutionContext:
    values: dict[str, object] = {
        "run_id": "validation-run",
        "scenario_id": "validation-scenario",
        "round_id": 2,
        "known_agent_ids": ["agent-a", "agent-b"],
        "visible_evidence_ids": ["evidence-visible"],
        "visible_evidence_introduced_rounds": {"evidence-visible": 1},
        "candidate_peer_turn_ids": ["turn-r1-agent-b"],
        "candidate_peer_turn_rounds": {"turn-r1-agent-b": 1},
        "current_budget": 1.0,
        "max_target_count": 1,
        "intervention_history": [],
        "last_action": None,
        "metadata": {},
    }
    values.update(overrides)
    return ExecutionContext.model_validate(values)


def _decision() -> InterventionDecision:
    return InterventionDecision(
        action=InterventionAction.REQUEST_EVIDENCE,
        target_agent_ids=[],
        reason_codes=[PolicyReason.INSUFFICIENT_EVIDENCE_QUALITY],
        risk_score=0.8,
        estimated_cost=0.2,
        round_id=2,
        policy_version="validation-v1",
        cooldown_until=3,
        valid=True,
    )


def test_plan_validator_rejects_invisible_evidence_without_mutating_plan() -> None:
    plan = InterventionExecutor.plan(_decision(), _context())
    altered = plan.model_copy(update={"requested_evidence_ids": ["evidence-hidden"]})
    before = altered.model_dump()

    result = validate_intervention_plan(altered, _context())

    assert result.valid is False
    assert PlanValidationError.UNKNOWN_EVIDENCE in result.errors
    assert altered.model_dump() == before


def test_execution_context_rejects_future_peer_turns_and_private_payloads() -> None:
    with pytest.raises(ValidationError, match="strictly earlier"):
        _context(candidate_peer_turn_rounds={"turn-r1-agent-b": 2})
    with pytest.raises(ValidationError, match="private or prompt"):
        _context(metadata={"hidden_probe_response": "private"})


def test_plan_validator_rejects_an_unknown_or_future_peer_reference() -> None:
    decision = InterventionDecision(
        action=InterventionAction.ADAPTIVE_EXPOSURE,
        target_agent_ids=["agent-a"],
        reason_codes=[PolicyReason.HIGH_RISK],
        risk_score=0.8,
        estimated_cost=0.2,
        round_id=2,
        policy_version="validation-v1",
        cooldown_until=3,
        valid=True,
    )
    plan = InterventionExecutor.plan(decision, _context())
    override = plan.visibility_overrides[0].model_copy(
        update={"visible_peer_turn_ids": ["turn-future"]}
    )
    altered = plan.model_copy(update={"visibility_overrides": [override]})

    result = validate_intervention_plan(altered, _context())

    assert result.valid is False
    assert PlanValidationError.UNKNOWN_PEER_TURN in result.errors


def test_context_rejects_duplicate_agents_and_missing_round_map_entries() -> None:
    with pytest.raises(ValidationError, match="known_agent_ids"):
        _context(known_agent_ids=["agent-a", "agent-a"])
    with pytest.raises(ValidationError, match="visible_evidence_introduced_rounds"):
        _context(visible_evidence_introduced_rounds={})
