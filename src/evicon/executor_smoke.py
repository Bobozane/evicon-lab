"""Local-only smoke cases for structured InterventionPlan compilation."""

from __future__ import annotations

import json

from .executor import (
    ExecutionContext,
    InterventionExecutor,
    InterventionPlan,
    MediatorEventType,
    validate_intervention_plan,
)
from .models.protocols import InterventionAction
from .policy import InterventionDecision, PolicyReason


def smoke_context(**overrides: object) -> ExecutionContext:
    """Return observable local data only; no dialogue, prompt, or probe data."""
    values: dict[str, object] = {
        "run_id": "executor-smoke",
        "scenario_id": "executor-smoke-scenario",
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
        "metadata": {"purpose": "local smoke"},
    }
    values.update(overrides)
    return ExecutionContext.model_validate(values)


def smoke_decision(
    action: InterventionAction,
    *,
    targets: list[str] | None = None,
    cost: float | None = None,
) -> InterventionDecision:
    """Construct a policy-shaped decision without running a monitor or policy."""
    return InterventionDecision(
        action=action,
        target_agent_ids=targets or [],
        reason_codes=[PolicyReason.HIGH_RISK],
        risk_score=0.8,
        estimated_cost=0.0 if action is InterventionAction.NO_OP else (0.2 if cost is None else cost),
        round_id=3,
        policy_version="executor-smoke-v1",
        cooldown_until=5,
        valid=True,
    )


def smoke_cases() -> list[tuple[str, InterventionPlan]]:
    """Compile seven valid and three validation-only local examples."""
    current = smoke_context()
    cases = [
        ("no_op", smoke_decision(InterventionAction.NO_OP)),
        ("request_evidence", smoke_decision(InterventionAction.REQUEST_EVIDENCE)),
        (
            "blind_evidence_reflection",
            smoke_decision(InterventionAction.BLIND_EVIDENCE_REFLECTION, targets=["agent-a"]),
        ),
        ("solicit_dissent", smoke_decision(InterventionAction.SOLICIT_DISSENT, targets=["agent-b"])),
        (
            "adaptive_exposure",
            smoke_decision(InterventionAction.ADAPTIVE_EXPOSURE, targets=["agent-a"]),
        ),
        ("minority_report", smoke_decision(InterventionAction.MINORITY_REPORT, targets=["agent-c"])),
        ("restructure", smoke_decision(InterventionAction.RESTRUCTURE, targets=["agent-c"])),
        ("invalid_target_agent", smoke_decision(InterventionAction.SOLICIT_DISSENT, targets=["unknown"])),
        ("over_budget_action", smoke_decision(InterventionAction.REQUEST_EVIDENCE, cost=0.2)),
    ]
    plans = [
        (
            label,
            InterventionExecutor.plan(
                item,
                smoke_context(current_budget=0.1) if label == "over_budget_action" else current,
            ),
        )
        for label, item in cases
    ]

    visible_request = InterventionExecutor.plan(
        smoke_decision(InterventionAction.REQUEST_EVIDENCE), current
    )
    invalid_evidence = visible_request.model_copy(
        update={"requested_evidence_ids": ["evidence-not-visible"]}
    )
    validation = validate_intervention_plan(invalid_evidence, current)
    plans.append(
        (
            "invisible_evidence",
            invalid_evidence.model_copy(
                update={
                    "mediator_event_type": MediatorEventType.INVALID,
                    "valid": False,
                    "validation_errors": validation.errors,
                }
            ),
        )
    )
    return plans


def main() -> int:
    for label, plan in smoke_cases():
        print(
            json.dumps(
                {
                    "case": label,
                    "action": plan.action.value,
                    "target_agent_ids": plan.target_agent_ids,
                    "mediator_event_type": plan.mediator_event_type.value,
                    "visibility_overrides": [
                        override.model_dump(mode="json") for override in plan.visibility_overrides
                    ],
                    "speaking_priority_agents": plan.speaking_priority_agents,
                    "turn_order_override": plan.turn_order_override,
                    "estimated_cost": plan.estimated_cost,
                    "valid": plan.valid,
                    "validation_errors": [error.value for error in plan.validation_errors],
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
