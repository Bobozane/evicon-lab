"""Public pure executor contracts and deterministic action-plan compiler."""

from __future__ import annotations

from . import intervention_executor_core as _core
from .models.protocols import InterventionAction
from .policy import InterventionDecision

ExecutionContext = _core.ExecutionContext
InterventionPlan = _core.InterventionPlan
MediatorEventType = _core.MediatorEventType
PlanValidationError = _core.PlanValidationError
PlanValidationResult = _core.PlanValidationResult
VisibilityOverride = _core.VisibilityOverride


class InterventionExecutor:
    """Compile policy decisions into plans without applying them to state."""

    @staticmethod
    def plan(decision: InterventionDecision, context: ExecutionContext) -> InterventionPlan:
        """Return a complete future-action description or an inert invalid plan."""
        errors = _core._decision_context_errors(decision, context)
        if errors:
            return _invalid_plan(decision, context, errors)
        action_fields, action_errors = _action_fields(decision, context)
        if action_errors:
            return _invalid_plan(decision, context, action_errors)
        plan = InterventionPlan(
            plan_id=_core._plan_id(decision),
            run_id=context.run_id,
            scenario_id=context.scenario_id,
            round_id=decision.round_id,
            action=decision.action,
            target_agent_ids=list(decision.target_agent_ids),
            estimated_cost=decision.estimated_cost,
            reason_codes=[reason.value for reason in decision.reason_codes],
            policy_version=decision.policy_version,
            valid=True,
            validation_errors=[],
            **action_fields,
        )
        validation = validate_intervention_plan(plan, context)
        return plan if validation.valid else _invalid_plan(decision, context, validation.errors)


def validate_intervention_plan(
    plan: InterventionPlan,
    context: ExecutionContext,
) -> PlanValidationResult:
    """Purely validate plan/context consistency; no runtime effect is performed."""
    errors: list[PlanValidationError] = list(plan.validation_errors)
    if not plan.plan_id.startswith("plan-"):
        errors.append(PlanValidationError.PLAN_ID_MISMATCH)
    if (
        plan.run_id != context.run_id
        or plan.scenario_id != context.scenario_id
        or plan.round_id != context.round_id
    ):
        errors.append(PlanValidationError.CONTEXT_MISMATCH)
    if len(plan.target_agent_ids) != len(set(plan.target_agent_ids)):
        errors.append(PlanValidationError.DUPLICATE_TARGET_AGENT)
    if any(agent_id not in context.known_agent_ids for agent_id in plan.target_agent_ids):
        errors.append(PlanValidationError.UNKNOWN_TARGET_AGENT)
    if len(plan.target_agent_ids) > context.max_target_count:
        errors.append(PlanValidationError.TARGET_LIMIT_EXCEEDED)
    if plan.estimated_cost > context.current_budget:
        errors.append(PlanValidationError.COST_EXCEEDS_BUDGET)
    if any(evidence_id not in context.visible_evidence_ids for evidence_id in plan.requested_evidence_ids):
        errors.append(PlanValidationError.UNKNOWN_EVIDENCE)
    errors.extend(_core._validate_overrides(plan, context))
    errors.extend(_validate_action_semantics(plan, context))
    unique_errors = _core._unique_errors(errors)
    return PlanValidationResult(valid=not unique_errors, errors=unique_errors, warnings=[])


def _action_fields(
    decision: InterventionDecision,
    context: ExecutionContext,
) -> tuple[dict[str, object], list[PlanValidationError]]:
    """Define each action's structural semantics without producing language."""
    action = decision.action
    targets = list(decision.target_agent_ids)
    fields: dict[str, object] = {
        "mediator_event_type": _event_type(action),
        "requested_evidence_ids": [],
        "visibility_overrides": [],
        "speaking_priority_agents": [],
        "turn_order_override": [],
        "requires_target_response": False,
        "requires_final_minority_report": False,
    }
    if action is InterventionAction.NO_OP:
        if targets or decision.estimated_cost != 0.0:
            return fields, [PlanValidationError.INVALID_NO_OP]
        return fields, []
    if action is InterventionAction.REQUEST_EVIDENCE:
        fields["requires_target_response"] = True
        return fields, []
    if action is InterventionAction.BLIND_EVIDENCE_REFLECTION:
        if not targets:
            return fields, [PlanValidationError.INVALID_ACTION_SEMANTICS]
        fields["visibility_overrides"] = [
            VisibilityOverride(
                agent_id=agent_id,
                visible_evidence_ids=list(context.visible_evidence_ids),
                visible_peer_turn_ids=[],
                preserve_own_history=True,
            )
            for agent_id in targets
        ]
        fields["requires_target_response"] = True
        return fields, []
    if action is InterventionAction.SOLICIT_DISSENT:
        if not targets:
            return fields, [PlanValidationError.INVALID_ACTION_SEMANTICS]
        fields["speaking_priority_agents"] = targets
        fields["requires_target_response"] = True
        return fields, []
    if action is InterventionAction.ADAPTIVE_EXPOSURE:
        if not targets or not context.candidate_peer_turn_ids:
            return fields, [PlanValidationError.INVALID_ACTION_SEMANTICS]
        fields["visibility_overrides"] = [
            VisibilityOverride(
                agent_id=agent_id,
                visible_evidence_ids=list(context.visible_evidence_ids),
                visible_peer_turn_ids=list(context.candidate_peer_turn_ids),
                preserve_own_history=True,
            )
            for agent_id in targets
        ]
        fields["requires_target_response"] = True
        return fields, []
    if action is InterventionAction.MINORITY_REPORT:
        if not targets:
            return fields, [PlanValidationError.INVALID_ACTION_SEMANTICS]
        fields["requires_final_minority_report"] = True
        return fields, []
    if action is InterventionAction.RESTRUCTURE:
        if not targets:
            return fields, [PlanValidationError.INVALID_ACTION_SEMANTICS]
        fields["turn_order_override"] = targets + [
            agent_id for agent_id in context.known_agent_ids if agent_id not in targets
        ]
        fields["requires_target_response"] = True
        return fields, []
    return fields, [PlanValidationError.INVALID_ACTION_SEMANTICS]


def _validate_action_semantics(
    plan: InterventionPlan,
    context: ExecutionContext,
) -> list[PlanValidationError]:
    errors: list[PlanValidationError] = []
    if plan.mediator_event_type is not _event_type(plan.action):
        errors.append(PlanValidationError.INVALID_ACTION_SEMANTICS)
    override_ids = [override.agent_id for override in plan.visibility_overrides]
    if plan.action is InterventionAction.NO_OP:
        if (
            plan.target_agent_ids
            or plan.estimated_cost != 0.0
            or plan.requested_evidence_ids
            or plan.visibility_overrides
            or plan.speaking_priority_agents
            or plan.turn_order_override
            or plan.requires_target_response
            or plan.requires_final_minority_report
        ):
            errors.append(PlanValidationError.INVALID_NO_OP)
    elif plan.action is InterventionAction.REQUEST_EVIDENCE:
        if not plan.requires_target_response or plan.visibility_overrides:
            errors.append(PlanValidationError.INVALID_ACTION_SEMANTICS)
    elif plan.action is InterventionAction.BLIND_EVIDENCE_REFLECTION:
        if (
            not plan.target_agent_ids
            or override_ids != plan.target_agent_ids
            or not plan.requires_target_response
            or any(override.visible_peer_turn_ids for override in plan.visibility_overrides)
        ):
            errors.append(PlanValidationError.INVALID_VISIBILITY_OVERRIDE)
    elif plan.action is InterventionAction.SOLICIT_DISSENT:
        if (
            not plan.target_agent_ids
            or plan.speaking_priority_agents != plan.target_agent_ids
            or not plan.requires_target_response
            or plan.visibility_overrides
        ):
            errors.append(PlanValidationError.INVALID_ACTION_SEMANTICS)
    elif plan.action is InterventionAction.ADAPTIVE_EXPOSURE:
        if (
            not plan.target_agent_ids
            or not context.candidate_peer_turn_ids
            or override_ids != plan.target_agent_ids
            or any(not override.visible_peer_turn_ids for override in plan.visibility_overrides)
            or not plan.requires_target_response
        ):
            errors.append(PlanValidationError.INVALID_VISIBILITY_OVERRIDE)
    elif plan.action is InterventionAction.MINORITY_REPORT:
        if not plan.target_agent_ids or not plan.requires_final_minority_report:
            errors.append(PlanValidationError.INVALID_ACTION_SEMANTICS)
    elif plan.action is InterventionAction.RESTRUCTURE:
        expected_order = plan.target_agent_ids + [
            agent_id for agent_id in context.known_agent_ids if agent_id not in plan.target_agent_ids
        ]
        if not plan.target_agent_ids or plan.turn_order_override != expected_order:
            errors.append(PlanValidationError.INVALID_RESTRUCTURE_ORDER)
    return errors


def _invalid_plan(
    decision: InterventionDecision,
    context: ExecutionContext,
    errors: list[PlanValidationError],
) -> InterventionPlan:
    """Create an inert audit result when no fully valid plan can be compiled."""
    return InterventionPlan(
        plan_id=_core._plan_id(decision),
        run_id=context.run_id,
        scenario_id=context.scenario_id,
        round_id=context.round_id,
        action=decision.action,
        target_agent_ids=list(decision.target_agent_ids),
        mediator_event_type=MediatorEventType.INVALID,
        requested_evidence_ids=[],
        visibility_overrides=[],
        speaking_priority_agents=[],
        turn_order_override=[],
        requires_target_response=False,
        requires_final_minority_report=False,
        estimated_cost=decision.estimated_cost,
        reason_codes=[reason.value for reason in decision.reason_codes],
        policy_version=decision.policy_version,
        valid=False,
        validation_errors=_core._unique_errors(errors),
    )


def _event_type(action: InterventionAction) -> MediatorEventType:
    return (
        MediatorEventType.NONE
        if action is InterventionAction.NO_OP
        else MediatorEventType(action.value)
    )


__all__ = [
    "ExecutionContext",
    "InterventionExecutor",
    "InterventionPlan",
    "MediatorEventType",
    "PlanValidationError",
    "PlanValidationResult",
    "VisibilityOverride",
    "validate_intervention_plan",
]
