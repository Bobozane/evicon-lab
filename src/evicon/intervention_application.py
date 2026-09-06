"""Pure validation and prospective application of structured intervention plans."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .executor import (
    ExecutionContext,
    InterventionPlan,
    PlanValidationError,
    VisibilityOverride,
    validate_intervention_plan,
)
from .models import DialogueState, InterventionAction, ScenarioSpec
from .models._validation import Metadata, identifier_list, normalized_text


class InterventionSchedule(BaseModel):
    """A validated, serializable, one-plan-per-round controlled-run schedule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schedule_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    plans: list[InterventionPlan] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("schedule_id", "run_id", "scenario_id")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @model_validator(mode="after")
    def validate_plan_alignment(self) -> "InterventionSchedule":
        plan_rounds = [plan.round_id for plan in self.plans]
        if len(plan_rounds) != len(set(plan_rounds)):
            raise ValueError("InterventionSchedule plans must have unique round_id values")
        for plan in self.plans:
            if not plan.valid:
                raise ValueError("InterventionSchedule plans must be valid before scheduling")
            if plan.run_id != self.run_id:
                raise ValueError("InterventionSchedule plan.run_id must match schedule.run_id")
            if plan.scenario_id != self.scenario_id:
                raise ValueError("InterventionSchedule plan.scenario_id must match schedule.scenario_id")
        return self

    def plan_for_round(self, round_id: int) -> InterventionPlan | None:
        """Return the one decision observed at a given round, if any."""
        if round_id < 0:
            raise ValueError("round_id must be non-negative")
        return next((plan for plan in self.plans if plan.round_id == round_id), None)


class AppliedIntervention(BaseModel):
    """Prospective, non-mutating result of applying one plan at a round boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str = Field(min_length=1)
    action: InterventionAction
    affected_agent_ids: list[str] = Field(default_factory=list)
    next_visibility_overrides: list[VisibilityOverride] = Field(default_factory=list)
    next_turn_order: list[str] = Field(default_factory=list)
    requested_evidence_ids: list[str] = Field(default_factory=list)
    evidence_request_agent_ids: list[str] = Field(default_factory=list)
    final_report_requirements: list[str] = Field(default_factory=list)
    applied: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("plan_id")
    @classmethod
    def validate_plan_id(cls, value: str) -> str:
        return normalized_text(value, "plan_id")

    @field_validator(
        "affected_agent_ids",
        "next_turn_order",
        "requested_evidence_ids",
        "evidence_request_agent_ids",
        "final_report_requirements",
        "errors",
        "warnings",
    )
    @classmethod
    def validate_identifiers(cls, value: list[str], info: object) -> list[str]:
        return identifier_list(value, getattr(info, "field_name", "identifiers"))


def apply_intervention_plan(
    state: DialogueState,
    plan: InterventionPlan,
    scenario: ScenarioSpec,
) -> AppliedIntervention:
    """Validate a plan and describe only its next-round effect; do not mutate inputs."""
    errors: list[str] = []
    if state.run_id != plan.run_id or state.scenario_id != plan.scenario_id:
        errors.append(PlanValidationError.CONTEXT_MISMATCH.value)
    if state.scenario_id != scenario.scenario_id or plan.round_id != state.current_round:
        errors.append(PlanValidationError.CONTEXT_MISMATCH.value)
    if not plan.valid:
        errors.extend(error.value for error in plan.validation_errors)

    context = _execution_context(state)
    validation = validate_intervention_plan(plan, context)
    errors.extend(error.value for error in validation.errors)
    errors.extend(_validate_plan_visibility(plan, state, scenario))
    errors = _unique(errors)
    if errors:
        return _rejected(plan, errors)

    affected = list(plan.target_agent_ids)
    if plan.action is InterventionAction.NO_OP:
        return _applied(plan, affected_agent_ids=[])
    if plan.action is InterventionAction.REQUEST_EVIDENCE:
        return _applied(
            plan,
            affected_agent_ids=affected,
            requested_evidence_ids=list(plan.requested_evidence_ids),
            evidence_request_agent_ids=affected,
        )
    if plan.action is InterventionAction.BLIND_EVIDENCE_REFLECTION:
        return _applied(
            plan,
            affected_agent_ids=affected,
            next_visibility_overrides=list(plan.visibility_overrides),
        )
    if plan.action is InterventionAction.SOLICIT_DISSENT:
        return _applied(
            plan,
            affected_agent_ids=affected,
            next_turn_order=_priority_order(plan.speaking_priority_agents, state),
        )
    if plan.action is InterventionAction.ADAPTIVE_EXPOSURE:
        return _applied(
            plan,
            affected_agent_ids=affected,
            next_visibility_overrides=list(plan.visibility_overrides),
        )
    if plan.action is InterventionAction.MINORITY_REPORT:
        return _applied(
            plan,
            affected_agent_ids=affected,
            final_report_requirements=affected,
        )
    if plan.action is InterventionAction.RESTRUCTURE:
        return _applied(
            plan,
            affected_agent_ids=affected,
            next_turn_order=list(plan.turn_order_override),
        )
    return _rejected(plan, [PlanValidationError.INVALID_ACTION_SEMANTICS.value])


def _execution_context(state: DialogueState) -> ExecutionContext:
    """Build only observable identifiers and chronology facts from a state snapshot."""
    return ExecutionContext(
        run_id=state.run_id,
        scenario_id=state.scenario_id,
        round_id=state.current_round,
        known_agent_ids=[agent.agent_id for agent in state.agents],
        visible_evidence_ids=[card.evidence_id for card in state.evidence_cards],
        visible_evidence_introduced_rounds={
            card.evidence_id: card.introduced_round for card in state.evidence_cards
        },
        candidate_peer_turn_ids=[turn.turn_id for turn in state.turns if turn.round_id < state.current_round],
        candidate_peer_turn_rounds={
            turn.turn_id: turn.round_id for turn in state.turns if turn.round_id < state.current_round
        },
        current_budget=state.intervention_budget,
        max_target_count=len(state.agents),
        intervention_history=[],
        last_action=None,
        metadata={},
    )


def _validate_plan_visibility(
    plan: InterventionPlan,
    state: DialogueState,
    scenario: ScenarioSpec,
) -> list[str]:
    """Ensure a plan never grants a target evidence it cannot already see."""
    errors: list[str] = []
    cards = {card.evidence_id: card for card in scenario.evidence_cards}
    available = {card.evidence_id for card in state.evidence_cards}
    for override in plan.visibility_overrides:
        for evidence_id in override.visible_evidence_ids:
            card = cards.get(evidence_id)
            if card is None or evidence_id not in available or not card.is_visible_to(override.agent_id):
                errors.append(PlanValidationError.UNKNOWN_EVIDENCE.value)
    return errors


def _priority_order(priority_agents: list[str], state: DialogueState) -> list[str]:
    return priority_agents + [
        agent.agent_id for agent in state.agents if agent.agent_id not in priority_agents
    ]


def _applied(
    plan: InterventionPlan,
    *,
    affected_agent_ids: list[str],
    next_visibility_overrides: list[VisibilityOverride] | None = None,
    next_turn_order: list[str] | None = None,
    requested_evidence_ids: list[str] | None = None,
    evidence_request_agent_ids: list[str] | None = None,
    final_report_requirements: list[str] | None = None,
) -> AppliedIntervention:
    return AppliedIntervention(
        plan_id=plan.plan_id,
        action=plan.action,
        affected_agent_ids=affected_agent_ids,
        next_visibility_overrides=next_visibility_overrides or [],
        next_turn_order=next_turn_order or [],
        requested_evidence_ids=requested_evidence_ids or [],
        evidence_request_agent_ids=evidence_request_agent_ids or [],
        final_report_requirements=final_report_requirements or [],
        applied=True,
        errors=[],
        warnings=[],
    )


def _rejected(plan: InterventionPlan, errors: list[str]) -> AppliedIntervention:
    return AppliedIntervention(
        plan_id=plan.plan_id,
        action=plan.action,
        affected_agent_ids=list(plan.target_agent_ids),
        next_visibility_overrides=[],
        next_turn_order=[],
        requested_evidence_ids=[],
        evidence_request_agent_ids=[],
        final_report_requirements=[],
        applied=False,
        errors=_unique(errors),
        warnings=[],
    )


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


__all__ = ["AppliedIntervention", "InterventionSchedule", "apply_intervention_plan"]
