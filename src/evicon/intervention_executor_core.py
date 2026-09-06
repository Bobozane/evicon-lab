"""Pure compilation of policy decisions into auditable intervention plans."""

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models._validation import Metadata, identifier_list, normalized_text
from .models.protocols import InterventionAction
from .policy import InterventionDecision, InterventionHistoryEntry


class MediatorEventType(str, Enum):
    """Machine-readable event kinds for future runtime integration."""

    NONE = "none"
    REQUEST_EVIDENCE = "request_evidence"
    BLIND_EVIDENCE_REFLECTION = "blind_evidence_reflection"
    SOLICIT_DISSENT = "solicit_dissent"
    ADAPTIVE_EXPOSURE = "adaptive_exposure"
    MINORITY_REPORT = "minority_report"
    RESTRUCTURE = "restructure"
    INVALID = "invalid"


class PlanValidationError(str, Enum):
    """Stable error codes for rejected plans; none contains free-form prompts."""

    CONTEXT_MISMATCH = "context_mismatch"
    PLAN_ID_MISMATCH = "plan_id_mismatch"
    UNKNOWN_TARGET_AGENT = "unknown_target_agent"
    DUPLICATE_TARGET_AGENT = "duplicate_target_agent"
    TARGET_LIMIT_EXCEEDED = "target_limit_exceeded"
    COST_EXCEEDS_BUDGET = "cost_exceeds_budget"
    UNKNOWN_EVIDENCE = "unknown_evidence"
    UNKNOWN_PEER_TURN = "unknown_peer_turn"
    FUTURE_PEER_TURN = "future_peer_turn"
    INVALID_NO_OP = "invalid_no_op"
    INVALID_ACTION_SEMANTICS = "invalid_action_semantics"
    INVALID_VISIBILITY_OVERRIDE = "invalid_visibility_override"
    INVALID_RESTRUCTURE_ORDER = "invalid_restructure_order"
    INVALID_PLAN_DECLARATION = "invalid_plan_declaration"


class VisibilityOverride(BaseModel):
    """Explicit next-turn visibility constraint for one target agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str = Field(min_length=1)
    visible_evidence_ids: list[str] = Field(default_factory=list)
    visible_peer_turn_ids: list[str] = Field(default_factory=list)
    preserve_own_history: bool = True

    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, value: str) -> str:
        return normalized_text(value, "agent_id")

    @field_validator("visible_evidence_ids", "visible_peer_turn_ids")
    @classmethod
    def validate_identifiers(cls, value: list[str], info: object) -> list[str]:
        return identifier_list(value, getattr(info, "field_name", "identifiers"))


class ExecutionContext(BaseModel):
    """Current observable state used to validate, but never execute, a plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    known_agent_ids: list[str] = Field(default_factory=list)
    visible_evidence_ids: list[str] = Field(default_factory=list)
    visible_evidence_introduced_rounds: dict[str, int] = Field(default_factory=dict)
    candidate_peer_turn_ids: list[str] = Field(default_factory=list)
    candidate_peer_turn_rounds: dict[str, int] = Field(default_factory=dict)
    current_budget: float = Field(ge=0.0)
    max_target_count: int = Field(ge=0)
    intervention_history: list[InterventionHistoryEntry] = Field(default_factory=list)
    last_action: InterventionAction | None = None
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("run_id", "scenario_id")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("known_agent_ids", "visible_evidence_ids", "candidate_peer_turn_ids")
    @classmethod
    def validate_identifiers(cls, value: list[str], info: object) -> list[str]:
        return identifier_list(value, getattr(info, "field_name", "identifiers"))

    @field_validator("round_id", "max_target_count", mode="before")
    @classmethod
    def validate_nonnegative_integer(cls, value: object, info: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{getattr(info, 'field_name', 'value')} must be a non-negative integer")
        return value

    @field_validator("current_budget", mode="before")
    @classmethod
    def validate_budget(cls, value: object) -> float:
        if isinstance(value, bool):
            raise ValueError("current_budget must be a number")
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("current_budget must be a number") from exc
        if not math.isfinite(normalized) or normalized < 0.0:
            raise ValueError("current_budget must be a finite non-negative number")
        return normalized

    @field_validator("visible_evidence_introduced_rounds", "candidate_peer_turn_rounds")
    @classmethod
    def validate_round_maps(cls, value: dict[str, int], info: object) -> dict[str, int]:
        normalized: dict[str, int] = {}
        for identifier, round_id in value.items():
            key = normalized_text(identifier, getattr(info, "field_name", "identifier"))
            if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
                raise ValueError(f"{getattr(info, 'field_name', 'rounds')} values must be non-negative integers")
            normalized[key] = round_id
        return normalized

    @field_validator("metadata")
    @classmethod
    def reject_private_metadata(cls, value: Metadata) -> Metadata:
        _reject_private_fields(value, "ExecutionContext.metadata")
        return value

    @model_validator(mode="after")
    def validate_observable_references(self) -> "ExecutionContext":
        if set(self.visible_evidence_ids) != set(self.visible_evidence_introduced_rounds):
            raise ValueError("visible_evidence_introduced_rounds must match visible_evidence_ids")
        if set(self.candidate_peer_turn_ids) != set(self.candidate_peer_turn_rounds):
            raise ValueError("candidate_peer_turn_rounds must match candidate_peer_turn_ids")
        if any(round_id > self.round_id for round_id in self.visible_evidence_introduced_rounds.values()):
            raise ValueError("visible evidence must be introduced no later than the current round")
        if any(round_id >= self.round_id for round_id in self.candidate_peer_turn_rounds.values()):
            raise ValueError("candidate peer turns must be strictly earlier than the current round")
        if any(entry.round_id > self.round_id for entry in self.intervention_history):
            raise ValueError("intervention_history must not contain future-round entries")
        return self


class InterventionPlan(BaseModel):
    """An auditable description of a future action, never an executed action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    action: InterventionAction
    target_agent_ids: list[str] = Field(default_factory=list)
    mediator_event_type: MediatorEventType
    requested_evidence_ids: list[str] = Field(default_factory=list)
    visibility_overrides: list[VisibilityOverride] = Field(default_factory=list)
    speaking_priority_agents: list[str] = Field(default_factory=list)
    turn_order_override: list[str] = Field(default_factory=list)
    requires_target_response: bool = False
    requires_final_minority_report: bool = False
    estimated_cost: float = Field(ge=0.0)
    reason_codes: list[str] = Field(default_factory=list)
    policy_version: str = Field(min_length=1)
    valid: bool
    validation_errors: list[PlanValidationError] = Field(default_factory=list)

    @field_validator("plan_id", "run_id", "scenario_id", "policy_version")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator(
        "target_agent_ids",
        "requested_evidence_ids",
        "speaking_priority_agents",
        "turn_order_override",
        "reason_codes",
    )
    @classmethod
    def validate_identifiers(cls, value: list[str], info: object) -> list[str]:
        return identifier_list(value, getattr(info, "field_name", "identifiers"))

    @field_validator("estimated_cost", mode="before")
    @classmethod
    def validate_cost(cls, value: object) -> float:
        if isinstance(value, bool):
            raise ValueError("estimated_cost must be a number")
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("estimated_cost must be a number") from exc
        if not math.isfinite(normalized) or normalized < 0.0:
            raise ValueError("estimated_cost must be a finite non-negative number")
        return normalized

    @model_validator(mode="after")
    def validate_error_declaration(self) -> "InterventionPlan":
        if self.valid and self.validation_errors:
            raise ValueError("valid plans must not declare validation_errors")
        if not self.valid and not self.validation_errors:
            raise ValueError("invalid plans must declare validation_errors")
        return self


class PlanValidationResult(BaseModel):
    """Pure validation result for a plan against a current execution context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    errors: list[PlanValidationError] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_consistency(self) -> "PlanValidationResult":
        if self.valid and self.errors:
            raise ValueError("valid validation results must not contain errors")
        if not self.valid and not self.errors:
            raise ValueError("invalid validation results must contain errors")
        return self


class InterventionExecutor:
    """Compile a decision into a deterministic plan without side effects."""

    @staticmethod
    def plan(decision: InterventionDecision, context: ExecutionContext) -> InterventionPlan:
        """Return an executable-description plan or a fully inert invalid plan."""
        errors = _decision_context_errors(decision, context)
        if errors:
            return _invalid_plan(decision, context, errors)

        plan_fields, action_errors = _action_fields(decision, context)
        if action_errors:
            return _invalid_plan(decision, context, action_errors)
        plan = InterventionPlan(
            plan_id=_plan_id(decision),
            run_id=decision.run_id if hasattr(decision, "run_id") else context.run_id,
            scenario_id=context.scenario_id,
            round_id=decision.round_id,
            action=decision.action,
            target_agent_ids=list(decision.target_agent_ids),
            estimated_cost=decision.estimated_cost,
            reason_codes=[reason.value for reason in decision.reason_codes],
            policy_version=decision.policy_version,
            valid=True,
            validation_errors=[],
            **plan_fields,
        )
        validation = validate_intervention_plan(plan, context)
        return plan if validation.valid else _invalid_plan(decision, context, validation.errors)


def validate_intervention_plan(
    plan: InterventionPlan,
    context: ExecutionContext,
) -> PlanValidationResult:
    """Validate a plan without changing it, the context, files, or runtime state."""
    errors: list[PlanValidationError] = list(plan.validation_errors)
    if plan.plan_id != _plan_id_from_fields(
        action=plan.action,
        target_agent_ids=plan.target_agent_ids,
        risk_score=None,
        estimated_cost=plan.estimated_cost,
        round_id=plan.round_id,
        policy_version=plan.policy_version,
        reason_codes=plan.reason_codes,
    ):
        # Full decision data is intentionally not stored in a plan. This check
        # is limited to the stable plan identity convention when applicable.
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
    errors.extend(_validate_overrides(plan, context))
    errors.extend(_validate_action_semantics(plan, context))
    unique_errors = _unique_errors(errors)
    return PlanValidationResult(valid=not unique_errors, errors=unique_errors, warnings=[])


def _decision_context_errors(
    decision: InterventionDecision,
    context: ExecutionContext,
) -> list[PlanValidationError]:
    errors: list[PlanValidationError] = []
    if decision.round_id != context.round_id:
        errors.append(PlanValidationError.CONTEXT_MISMATCH)
    if any(agent_id not in context.known_agent_ids for agent_id in decision.target_agent_ids):
        errors.append(PlanValidationError.UNKNOWN_TARGET_AGENT)
    if len(decision.target_agent_ids) > context.max_target_count:
        errors.append(PlanValidationError.TARGET_LIMIT_EXCEEDED)
    if decision.estimated_cost > context.current_budget:
        errors.append(PlanValidationError.COST_EXCEEDS_BUDGET)
    return errors


def _action_fields(
    decision: InterventionDecision,
    context: ExecutionContext,
) -> tuple[dict[str, object], list[PlanValidationError]]:
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


def _validate_overrides(
    plan: InterventionPlan,
    context: ExecutionContext,
) -> list[PlanValidationError]:
    errors: list[PlanValidationError] = []
    override_agents = [override.agent_id for override in plan.visibility_overrides]
    if len(override_agents) != len(set(override_agents)):
        errors.append(PlanValidationError.INVALID_VISIBILITY_OVERRIDE)
    for override in plan.visibility_overrides:
        if override.agent_id not in context.known_agent_ids:
            errors.append(PlanValidationError.UNKNOWN_TARGET_AGENT)
        if any(item not in context.visible_evidence_ids for item in override.visible_evidence_ids):
            errors.append(PlanValidationError.UNKNOWN_EVIDENCE)
        for turn_id in override.visible_peer_turn_ids:
            if turn_id not in context.candidate_peer_turn_ids:
                errors.append(PlanValidationError.UNKNOWN_PEER_TURN)
            elif context.candidate_peer_turn_rounds[turn_id] >= context.round_id:
                errors.append(PlanValidationError.FUTURE_PEER_TURN)
    return errors


def _validate_action_semantics(
    plan: InterventionPlan,
    context: ExecutionContext,
) -> list[PlanValidationError]:
    errors: list[PlanValidationError] = []
    expected_event = _event_type(plan.action)
    if plan.mediator_event_type is not expected_event:
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
    """Return one inert audit record rather than a partially applicable plan."""
    return InterventionPlan(
        plan_id=_plan_id(decision),
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
        validation_errors=_unique_errors(errors),
    )


def _event_type(action: InterventionAction) -> MediatorEventType:
    return MediatorEventType(action.value)


def _plan_id(decision: InterventionDecision) -> str:
    payload = {
        "action": decision.action.value,
        "estimated_cost": decision.estimated_cost,
        "policy_version": decision.policy_version,
        "reason_codes": [reason.value for reason in decision.reason_codes],
        "risk_score": decision.risk_score,
        "round_id": decision.round_id,
        "target_agent_ids": decision.target_agent_ids,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return f"plan-{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:20]}"


def _plan_id_from_fields(
    *,
    action: InterventionAction,
    target_agent_ids: list[str],
    risk_score: float | None,
    estimated_cost: float,
    round_id: int,
    policy_version: str,
    reason_codes: list[str],
) -> str:
    """Best-effort identity check for standalone plans without private inputs."""
    payload = {
        "action": action.value,
        "estimated_cost": estimated_cost,
        "policy_version": policy_version,
        "reason_codes": reason_codes,
        "risk_score": risk_score,
        "round_id": round_id,
        "target_agent_ids": target_agent_ids,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return f"plan-{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:20]}"


def _unique_errors(errors: list[PlanValidationError]) -> list[PlanValidationError]:
    return list(dict.fromkeys(errors))


def _reject_private_fields(value: object, field_name: str) -> None:
    """Reject private/control payloads even when nested inside metadata."""
    prohibited = ("prompt", "hidden_probe", "hidden_profile", "value_profile", "dialogue_state")
    if isinstance(value, dict):
        for key, nested_value in value.items():
            normalized_key = str(key).lower()
            if any(token in normalized_key for token in prohibited):
                raise ValueError(f"{field_name} must not contain private or prompt data")
            _reject_private_fields(nested_value, field_name)
    elif isinstance(value, list):
        for nested_value in value:
            _reject_private_fields(nested_value, field_name)
