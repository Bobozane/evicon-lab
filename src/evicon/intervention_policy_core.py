"""Pure target selection and structured intervention eligibility decisions."""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .conformity_monitor import MonitorReason, MonitorResult, RiskLevel
from .models._validation import Metadata, identifier_list, normalized_text
from .models.protocols import InterventionAction


class TargetSelectionReason(str, Enum):
    """Stable explanations produced by the deterministic target selector."""

    TARGETS_SELECTED = "targets_selected"
    EMPTY_CANDIDATES = "empty_candidates"
    NO_ELIGIBLE_CANDIDATES = "no_eligible_candidates"
    ALL_BELOW_MINIMUM_GAIN = "all_below_minimum_gain"
    ALL_ABOVE_SAFETY_LIMIT = "all_above_safety_limit"


class PolicyReason(str, Enum):
    """Stable policy explanations; they deliberately contain no natural language."""

    RISK_BELOW_THRESHOLD = "risk_below_threshold"
    EVIDENCE_SUPPORTED_UPDATE = "evidence_supported_update"
    INVALID_MONITOR_RESULT = "invalid_monitor_result"
    BUDGET_EXHAUSTED = "budget_exhausted"
    COOLDOWN_ACTIVE = "cooldown_active"
    HIGH_RISK = "high_risk"
    MINORITY_LOSS = "minority_loss"
    HARM_RISK = "harm_risk"
    INSUFFICIENT_EVIDENCE_QUALITY = "insufficient_evidence_quality"
    NO_ELIGIBLE_TARGETS = "no_eligible_targets"
    COST_EXCEEDS_BUDGET = "cost_exceeds_budget"
    REPEATED_INTERVENTION_FAILURES = "repeated_intervention_failures"
    RESTRUCTURE_LIMIT_REACHED = "restructure_limit_reached"
    TARGETS_SELECTED = "targets_selected"


class TargetCandidate(BaseModel):
    """A caller-supplied, online-observable candidate with no private probe data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str = Field(min_length=1)
    marginal_coverage_gain: float = Field(ge=0.0, le=1.0)
    minority_representation_score: float = Field(ge=0.0, le=1.0)
    safety_risk: float = Field(ge=0.0, le=1.0)
    estimated_cost: float = Field(ge=0.0, le=1.0)
    eligible: bool
    reason_codes: list[str] = Field(default_factory=list)

    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, value: str) -> str:
        return normalized_text(value, "agent_id")

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "reason_codes")

    @field_validator(
        "marginal_coverage_gain",
        "minority_representation_score",
        "safety_risk",
        "estimated_cost",
        mode="before",
    )
    @classmethod
    def validate_finite_unit_number(cls, value: object, info: object) -> float:
        return _finite_number(value, getattr(info, "field_name", "value"))


class InterventionHistoryEntry(BaseModel):
    """Structured outcome of a past intervention, used only for bounded escalation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: InterventionAction
    round_id: int = Field(ge=0)
    succeeded: bool

    @field_validator("round_id", mode="before")
    @classmethod
    def validate_round_id(cls, value: object) -> int:
        return _nonnegative_integer(value, "round_id")


class PolicyInput(BaseModel):
    """Online policy input; it cannot contain prompts, hidden probes, or profiles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    monitor_result: MonitorResult
    target_candidates: list[TargetCandidate] = Field(default_factory=list)
    remaining_budget: float = Field(ge=0.0)
    last_intervention_round: int | None = Field(default=None, ge=0)
    cooldown_rounds: int = Field(ge=0)
    intervention_history: list[InterventionHistoryEntry] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("run_id", "scenario_id")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("round_id", "cooldown_rounds", "last_intervention_round", mode="before")
    @classmethod
    def validate_optional_round(cls, value: object, info: object) -> int | None:
        if value is None:
            return None
        return _nonnegative_integer(value, getattr(info, "field_name", "round_id"))

    @field_validator("remaining_budget", mode="before")
    @classmethod
    def validate_remaining_budget(cls, value: object) -> float:
        return _finite_number(value, "remaining_budget")

    @model_validator(mode="after")
    def validate_cross_field_alignment(self) -> "PolicyInput":
        result = self.monitor_result
        if (
            result.run_id != self.run_id
            or result.scenario_id != self.scenario_id
            or result.round_id != self.round_id
        ):
            raise ValueError("monitor_result run_id, scenario_id, and round_id must match PolicyInput")
        candidate_ids = [candidate.agent_id for candidate in self.target_candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("target_candidates must not contain duplicate agent_id values")
        if self.last_intervention_round is not None and self.last_intervention_round > self.round_id:
            raise ValueError("last_intervention_round must not be in a future round")
        history_rounds = [entry.round_id for entry in self.intervention_history]
        if any(round_id > self.round_id for round_id in history_rounds):
            raise ValueError("intervention_history must not contain future-round entries")
        if history_rounds != sorted(history_rounds):
            raise ValueError("intervention_history must be ordered by round_id")
        return self


class PolicyConfig(BaseModel):
    """All action costs and policy thresholds are explicit configuration inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_targets: int = Field(ge=0)
    min_marginal_gain: float = Field(ge=0.0, le=1.0)
    max_target_safety_risk: float = Field(ge=0.0, le=1.0)
    request_evidence_cost: float = Field(ge=0.0, le=1.0)
    blind_reflection_cost: float = Field(ge=0.0, le=1.0)
    solicit_dissent_cost: float = Field(ge=0.0, le=1.0)
    adaptive_exposure_cost: float = Field(ge=0.0, le=1.0)
    minority_report_cost: float = Field(ge=0.0, le=1.0)
    restructure_cost: float = Field(ge=0.0, le=1.0)
    high_risk_restructure_threshold: float = Field(ge=0.0, le=1.0)
    cooldown_rounds: int = Field(ge=0)
    version: str = Field(min_length=1)
    restructure_failure_threshold: int | None = Field(default=None, ge=1)

    @field_validator("max_targets", "cooldown_rounds", "restructure_failure_threshold", mode="before")
    @classmethod
    def validate_optional_integer(cls, value: object, info: object) -> int | None:
        if value is None:
            return None
        return _nonnegative_integer(value, getattr(info, "field_name", "integer"))

    @field_validator(
        "min_marginal_gain",
        "max_target_safety_risk",
        "request_evidence_cost",
        "blind_reflection_cost",
        "solicit_dissent_cost",
        "adaptive_exposure_cost",
        "minority_report_cost",
        "restructure_cost",
        "high_risk_restructure_threshold",
        mode="before",
    )
    @classmethod
    def validate_finite_unit_number(cls, value: object, info: object) -> float:
        return _finite_number(value, getattr(info, "field_name", "value"))

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return normalized_text(value, "version")


class TargetSelection(BaseModel):
    """Structured result of target filtering and deterministic ranking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_agent_ids: list[str] = Field(default_factory=list)
    reason_codes: list[TargetSelectionReason] = Field(default_factory=list)
    valid: bool

    @field_validator("target_agent_ids")
    @classmethod
    def validate_target_agent_ids(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "target_agent_ids")


class InterventionDecision(BaseModel):
    """Policy-specific structured decision; it does not execute an intervention."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: InterventionAction
    target_agent_ids: list[str] = Field(default_factory=list)
    reason_codes: list[PolicyReason] = Field(default_factory=list)
    risk_score: float = Field(ge=0.0, le=1.0)
    estimated_cost: float = Field(ge=0.0, le=1.0)
    round_id: int = Field(ge=0)
    policy_version: str = Field(min_length=1)
    cooldown_until: int | None = Field(default=None, ge=0)
    valid: bool

    @field_validator("target_agent_ids")
    @classmethod
    def validate_target_agent_ids(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "target_agent_ids")

    @field_validator("policy_version")
    @classmethod
    def validate_policy_version(cls, value: str) -> str:
        return normalized_text(value, "policy_version")

    @model_validator(mode="after")
    def validate_no_op_cost(self) -> "InterventionDecision":
        if self.action is InterventionAction.NO_OP:
            if self.target_agent_ids:
                raise ValueError("no_op decisions must not name target agents")
            if self.estimated_cost != 0.0:
                raise ValueError("no_op decisions must have estimated_cost=0")
        return self


def select_targets(
    candidates: list[TargetCandidate],
    policy_config: PolicyConfig,
    max_targets: int | None = None,
) -> TargetSelection:
    """Filter and stably rank targets without modifying caller-owned candidates."""
    limit = policy_config.max_targets if max_targets is None else max_targets
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("max_targets must be a non-negative integer")
    if not candidates:
        return TargetSelection(
            target_agent_ids=[],
            reason_codes=[TargetSelectionReason.EMPTY_CANDIDATES],
            valid=True,
        )
    eligible = [candidate for candidate in candidates if candidate.eligible]
    if not eligible:
        return TargetSelection(
            target_agent_ids=[],
            reason_codes=[TargetSelectionReason.NO_ELIGIBLE_CANDIDATES],
            valid=True,
        )
    gain_qualified = [
        candidate
        for candidate in eligible
        if candidate.marginal_coverage_gain >= policy_config.min_marginal_gain
    ]
    if not gain_qualified:
        return TargetSelection(
            target_agent_ids=[],
            reason_codes=[TargetSelectionReason.ALL_BELOW_MINIMUM_GAIN],
            valid=True,
        )
    safe = [
        candidate
        for candidate in gain_qualified
        if candidate.safety_risk <= policy_config.max_target_safety_risk
    ]
    if not safe or limit == 0:
        return TargetSelection(
            target_agent_ids=[],
            reason_codes=[TargetSelectionReason.ALL_ABOVE_SAFETY_LIMIT],
            valid=True,
        )
    ranked = sorted(
        safe,
        key=lambda candidate: (
            -candidate.marginal_coverage_gain,
            -candidate.minority_representation_score,
            candidate.safety_risk,
            candidate.estimated_cost,
            candidate.agent_id,
        ),
    )
    return TargetSelection(
        target_agent_ids=[candidate.agent_id for candidate in ranked[:limit]],
        reason_codes=[TargetSelectionReason.TARGETS_SELECTED],
        valid=True,
    )


class InterventionPolicy:
    """Side-effect-free policy that emits a decision but never applies it."""

    @staticmethod
    def decide(policy_input: PolicyInput, config: PolicyConfig) -> InterventionDecision:
        """Choose an eligible structured action from online monitor signals only."""
        monitor = policy_input.monitor_result
        inactive_cooldown = _cooldown_until(
            policy_input.last_intervention_round,
            policy_input.cooldown_rounds,
        )
        if not monitor.valid or monitor.risk_level is RiskLevel.INVALID:
            return _no_op(policy_input, config, [PolicyReason.INVALID_MONITOR_RESULT], inactive_cooldown)
        if not monitor.should_intervene:
            reason = (
                PolicyReason.EVIDENCE_SUPPORTED_UPDATE
                if monitor.evidence_supported_update
                else PolicyReason.RISK_BELOW_THRESHOLD
            )
            return _no_op(policy_input, config, [reason], inactive_cooldown)
        if policy_input.remaining_budget <= 0.0:
            return _no_op(policy_input, config, [PolicyReason.BUDGET_EXHAUSTED], inactive_cooldown)
        if inactive_cooldown is not None and policy_input.round_id <= inactive_cooldown:
            return _no_op(policy_input, config, [PolicyReason.COOLDOWN_ACTIVE], inactive_cooldown)

        selection = select_targets(policy_input.target_candidates, config, config.max_targets)
        selection_reasons = _selection_policy_reasons(selection)
        failure_count = _consecutive_failures(policy_input.intervention_history)
        has_restructure = any(
            entry.action is InterventionAction.RESTRUCTURE
            for entry in policy_input.intervention_history
        )
        can_restructure = (
            config.restructure_failure_threshold is not None
            and failure_count >= config.restructure_failure_threshold
            and monitor.risk_score >= config.high_risk_restructure_threshold
            and not has_restructure
            and bool(selection.target_agent_ids)
        )
        if can_restructure:
            return _with_budget(
                policy_input,
                config,
                action=InterventionAction.RESTRUCTURE,
                target_agent_ids=selection.target_agent_ids,
                reason_codes=[PolicyReason.HIGH_RISK, PolicyReason.REPEATED_INTERVENTION_FAILURES, *selection_reasons],
            )

        harm_present = MonitorReason.HARM_RISK in monitor.reasons
        evidence_insufficient = MonitorReason.INSUFFICIENT_EVIDENCE in monitor.reasons
        minority_loss = monitor.signals.minority_loss > 0.0
        if harm_present:
            action = (
                InterventionAction.MINORITY_REPORT
                if selection.target_agent_ids
                else InterventionAction.REQUEST_EVIDENCE
            )
            return _with_budget(
                policy_input,
                config,
                action=action,
                target_agent_ids=selection.target_agent_ids if action is InterventionAction.MINORITY_REPORT else [],
                reason_codes=[PolicyReason.HARM_RISK, *selection_reasons],
            )
        if evidence_insufficient:
            return _with_budget(
                policy_input,
                config,
                action=InterventionAction.REQUEST_EVIDENCE,
                target_agent_ids=[],
                reason_codes=[PolicyReason.HIGH_RISK, PolicyReason.INSUFFICIENT_EVIDENCE_QUALITY, *selection_reasons],
            )
        if minority_loss and selection.target_agent_ids:
            return _with_budget(
                policy_input,
                config,
                action=InterventionAction.SOLICIT_DISSENT,
                target_agent_ids=selection.target_agent_ids,
                reason_codes=[PolicyReason.MINORITY_LOSS, *selection_reasons],
            )
        if monitor.risk_level is RiskLevel.HIGH and selection.target_agent_ids:
            return _with_budget(
                policy_input,
                config,
                action=InterventionAction.SOLICIT_DISSENT,
                target_agent_ids=selection.target_agent_ids,
                reason_codes=[PolicyReason.HIGH_RISK, *selection_reasons],
            )
        return _with_budget(
            policy_input,
            config,
            action=InterventionAction.REQUEST_EVIDENCE,
            target_agent_ids=[],
            reason_codes=[PolicyReason.NO_ELIGIBLE_TARGETS, *selection_reasons],
        )


def _with_budget(
    policy_input: PolicyInput,
    config: PolicyConfig,
    *,
    action: InterventionAction,
    target_agent_ids: list[str],
    reason_codes: list[PolicyReason],
) -> InterventionDecision:
    cost = _action_cost(action, config)
    if cost > policy_input.remaining_budget:
        return _no_op(
            policy_input,
            config,
            [PolicyReason.COST_EXCEEDS_BUDGET],
            _cooldown_until(policy_input.last_intervention_round, policy_input.cooldown_rounds),
        )
    return InterventionDecision(
        action=action,
        target_agent_ids=target_agent_ids,
        reason_codes=_unique_reasons(reason_codes),
        risk_score=policy_input.monitor_result.risk_score,
        estimated_cost=cost,
        round_id=policy_input.round_id,
        policy_version=config.version,
        cooldown_until=policy_input.round_id + config.cooldown_rounds,
        valid=True,
    )


def _no_op(
    policy_input: PolicyInput,
    config: PolicyConfig,
    reasons: list[PolicyReason],
    cooldown_until: int | None,
) -> InterventionDecision:
    return InterventionDecision(
        action=InterventionAction.NO_OP,
        target_agent_ids=[],
        reason_codes=_unique_reasons(reasons),
        risk_score=policy_input.monitor_result.risk_score,
        estimated_cost=0.0,
        round_id=policy_input.round_id,
        policy_version=config.version,
        cooldown_until=cooldown_until,
        valid=True,
    )


def _action_cost(action: InterventionAction, config: PolicyConfig) -> float:
    costs = {
        InterventionAction.REQUEST_EVIDENCE: config.request_evidence_cost,
        InterventionAction.BLIND_EVIDENCE_REFLECTION: config.blind_reflection_cost,
        InterventionAction.SOLICIT_DISSENT: config.solicit_dissent_cost,
        InterventionAction.ADAPTIVE_EXPOSURE: config.adaptive_exposure_cost,
        InterventionAction.MINORITY_REPORT: config.minority_report_cost,
        InterventionAction.RESTRUCTURE: config.restructure_cost,
    }
    if action is InterventionAction.NO_OP:
        return 0.0
    return costs[action]


def _selection_policy_reasons(selection: TargetSelection) -> list[PolicyReason]:
    return (
        [PolicyReason.TARGETS_SELECTED]
        if selection.target_agent_ids
        else [PolicyReason.NO_ELIGIBLE_TARGETS]
    )


def _consecutive_failures(history: list[InterventionHistoryEntry]) -> int:
    failures = 0
    for entry in reversed(history):
        if entry.action is InterventionAction.NO_OP:
            continue
        if entry.succeeded:
            break
        failures += 1
    return failures


def _cooldown_until(last_round: int | None, cooldown_rounds: int) -> int | None:
    return None if last_round is None else last_round + cooldown_rounds


def _unique_reasons(reasons: list[PolicyReason]) -> list[PolicyReason]:
    return list(dict.fromkeys(reasons))


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value
