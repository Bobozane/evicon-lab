"""Read-only replay validation for baseline, controlled, and adaptive runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import replay_controlled as _controlled
from .events import EventType
from .models import InterventionAction, RunRecord

ReplayValidationError = _controlled.ReplayValidationError
ReplayValidationReport = _controlled.ReplayValidationReport


def validate_replay(
    events_path: str | Path,
    record_path: str | Path,
) -> ReplayValidationReport:
    """Validate base replay plus the appropriate controlled or adaptive contract."""
    report = _controlled._core.validate_replay(events_path, record_path)
    events = _controlled._core._load_events(Path(events_path))
    record = _controlled._core._load_record(Path(record_path))
    if bool(events[0]["payload"].get("adaptive")):
        _validate_adaptive_events(events, record)
    else:
        _controlled._validate_intervention_events(events, record)
    return report


def _validate_adaptive_events(events: list[dict[str, Any]], record: RunRecord) -> None:
    """Validate online-observation, plan, budget, and cooldown causality in JSONL."""
    completed_sequences: dict[int, int] = {}
    observed: dict[str, tuple[int, int]] = {}
    monitored: dict[str, int] = {}
    policies: dict[str, int] = {}
    created: dict[str, tuple[int, str, int]] = {}
    scheduled: dict[str, tuple[int, str]] = {}
    applied: set[str] = set()
    rejected: set[str] = set()
    spent: set[str] = set()
    released: set[str] = set()
    cooldowns: set[str] = set()

    for event in events:
        event_type = event["event_type"]
        payload = event["payload"]
        sequence = event["sequence"]
        round_id = event["round_id"]
        if event_type == EventType.ROUND_COMPLETED.value:
            completed_sequences[round_id] = sequence
            continue
        if event_type == EventType.ONLINE_STATE_OBSERVED.value:
            correlation = _correlation(payload, event_type)
            if completed_sequences.get(round_id, 0) >= sequence:
                raise ReplayValidationError("online_state_observed must follow current round_completed")
            observed[correlation] = (round_id, sequence)
            _reject_private_adaptive_payload(payload, event_type)
        elif event_type == EventType.MONITOR_EVALUATED.value:
            correlation = _correlation(payload, event_type)
            _require_prior(correlation, observed, sequence, "monitor_evaluated must follow online_state_observed")
            if observed[correlation][0] != round_id:
                raise ReplayValidationError("monitor_evaluated must retain the observation round")
            monitored[correlation] = sequence
            _reject_private_adaptive_payload(payload, event_type)
        elif event_type == EventType.TARGET_CANDIDATES_BUILT.value:
            correlation = _correlation(payload, event_type)
            _require_prior(correlation, monitored, sequence, "candidates must follow monitor_evaluated")
            _reject_private_adaptive_payload(payload, event_type)
        elif event_type == EventType.POLICY_DECIDED.value:
            correlation = _correlation(payload, event_type)
            _require_prior(correlation, monitored, sequence, "policy_decided must follow monitor_evaluated")
            action = payload.get("action")
            if action not in {value.value for value in InterventionAction}:
                raise ReplayValidationError("policy_decided has an unknown action")
            policies[correlation] = sequence
            _reject_private_adaptive_payload(payload, event_type)
        elif event_type == EventType.INTERVENTION_PLAN_CREATED.value:
            correlation = _correlation(payload, event_type)
            _require_prior(correlation, policies, sequence, "plan_created must follow policy_decided")
            plan_id, action, scheduled_round = _plan_fields(payload, event_type)
            if plan_id in created:
                raise ReplayValidationError("adaptive plan_id may be created only once")
            created[plan_id] = (round_id, action, scheduled_round)
            _reject_private_adaptive_payload(payload, event_type)
        elif event_type == EventType.INTERVENTION_PLAN_SCHEDULED.value:
            correlation = _correlation(payload, event_type)
            _require_prior(correlation, policies, sequence, "plan_scheduled must follow policy_decided")
            plan_id, action, scheduled_round = _plan_fields(payload, event_type)
            created_value = created.get(plan_id)
            if created_value is None:
                raise ReplayValidationError("plan_scheduled must reference a created plan")
            if scheduled_round <= round_id or scheduled_round != created_value[2]:
                raise ReplayValidationError("plan_scheduled must affect a future round")
            scheduled[plan_id] = (scheduled_round, action)
            _reject_private_adaptive_payload(payload, event_type)
        elif event_type == EventType.INTERVENTION_PLAN_APPLIED.value:
            plan_id, action, _ = _plan_fields(payload, event_type)
            schedule = scheduled.get(plan_id)
            if schedule is None or schedule[0] != round_id or schedule[1] != action:
                raise ReplayValidationError("plan_applied must match a scheduled plan at its future round")
            if plan_id in applied or plan_id in rejected:
                raise ReplayValidationError("adaptive plan must not be applied or rejected twice")
            applied.add(plan_id)
            _reject_private_adaptive_payload(payload, event_type)
        elif event_type == EventType.INTERVENTION_PLAN_REJECTED.value:
            plan_id, _, _ = _plan_fields(payload, event_type)
            if plan_id not in scheduled or plan_id in applied or plan_id in rejected:
                raise ReplayValidationError("plan_rejected must match one pending scheduled plan")
            rejected.add(plan_id)
            _reject_private_adaptive_payload(payload, event_type)
        elif event_type in {EventType.BUDGET_RESERVED.value, EventType.BUDGET_SPENT.value, EventType.BUDGET_RELEASED.value}:
            plan_id = payload.get("plan_id")
            if not isinstance(plan_id, str):
                raise ReplayValidationError(f"{event_type} must include plan_id")
            _validate_budget_payload(payload, event_type)
            if event_type == EventType.BUDGET_RESERVED.value:
                if plan_id not in scheduled:
                    raise ReplayValidationError("budget_reserved must correspond to a scheduled plan")
            elif event_type == EventType.BUDGET_SPENT.value:
                if plan_id not in applied or plan_id in spent:
                    raise ReplayValidationError("budget_spent must correspond once to an applied plan")
                if scheduled[plan_id][1] == InterventionAction.NO_OP.value:
                    raise ReplayValidationError("no_op must not produce budget_spent")
                spent.add(plan_id)
            else:
                if plan_id not in scheduled or plan_id in released:
                    raise ReplayValidationError("budget_released must correspond once to a scheduled plan")
                released.add(plan_id)
            _reject_private_adaptive_payload(payload, event_type)
        elif event_type == EventType.COOLDOWN_UPDATED.value:
            plan_id = payload.get("plan_id")
            if not isinstance(plan_id, str) or plan_id not in applied or plan_id in cooldowns:
                raise ReplayValidationError("cooldown_updated must follow one applied plan")
            if payload.get("cooldown_until") is None:
                raise ReplayValidationError("cooldown_updated must include cooldown_until")
            cooldowns.add(plan_id)
            _reject_private_adaptive_payload(payload, event_type)

    if spent != applied:
        raise ReplayValidationError("every applied adaptive plan must have exactly one budget_spent")
    if not rejected.issubset(released):
        raise ReplayValidationError("every rejected adaptive plan must release its reservation")
    unresolved = set(scheduled) - applied - rejected
    if unresolved and record.status.value == "completed":
        raise ReplayValidationError("completed adaptive runs must not retain unresolved scheduled plans")
    if unresolved and not unresolved.issubset(released):
        raise ReplayValidationError("unapplied adaptive plans must release their reservation")


def _correlation(payload: dict[str, Any], event_type: str) -> str:
    value = payload.get("correlation_id")
    if not isinstance(value, str) or not value:
        raise ReplayValidationError(f"{event_type} must include correlation_id")
    return value


def _require_prior(
    key: str,
    values: dict[str, Any],
    sequence: int,
    message: str,
) -> None:
    prior = values.get(key)
    if prior is None:
        raise ReplayValidationError(message)
    prior_sequence = prior[-1] if isinstance(prior, tuple) else prior
    if prior_sequence >= sequence:
        raise ReplayValidationError(message)


def _plan_fields(payload: dict[str, Any], event_type: str) -> tuple[str, str, int]:
    plan_id = payload.get("plan_id")
    action = payload.get("action")
    scheduled_round = payload.get("scheduled_round")
    if not isinstance(plan_id, str) or not plan_id.startswith("plan-"):
        raise ReplayValidationError(f"{event_type} has an invalid plan_id")
    if action not in {value.value for value in InterventionAction}:
        raise ReplayValidationError(f"{event_type} has an invalid action")
    if type(scheduled_round) is not int or scheduled_round < 0:
        raise ReplayValidationError(f"{event_type} has an invalid scheduled_round")
    return plan_id, action, scheduled_round


def _validate_budget_payload(payload: dict[str, Any], event_type: str) -> None:
    for field in ("budget_before", "budget_after"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0.0:
            raise ReplayValidationError(f"{event_type} has an invalid {field}")


def _reject_private_adaptive_payload(payload: dict[str, Any], event_type: str) -> None:
    serialized = str(payload).lower()
    if any(token in serialized for token in ("hidden_probe", "hidden_profile", "valueprobe", "prompt")):
        raise ReplayValidationError(f"{event_type} contains prohibited private or prompt material")


__all__ = ["ReplayValidationError", "ReplayValidationReport", "validate_replay"]
