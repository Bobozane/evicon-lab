"""Read-only base and controlled-intervention replay validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import replay_core as _core
from .events import EventType
from .models import InterventionAction, RunRecord

ReplayValidationError = _core.ReplayValidationError
ReplayValidationReport = _core.ReplayValidationReport


def validate_replay(
    events_path: str | Path,
    record_path: str | Path,
) -> ReplayValidationReport:
    """Validate base replay invariants plus any opt-in controlled plan events."""
    report = _core.validate_replay(events_path, record_path)
    events = _core._load_events(Path(events_path))
    record = _core._load_record(Path(record_path))
    _validate_intervention_events(events, record)
    return report


def _validate_intervention_events(events: list[dict[str, Any]], record: RunRecord) -> None:
    """Cross-check logged schedules and their prospective effects without execution."""
    seen: dict[str, tuple[int, Mapping[str, Any]]] = {}
    applied: set[str] = set()
    rejected: set[str] = set()
    effect_events = {
        EventType.VISIBILITY_OVERRIDE_APPLIED.value,
        EventType.TURN_ORDER_OVERRIDE_APPLIED.value,
        EventType.EVIDENCE_REQUEST_CREATED.value,
        EventType.MINORITY_REPORT_REQUIREMENT_CREATED.value,
    }
    for event in events:
        event_type = event["event_type"]
        if event_type not in {
            EventType.INTERVENTION_PLAN_SEEN.value,
            EventType.INTERVENTION_PLAN_APPLIED.value,
            EventType.INTERVENTION_PLAN_REJECTED.value,
            *effect_events,
        }:
            continue
        payload = event["payload"]
        _validate_intervention_payload(payload, record, event_type)
        plan_id = payload["plan_id"]
        if event_type == EventType.INTERVENTION_PLAN_SEEN.value:
            if plan_id in seen:
                raise ReplayValidationError("an intervention plan_id may be seen only once")
            seen[plan_id] = (event["round_id"], payload)
        elif plan_id not in seen:
            raise ReplayValidationError(f"{event_type} must reference an earlier intervention_plan_seen")
        else:
            seen_round, seen_payload = seen[plan_id]
            if event["round_id"] != seen_round:
                raise ReplayValidationError("intervention plan events must retain the scheduled round_id")
            _validate_matching_plan_payload(seen_payload, payload)
            if event_type == EventType.INTERVENTION_PLAN_APPLIED.value:
                if payload["applied"] is not True:
                    raise ReplayValidationError("intervention_plan_applied payload.applied must be true")
                if plan_id in applied or plan_id in rejected:
                    raise ReplayValidationError("an intervention plan_id may not be applied more than once")
                applied.add(plan_id)
            elif event_type == EventType.INTERVENTION_PLAN_REJECTED.value:
                if payload["applied"] is not False:
                    raise ReplayValidationError("intervention_plan_rejected payload.applied must be false")
                if plan_id in applied or plan_id in rejected:
                    raise ReplayValidationError("a rejected intervention plan_id must be unique")
                rejected.add(plan_id)
            else:
                if plan_id not in applied:
                    raise ReplayValidationError("effect events require a prior applied intervention plan")

    for plan_id in rejected:
        if plan_id in applied:
            raise ReplayValidationError("a rejected plan must not later be applied")
    _validate_applied_effects(seen, applied, events, record)


def _validate_intervention_payload(
    payload: Mapping[str, Any],
    record: RunRecord,
    event_type: str,
) -> None:
    required = {
        "scenario_id",
        "plan_id",
        "action",
        "target_agent_ids",
        "applied",
        "reason_codes",
        "warnings",
        "errors",
        "visibility_overrides",
        "turn_order_override",
        "requested_evidence_ids",
        "final_report_requirements",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ReplayValidationError(f"{event_type} lacks required intervention fields: {missing}")
    if payload["scenario_id"] != record.config.scenario_id:
        raise ReplayValidationError("intervention event scenario_id does not match run_record")
    if not isinstance(payload["plan_id"], str) or not payload["plan_id"].startswith("plan-"):
        raise ReplayValidationError("intervention event has an invalid plan_id")
    if payload["action"] not in {action.value for action in InterventionAction}:
        raise ReplayValidationError("intervention event has an unknown action")
    for field in (
        "target_agent_ids",
        "reason_codes",
        "warnings",
        "errors",
        "visibility_overrides",
        "turn_order_override",
        "requested_evidence_ids",
        "final_report_requirements",
    ):
        if not isinstance(payload[field], list):
            raise ReplayValidationError(f"intervention event payload.{field} must be a list")
    if not isinstance(payload["applied"], bool):
        raise ReplayValidationError("intervention event payload.applied must be a boolean")
    if any("prompt" in str(key).lower() for key in payload):
        raise ReplayValidationError("intervention events must not include prompt data")


def _validate_matching_plan_payload(seen: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    for field in (
        "scenario_id",
        "plan_id",
        "action",
        "target_agent_ids",
        "reason_codes",
        "visibility_overrides",
        "turn_order_override",
        "requested_evidence_ids",
    ):
        if seen[field] != current[field]:
            raise ReplayValidationError(f"intervention plan payload changed field {field}")


def _validate_applied_effects(
    seen: Mapping[str, tuple[int, Mapping[str, Any]]],
    applied: set[str],
    events: list[dict[str, Any]],
    record: RunRecord,
) -> None:
    turns_by_id = {turn.turn_id: turn for turn in record.turns}
    turns_by_round: dict[int, list[Any]] = {}
    for turn in record.turns:
        turns_by_round.setdefault(turn.round_id, []).append(turn)
    event_sequences = {
        (event["round_id"], event["payload"].get("turn_id")): event["sequence"]
        for event in events
        if event["event_type"] == EventType.TURN_COMPLETED.value
    }
    for plan_id in applied:
        round_id, payload = seen[plan_id]
        action = payload["action"]
        next_round = round_id + 1
        if next_round >= record.config.max_rounds:
            continue
        next_turns = turns_by_round.get(next_round, [])
        targets = payload["target_agent_ids"]
        if action == InterventionAction.BLIND_EVIDENCE_REFLECTION.value:
            for target in targets:
                target_turn = _find_turn(next_turns, target)
                if target_turn is not None and target_turn.visible_peer_turn_ids:
                    raise ReplayValidationError("blind reflection target saw peer turns in its next round")
        if action == InterventionAction.ADAPTIVE_EXPOSURE.value:
            for override in payload["visibility_overrides"]:
                if not isinstance(override, dict):
                    raise ReplayValidationError("adaptive exposure override must be an object")
                for peer_id in override.get("visible_peer_turn_ids", []):
                    peer = turns_by_id.get(peer_id)
                    if peer is None or peer.round_id >= round_id:
                        raise ReplayValidationError("adaptive exposure must reference a strictly earlier peer turn")
        if action in {
            InterventionAction.SOLICIT_DISSENT.value,
            InterventionAction.RESTRUCTURE.value,
        }:
            expected_order = payload["turn_order_override"]
            observed_order = [turn.speaker_id for turn in next_turns]
            if expected_order and observed_order != expected_order:
                raise ReplayValidationError("turn order override did not apply to the next round")
        for turn in next_turns:
            sequence = event_sequences.get((turn.round_id, turn.turn_id))
            if sequence is None:
                raise ReplayValidationError("controlled next-round turn lacks turn_completed event")


def _find_turn(turns: list[Any], agent_id: str) -> Any | None:
    matches = [turn for turn in turns if turn.speaker_id == agent_id]
    if len(matches) > 1:
        raise ReplayValidationError("multiple next-round turns for an intervention target")
    return matches[0] if matches else None


__all__ = ["ReplayValidationError", "ReplayValidationReport", "validate_replay"]
