"""Read-only replay validation including adaptive audit-envelope checks."""

from __future__ import annotations

from pathlib import Path

from . import replay_stage13_previous as _previous
from .events import EventType

ReplayValidationError = _previous.ReplayValidationError
ReplayValidationReport = _previous.ReplayValidationReport

_ADAPTIVE_EVENT_TYPES = frozenset(
    {
        EventType.ONLINE_STATE_OBSERVED.value,
        EventType.MONITOR_EVALUATED.value,
        EventType.TARGET_CANDIDATES_BUILT.value,
        EventType.POLICY_DECIDED.value,
        EventType.INTERVENTION_PLAN_CREATED.value,
        EventType.INTERVENTION_PLAN_SCHEDULED.value,
        EventType.BUDGET_RESERVED.value,
        EventType.BUDGET_SPENT.value,
        EventType.BUDGET_RELEASED.value,
        EventType.COOLDOWN_UPDATED.value,
    }
)

_REQUIRED_AUDIT_FIELDS = frozenset(
    {
        "scenario_id",
        "action",
        "reason_codes",
        "budget_before",
        "budget_after",
        "cooldown_until",
        "warnings",
        "errors",
    }
)


def validate_replay(
    events_path: str | Path,
    record_path: str | Path,
) -> ReplayValidationReport:
    """Validate previous replay contracts plus the adaptive event envelope."""
    report = _previous.validate_replay(events_path, record_path)
    events = _previous._controlled._core._load_events(Path(events_path))
    record = _previous._controlled._core._load_record(Path(record_path))
    if bool(events[0]["payload"].get("adaptive")):
        _validate_adaptive_audit_envelope(events, record.config.scenario_id)
    return report


def _validate_adaptive_audit_envelope(events: list[dict[str, object]], scenario_id: str) -> None:
    for event in events:
        if event["event_type"] not in _ADAPTIVE_EVENT_TYPES:
            continue
        if event.get("scenario_id") != scenario_id:
            raise ReplayValidationError("adaptive event has a mismatched scenario_id")
        payload = event["payload"]
        if not isinstance(payload, dict):
            raise ReplayValidationError("adaptive event payload must be an object")
        missing = _REQUIRED_AUDIT_FIELDS - set(payload)
        if missing:
            raise ReplayValidationError(
                f"adaptive event is missing audit fields: {', '.join(sorted(missing))}"
            )
        if payload["scenario_id"] != scenario_id:
            raise ReplayValidationError("adaptive event payload has a mismatched scenario_id")
        if not isinstance(payload["reason_codes"], list):
            raise ReplayValidationError("adaptive event reason_codes must be a list")


__all__ = ["ReplayValidationError", "ReplayValidationReport", "validate_replay"]
