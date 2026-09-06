"""Append-only deterministic JSONL events for base, controlled, and adaptive runs."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any


class EventType(str, Enum):
    RUN_STARTED = "run_started"
    ROUND_STARTED = "round_started"
    EXPOSURE_CREATED = "exposure_created"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    TURN_COMPLETED = "turn_completed"
    ROUND_COMPLETED = "round_completed"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    INTERVENTION_PLAN_SEEN = "intervention_plan_seen"
    INTERVENTION_PLAN_APPLIED = "intervention_plan_applied"
    INTERVENTION_PLAN_REJECTED = "intervention_plan_rejected"
    VISIBILITY_OVERRIDE_APPLIED = "visibility_override_applied"
    TURN_ORDER_OVERRIDE_APPLIED = "turn_order_override_applied"
    EVIDENCE_REQUEST_CREATED = "evidence_request_created"
    MINORITY_REPORT_REQUIREMENT_CREATED = "minority_report_requirement_created"
    ONLINE_STATE_OBSERVED = "online_state_observed"
    MONITOR_EVALUATED = "monitor_evaluated"
    TARGET_CANDIDATES_BUILT = "target_candidates_built"
    POLICY_DECIDED = "policy_decided"
    INTERVENTION_PLAN_CREATED = "intervention_plan_created"
    INTERVENTION_PLAN_SCHEDULED = "intervention_plan_scheduled"
    BUDGET_RESERVED = "budget_reserved"
    BUDGET_SPENT = "budget_spent"
    BUDGET_RELEASED = "budget_released"
    COOLDOWN_UPDATED = "cooldown_updated"


_ADAPTIVE_EVENT_TYPES = frozenset(
    {
        EventType.ONLINE_STATE_OBSERVED,
        EventType.MONITOR_EVALUATED,
        EventType.TARGET_CANDIDATES_BUILT,
        EventType.POLICY_DECIDED,
        EventType.INTERVENTION_PLAN_CREATED,
        EventType.INTERVENTION_PLAN_SCHEDULED,
        EventType.BUDGET_RESERVED,
        EventType.BUDGET_SPENT,
        EventType.BUDGET_RELEASED,
        EventType.COOLDOWN_UPDATED,
    }
)


class JsonlEventLog:
    """Write one stable JSON object per line using a monotonic sequence number."""

    def __init__(self, path: str | Path, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self._sequence = 0
        self._scenario_id: str | None = None

    def append(self, event_type: EventType, *, round_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        """Append a deterministic event without clock or environment data."""
        if self._scenario_id is None:
            scenario_id = payload.get("scenario_id")
            if isinstance(scenario_id, str) and scenario_id:
                self._scenario_id = scenario_id

        normalized_payload = dict(payload)
        if event_type in _ADAPTIVE_EVENT_TYPES:
            normalized_payload.setdefault("scenario_id", self._scenario_id)
            normalized_payload.setdefault("action", None)
            normalized_payload.setdefault("reason_codes", [])
            normalized_payload.setdefault("warnings", [])
            normalized_payload.setdefault("errors", [])

        self._sequence += 1
        event = {
            "event_id": f"{self.run_id}:{self._sequence:06d}",
            "event_type": event_type.value,
            "run_id": self.run_id,
            "scenario_id": self._scenario_id,
            "round_id": round_id,
            "sequence": self._sequence,
            "payload": normalized_payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")
        return event
