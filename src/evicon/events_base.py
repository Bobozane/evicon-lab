"""Append-only deterministic JSONL events, including controlled-plan audit events."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any


class EventType(str, Enum):
    """Event vocabulary for base protocol and opt-in controlled runs."""

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


class JsonlEventLog:
    """Write one stable JSON object per line using a monotonic sequence number."""

    def __init__(self, path: str | Path, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self._sequence = 0

    def append(self, event_type: EventType, *, round_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        """Append and return a deterministic event without clock or environment data."""
        self._sequence += 1
        event = {
            "event_id": f"{self.run_id}:{self._sequence:06d}",
            "event_type": event_type.value,
            "run_id": self.run_id,
            "round_id": round_id,
            "sequence": self._sequence,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")
        return event
