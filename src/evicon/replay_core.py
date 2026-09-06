"""Read-only validation for deterministic protocol event logs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .events import EventType
from .models import DialogueTurn, RunRecord, RunStatus, ScenarioSpec


class ReplayValidationError(ValueError):
    """Raised when a JSONL event stream cannot faithfully describe its record."""


@dataclass(frozen=True)
class ReplayValidationReport:
    """A small summary returned after a log passes every consistency check."""

    run_id: str
    event_count: int
    round_count: int
    turn_count: int
    status: RunStatus
    validation: str


def validate_replay(
    events_path: str | Path,
    record_path: str | Path,
) -> ReplayValidationReport:
    """Validate an existing event log without replaying a provider or writing files."""
    events = _load_events(Path(events_path))
    record = _load_record(Path(record_path))
    scenario = record.scenario
    if scenario is None:
        raise ReplayValidationError("run_record must include a scenario snapshot")
    if not events:
        raise ReplayValidationError("events.jsonl must contain at least run_started")

    _reject_probe_material(events)
    _validate_envelope(events, record)
    _validate_event_lifecycle(events, record)
    _validate_turns_and_visibility(events, record, scenario)
    return ReplayValidationReport(
        run_id=record.config.run_id,
        event_count=len(events),
        round_count=record.config.max_rounds,
        turn_count=len(record.turns),
        status=record.status,
        validation="passed" if record.status == RunStatus.COMPLETED else "passed_with_failure",
    )


def _load_events(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReplayValidationError(f"cannot read events {path}: {exc}") from exc

    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ReplayValidationError(f"{path}:{line_number}: blank JSONL lines are not allowed")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayValidationError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ReplayValidationError(f"{path}:{line_number}: event must be an object")
        events.append(value)
    return events


def _load_record(path: Path) -> RunRecord:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ReplayValidationError(f"cannot read run record {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ReplayValidationError(f"invalid run record {path}: {exc.msg}") from exc
    try:
        return RunRecord.model_validate(payload)
    except ValidationError as exc:
        raise ReplayValidationError(f"invalid run record {path}: {exc}") from exc


def _validate_envelope(events: list[dict[str, Any]], record: RunRecord) -> None:
    run_id = record.config.run_id
    for expected_sequence, event in enumerate(events, start=1):
        sequence = event.get("sequence")
        if type(sequence) is not int or sequence != expected_sequence:
            raise ReplayValidationError(
                f"event sequence must be continuous from 1; expected {expected_sequence}, got {sequence!r}"
            )
        if event.get("event_id") != f"{run_id}:{expected_sequence:06d}":
            raise ReplayValidationError(f"event {expected_sequence} has an inconsistent event_id")
        if event.get("run_id") != run_id:
            raise ReplayValidationError(f"event {expected_sequence} has a mismatched run_id")
        if event.get("event_type") not in {item.value for item in EventType}:
            raise ReplayValidationError(f"event {expected_sequence} has an unknown event_type")
        if type(event.get("round_id")) is not int or event["round_id"] < 0:
            raise ReplayValidationError(f"event {expected_sequence} has an invalid round_id")
        if not isinstance(event.get("payload"), dict):
            raise ReplayValidationError(f"event {expected_sequence} payload must be an object")

    if events[0]["event_type"] != EventType.RUN_STARTED.value:
        raise ReplayValidationError("run_started must be the first event")
    failed_positions = [
        index for index, event in enumerate(events) if event["event_type"] == EventType.RUN_FAILED.value
    ]
    completed_positions = [
        index for index, event in enumerate(events) if event["event_type"] == EventType.RUN_COMPLETED.value
    ]
    if len(failed_positions) > 1 or len(completed_positions) > 1:
        raise ReplayValidationError("a run may contain only one terminal event")
    if failed_positions and failed_positions[0] != len(events) - 1:
        raise ReplayValidationError("run_failed must be final; no business event may follow it")
    if completed_positions and completed_positions[0] != len(events) - 1:
        raise ReplayValidationError("run_completed must be final")
    if events[-1]["event_type"] not in {EventType.RUN_COMPLETED.value, EventType.RUN_FAILED.value}:
        raise ReplayValidationError("run_completed or run_failed must be the last event")

    started_payload = events[0]["payload"]
    if started_payload.get("scenario_id") != record.config.scenario_id:
        raise ReplayValidationError("run_started scenario_id does not match run_record")
    if started_payload.get("agent_count") != record.config.agent_count:
        raise ReplayValidationError("run_started agent_count does not match run_record")
    if started_payload.get("max_rounds") != record.config.max_rounds:
        raise ReplayValidationError("run_started max_rounds does not match run_record")


def _validate_event_lifecycle(events: list[dict[str, Any]], record: RunRecord) -> None:
    started_rounds: set[int] = set()
    completed_rounds: set[int] = set()
    request_sequence: dict[tuple[int, str], int] = {}
    response_sequence: dict[tuple[int, str], int] = {}

    for event in events:
        event_type = event["event_type"]
        round_id = event["round_id"]
        payload = event["payload"]
        if event_type in {EventType.ROUND_STARTED.value, EventType.ROUND_COMPLETED.value}:
            _require_valid_round(round_id, record, event_type)
        if event_type == EventType.ROUND_STARTED.value:
            if round_id in started_rounds:
                raise ReplayValidationError(f"round {round_id} has multiple round_started events")
            started_rounds.add(round_id)
        elif event_type == EventType.ROUND_COMPLETED.value:
            if round_id in completed_rounds:
                raise ReplayValidationError(f"round {round_id} has multiple round_completed events")
            completed_rounds.add(round_id)
        elif event_type in {EventType.LLM_REQUEST.value, EventType.LLM_RESPONSE.value}:
            agent_id = payload.get("agent_id")
            if not isinstance(agent_id, str):
                raise ReplayValidationError(f"{event_type} must include payload.agent_id")
            _require_valid_round(round_id, record, event_type)
            key = (round_id, agent_id)
            target = request_sequence if event_type == EventType.LLM_REQUEST.value else response_sequence
            if key in target:
                raise ReplayValidationError(f"duplicate {event_type} for round {round_id}, agent {agent_id}")
            target[key] = event["sequence"]

    if completed_rounds - started_rounds:
        raise ReplayValidationError("round_completed must have a corresponding round_started")
    unclosed_rounds = started_rounds - completed_rounds
    expected_rounds = set(range(record.config.max_rounds))
    if record.status == RunStatus.COMPLETED:
        if unclosed_rounds:
            raise ReplayValidationError("completed runs must not contain unclosed rounds")
        if completed_rounds != expected_rounds:
            raise ReplayValidationError("completed runs must include every configured round")
    elif len(unclosed_rounds) > 1 or (
        unclosed_rounds and unclosed_rounds != {max(started_rounds)}
    ):
        raise ReplayValidationError("failed runs may leave only their final round unclosed")
    if request_sequence.keys() != response_sequence.keys():
        raise ReplayValidationError("every llm_request must have exactly one llm_response")
    for key, request_position in request_sequence.items():
        if response_sequence[key] <= request_position:
            raise ReplayValidationError(f"llm_response precedes its llm_request for {key}")

    terminal_type = events[-1]["event_type"]
    expected_terminal = (
        EventType.RUN_COMPLETED.value
        if record.status == RunStatus.COMPLETED
        else EventType.RUN_FAILED.value
    )
    if terminal_type != expected_terminal:
        raise ReplayValidationError("terminal event does not match run_record status")
    if terminal_type == EventType.RUN_COMPLETED.value:
        if events[-1]["round_id"] != record.config.max_rounds - 1:
            raise ReplayValidationError("run_completed has an invalid round_id")
        if events[-1]["payload"].get("turn_count") != len(record.turns):
            raise ReplayValidationError("run_completed turn_count does not match run_record")


def _validate_turns_and_visibility(
    events: list[dict[str, Any]],
    record: RunRecord,
    scenario: ScenarioSpec,
) -> None:
    turns_by_id: dict[str, DialogueTurn] = {}
    for turn in record.turns:
        if turn.turn_id in turns_by_id:
            raise ReplayValidationError(f"run_record has duplicate turn_id {turn.turn_id}")
        turns_by_id[turn.turn_id] = turn

    known_agents = {agent.agent_id for agent in scenario.agents}
    evidence_by_id = {card.evidence_id: card for card in scenario.evidence_cards}
    for turn in record.turns:
        _require_valid_round(turn.round_id, record, f"turn {turn.turn_id}")
        if turn.speaker_id not in known_agents:
            raise ReplayValidationError(f"turn {turn.turn_id} has an unknown speaker")
        _validate_visibility_references(turn, turns_by_id, evidence_by_id)

    completed_turns: set[str] = set()
    requests_by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    responses_by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for event in events:
        event_type = event["event_type"]
        payload = event["payload"]
        if event_type == EventType.TURN_COMPLETED.value:
            turn_id = payload.get("turn_id")
            if not isinstance(turn_id, str) or turn_id not in turns_by_id:
                raise ReplayValidationError("turn_completed references an unknown turn_id")
            if turn_id in completed_turns:
                raise ReplayValidationError(f"turn_completed duplicates turn_id {turn_id}")
            turn = turns_by_id[turn_id]
            if event["round_id"] != turn.round_id or payload.get("speaker_id") != turn.speaker_id:
                raise ReplayValidationError(f"turn_completed does not match {turn_id}")
            _validate_payload_visibility(payload, turn, turns_by_id, evidence_by_id)
            completed_turns.add(turn_id)
        elif event_type in {EventType.EXPOSURE_CREATED.value, EventType.LLM_REQUEST.value}:
            agent_id = payload.get("agent_id")
            if not isinstance(agent_id, str) or agent_id not in known_agents:
                raise ReplayValidationError(f"{event_type} has an unknown agent")
            request_turn = _optional_turn_for(record.turns, event["round_id"], agent_id)
            if request_turn is None:
                if record.status != RunStatus.FAILED:
                    raise ReplayValidationError(
                        f"expected one turn for round {event['round_id']}, agent {agent_id}"
                    )
                _validate_partial_visibility_payload(
                    payload,
                    round_id=event["round_id"],
                    agent_id=agent_id,
                    turns_by_id=turns_by_id,
                    evidence_by_id=evidence_by_id,
                )
            else:
                _validate_payload_visibility(payload, request_turn, turns_by_id, evidence_by_id)
            if event_type == EventType.LLM_REQUEST.value:
                requests_by_key[(event["round_id"], agent_id)] = payload
        elif event_type == EventType.LLM_RESPONSE.value:
            agent_id = payload.get("agent_id")
            if not isinstance(agent_id, str):
                raise ReplayValidationError("llm_response must include payload.agent_id")
            responses_by_key[(event["round_id"], agent_id)] = payload

    if record.status == RunStatus.COMPLETED and completed_turns != set(turns_by_id):
        raise ReplayValidationError("turn_completed events do not match run_record turns")
    for key, request in requests_by_key.items():
        response = responses_by_key.get(key)
        if response is None:
            continue
        if response.get("visible_peer_turn_ids") != request.get("visible_peer_turn_ids"):
            raise ReplayValidationError(f"llm_response peer visibility differs from request for {key}")
        if response.get("visible_evidence_ids") != request.get("visible_evidence_ids"):
            raise ReplayValidationError(f"llm_response evidence visibility differs from request for {key}")


def _turn_for(turns: list[DialogueTurn], round_id: int, agent_id: str) -> DialogueTurn:
    matching = _matching_turns(turns, round_id, agent_id)
    if len(matching) != 1:
        raise ReplayValidationError(f"expected one turn for round {round_id}, agent {agent_id}")
    return matching[0]


def _optional_turn_for(
    turns: list[DialogueTurn], round_id: int, agent_id: str
) -> DialogueTurn | None:
    matching = _matching_turns(turns, round_id, agent_id)
    if len(matching) > 1:
        raise ReplayValidationError(f"multiple turns exist for round {round_id}, agent {agent_id}")
    return matching[0] if matching else None


def _matching_turns(turns: list[DialogueTurn], round_id: int, agent_id: str) -> list[DialogueTurn]:
    return [turn for turn in turns if turn.round_id == round_id and turn.speaker_id == agent_id]


def _validate_payload_visibility(
    payload: Mapping[str, Any],
    turn: DialogueTurn,
    turns_by_id: Mapping[str, DialogueTurn],
    evidence_by_id: Mapping[str, Any],
) -> None:
    peers = payload.get("visible_peer_turn_ids")
    evidence_ids = payload.get("visible_evidence_ids")
    if peers != turn.visible_peer_turn_ids or evidence_ids != turn.visible_evidence_ids:
        raise ReplayValidationError(f"visibility payload does not match turn {turn.turn_id}")
    _validate_visibility_references(turn, turns_by_id, evidence_by_id)
    history_ids = payload.get("visible_history_ids")
    if history_ids is not None:
        if not isinstance(history_ids, list):
            raise ReplayValidationError("visible_history_ids must be a list")
        for history_id in history_ids:
            history_turn = turns_by_id.get(history_id)
            if history_turn is None or history_turn.round_id >= turn.round_id:
                raise ReplayValidationError("visible history must reference an earlier turn")


def _validate_partial_visibility_payload(
    payload: Mapping[str, Any],
    *,
    round_id: int,
    agent_id: str,
    turns_by_id: Mapping[str, DialogueTurn],
    evidence_by_id: Mapping[str, Any],
) -> None:
    """Validate final failed-round context that has no completed DialogueTurn."""
    peer_ids = payload.get("visible_peer_turn_ids")
    evidence_ids = payload.get("visible_evidence_ids")
    if not isinstance(peer_ids, list) or not isinstance(evidence_ids, list):
        raise ReplayValidationError("partial-round visibility IDs must be lists")
    for peer_id in peer_ids:
        peer = turns_by_id.get(peer_id)
        if peer is None or peer.speaker_id == agent_id or peer.round_id >= round_id:
            raise ReplayValidationError("partial-round peer visibility must reference an earlier peer turn")
    for evidence_id in evidence_ids:
        card = evidence_by_id.get(evidence_id)
        if card is None:
            raise ReplayValidationError("partial-round visibility references an unknown evidence_id")
        if card.introduced_round > round_id:
            raise ReplayValidationError("partial-round visibility references evidence before introduction")
        if not card.is_visible_to(agent_id):
            raise ReplayValidationError("partial-round visibility references invisible evidence")
    history_ids = payload.get("visible_history_ids")
    if history_ids is not None:
        if not isinstance(history_ids, list):
            raise ReplayValidationError("visible_history_ids must be a list")
        for history_id in history_ids:
            history_turn = turns_by_id.get(history_id)
            if history_turn is None or history_turn.round_id >= round_id:
                raise ReplayValidationError("visible history must reference an earlier turn")


def _validate_visibility_references(
    turn: DialogueTurn,
    turns_by_id: Mapping[str, DialogueTurn],
    evidence_by_id: Mapping[str, Any],
) -> None:
    for peer_id in turn.visible_peer_turn_ids:
        peer = turns_by_id.get(peer_id)
        if peer is None:
            raise ReplayValidationError(f"turn {turn.turn_id} references an unknown peer turn")
        if peer.speaker_id == turn.speaker_id or peer.round_id >= turn.round_id:
            raise ReplayValidationError(f"turn {turn.turn_id} has a non-prior peer turn reference")
    for evidence_id in turn.visible_evidence_ids:
        card = evidence_by_id.get(evidence_id)
        if card is None:
            raise ReplayValidationError(f"turn {turn.turn_id} references an unknown evidence_id")
        if card.introduced_round > turn.round_id:
            raise ReplayValidationError(f"turn {turn.turn_id} references evidence before introduction")
        if not card.is_visible_to(turn.speaker_id):
            raise ReplayValidationError(f"turn {turn.turn_id} references evidence not visible to its speaker")


def _require_valid_round(round_id: int, record: RunRecord, label: str) -> None:
    if round_id < 0 or round_id >= record.config.max_rounds:
        raise ReplayValidationError(f"{label} has a round_id outside configured bounds")


def _reject_probe_material(events: list[dict[str, Any]]) -> None:
    for event_number, event in enumerate(events, start=1):
        for value in _walk_values(event):
            if isinstance(value, str) and ("holdout" in value.lower() or "valueprobe" in value.lower()):
                raise ReplayValidationError(
                    f"event {event_number} contains prohibited holdout probe material"
                )


def _walk_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        values: list[Any] = []
        for key, nested in value.items():
            values.extend([key])
            values.extend(_walk_values(nested))
        return values
    if isinstance(value, list):
        return [item for nested in value for item in _walk_values(nested)]
    return [value]
