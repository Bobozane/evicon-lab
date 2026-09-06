"""Read-only manifest and artifact loaders for offline evaluation."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..models import ProbeResult, RunRecord, RunStatus
from .models import (
    EvaluationManifest,
    EvaluationRunEntry,
    InputFileAudit,
)


class EvaluationInputError(ValueError):
    """Raised when a declared local evaluation input cannot be trusted."""


@dataclass(frozen=True)
class LoadedRunRecord:
    """A parsed run record plus the immutable file audit used in the report."""

    record: RunRecord
    audit: InputFileAudit


@dataclass(frozen=True)
class LoadedProbeResults:
    """One private result file read only by the offline evaluator."""

    results: tuple[ProbeResult, ...]
    audit: InputFileAudit


@dataclass(frozen=True)
class LoadedEvaluationRun:
    """All validated artifacts that belong to one explicit manifest entry."""

    entry: EvaluationRunEntry
    record: RunRecord
    events: tuple[dict[str, Any], ...]
    probe_results: tuple[ProbeResult, ...]
    input_audit: tuple[InputFileAudit, ...]

    @property
    def agent_ids(self) -> tuple[str, ...]:
        scenario = self.record.scenario
        assert scenario is not None  # established by ``load_declared_run``
        return tuple(sorted(agent.agent_id for agent in scenario.agents))

    @property
    def probe_set_id(self) -> str | None:
        if not self.probe_results:
            return None
        return self.probe_results[0].probe_set_id


class EvaluationManifestLoader:
    """Load a strict JSON or TOML manifest without resolving any input implicitly."""

    @staticmethod
    def load(path: str | Path) -> EvaluationManifest:
        manifest_path = Path(path)
        raw = _read_bytes(manifest_path, "evaluation manifest")
        payload = _parse_object(manifest_path, raw, "evaluation manifest")
        try:
            return EvaluationManifest.model_validate(payload)
        except ValidationError as exc:
            raise EvaluationInputError(
                f"invalid evaluation manifest {manifest_path}: {_validation_details(exc)}"
            ) from exc


class RunRecordLoader:
    """Read a persisted RunRecord and retain an SHA-256 audit entry."""

    @staticmethod
    def load(path: str | Path) -> LoadedRunRecord:
        record_path = Path(path)
        raw = _read_bytes(record_path, "run record")
        payload = _parse_json_object(record_path, raw, "run record")
        try:
            record = RunRecord.model_validate(payload)
        except ValidationError as exc:
            raise EvaluationInputError(
                f"invalid run record {record_path}: {_validation_details(exc)}"
            ) from exc
        return LoadedRunRecord(record=record, audit=_file_audit(record_path, raw, "run_record"))


class ProbeResultLoader:
    """Read isolated ProbeResult JSONL without exposing it to online components."""

    @staticmethod
    def load(path: str | Path) -> LoadedProbeResults:
        result_path = Path(path)
        raw = _read_bytes(result_path, "probe results")
        try:
            lines = raw.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise EvaluationInputError(f"cannot decode probe results {result_path}: {exc}") from exc
        if not lines:
            raise EvaluationInputError(f"probe results {result_path} must not be empty")

        results: list[ProbeResult] = []
        for line_number, line in enumerate(lines, start=1):
            if not line:
                raise EvaluationInputError(
                    f"probe results {result_path}:{line_number}: blank JSONL lines are not allowed"
                )
            try:
                results.append(ProbeResult.model_validate_json(line))
            except (ValidationError, json.JSONDecodeError) as exc:
                raise EvaluationInputError(
                    f"invalid ProbeResult {result_path}:{line_number}: {exc}"
                ) from exc
        _validate_probe_result_file(result_path, results)
        return LoadedProbeResults(
            results=tuple(results),
            audit=_file_audit(result_path, raw, "probe_results"),
        )


def load_declared_run(
    entry: EvaluationRunEntry,
    manifest: EvaluationManifest,
    *,
    manifest_directory: str | Path,
) -> LoadedEvaluationRun:
    """Resolve and cross-check one explicit entry without creating or changing files."""
    base_directory = Path(manifest_directory)
    record_loaded = RunRecordLoader.load(_resolve_path(entry.run_record_path, base_directory))
    record = record_loaded.record
    scenario = record.scenario
    if scenario is None:
        raise EvaluationInputError(
            f"run record {record_loaded.audit.path}: scenario snapshot is required for evaluation"
        )
    _validate_entry_against_record(entry, manifest, record, record_loaded.audit.path)

    events_path = _resolve_path(entry.events_path, base_directory)
    events_raw = _read_bytes(events_path, "events")
    events = _load_events(events_path, events_raw, record.config.run_id)
    event_audit = _file_audit(events_path, events_raw, "events")

    probe_results: tuple[ProbeResult, ...] = ()
    audits = [record_loaded.audit, event_audit]
    if entry.probe_result_path is not None:
        probes_loaded = ProbeResultLoader.load(_resolve_path(entry.probe_result_path, base_directory))
        probe_results = probes_loaded.results
        _validate_probes_against_run(
            probes_loaded.audit.path,
            probe_results,
            record,
            manifest,
        )
        audits.append(probes_loaded.audit)

    return LoadedEvaluationRun(
        entry=entry,
        record=record,
        events=tuple(events),
        probe_results=probe_results,
        input_audit=tuple(audits),
    )


def _validate_entry_against_record(
    entry: EvaluationRunEntry,
    manifest: EvaluationManifest,
    record: RunRecord,
    path: str,
) -> None:
    scenario = record.scenario
    assert scenario is not None
    checks = (
        (record.config.run_id, entry.run_id, "run_id"),
        (record.config.scenario_id, manifest.scenario_id, "scenario_id"),
        (scenario.scenario_id, manifest.scenario_id, "scenario.scenario_id"),
        (record.config.protocol, entry.protocol, "protocol"),
        (record.config.seed, entry.seed, "seed"),
        (record.config.model_name, entry.model_name, "model_name"),
    )
    for actual, expected, field_name in checks:
        if actual != expected:
            actual_value = actual.value if hasattr(actual, "value") else actual
            expected_value = expected.value if hasattr(expected, "value") else expected
            raise EvaluationInputError(
                f"run record {path}: {field_name} mismatch; "
                f"manifest={expected_value!r}, record={actual_value!r}"
            )
    agent_ids = [agent.agent_id for agent in scenario.agents]
    if record.config.agent_count != len(agent_ids):
        raise EvaluationInputError(
            f"run record {path}: config.agent_count does not match scenario agents"
        )
    if record.status is not RunStatus.COMPLETED and not entry.allow_failed_for_audit:
        raise EvaluationInputError(
            f"run record {path}: status={record.status.value!r} is not eligible; "
            "set allow_failed_for_audit only for partial audit input"
        )


def _validate_probe_result_file(path: Path, results: list[ProbeResult]) -> None:
    agent_ids = [result.agent_id for result in results]
    if len(agent_ids) != len(set(agent_ids)):
        raise EvaluationInputError(f"probe results {path}: duplicate agent_id values")
    probe_set_ids = {result.probe_set_id for result in results}
    if len(probe_set_ids) != 1:
        raise EvaluationInputError(f"probe results {path}: probe_set_id must be consistent")
    rounds = {result.round_id for result in results}
    if len(rounds) != 1:
        raise EvaluationInputError(f"probe results {path}: round_id must be consistent")
    holdout_values = {result.is_holdout for result in results}
    if len(holdout_values) != 1:
        raise EvaluationInputError(f"probe results {path}: is_holdout must be consistent")
    if any(not result.completed for result in results):
        raise EvaluationInputError(f"probe results {path}: all ProbeResult entries must be completed")


def _validate_probes_against_run(
    path: str,
    results: tuple[ProbeResult, ...],
    record: RunRecord,
    manifest: EvaluationManifest,
) -> None:
    scenario = record.scenario
    assert scenario is not None
    expected_agents = {agent.agent_id for agent in scenario.agents}
    actual_agents = {result.agent_id for result in results}
    if actual_agents != expected_agents:
        raise EvaluationInputError(
            f"probe results {path}: agent_ids do not match scenario; "
            f"missing={sorted(expected_agents - actual_agents)}, "
            f"unexpected={sorted(actual_agents - expected_agents)}"
        )
    probe_set_id = results[0].probe_set_id
    if probe_set_id not in manifest.probe_sets:
        raise EvaluationInputError(
            f"probe results {path}: probe_set_id {probe_set_id!r} is absent from manifest.probe_sets"
        )
    for result in results:
        if result.value_profile.probe_id != probe_set_id:
            raise EvaluationInputError(
                f"probe results {path}: value_profile.probe_id must match probe_set_id"
            )


def _load_events(path: Path, raw: bytes, expected_run_id: str) -> list[dict[str, Any]]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise EvaluationInputError(f"cannot decode events {path}: {exc}") from exc
    if not lines:
        raise EvaluationInputError(f"events {path} must not be empty")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise EvaluationInputError(f"events {path}:{line_number}: blank JSONL lines are not allowed")
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationInputError(f"events {path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(event, dict):
            raise EvaluationInputError(f"events {path}:{line_number}: event must be an object")
        if event.get("run_id") != expected_run_id:
            raise EvaluationInputError(
                f"events {path}:{line_number}: run_id does not match run record"
            )
        if not isinstance(event.get("round_id"), int) or event["round_id"] < 0:
            raise EvaluationInputError(f"events {path}:{line_number}: round_id must be non-negative")
        if not isinstance(event.get("payload"), dict):
            raise EvaluationInputError(f"events {path}:{line_number}: payload must be an object")
        events.append(event)
    first = events[0]
    if first.get("event_type") != "run_started":
        raise EvaluationInputError(f"events {path}: first event must be run_started")
    return events


def _resolve_path(value: str, base_directory: Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else base_directory / candidate


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise EvaluationInputError(f"cannot read {label} {path}: {exc.strerror or exc}") from exc


def _parse_object(path: Path, raw: bytes, label: str) -> dict[str, Any]:
    try:
        if path.suffix == ".toml":
            payload: Any = tomllib.loads(raw.decode("utf-8"))
        elif path.suffix == ".json":
            payload = json.loads(raw.decode("utf-8"))
        else:
            raise EvaluationInputError(f"{label} {path} must use a .toml or .json extension")
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationInputError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationInputError(f"{label} {path}: root must be an object")
    return payload


def _parse_json_object(path: Path, raw: bytes, label: str) -> dict[str, Any]:
    try:
        payload: Any = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationInputError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationInputError(f"{label} {path}: root must be an object")
    return payload


def _file_audit(path: Path, raw: bytes, kind: str) -> InputFileAudit:
    return InputFileAudit(
        path=str(path.resolve()),
        kind=kind,
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_count=len(raw),
    )


def _validation_details(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in error.errors()
    )
