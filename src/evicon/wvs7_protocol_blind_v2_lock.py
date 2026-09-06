"""Read-only lock validation for the completed protocol-blind v2 calibration.

This module consumes safe receipts, batch metadata, replay results, and the
content-free request ledger.  It never loads prompts, responses, probe text,
or provider configuration.  A successful validation writes one immutable
lock receipt under ``outputs/study-locks``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .batch_experiment import BatchExperimentManifest, BatchExperimentRecord, BatchRunStatus
from .frozen_wvs7_manifest import sha256_file
from .replay import validate_replay
from .request_ledger import RequestLedger, RequestLedgerEntry, RequestLedgerStatus


EXPECTED_RUN_ID = "wvs7-real-baseline-pilot-seed-003"
EXPECTED_PROBE_SHA256 = "181fc27816565e22d23689bb2fd3da5fe693db7943c734a0ccfc355aa40190f2"
EXPECTED_PROBE_TEMPLATE = "contextual_value_probe.v2"
EXPECTED_CONDITIONS = ("independent", "social_only", "evidence_only", "evidence_social")
EXPECTED_LOGICAL_REQUESTS = 384
EXPECTED_PRE_COORDINATES = 46
_ROOT = Path(__file__).resolve().parents[2]


class CalibrationLockError(ValueError):
    """Stable, content-free lock failure."""


class ProtocolBlindAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pre_probe_protocol_blind: bool
    pre_probe_coordinate_count: int = Field(ge=0)
    pre_probe_hash_mismatch_count: int = Field(ge=0)
    recovered_failure_attempt_count: int = Field(ge=0)
    all_logical_requests_completed: bool
    all_replays_passed: bool


class CalibrationLockReceipt(BaseModel):
    """Safe, immutable metadata proving that the v2 calibration can be locked."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lock_id: str = Field(min_length=1)
    study_id: str = Field(min_length=1)
    status: Literal["locked"]
    development_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    v1_002_excluded_from_v2_main_analysis: Literal[True]
    pilot_version: Literal["protocol_blind_v2"]
    run_id: str = Field(min_length=1)
    condition_count: int = Field(ge=0)
    completed_run_count: int = Field(ge=0)
    frozen_probe_set_sha256: str = Field(min_length=64, max_length=64)
    input_file_sha256: dict[str, str]
    logical_request_count: int = Field(ge=0)
    completed_logical_request_count: int = Field(ge=0)
    recovered_failure_attempt_count: int = Field(ge=0)
    probe_template_versions: list[str]
    protocol_blind_audit: ProtocolBlindAudit


class CalibrationLockResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["locked", "blocked"]
    lock_receipt_path: str | None = None
    blocking_reasons: list[str] = Field(default_factory=list)
    run_id: str
    logical_request_count: int = Field(ge=0)
    recovered_failure_attempt_count: int = Field(ge=0)


def _load_json(path: Path, code: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationLockError(code) from exc
    if not isinstance(payload, dict):
        raise CalibrationLockError(code)
    return payload


def _hash_inputs(paths: dict[str, Path]) -> dict[str, str]:
    try:
        return {name: sha256_file(path) for name, path in paths.items()}
    except (OSError, ValueError) as exc:
        raise CalibrationLockError("lock_input_missing") from exc


def _validate_ledger(path: Path, manifest: BatchExperimentManifest) -> tuple[list[RequestLedgerEntry], int, int, ProtocolBlindAudit]:
    try:
        entries = RequestLedger(path).entries()
    except Exception as exc:
        raise CalibrationLockError("request_ledger_invalid") from exc
    started = [entry for entry in entries if entry.status is RequestLedgerStatus.STARTED]
    fingerprints = {entry.fingerprint for entry in started}
    if len(fingerprints) != EXPECTED_LOGICAL_REQUESTS:
        raise CalibrationLockError("logical_request_count_mismatch")
    if {condition.value for condition in manifest.conditions} != set(EXPECTED_CONDITIONS):
        raise CalibrationLockError("calibration_conditions_mismatch")
    if len({entry.request_key for entry in started}) != EXPECTED_LOGICAL_REQUESTS:
        raise CalibrationLockError("logical_request_coordinate_mismatch")
    by_fingerprint: dict[str, list[RequestLedgerEntry]] = defaultdict(list)
    for entry in entries:
        by_fingerprint[entry.fingerprint].append(entry)
    recovered_failures = 0
    terminal_completed: dict[str, RequestLedgerEntry] = {}
    for fingerprint in fingerprints:
        rows = by_fingerprint[fingerprint]
        terminals = [row for row in rows if row.status in {RequestLedgerStatus.COMPLETED, RequestLedgerStatus.FAILED}]
        if not terminals:
            raise CalibrationLockError("logical_request_not_completed")
        latest = max(terminals, key=lambda row: row.attempt_count)
        if latest.status is not RequestLedgerStatus.COMPLETED:
            raise CalibrationLockError("logical_request_not_completed")
        recovered_failures += sum(row.status is RequestLedgerStatus.FAILED for row in terminals)
        terminal_completed[fingerprint] = latest
    if any(entry.template_version not in {"agent_turn.v1", EXPECTED_PROBE_TEMPLATE} for entry in entries):
        raise CalibrationLockError("unknown_request_template")
    probe_entries = [entry for entry in entries if entry.phase == "probe"]
    if not probe_entries or any(entry.template_version != EXPECTED_PROBE_TEMPLATE for entry in probe_entries):
        raise CalibrationLockError("probe_template_version_mismatch")
    expected_per_condition = EXPECTED_LOGICAL_REQUESTS // len(EXPECTED_CONDITIONS)
    if any(sum(entry.condition == condition for entry in terminal_completed.values()) != expected_per_condition for condition in EXPECTED_CONDITIONS):
        raise CalibrationLockError("condition_request_count_mismatch")

    coordinates: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for entry in terminal_completed.values():
        if entry.phase == "probe" and entry.round_id == 0 and entry.agent_id and entry.probe_id:
            coordinates[(entry.agent_id, entry.probe_id)][entry.condition].add(entry.public_content_sha256)
    expected_coordinates = {
        (agent_id, probe_id)
        for agent_id in manifest.agent_ids
        for probe_id in manifest.probe_item_ids
    }
    if len(coordinates) != EXPECTED_PRE_COORDINATES or set(coordinates) != expected_coordinates:
        raise CalibrationLockError("pre_probe_coordinate_count_mismatch")
    mismatch_count = 0
    for by_condition in coordinates.values():
        if set(by_condition) != set(EXPECTED_CONDITIONS) or len({next(iter(values)) for values in by_condition.values() if len(values) == 1}) != 1 or any(len(values) != 1 for values in by_condition.values()):
            mismatch_count += 1
    if mismatch_count:
        raise CalibrationLockError("pre_probe_public_hash_mismatch")
    return entries, len(fingerprints), recovered_failures, ProtocolBlindAudit(
        pre_probe_protocol_blind=True,
        pre_probe_coordinate_count=len(coordinates),
        pre_probe_hash_mismatch_count=mismatch_count,
        recovered_failure_attempt_count=recovered_failures,
        all_logical_requests_completed=True,
        all_replays_passed=True,
    )


def lock_protocol_blind_v2_calibration(
    calibration_dir: str | Path = "results/batches/wvs7-real-baseline-pilot-seed-003",
    *,
    output_path: str | Path = "outputs/study-locks/wvs7_protocol_blind_v2_calibration_lock.json",
) -> CalibrationLockResult:
    """Validate 003 and write a new lock receipt without modifying its inputs."""
    root = Path(calibration_dir)
    receipt_path = root / "protocol_blind_v2_calibration_receipt.json"
    batch_manifest_path = root / "batch_manifest.json"
    batch_record_path = root / "batch_record.json"
    ledger_path = root / "request_ledger.jsonl"
    report_path = root / "offline_evaluation_report_initial_final_v2.json"
    initial_final_path = root / "protocol_blind_v2_initial_final_report.json"
    receipt = _load_json(receipt_path, "calibration_receipt_invalid")
    if receipt.get("run_id") != EXPECTED_RUN_ID or receipt.get("status") != "calibration":
        raise CalibrationLockError("calibration_receipt_contract_mismatch")
    if receipt.get("frozen_probe_set_sha256") != EXPECTED_PROBE_SHA256:
        raise CalibrationLockError("frozen_probe_set_hash_mismatch")
    if not all(receipt.get(flag) is True for flag in ("development_only", "not_paper_result", "no_causal_conclusion", "v1_002_excluded_from_v2_main_analysis")):
        raise CalibrationLockError("calibration_receipt_safety_flags_invalid")
    manifest_payload = _load_json(batch_manifest_path, "batch_manifest_invalid")
    record_payload = _load_json(batch_record_path, "batch_record_invalid")
    if tuple(manifest_payload.get("conditions", [])) != EXPECTED_CONDITIONS:
        raise CalibrationLockError("calibration_conditions_mismatch")
    try:
        manifest = BatchExperimentManifest.model_validate(manifest_payload)
        record = BatchExperimentRecord.model_validate(record_payload)
    except ValidationError as exc:
        raise CalibrationLockError("batch_metadata_invalid") from exc
    if manifest.batch_id != EXPECTED_RUN_ID or record.batch_id != EXPECTED_RUN_ID:
        raise CalibrationLockError("batch_run_id_mismatch")
    if (
        manifest.probe_set_id != "wvs7-english-core-23"
        or manifest.max_rounds != 2
        or len(manifest.agent_ids) != 2
        or manifest.seeds != [20260819]
        or manifest.metadata.get("frozen_probe_set_sha256") != EXPECTED_PROBE_SHA256
    ):
        raise CalibrationLockError("batch_manifest_contract_mismatch")
    if tuple(run.condition.value for run in manifest.runs) != EXPECTED_CONDITIONS:
        raise CalibrationLockError("calibration_conditions_mismatch")
    if record.status.value != "completed" or record.completed_run_count != 4 or record.failed_run_count != 0:
        raise CalibrationLockError("completed_matched_group_required")
    if (
        len(record.runs) != 4
        or any(run.status is not BatchRunStatus.COMPLETED for run in record.runs)
        or len(record.matched_groups) != 1
        or not all(group.is_complete for group in record.matched_groups)
    ):
        raise CalibrationLockError("condition_not_completed")
    replay_statuses: dict[str, str] = {}
    for spec in manifest.runs:
        result = validate_replay(spec.artifacts.events_path, spec.artifacts.run_record_path)
        replay_statuses[spec.run_id] = result.validation
    if set(replay_statuses.values()) != {"passed"}:
        raise CalibrationLockError("calibration_replay_validation_failed")
    probe_path = _ROOT / "outputs/wvs7-frozen/english_core_wvs7_23_frozen.json"
    if not probe_path.exists() or sha256_file(probe_path) != EXPECTED_PROBE_SHA256:
        raise CalibrationLockError("frozen_probe_set_hash_mismatch")
    entries, logical_count, recovered_failures, audit = _validate_ledger(ledger_path, manifest)
    if _load_json(initial_final_path, "initial_final_report_invalid").get("profile_drift_condition_count") != 4:
        raise CalibrationLockError("profile_drift_condition_count_mismatch")
    evaluation_report = _load_json(report_path, "initial_final_report_invalid")
    metrics = evaluation_report.get("metrics")
    profile_drift_count = sum(
        isinstance(metric, dict) and metric.get("metric_name") == "profile_drift"
        for metric in metrics if isinstance(metrics, list)
    )
    if profile_drift_count != 4:
        raise CalibrationLockError("profile_drift_condition_count_mismatch")
    input_paths = {
        "calibration_receipt": receipt_path,
        "batch_manifest": batch_manifest_path,
        "batch_record": batch_record_path,
        "request_ledger": ledger_path,
        "initial_final_report": report_path,
        "probe_set": probe_path,
    }
    output = Path(output_path)
    if output.exists():
        raise CalibrationLockError("lock_output_already_exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = CalibrationLockReceipt(
        lock_id="wvs7-protocol-blind-v2-calibration-lock-003",
        study_id=str(receipt["study_id"]),
        status="locked",
        development_only=True,
        not_paper_result=True,
        no_causal_conclusion=True,
        v1_002_excluded_from_v2_main_analysis=True,
        pilot_version="protocol_blind_v2",
        run_id=EXPECTED_RUN_ID,
        condition_count=4,
        completed_run_count=4,
        frozen_probe_set_sha256=EXPECTED_PROBE_SHA256,
        input_file_sha256=_hash_inputs(input_paths),
        logical_request_count=logical_count,
        completed_logical_request_count=logical_count,
        recovered_failure_attempt_count=recovered_failures,
        probe_template_versions=[EXPECTED_PROBE_TEMPLATE],
        protocol_blind_audit=audit.model_copy(update={"all_replays_passed": True}),
    )
    output.write_text(lock.model_dump_json(indent=2), encoding="utf-8")
    return CalibrationLockResult(status="locked", lock_receipt_path=str(output), run_id=EXPECTED_RUN_ID, logical_request_count=logical_count, recovered_failure_attempt_count=recovered_failures)


def validate_calibration_lock(
    calibration_dir: str | Path = "results/batches/wvs7-real-baseline-pilot-seed-003",
    *,
    output_path: str | Path = "outputs/study-locks/wvs7_protocol_blind_v2_calibration_lock.json",
) -> CalibrationLockResult:
    """Descriptive alias for :func:`lock_protocol_blind_v2_calibration`."""
    return lock_protocol_blind_v2_calibration(calibration_dir, output_path=output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lock a completed protocol-blind v2 calibration using local safe receipts only.")
    parser.add_argument("--calibration-dir", default="results/batches/wvs7-real-baseline-pilot-seed-003")
    parser.add_argument("--output", default="outputs/study-locks/wvs7_protocol_blind_v2_calibration_lock.json")
    args = parser.parse_args(argv)
    try:
        result = lock_protocol_blind_v2_calibration(args.calibration_dir, output_path=args.output)
    except CalibrationLockError as exc:
        print(json.dumps({"status": "blocked", "blocking_reasons": [str(exc)]}, sort_keys=True))
        return 1
    print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CalibrationLockError",
    "CalibrationLockReceipt",
    "CalibrationLockResult",
    "ProtocolBlindAudit",
    "lock_protocol_blind_v2_calibration",
    "validate_calibration_lock",
]
