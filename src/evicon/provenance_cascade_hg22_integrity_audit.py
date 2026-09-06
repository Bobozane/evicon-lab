"""Read-only integrity audit for the completed H-G.2.2 eligibility calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .cascade_hg1_replay import HG1OutcomeReplayValidator
from .cascade_outcome_replay import CascadeOutcomeReplayStatus
from .cascade_real_agent_runner import CascadeAgentCheckpoint, CascadeRealAgentRunRecord
from .provenance_cascade_hg2 import sha256_file
from .provenance_cascade_hg21_calibration import (
    CalibrationStatus, HG21BatchRecord, HG21CalibrationReceipt, HG21ExecutionApproval,
    load_decisions, _hash_json,
)
from .provenance_cascade_hg22 import (
    DEFAULT_AMENDMENT_RECEIPT, DEFAULT_APPROVAL, DEFAULT_COMPATIBILITY_RECEIPT,
    DEFAULT_CONFIG, load_config,
)
from .provenance_cascade_hg22_calibration import (
    CONTROLLER_PATH, DEFAULT_EXECUTION_APPROVAL, REPLAY_PATH, HG22CalibrationRunner,
)
from .request_ledger import RequestLedger, RequestLedgerStatus

_ROOT = Path(__file__).resolve().parents[2]
PILOT_ROOT = _ROOT / "results/provenance-cascade-hg22-eligibility-calibration-v1"
BATCH_PATH = PILOT_ROOT / "calibration_batch_record.json"
CALIBRATION_RECEIPT_PATH = PILOT_ROOT / "calibration_receipt.json"
AUDIT_RECEIPT_PATH = _ROOT / "outputs/study-locks/provenance_cascade_hg22_resume_cap_technical_amendment_receipt.json"
AUDIT_VERSION = "provenance_cascade_hg22_resume_cap_audit.v1"
PRE_FIX_REQUEST_LEDGER_SHA256 = "689008ca9bdb0c90e6bebdf31c5d4e683b3da1165f42e21d819842e69169f2c9"
POST_FIX_REQUEST_LEDGER_SHA256 = "329709bedb436cc0835feb9c1e9608c24d8144f74f4f5680cb0bdb503a79b466"
FIX_APPLIED_AT = datetime.fromisoformat("2026-08-25T13:44:58.749936+00:00")
_FORBIDDEN = {
    "ground_truth_label", "source_independence_label", "system_prompt", "user_prompt",
    "prompt", "raw_response", "provider_metadata", "api_key", "authorization",
}


class HG22IntegrityAuditError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HG22TechnicalAuditReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    receipt_id: Literal["provenance-cascade-hg22-resume-cap-technical-amendment-v1"]
    audit_version: Literal["provenance_cascade_hg22_resume_cap_audit.v1"]
    status: Literal["passed_development_analysis_eligible"]
    analysis_input_status: Literal["development_only_evaluator_analysis_allowed"]
    study_id: str
    config_sha256: str
    protocol_sha256: str
    calibration_runner_sha256: str
    controller_sha256: str
    replay_sha256: str
    request_ledger_pre_fix_sha256: str
    request_ledger_post_fix_sha256: str
    calibration_receipt_sha256: str
    batch_record_sha256: str
    design_approval_sha256: str
    execution_approval_sha256: str
    compatibility_receipt_sha256: str
    amendment_receipt_sha256: str
    first_request_started_at: datetime
    implementation_fix_applied_at: datetime
    final_request_completed_at: datetime
    fix_occurred_after_batch_start: Literal[True]
    fix_occurred_before_batch_completion: Literal[True]
    fix_scope: tuple[str, ...]
    prompt_or_request_content_changed: Literal[False]
    fingerprint_algorithm_changed: Literal[False]
    parser_or_output_contract_changed: Literal[False]
    scenario_condition_seed_metric_changed: Literal[False]
    logical_request_cap_changed: Literal[False]
    completed_fingerprint_replayed: Literal[False]
    original_receipt_bound_request_ledger_implementation: Literal[False]
    technical_receipt_supplies_missing_implementation_binding: Literal[True]
    pre_fix_ledger_snapshot_hash_available: Literal[False]
    pre_fix_completed_request_count: int = Field(ge=0)
    pre_fix_failed_attempt_count: int = Field(ge=0)
    run_count: Literal[16]
    matched_group_count: Literal[4]
    unique_logical_request_count: Literal[288]
    completed_logical_request_count: Literal[288]
    transport_attempt_count: Literal[292]
    failed_transport_attempt_count: Literal[4]
    recovered_failed_fingerprint_count: Literal[4]
    duplicate_completed_fingerprint_count: Literal[0]
    orphaned_started_attempt_count: Literal[0]
    behavior_decision_count: Literal[288]
    checkpoint_entry_count: Literal[288]
    replay_passed_count: Literal[16]
    actual_prompt_token_count: Literal[218995]
    actual_completion_token_count: Literal[49981]
    actual_total_token_count: Literal[268976]
    limitation_codes: tuple[str, ...]
    engineering_integrity_passed: Literal[True]
    behavior_or_mechanism_conclusion_permitted: Literal[False]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    private_truth_loaded: Literal[False]
    network_used_for_audit: Literal[False]


def _require(value: bool, code: str) -> None:
    if not value:
        raise HG22IntegrityAuditError(code)


def _read_model(path: Path, model: type[BaseModel], code: str) -> BaseModel:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HG22IntegrityAuditError(code) from exc


def _safe(value: object) -> None:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True).lower()
    _require(not any(f'"{key}"' in payload for key in _FORBIDDEN), "sensitive_public_artifact_field")


def _checkpoint(path: Path) -> CascadeAgentCheckpoint:
    try:
        return CascadeAgentCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HG22IntegrityAuditError("checkpoint_invalid") from exc


def audit_hg22_integrity() -> HG22TechnicalAuditReceipt:
    config, config_path = load_config(DEFAULT_CONFIG)
    runner = HG22CalibrationRunner(config_path)
    receipt = _read_model(CALIBRATION_RECEIPT_PATH, HG21CalibrationReceipt, "calibration_receipt_invalid")
    batch = _read_model(BATCH_PATH, HG21BatchRecord, "batch_record_invalid")
    assert isinstance(receipt, HG21CalibrationReceipt) and isinstance(batch, HG21BatchRecord)
    execution = HG21ExecutionApproval.model_validate(
        tomllib.loads((_ROOT / DEFAULT_EXECUTION_APPROVAL).read_text(encoding="utf-8"))
    )
    expected_hashes = {
        "config_sha256": sha256_file(config_path),
        "protocol_sha256": sha256_file(_ROOT / config.protocol_path),
        "controller_sha256": sha256_file(_ROOT / CONTROLLER_PATH),
        "replay_sha256": sha256_file(_ROOT / REPLAY_PATH),
        "amendment_receipt_sha256": sha256_file(_ROOT / DEFAULT_AMENDMENT_RECEIPT),
        "design_approval_sha256": sha256_file(_ROOT / DEFAULT_APPROVAL),
        "execution_approval_sha256": sha256_file(_ROOT / DEFAULT_EXECUTION_APPROVAL),
        "compatibility_receipt_sha256": sha256_file(_ROOT / DEFAULT_COMPATIBILITY_RECEIPT),
    }
    for field, expected in expected_hashes.items():
        _require(getattr(receipt, field) == expected and getattr(batch, field) == expected, f"{field}_mismatch")
    _require(execution.runner_sha256 == sha256_file(_ROOT / "src/evicon/provenance_cascade_hg22_calibration.py"), "runner_hash_mismatch")
    _require(sha256_file(_ROOT / "src/evicon/request_ledger.py") == POST_FIX_REQUEST_LEDGER_SHA256, "request_ledger_post_fix_hash_mismatch")
    _require(receipt.status == "eligibility_calibration_completed" and batch.status is CalibrationStatus.COMPLETED, "calibration_incomplete")
    _require(receipt.run_count == receipt.completed_run_count == len(batch.runs) == 16, "run_count_mismatch")
    _require(receipt.matched_group_count == 4 and receipt.replay_passed_count == 16, "matched_or_replay_count_mismatch")

    states = {item.run_id: item for item in batch.runs}
    all_started = []
    all_terminal = []
    aggregate_decisions = 0
    aggregate_checkpoints = 0
    aggregate_records: list[CascadeRealAgentRunRecord] = []
    failed_fingerprints: set[str] = set()
    completed_fingerprints: set[str] = set()
    duplicate_completed = 0
    orphaned = 0
    for spec in runner.runs:
        _require(spec.run_id in states, "run_set_incomplete")
        state = states[spec.run_id]
        _require(
            state.status is CalibrationStatus.COMPLETED and state.error_code is None
            and state.logical_request_count == 18 and state.replay_status == "passed",
            "run_state_incomplete",
        )
        run_dir = PILOT_ROOT / spec.run_id
        run_path = run_dir / "run_record.json"
        _require(sha256_file(run_path) == state.run_record_sha256, "run_record_hash_mismatch")
        record = _read_model(run_path, CascadeRealAgentRunRecord, "run_record_invalid")
        assert isinstance(record, CascadeRealAgentRunRecord)
        _safe(record.model_dump(mode="json"))
        _require(
            record.run_id == spec.run_id and record.scenario_id == spec.scenario_id
            and record.seed == spec.seed and record.condition is spec.condition,
            "run_coordinate_mismatch",
        )
        _require(record.status == "completed" and record.error_code is None, "run_record_incomplete")
        replay = HG1OutcomeReplayValidator.validate(
            runner.scenarios[spec.scenario_id].graph,
            record.exposure_ledger,
            record.outcome_ledger,
            record.application_ledger,
            record.round_contexts,
        )
        _require(
            replay.status is CascadeOutcomeReplayStatus.PASSED
            and replay.cascade_replay.status.value == "passed"
            and replay.application_replay.status.value == "passed",
            "replay_not_passed",
        )
        ledger = RequestLedger(run_dir / "request_ledger.jsonl")
        entries = ledger.entries()
        _safe([item.model_dump(mode="json") for item in entries])
        started = [item for item in entries if item.status is RequestLedgerStatus.STARTED]
        terminal = [item for item in entries if item.status is not RequestLedgerStatus.STARTED]
        by_fingerprint: dict[str, list] = defaultdict(list)
        for item in entries:
            by_fingerprint[item.fingerprint].append(item)
        _require(len(by_fingerprint) == 18, "logical_request_count_mismatch")
        for fingerprint, rows in by_fingerprint.items():
            starts = [item for item in rows if item.status is RequestLedgerStatus.STARTED]
            terminals = [item for item in rows if item.status is not RequestLedgerStatus.STARTED]
            attempts = {item.attempt_count for item in starts}
            _require(attempts == set(range(1, len(starts) + 1)), "attempt_sequence_invalid")
            _require(len(terminals) == len(starts) and {item.attempt_count for item in terminals} == attempts, "orphaned_started_attempt")
            completed = [item for item in terminals if item.status is RequestLedgerStatus.COMPLETED]
            failed = [item for item in terminals if item.status is RequestLedgerStatus.FAILED]
            duplicate_completed += max(0, len(completed) - 1)
            _require(len(completed) == 1 and completed[0].attempt_count == max(attempts), "logical_request_not_finally_completed")
            if failed:
                failed_fingerprints.add(fingerprint)
                _require(max(item.attempt_count for item in failed) < completed[0].attempt_count, "failed_attempt_not_recovered")
            completed_fingerprints.add(fingerprint)
        summary = ledger.summary(request_cap=18, completion_reservation_cap=18432)
        _require(
            summary.unique_logical_request_count == 18
            and summary.transport_attempt_count == state.transport_attempt_count,
            "ledger_state_count_mismatch",
        )
        binding = runner._run_binding(spec, config.required_model_name)
        decisions = load_decisions(run_dir / "behavior_decisions.jsonl", binding)
        checkpoint = _checkpoint(run_dir / "agent_checkpoint.json")
        _require(len(decisions) == len(checkpoint.entries) == 18, "decision_checkpoint_incomplete")
        _require({item.fingerprint for item in decisions} == completed_fingerprints.intersection(by_fingerprint), "decision_fingerprint_mismatch")
        _require({item.fingerprint for item in checkpoint.entries} == completed_fingerprints.intersection(by_fingerprint), "checkpoint_fingerprint_mismatch")
        aggregate_decisions += len(decisions)
        aggregate_checkpoints += len(checkpoint.entries)
        aggregate_records.append(record)
        all_started.extend(started)
        all_terminal.extend(terminal)

    first_started = min(item.completed_at for item in all_started)
    final_terminal = max(item.completed_at for item in all_terminal)
    pre_fix_completed = sum(
        item.status is RequestLedgerStatus.COMPLETED and item.completed_at <= FIX_APPLIED_AT
        for item in all_terminal
    )
    pre_fix_failed = sum(
        item.status is RequestLedgerStatus.FAILED and item.completed_at <= FIX_APPLIED_AT
        for item in all_terminal
    )
    _require(first_started < FIX_APPLIED_AT < final_terminal, "fix_timeline_not_within_batch")
    _require(len(completed_fingerprints) == 288 and len(all_started) == 292, "aggregate_request_count_mismatch")
    _require(len(failed_fingerprints) == 4 and sum(item.status is RequestLedgerStatus.FAILED for item in all_terminal) == 4, "recovery_count_mismatch")
    _require(duplicate_completed == 0 and orphaned == 0, "completed_fingerprint_replayed")
    totals = runner._ledger_totals(PILOT_ROOT)
    _require(totals["logical_request_count"] == receipt.logical_request_count == 288, "receipt_logical_count_mismatch")
    _require(totals["transport_attempt_count"] == receipt.transport_attempt_count == 292, "receipt_transport_count_mismatch")
    _require(totals["actual_prompt_token_count"] == receipt.actual_prompt_token_count == 218995, "receipt_prompt_token_mismatch")
    _require(totals["actual_completion_token_count"] == receipt.actual_completion_token_count == 49981, "receipt_completion_token_mismatch")
    _require(totals["actual_total_token_count"] == receipt.actual_total_token_count == 268976, "receipt_total_token_mismatch")
    expected_ledger_hash = _hash_json([
        (item.run_id, item.exposure_ledger_sha256, item.application_ledger_sha256, item.outcome_ledger_sha256)
        for item in aggregate_records
    ])
    _require(expected_ledger_hash == receipt.ledger_hash, "receipt_run_aggregate_hash_mismatch")

    result = HG22TechnicalAuditReceipt(
        receipt_id="provenance-cascade-hg22-resume-cap-technical-amendment-v1",
        audit_version=AUDIT_VERSION,
        status="passed_development_analysis_eligible",
        analysis_input_status="development_only_evaluator_analysis_allowed",
        study_id=config.study_id,
        config_sha256=expected_hashes["config_sha256"], protocol_sha256=expected_hashes["protocol_sha256"],
        calibration_runner_sha256=execution.runner_sha256,
        controller_sha256=expected_hashes["controller_sha256"], replay_sha256=expected_hashes["replay_sha256"],
        request_ledger_pre_fix_sha256=PRE_FIX_REQUEST_LEDGER_SHA256,
        request_ledger_post_fix_sha256=POST_FIX_REQUEST_LEDGER_SHA256,
        calibration_receipt_sha256=sha256_file(CALIBRATION_RECEIPT_PATH), batch_record_sha256=sha256_file(BATCH_PATH),
        design_approval_sha256=expected_hashes["design_approval_sha256"],
        execution_approval_sha256=expected_hashes["execution_approval_sha256"],
        compatibility_receipt_sha256=expected_hashes["compatibility_receipt_sha256"],
        amendment_receipt_sha256=expected_hashes["amendment_receipt_sha256"],
        first_request_started_at=first_started, implementation_fix_applied_at=FIX_APPLIED_AT,
        final_request_completed_at=final_terminal,
        fix_occurred_after_batch_start=True, fix_occurred_before_batch_completion=True,
        fix_scope=("failed_fingerprint_resume_cap_accounting", "existing_logical_reservation_not_double_counted"),
        prompt_or_request_content_changed=False, fingerprint_algorithm_changed=False,
        parser_or_output_contract_changed=False, scenario_condition_seed_metric_changed=False,
        logical_request_cap_changed=False, completed_fingerprint_replayed=False,
        original_receipt_bound_request_ledger_implementation=False,
        technical_receipt_supplies_missing_implementation_binding=True,
        pre_fix_ledger_snapshot_hash_available=False,
        pre_fix_completed_request_count=pre_fix_completed, pre_fix_failed_attempt_count=pre_fix_failed,
        run_count=16, matched_group_count=4, unique_logical_request_count=288,
        completed_logical_request_count=288, transport_attempt_count=292,
        failed_transport_attempt_count=4, recovered_failed_fingerprint_count=4,
        duplicate_completed_fingerprint_count=0, orphaned_started_attempt_count=0,
        behavior_decision_count=aggregate_decisions, checkpoint_entry_count=aggregate_checkpoints,
        replay_passed_count=16, actual_prompt_token_count=218995,
        actual_completion_token_count=49981, actual_total_token_count=268976,
        limitation_codes=("pre_fix_ledger_file_hash_snapshot_unavailable", "single_seed_eligibility_calibration"),
        engineering_integrity_passed=True, behavior_or_mechanism_conclusion_permitted=False,
        development_only=True, calibration_only=True, not_paper_result=True,
        no_causal_conclusion=True, private_truth_loaded=False, network_used_for_audit=False,
    )
    _safe(result.model_dump(mode="json"))
    return result


def write_audit_receipt(path: str | Path = AUDIT_RECEIPT_PATH) -> HG22TechnicalAuditReceipt:
    receipt = audit_hg22_integrity()
    destination = Path(path)
    if destination.exists():
        existing = HG22TechnicalAuditReceipt.model_validate_json(destination.read_text(encoding="utf-8"))
        if existing != receipt:
            raise HG22IntegrityAuditError("technical_audit_receipt_binding_mismatch")
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(destination)
    return receipt


def safe_summary(receipt: HG22TechnicalAuditReceipt, *, receipt_path: Path | None = None) -> dict[str, object]:
    return {
        "status": receipt.status,
        "analysis_input_status": receipt.analysis_input_status,
        "run_count": receipt.run_count,
        "matched_group_count": receipt.matched_group_count,
        "logical_request_count": receipt.unique_logical_request_count,
        "transport_attempt_count": receipt.transport_attempt_count,
        "recovered_failed_fingerprint_count": receipt.recovered_failed_fingerprint_count,
        "replay_passed_count": receipt.replay_passed_count,
        "fix_occurred_after_batch_start": receipt.fix_occurred_after_batch_start,
        "receipt_sha256": sha256_file(receipt_path) if receipt_path and receipt_path.exists() else None,
        "network": "disabled",
        "private_truth_loaded": False,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the completed H-G.2.2 calibration without network access.")
    parser.add_argument("--write-receipt", action="store_true")
    args = parser.parse_args(argv)
    try:
        receipt = write_audit_receipt() if args.write_receipt else audit_hg22_integrity()
        print(json.dumps(safe_summary(receipt, receipt_path=AUDIT_RECEIPT_PATH if args.write_receipt else None), sort_keys=True))
        return 0
    except HG22IntegrityAuditError as exc:
        print(json.dumps({"status": "blocked", "error_code": exc.code, "network": "disabled", "private_truth_loaded": False}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
