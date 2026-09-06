"""Offline-only technical amendment for the H-G.2.3.2 timeout resume gate."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cascade_hg1_replay import HG1OutcomeReplayValidator
from .cascade_real_agent_runner import CascadeRealAgentRunRecord
from .provenance_cascade_hg2 import sha256_file
from .provenance_cascade_hg21_calibration import HG21BatchRecord
from .request_ledger import RequestLedger, RequestLedgerStatus

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = "results/provenance-cascade-hg232-adoption-identifiability-v1"
DEFAULT_BATCH_RECORD = f"{DEFAULT_ROOT}/calibration_batch_record.json"
DEFAULT_RECEIPT = "outputs/study-locks/provenance_cascade_hg232_timeout_resume_technical_amendment.json"
DEFAULT_APPROVAL = "configs/provenance_cascade/pilot/provenance_cascade_hg232_timeout_resume_approval.toml"
DEFAULT_CONFIG = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg232_calibration.v1.toml"
DEFAULT_EXECUTION_APPROVAL = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg232_execution_approval.toml"
DEFAULT_COMPATIBILITY_RECEIPT = "outputs/study-locks/provenance_cascade_hg232_compatibility_receipt.json"
DEFAULT_PROBE_RECEIPT = "results/provenance-cascade-hg232-protocol-stability-probe-v1/protocol_stability_probe_receipt.json"
RUNNER_PATH = "src/evicon/provenance_cascade_hg232_calibration.py"
OLD_RUNNER_SHA256 = "041f85a0d8889377b6bb63a0237b5d212e93081b9cce7f3ab6cd4ad24f30a06f"
PRE_AMENDMENT_BATCH_SHA256 = "9c7fc4313fad9833c6bb464e7e78f387aeb0458798eec016ab0c8dd86fcb5273"
FAILED_RUN_ID = "hg232-cascade-hg1-true-minority-correction-20261031-provenance_aware_controller"
FAILED_FINGERPRINT = "5f2a4650dad79abe1a2ff28f57092228f892d5f9d09248278eced91c1b1c747f"


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


class HG232ResumeAmendmentError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HG232ResumeAmendmentReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    amendment_id: Literal["provenance-cascade-hg232-timeout-resume-gate-v1"]
    status: Literal["offline_validated_pending_exact_hash_approval"]
    pre_amendment_batch_record_sha256: Literal[PRE_AMENDMENT_BATCH_SHA256]
    config_sha256: str = Field(min_length=64, max_length=64)
    protocol_sha256: str = Field(min_length=64, max_length=64)
    execution_approval_sha256: str = Field(min_length=64, max_length=64)
    compatibility_receipt_sha256: str = Field(min_length=64, max_length=64)
    protocol_stability_probe_receipt_sha256: str = Field(min_length=64, max_length=64)
    old_runner_sha256: Literal[OLD_RUNNER_SHA256]
    new_runner_sha256: str = Field(min_length=64, max_length=64)
    failed_run_id: Literal[FAILED_RUN_ID]
    failed_fingerprint: Literal[FAILED_FINGERPRINT]
    prior_error_code: Literal["timeout"]
    prior_attempt_count: Literal[1]
    completed_run_count: Literal[7]
    failed_run_count: Literal[1]
    planned_run_count: Literal[8]
    failed_run_unique_logical_request_count: Literal[9]
    failed_run_completed_request_count: Literal[8]
    failed_run_failed_request_count: Literal[1]
    failed_run_checkpoint_entry_count: Literal[8]
    failed_run_transport_attempt_count: Literal[9]
    max_additional_timeout_transport_attempts: Literal[1]
    request_cap: Literal[288]
    completion_reservation_cap: Literal[589824]
    prompt_changed: Literal[False]
    schema_changed: Literal[False]
    model_changed: Literal[False]
    seed_condition_scenario_changed: Literal[False]
    metric_or_controller_changed: Literal[False]
    fingerprint_algorithm_changed: Literal[False]
    batch_binding_changed: Literal[False]
    execution_approval_changed: Literal[False]
    completed_runs_replayed: Literal[False]
    parser_recovery_enabled: Literal[False]
    changes_resume_gate_only: Literal[True]
    existing_results_rewritten: Literal[False]
    network: Literal["disabled"]
    provider_constructed: Literal[False]
    private_truth_exposed: Literal[False]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]


class HG232ResumeAmendmentApproval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    approval_id: Literal["provenance-cascade-hg232-timeout-resume-approval-v1"]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    amendment_receipt_sha256: str = Field(min_length=64, max_length=64)
    pre_amendment_batch_record_sha256: Literal[PRE_AMENDMENT_BATCH_SHA256]
    old_runner_sha256: Literal[OLD_RUNNER_SHA256]
    new_runner_sha256: str = Field(min_length=64, max_length=64)
    failed_run_id: Literal[FAILED_RUN_ID]
    failed_fingerprint: Literal[FAILED_FINGERPRINT]
    max_additional_timeout_transport_attempts: Literal[1]
    confirm_resume_gate_only: bool
    confirm_completed_runs_not_replayed: bool
    confirm_batch_binding_unchanged: bool
    confirm_caps_unchanged: bool
    confirm_no_parser_recovery: bool
    network_resume_authorized: bool

    @model_validator(mode="after")
    def accepted(self) -> "HG232ResumeAmendmentApproval":
        confirmations = (
            self.confirm_resume_gate_only,
            self.confirm_completed_runs_not_replayed,
            self.confirm_batch_binding_unchanged,
            self.confirm_caps_unchanged,
            self.confirm_no_parser_recovery,
            self.network_resume_authorized,
        )
        if self.acceptance_status == "accepted" and (not self.accepted_by.strip() or not all(confirmations)):
            raise ValueError("resume amendment approval incomplete")
        return self


def _load_batch() -> HG21BatchRecord:
    path = _path(DEFAULT_BATCH_RECORD)
    if sha256_file(path) != PRE_AMENDMENT_BATCH_SHA256:
        raise HG232ResumeAmendmentError("resume_pre_amendment_batch_hash_mismatch")
    try:
        return HG21BatchRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HG232ResumeAmendmentError("resume_batch_record_invalid") from exc


def audit_failed_state(*, expected_new_runner_sha256: str | None = None) -> dict[str, object]:
    batch = _load_batch()
    states = {item.run_id: item for item in batch.runs}
    failed = states.get(FAILED_RUN_ID)
    if (
        batch.status.value != "failed"
        or batch.failure_code != "timeout"
        or failed is None
        or failed.status.value != "failed"
        or failed.error_code != "timeout"
    ):
        raise HG232ResumeAmendmentError("resume_failed_state_mismatch")
    completed = [item for item in batch.runs if item.status.value == "completed"]
    planned = [item for item in batch.runs if item.status.value == "planned"]
    if len(completed) != 7 or len(planned) != 8:
        raise HG232ResumeAmendmentError("resume_run_counts_mismatch")
    root = _path(DEFAULT_ROOT)
    for state in completed:
        record_path = root / state.run_id / "run_record.json"
        try:
            record = CascadeRealAgentRunRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HG232ResumeAmendmentError("resume_completed_run_record_invalid") from exc
        if record.run_id != state.run_id or record.replay is None or record.replay.status.value != "passed":
            raise HG232ResumeAmendmentError("resume_completed_run_replay_not_passed")
    run_dir = root / FAILED_RUN_ID
    ledger = RequestLedger(run_dir / "request_ledger.jsonl")
    entries = ledger.entries()
    summary = ledger.summary(request_cap=18, completion_reservation_cap=36864)
    terminal_failed = [item for item in entries if item.status is RequestLedgerStatus.FAILED]
    if len(terminal_failed) != 1:
        raise HG232ResumeAmendmentError("resume_failed_attempt_count_mismatch")
    failure = terminal_failed[0]
    if (
        failure.fingerprint != FAILED_FINGERPRINT
        or failure.attempt_count != 1
        or failure.error_code != "timeout"
        or summary.unique_logical_request_count != 9
        or summary.transport_attempt_count != 9
        or summary.completed_count != 8
        or summary.failed_count != 1
    ):
        raise HG232ResumeAmendmentError("resume_failed_fingerprint_state_mismatch")
    checkpoint_path = run_dir / "agent_checkpoint.json"
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint_count = len(checkpoint["entries"])
    except Exception as exc:
        raise HG232ResumeAmendmentError("resume_checkpoint_invalid") from exc
    if checkpoint_count != 8:
        raise HG232ResumeAmendmentError("resume_checkpoint_count_mismatch")
    current_runner_sha = sha256_file(RUNNER_PATH)
    if expected_new_runner_sha256 is not None and current_runner_sha != expected_new_runner_sha256:
        raise HG232ResumeAmendmentError("resume_runner_hash_mismatch")
    return {
        "batch": batch,
        "current_runner_sha256": current_runner_sha,
        "checkpoint_entry_count": checkpoint_count,
        "summary": summary,
    }


def build_receipt() -> HG232ResumeAmendmentReceipt:
    audit = audit_failed_state()
    batch = audit["batch"]
    return HG232ResumeAmendmentReceipt(
        amendment_id="provenance-cascade-hg232-timeout-resume-gate-v1",
        status="offline_validated_pending_exact_hash_approval",
        pre_amendment_batch_record_sha256=PRE_AMENDMENT_BATCH_SHA256,
        config_sha256=batch.config_sha256,
        protocol_sha256=batch.protocol_sha256,
        execution_approval_sha256=sha256_file(DEFAULT_EXECUTION_APPROVAL),
        compatibility_receipt_sha256=sha256_file(DEFAULT_COMPATIBILITY_RECEIPT),
        protocol_stability_probe_receipt_sha256=sha256_file(DEFAULT_PROBE_RECEIPT),
        old_runner_sha256=OLD_RUNNER_SHA256,
        new_runner_sha256=audit["current_runner_sha256"],
        failed_run_id=FAILED_RUN_ID,
        failed_fingerprint=FAILED_FINGERPRINT,
        prior_error_code="timeout",
        prior_attempt_count=1,
        completed_run_count=7,
        failed_run_count=1,
        planned_run_count=8,
        failed_run_unique_logical_request_count=9,
        failed_run_completed_request_count=8,
        failed_run_failed_request_count=1,
        failed_run_checkpoint_entry_count=8,
        failed_run_transport_attempt_count=9,
        max_additional_timeout_transport_attempts=1,
        request_cap=288,
        completion_reservation_cap=589824,
        prompt_changed=False,
        schema_changed=False,
        model_changed=False,
        seed_condition_scenario_changed=False,
        metric_or_controller_changed=False,
        fingerprint_algorithm_changed=False,
        batch_binding_changed=False,
        execution_approval_changed=False,
        completed_runs_replayed=False,
        parser_recovery_enabled=False,
        changes_resume_gate_only=True,
        existing_results_rewritten=False,
        network="disabled",
        provider_constructed=False,
        private_truth_exposed=False,
        development_only=True,
        calibration_only=True,
        not_paper_result=True,
        no_causal_conclusion=True,
    )


def write_receipt() -> tuple[HG232ResumeAmendmentReceipt, str]:
    receipt = build_receipt()
    target = _path(DEFAULT_RECEIPT)
    content = receipt.model_dump_json(indent=2) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != content:
        raise HG232ResumeAmendmentError("resume_amendment_receipt_conflict")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return receipt, sha256_file(target)


def validate_resume_amendment() -> tuple[HG232ResumeAmendmentReceipt, HG232ResumeAmendmentApproval]:
    receipt_path = _path(DEFAULT_RECEIPT)
    approval_path = _path(DEFAULT_APPROVAL)
    if not receipt_path.is_file():
        raise HG232ResumeAmendmentError("timeout_resume_amendment_required")
    try:
        receipt = HG232ResumeAmendmentReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
        approval = HG232ResumeAmendmentApproval.model_validate(tomllib.loads(approval_path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise HG232ResumeAmendmentError("timeout_resume_amendment_invalid") from exc
    if (
        approval.acceptance_status != "accepted"
        or not approval.network_resume_authorized
        or approval.amendment_receipt_sha256 != sha256_file(receipt_path)
        or approval.pre_amendment_batch_record_sha256 != receipt.pre_amendment_batch_record_sha256
        or approval.old_runner_sha256 != receipt.old_runner_sha256
        or approval.new_runner_sha256 != receipt.new_runner_sha256
        or approval.failed_run_id != receipt.failed_run_id
        or approval.failed_fingerprint != receipt.failed_fingerprint
    ):
        raise HG232ResumeAmendmentError("timeout_resume_amendment_approval_required")
    audit_failed_state(expected_new_runner_sha256=receipt.new_runner_sha256)
    return receipt, approval


__all__ = [
    "DEFAULT_APPROVAL", "DEFAULT_RECEIPT", "FAILED_FINGERPRINT", "FAILED_RUN_ID",
    "HG232ResumeAmendmentApproval", "HG232ResumeAmendmentError", "HG232ResumeAmendmentReceipt",
    "audit_failed_state", "build_receipt", "validate_resume_amendment", "write_receipt",
]
