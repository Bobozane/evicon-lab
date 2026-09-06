"""Offline audit and isolated timeout-only resume gate for H-G.2.3.2."""
from __future__ import annotations

import argparse
import json
import os
import tomllib
from pathlib import Path
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .provenance_cascade_hg2 import sha256_file
from .provenance_cascade_hg21_calibration import CalibrationStatus, HG21BatchRecord, HG21ExecutionSummary, _atomic
from .provenance_cascade_hg232_calibration import execute_real as execute_historical_runner
from .request_ledger import RequestLedger, RequestLedgerStatus

_ROOT = Path(__file__).resolve().parents[2]
BATCH_ROOT = "results/provenance-cascade-hg232-adoption-identifiability-v1"
BATCH_RECORD = f"{BATCH_ROOT}/calibration_batch_record.json"
FAILED_RUN_ID = "hg232-cascade-hg1-true-minority-correction-20261031-provenance_aware_controller"
FAILED_RUN_ROOT = f"{BATCH_ROOT}/{FAILED_RUN_ID}"
FAILED_LEDGER = f"{FAILED_RUN_ROOT}/request_ledger.jsonl"
FAILED_CHECKPOINT = f"{FAILED_RUN_ROOT}/agent_checkpoint.json"
FAILED_DECISIONS = f"{FAILED_RUN_ROOT}/behavior_decisions.jsonl"
FAILED_RESPONSE_AUDIT = f"{FAILED_RUN_ROOT}/response_audit.jsonl"
HISTORICAL_RUNNER = "src/evicon/provenance_cascade_hg232_calibration.py"
HISTORICAL_PROBE_RUNNER = "src/evicon/provenance_cascade_hg232_stability_probe.py"
CONFIG_PATH = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg232_calibration.v1.toml"
PROTOCOL_PATH = "src/evicon/cascade_agent_protocol_hg232.py"
EXECUTION_APPROVAL = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg232_execution_approval.toml"
PROBE_RECEIPT = "results/provenance-cascade-hg232-protocol-stability-probe-v1/protocol_stability_probe_receipt.json"
COMPATIBILITY_RECEIPT = "outputs/study-locks/provenance_cascade_hg232_compatibility_receipt.json"
AMENDMENT_DOC = "docs/provenance-cascade-hg232-timeout-resume-amendment.md"
AMENDMENT_RECEIPT = "outputs/study-locks/provenance_cascade_hg232_timeout_resume_amendment.json"
RESUME_APPROVAL = "configs/provenance_cascade/pilot/provenance_cascade_hg232_timeout_resume_approval.toml"
TIMEOUT_FINGERPRINT = "5f2a4650dad79abe1a2ff28f57092228f892d5f9d09248278eced91c1b1c747f"
EXPECTED_HISTORICAL_RUNNER_SHA256 = "041f85a0d8889377b6bb63a0237b5d212e93081b9cce7f3ab6cd4ad24f30a06f"
EXPECTED_HISTORICAL_PROBE_RUNNER_SHA256 = "12e602a98d892b25a3fe6068dc5f9c1c5174de24024a8fd74be68c6843bd98b1"
EXPECTED_BATCH_SHA256 = "9c7fc4313fad9833c6bb464e7e78f387aeb0458798eec016ab0c8dd86fcb5273"
EXPECTED_FAILED_LEDGER_SHA256 = "b79314ff70cddb350d14def62971a8abfce8eb68fae1c2bcf66b390a27db15bc"
EXPECTED_FAILED_CHECKPOINT_SHA256 = "04ae613b425d1ee349ef2aa249afefbd2e30bf8b364bcef4718249810eea6e91"
EXPECTED_FAILED_DECISIONS_SHA256 = "4ce5e07ae7e9eb843cc3ad921d94430df7681267911f0b7778ff4033fd21fda0"
EXPECTED_FAILED_RESPONSE_AUDIT_SHA256 = "a66999aa78efb7f6a8ae7f30cbccba98440840b0b9e410c47dcca2e18001bf10"


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


class HG232TimeoutResumeError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HG232TimeoutResumeAmendment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    amendment_id: Literal["provenance-cascade-hg232-timeout-resume.v1"]
    status: Literal["offline_validated_pending_network_resume_authorization"]
    batch_record_sha256: str
    amendment_document_sha256: str
    config_sha256: str
    protocol_sha256: str
    execution_approval_sha256: str
    compatibility_receipt_sha256: str
    probe_receipt_sha256: str
    historical_runner_sha256: str
    historical_probe_runner_sha256: str
    failed_ledger_sha256: str
    failed_checkpoint_sha256: str
    failed_decisions_sha256: str
    failed_response_audit_sha256: str
    failed_run_id: Literal[FAILED_RUN_ID]
    failed_fingerprint: Literal[TIMEOUT_FINGERPRINT]
    failed_agent_id: Literal["network-agent-03"]
    failed_round_id: Literal[1]
    failed_error_code: Literal["timeout"]
    completed_run_count: Literal[7]
    failed_run_count: Literal[1]
    planned_run_count: Literal[8]
    observed_unique_logical_request_count: Literal[135]
    observed_completed_request_count: Literal[134]
    observed_failed_request_count: Literal[1]
    observed_transport_attempt_count: Literal[135]
    failed_run_completed_fingerprint_count: Literal[8]
    failed_run_failed_fingerprint_count: Literal[1]
    logical_request_cap: Literal[288]
    completion_reservation_cap: Literal[589824]
    maximum_additional_timeout_attempts: Literal[1]
    completed_fingerprints_replayed: Literal[False]
    parser_recovery_enabled: Literal[False]
    semantic_recovery_enabled: Literal[False]
    prompt_changed: Literal[False]
    schema_changed: Literal[False]
    model_changed: Literal[False]
    seed_changed: Literal[False]
    condition_changed: Literal[False]
    metrics_changed: Literal[False]
    append_only_ledger_required: Literal[True]
    resume_invocation_single_use: Literal[True]
    network: Literal["disabled"]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    private_truth_exposed: Literal[False]


class HG232TimeoutResumeApproval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    approval_id: Literal["provenance-cascade-hg232-timeout-resume-execution-v1"]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    amendment_receipt_sha256: str
    resume_runner_sha256: str
    historical_runner_sha256: Literal[EXPECTED_HISTORICAL_RUNNER_SHA256]
    historical_probe_runner_sha256: Literal[EXPECTED_HISTORICAL_PROBE_RUNNER_SHA256]
    failed_fingerprint: Literal[TIMEOUT_FINGERPRINT]
    logical_request_cap: Literal[288]
    completion_reservation_cap: Literal[589824]
    maximum_additional_timeout_attempts: Literal[1]
    confirm_timeout_only: bool
    confirm_completed_fingerprints_not_replayed: bool
    confirm_no_parser_or_semantic_recovery: bool
    confirm_append_only_ledger: bool
    confirm_single_resume_invocation: bool
    network_resume_authorized: bool

    @model_validator(mode="after")
    def accepted(self) -> "HG232TimeoutResumeApproval":
        confirmations = (
            self.confirm_timeout_only,
            self.confirm_completed_fingerprints_not_replayed,
            self.confirm_no_parser_or_semantic_recovery,
            self.confirm_append_only_ledger,
            self.confirm_single_resume_invocation,
            self.network_resume_authorized,
        )
        if self.acceptance_status == "accepted" and (not self.accepted_by.strip() or not all(confirmations)):
            raise ValueError("timeout resume approval incomplete")
        return self


def _batch() -> HG21BatchRecord:
    try:
        return HG21BatchRecord.model_validate_json(_path(BATCH_RECORD).read_text(encoding="utf-8"))
    except Exception as exc:
        raise HG232TimeoutResumeError("timeout_resume_batch_invalid") from exc


def _validate_current_state(*, exact_hashes: bool = True) -> tuple[HG21BatchRecord, object]:
    required = {
        HISTORICAL_RUNNER: EXPECTED_HISTORICAL_RUNNER_SHA256,
        HISTORICAL_PROBE_RUNNER: EXPECTED_HISTORICAL_PROBE_RUNNER_SHA256,
        BATCH_RECORD: EXPECTED_BATCH_SHA256,
        FAILED_LEDGER: EXPECTED_FAILED_LEDGER_SHA256,
        FAILED_CHECKPOINT: EXPECTED_FAILED_CHECKPOINT_SHA256,
        FAILED_DECISIONS: EXPECTED_FAILED_DECISIONS_SHA256,
        FAILED_RESPONSE_AUDIT: EXPECTED_FAILED_RESPONSE_AUDIT_SHA256,
    }
    if exact_hashes and any(not _path(path).is_file() or sha256_file(_path(path)) != digest for path, digest in required.items()):
        raise HG232TimeoutResumeError("timeout_resume_input_hash_mismatch")
    if sha256_file(HISTORICAL_RUNNER) != EXPECTED_HISTORICAL_RUNNER_SHA256 or sha256_file(HISTORICAL_PROBE_RUNNER) != EXPECTED_HISTORICAL_PROBE_RUNNER_SHA256:
        raise HG232TimeoutResumeError("historical_runner_hash_mismatch")
    batch = _batch()
    states = {item.run_id: item for item in batch.runs}
    failed = states.get(FAILED_RUN_ID)
    if batch.status is not CalibrationStatus.FAILED or batch.failure_code != "timeout":
        raise HG232TimeoutResumeError("timeout_resume_batch_failure_mismatch")
    if failed is None or failed.status is not CalibrationStatus.FAILED or failed.error_code != "timeout":
        raise HG232TimeoutResumeError("timeout_resume_failed_run_mismatch")
    if sum(item.status is CalibrationStatus.COMPLETED for item in batch.runs) != 7:
        raise HG232TimeoutResumeError("timeout_resume_completed_run_count_mismatch")
    if sum(item.status is CalibrationStatus.PLANNED for item in batch.runs) != 8:
        raise HG232TimeoutResumeError("timeout_resume_planned_run_count_mismatch")
    ledger = RequestLedger(_path(FAILED_LEDGER))
    entries = ledger.entries()
    terminal = [item for item in entries if item.status in {RequestLedgerStatus.COMPLETED, RequestLedgerStatus.FAILED}]
    latest = {}
    for item in terminal:
        if item.fingerprint not in latest or item.attempt_count > latest[item.fingerprint].attempt_count:
            latest[item.fingerprint] = item
    failed_entries = [item for item in latest.values() if item.status is RequestLedgerStatus.FAILED]
    if len(failed_entries) != 1:
        raise HG232TimeoutResumeError("timeout_resume_failed_fingerprint_count_mismatch")
    item = failed_entries[0]
    if (
        item.fingerprint != TIMEOUT_FINGERPRINT
        or item.error_code != "timeout"
        or item.agent_id != "network-agent-03"
        or item.round_id != 1
        or item.attempt_count != 1
        or item.recovery_kind is not None
    ):
        raise HG232TimeoutResumeError("timeout_resume_failed_fingerprint_mismatch")
    if sum(value.status is RequestLedgerStatus.COMPLETED for value in latest.values()) != 8:
        raise HG232TimeoutResumeError("timeout_resume_checkpoint_coverage_mismatch")
    return batch, item


def prepare_amendment() -> tuple[HG232TimeoutResumeAmendment, str]:
    batch, _ = _validate_current_state(exact_hashes=True)
    receipt = HG232TimeoutResumeAmendment(
        amendment_id="provenance-cascade-hg232-timeout-resume.v1",
        status="offline_validated_pending_network_resume_authorization",
        batch_record_sha256=sha256_file(BATCH_RECORD),
        amendment_document_sha256=sha256_file(AMENDMENT_DOC),
        config_sha256=sha256_file(CONFIG_PATH),
        protocol_sha256=sha256_file(PROTOCOL_PATH),
        execution_approval_sha256=sha256_file(EXECUTION_APPROVAL),
        compatibility_receipt_sha256=sha256_file(COMPATIBILITY_RECEIPT),
        probe_receipt_sha256=sha256_file(PROBE_RECEIPT),
        historical_runner_sha256=sha256_file(HISTORICAL_RUNNER),
        historical_probe_runner_sha256=sha256_file(HISTORICAL_PROBE_RUNNER),
        failed_ledger_sha256=sha256_file(FAILED_LEDGER),
        failed_checkpoint_sha256=sha256_file(FAILED_CHECKPOINT),
        failed_decisions_sha256=sha256_file(FAILED_DECISIONS),
        failed_response_audit_sha256=sha256_file(FAILED_RESPONSE_AUDIT),
        failed_run_id=FAILED_RUN_ID,
        failed_fingerprint=TIMEOUT_FINGERPRINT,
        failed_agent_id="network-agent-03",
        failed_round_id=1,
        failed_error_code="timeout",
        completed_run_count=7,
        failed_run_count=1,
        planned_run_count=8,
        observed_unique_logical_request_count=135,
        observed_completed_request_count=134,
        observed_failed_request_count=1,
        observed_transport_attempt_count=135,
        failed_run_completed_fingerprint_count=8,
        failed_run_failed_fingerprint_count=1,
        logical_request_cap=288,
        completion_reservation_cap=589824,
        maximum_additional_timeout_attempts=1,
        completed_fingerprints_replayed=False,
        parser_recovery_enabled=False,
        semantic_recovery_enabled=False,
        prompt_changed=False,
        schema_changed=False,
        model_changed=False,
        seed_changed=False,
        condition_changed=False,
        metrics_changed=False,
        append_only_ledger_required=True,
        resume_invocation_single_use=True,
        network="disabled",
        development_only=True,
        calibration_only=True,
        not_paper_result=True,
        no_causal_conclusion=True,
        private_truth_exposed=False,
    )
    target = _path(AMENDMENT_RECEIPT)
    serialized = receipt.model_dump_json(indent=2)
    if target.exists():
        if target.read_text(encoding="utf-8") != serialized:
            raise HG232TimeoutResumeError("timeout_resume_amendment_receipt_conflict")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(serialized, encoding="utf-8")
    return receipt, sha256_file(target)


def validate_amendment() -> HG232TimeoutResumeAmendment:
    _validate_current_state(exact_hashes=True)
    try:
        receipt = HG232TimeoutResumeAmendment.model_validate_json(_path(AMENDMENT_RECEIPT).read_text(encoding="utf-8"))
    except Exception as exc:
        raise HG232TimeoutResumeError("timeout_resume_amendment_receipt_invalid") from exc
    expected = {
        "batch_record_sha256": sha256_file(BATCH_RECORD),
        "amendment_document_sha256": sha256_file(AMENDMENT_DOC),
        "config_sha256": sha256_file(CONFIG_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "execution_approval_sha256": sha256_file(EXECUTION_APPROVAL),
        "compatibility_receipt_sha256": sha256_file(COMPATIBILITY_RECEIPT),
        "probe_receipt_sha256": sha256_file(PROBE_RECEIPT),
        "historical_runner_sha256": sha256_file(HISTORICAL_RUNNER),
        "historical_probe_runner_sha256": sha256_file(HISTORICAL_PROBE_RUNNER),
        "failed_ledger_sha256": sha256_file(FAILED_LEDGER),
        "failed_checkpoint_sha256": sha256_file(FAILED_CHECKPOINT),
        "failed_decisions_sha256": sha256_file(FAILED_DECISIONS),
        "failed_response_audit_sha256": sha256_file(FAILED_RESPONSE_AUDIT),
    }
    if any(getattr(receipt, key) != value for key, value in expected.items()):
        raise HG232TimeoutResumeError("timeout_resume_amendment_binding_mismatch")
    return receipt


def _approval() -> HG232TimeoutResumeApproval:
    try:
        return HG232TimeoutResumeApproval.model_validate(tomllib.loads(_path(RESUME_APPROVAL).read_text(encoding="utf-8")))
    except Exception as exc:
        raise HG232TimeoutResumeError("timeout_resume_approval_invalid") from exc


def preflight() -> dict[str, object]:
    reasons: list[str] = []
    try:
        amendment = validate_amendment()
        approval = _approval()
        expected = {
            "amendment_receipt_sha256": sha256_file(AMENDMENT_RECEIPT),
            "resume_runner_sha256": sha256_file(__file__),
            "historical_runner_sha256": amendment.historical_runner_sha256,
            "historical_probe_runner_sha256": amendment.historical_probe_runner_sha256,
            "failed_fingerprint": amendment.failed_fingerprint,
        }
        if any(getattr(approval, key) != value for key, value in expected.items()):
            reasons.append("timeout_resume_approval_binding_mismatch")
        if approval.acceptance_status != "accepted" or not approval.network_resume_authorized:
            reasons.append("timeout_resume_network_authorization_required")
        return {
            "status": "ready_for_timeout_resume" if not reasons else "blocked",
            "ready_for_timeout_resume": not reasons,
            "blocking_reasons": sorted(set(reasons)),
            "failed_run_id": FAILED_RUN_ID,
            "failed_fingerprint": TIMEOUT_FINGERPRINT,
            "completed_run_count": 7,
            "observed_unique_logical_request_count": 135,
            "observed_transport_attempt_count": 135,
            "maximum_additional_timeout_attempts": 1,
            "logical_request_cap": 288,
            "completion_reservation_cap": 589824,
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "private_truth_exposed": False,
            "not_paper_result": True,
            "no_causal_conclusion": True,
        }
    except Exception as exc:
        return {
            "status": "blocked",
            "ready_for_timeout_resume": False,
            "blocking_reasons": [getattr(exc, "code", "timeout_resume_preflight_failed")],
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
        }


def execute_resume(
    *,
    allow_network: bool = False,
    confirm_run: bool = False,
    confirm_request_cap: int | None = None,
    confirm_completion_reservation_cap: int | None = None,
    confirm_timeout_fingerprint: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> HG21ExecutionSummary:
    checks = (
        (allow_network, "allow_network_required"),
        (confirm_run, "confirm_run_required"),
        (confirm_request_cap == 288, "confirm_request_cap_must_equal_288"),
        (confirm_completion_reservation_cap == 589824, "confirm_completion_reservation_cap_must_equal_589824"),
        (confirm_timeout_fingerprint == TIMEOUT_FINGERPRINT, "confirm_timeout_fingerprint_mismatch"),
    )
    for valid, code in checks:
        if not valid:
            return HG21ExecutionSummary(status="blocked", error_code=code, safety={"private_truth_exposed": False})
    gate = preflight()
    if not gate["ready_for_timeout_resume"]:
        return HG21ExecutionSummary(status="blocked", error_code=gate["blocking_reasons"][0], safety={"private_truth_exposed": False})
    # The historical runner keeps the exact prompt/schema/binding and performs
    # checkpoint-based skip of completed fingerprints. This wrapper adds only
    # the immutable timeout-only authorization gate above.
    return execute_historical_runner(
        allow_network=True,
        confirm_run=True,
        confirm_request_cap=288,
        confirm_completion_reservation_cap=589824,
        resume=True,
        environment=os.environ if environment is None else environment,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="H-G.2.3.2 timeout-only resume gate")
    parser.add_argument("--mode", choices=("prepare", "preflight", "resume"), default="preflight")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--confirm-request-cap", type=int)
    parser.add_argument("--confirm-completion-reservation-cap", type=int)
    parser.add_argument("--confirm-timeout-fingerprint")
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        receipt, digest = prepare_amendment()
        payload = {"status": receipt.status, "amendment_receipt_sha256": digest, "network": "disabled"}
    elif args.mode == "preflight":
        payload = preflight()
    else:
        payload = execute_resume(
            allow_network=args.allow_network,
            confirm_run=args.confirm_run,
            confirm_request_cap=args.confirm_request_cap,
            confirm_completion_reservation_cap=args.confirm_completion_reservation_cap,
            confirm_timeout_fingerprint=args.confirm_timeout_fingerprint,
        ).model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if payload.get("status") in {"offline_validated_pending_network_resume_authorization", "ready_for_timeout_resume", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
