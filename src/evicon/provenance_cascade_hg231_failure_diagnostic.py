"""Read-only, content-free diagnosis of the H-G.2.3.1 parser failure."""
from __future__ import annotations

import argparse
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from .cascade_real_agent_runner import CascadeAgentCheckpoint, CascadeRealAgentRunRecord
from .provenance_cascade_hg21_calibration import CalibrationStatus, HG21BatchRecord, _path
from .provenance_cascade_hg231 import (
    DEFAULT_AMENDMENT_RECEIPT,
    DEFAULT_APPROVAL,
    DEFAULT_COMPATIBILITY_RECEIPT,
    DEFAULT_CONFIG,
    load_config,
)
from .provenance_cascade_hg231_calibration import DEFAULT_EXECUTION_APPROVAL
from .provenance_cascade_hg2 import sha256_file
from .request_ledger import RequestLedger, RequestLedgerStatus

DIAGNOSTIC_VERSION = "provenance_cascade_hg231_parser_diagnostic.v1"
DEFAULT_OUTPUT = "outputs/study-locks/provenance_cascade_hg231_parser_failure_diagnostic.json"
FINISH_REASON_UNAVAILABLE = "unknown_not_persisted"
HTTP_STATUS_CLASS_UNAVAILABLE = "unknown_not_persisted"


class HG231FailureAssessment(str, Enum):
    TRUNCATION_MORE_CONSISTENT = "truncation_more_consistent"
    SPORADIC_MALFORMED_MORE_CONSISTENT = "sporadic_malformed_more_consistent"
    INDETERMINATE = "indeterminate"


class HG231FailureDiagnosticError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HG231ParserFailureDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    diagnostic_version: str = DIAGNOSTIC_VERSION
    status: str = "diagnosed_offline"
    study_id: str
    run_id: str
    scenario_id: str
    condition: str
    agent_id: str
    round_id: int = Field(ge=0)
    fingerprint: str = Field(min_length=64, max_length=64)
    parser_error_category: str
    ledger_terminal_status: str
    transport_completed: bool
    http_status_class: str
    http_status_available: bool
    finish_reason: str
    finish_reason_available: bool
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    configured_max_tokens: int = Field(gt=0)
    completion_token_limit_reached: bool | None
    request_attempt_count: int = Field(ge=1)
    response_format_mode: str
    response_schema_name: str
    assessment: HG231FailureAssessment
    reason_codes: tuple[str, ...]
    completed_run_count: int = Field(ge=0)
    failed_run_count: int = Field(ge=0)
    planned_run_count: int = Field(ge=0)
    unique_logical_request_count: int = Field(ge=0)
    transport_attempt_count: int = Field(ge=0)
    ledger_file_count: int = Field(ge=0)
    append_only_counts_reconstructed: bool
    checkpoint_entry_count: int = Field(ge=0)
    failed_fingerprint_checkpointed: bool
    completed_run_replay_passed_count: int = Field(ge=0)
    config_sha256: str
    protocol_sha256: str
    amendment_receipt_sha256: str
    design_approval_sha256: str
    compatibility_receipt_sha256: str
    execution_approval_sha256: str
    batch_record_sha256: str
    failed_request_ledger_sha256: str
    failed_checkpoint_sha256: str
    input_run_record_hashes: dict[str, str]
    recommended_path: str
    narrow_recovery_amendment_drafted: bool
    current_batch_must_not_resume: bool
    original_artifacts_preserved: bool = True
    network_used_for_diagnostic: bool = False
    provider_constructed: bool = False
    automatic_retry_performed: bool = False
    response_repair_performed: bool = False
    private_truth_exposed: bool = False
    development_only: bool = True
    calibration_only: bool = True
    not_paper_result: bool = True
    no_causal_conclusion: bool = True


def classify_failure(
    *, parser_error: str, finish_reason: str | None,
    completion_tokens: int | None, configured_max_tokens: int,
) -> tuple[HG231FailureAssessment, tuple[str, ...]]:
    at_limit = completion_tokens is not None and completion_tokens == configured_max_tokens
    reasons = [f"parser_{parser_error}"]
    if at_limit:
        reasons.append("completion_tokens_equal_configured_max")
    if finish_reason == "length":
        reasons.append("finish_reason_length")
    elif finish_reason == "stop":
        reasons.append("finish_reason_non_length")
    else:
        reasons.append("finish_reason_unavailable")
    if parser_error == "malformed_json" and (at_limit or finish_reason == "length"):
        return HG231FailureAssessment.TRUNCATION_MORE_CONSISTENT, tuple(reasons)
    if parser_error == "malformed_json" and finish_reason == "stop" and not at_limit:
        return HG231FailureAssessment.SPORADIC_MALFORMED_MORE_CONSISTENT, tuple(reasons)
    return HG231FailureAssessment.INDETERMINATE, tuple(reasons)


def diagnose() -> HG231ParserFailureDiagnostic:
    config, config_path = load_config(DEFAULT_CONFIG)
    root = _path(config.output_root)
    batch_path = root / "calibration_batch_record.json"
    if not batch_path.is_file():
        raise HG231FailureDiagnosticError("batch_record_missing")
    try:
        batch = HG21BatchRecord.model_validate_json(batch_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HG231FailureDiagnosticError("batch_record_invalid") from exc
    bindings = {
        "config_sha256": sha256_file(config_path),
        "protocol_sha256": config.protocol_sha256,
        "amendment_receipt_sha256": sha256_file(DEFAULT_AMENDMENT_RECEIPT),
        "design_approval_sha256": sha256_file(DEFAULT_APPROVAL),
        "compatibility_receipt_sha256": sha256_file(DEFAULT_COMPATIBILITY_RECEIPT),
        "execution_approval_sha256": sha256_file(DEFAULT_EXECUTION_APPROVAL),
    }
    if any(getattr(batch, key) != digest for key, digest in bindings.items()):
        raise HG231FailureDiagnosticError("batch_binding_mismatch")
    failed = [item for item in batch.runs if item.status is CalibrationStatus.FAILED]
    completed = [item for item in batch.runs if item.status is CalibrationStatus.COMPLETED]
    planned = [item for item in batch.runs if item.status is CalibrationStatus.PLANNED]
    if batch.status is not CalibrationStatus.FAILED or batch.failure_code != "malformed_json" or len(failed) != 1:
        raise HG231FailureDiagnosticError("expected_parser_failure_not_found")
    run_hashes: dict[str, str] = {}
    replay_passed = 0
    for state in completed:
        record_path = root / state.run_id / "run_record.json"
        if not record_path.is_file() or sha256_file(record_path) != state.run_record_sha256:
            raise HG231FailureDiagnosticError("completed_run_hash_mismatch")
        record = CascadeRealAgentRunRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
        if record.replay is None or record.replay.status.value != "passed":
            raise HG231FailureDiagnosticError("completed_run_replay_failed")
        replay_passed += 1
        run_hashes[state.run_id] = sha256_file(record_path)
    summaries = []
    ledger_paths: list[Path] = []
    for state in batch.runs:
        ledger_path = root / state.run_id / "request_ledger.jsonl"
        if ledger_path.is_file():
            ledger_paths.append(ledger_path)
            summaries.append(RequestLedger(ledger_path).summary(
                request_cap=18, completion_reservation_cap=18 * config.agent_max_tokens,
            ))
    unique_count = sum(item.unique_logical_request_count for item in summaries)
    attempt_count = sum(item.transport_attempt_count for item in summaries)
    if (len(completed), unique_count, attempt_count) != (5, 106, 107):
        raise HG231FailureDiagnosticError("append_only_count_reconstruction_mismatch")
    failed_state = failed[0]
    failed_dir = root / failed_state.run_id
    failed_ledger_path = failed_dir / "request_ledger.jsonl"
    checkpoint_path = failed_dir / "agent_checkpoint.json"
    if not checkpoint_path.is_file():
        raise HG231FailureDiagnosticError("failed_checkpoint_missing")
    checkpoint = CascadeAgentCheckpoint.model_validate_json(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint_fingerprints = {item.fingerprint for item in checkpoint.entries}
    terminal = {}
    for item in RequestLedger(failed_ledger_path).entries():
        terminal[item.fingerprint] = item
    candidates = [
        item for fingerprint, item in terminal.items()
        if item.status is RequestLedgerStatus.COMPLETED and fingerprint not in checkpoint_fingerprints
    ]
    if len(candidates) != 1:
        raise HG231FailureDiagnosticError("parser_failure_fingerprint_not_unique")
    item = candidates[0]
    assessment, reasons = classify_failure(
        parser_error=failed_state.error_code or batch.failure_code or "unknown",
        finish_reason=None,
        completion_tokens=item.completion_tokens,
        configured_max_tokens=config.agent_max_tokens,
    )
    at_limit = None if item.completion_tokens is None else item.completion_tokens == config.agent_max_tokens
    recommendation = (
        "new_version_new_namespace_full_rerun"
        if assessment is not HG231FailureAssessment.SPORADIC_MALFORMED_MORE_CONSISTENT
        else "single_fingerprint_recovery_requires_new_amendment"
    )
    return HG231ParserFailureDiagnostic(
        study_id=batch.study_id, run_id=failed_state.run_id,
        scenario_id=failed_state.scenario_id, condition=failed_state.condition,
        agent_id=item.agent_id or "unknown", round_id=item.round_id or 0,
        fingerprint=item.fingerprint, parser_error_category=failed_state.error_code or "malformed_json",
        ledger_terminal_status=item.status.value, transport_completed=True,
        http_status_class=HTTP_STATUS_CLASS_UNAVAILABLE, http_status_available=False,
        finish_reason=FINISH_REASON_UNAVAILABLE, finish_reason_available=False,
        prompt_tokens=item.prompt_tokens, completion_tokens=item.completion_tokens,
        total_tokens=item.total_tokens, configured_max_tokens=config.agent_max_tokens,
        completion_token_limit_reached=at_limit, request_attempt_count=item.attempt_count,
        response_format_mode=config.response_format, response_schema_name=config.response_schema_name,
        assessment=assessment, reason_codes=reasons,
        completed_run_count=len(completed), failed_run_count=len(failed), planned_run_count=len(planned),
        unique_logical_request_count=unique_count, transport_attempt_count=attempt_count,
        ledger_file_count=len(ledger_paths), append_only_counts_reconstructed=True,
        checkpoint_entry_count=len(checkpoint.entries), failed_fingerprint_checkpointed=False,
        completed_run_replay_passed_count=replay_passed,
        config_sha256=bindings["config_sha256"], protocol_sha256=bindings["protocol_sha256"],
        amendment_receipt_sha256=bindings["amendment_receipt_sha256"],
        design_approval_sha256=bindings["design_approval_sha256"],
        compatibility_receipt_sha256=bindings["compatibility_receipt_sha256"],
        execution_approval_sha256=bindings["execution_approval_sha256"],
        batch_record_sha256=sha256_file(batch_path),
        failed_request_ledger_sha256=sha256_file(failed_ledger_path),
        failed_checkpoint_sha256=sha256_file(checkpoint_path),
        input_run_record_hashes=run_hashes, recommended_path=recommendation,
        narrow_recovery_amendment_drafted=False, current_batch_must_not_resume=True,
    )


def write_receipt(receipt: HG231ParserFailureDiagnostic, output: str | Path = DEFAULT_OUTPUT) -> str:
    path = _path(output)
    if path.exists():
        try:
            existing = HG231ParserFailureDiagnostic.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HG231FailureDiagnosticError("diagnostic_receipt_invalid") from exc
        if existing != receipt:
            raise HG231FailureDiagnosticError("diagnostic_receipt_mismatch")
        return sha256_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(receipt.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return sha256_file(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only H-G.2.3.1 parser failure diagnostic")
    parser.add_argument("--write-receipt", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        receipt = diagnose()
        digest = write_receipt(receipt, args.output) if args.write_receipt else None
        payload = receipt.model_dump(mode="json")
        payload["diagnostic_receipt_sha256"] = digest
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        return 0
    except HG231FailureDiagnosticError as exc:
        print(json.dumps({
            "status": "blocked", "error_code": exc.code, "network": "disabled",
            "provider_constructed": False, "results_modified": False,
        }, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
