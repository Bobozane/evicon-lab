"""Offline, content-free diagnostics for H-D.2 parser-invalid failures."""
from __future__ import annotations

import argparse
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from .cascade_agent_protocol_v2 import V2_SCHEMA_NAME, V2_TEMPLATE_VERSION, load_hd2_config
from .cascade_agent_protocol_v2_compatibility import COMPATIBILITY_MAX_TOKENS
from .cascade_agent_protocol_v2_pilot import (
    HD2_CONFIG_RELATIVE,
    HD2_PROTOCOL_RELATIVE,
    HD2PilotBatchRecord,
    HD2PilotRunStatus,
    sha256_file,
)
from .cascade_real_agent_runner import CascadeAgentCheckpoint
from .request_ledger import RequestLedger, RequestLedgerStatus

_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_VERSION = "provenance_cascade_hd2_parser_diagnostic.v1"
DEFAULT_RESULT_ROOT = "results/provenance-cascade-pilot-hd2-v1"
DEFAULT_DIAGNOSTIC_FILENAME = "hd2_parser_failure_diagnostic.json"
FINISH_REASON_NOT_PERSISTED = "unknown_not_persisted"


class ParserFailureAssessment(str, Enum):
    TRUNCATION_MORE_CONSISTENT = "truncation_more_consistent_but_not_confirmed"
    INCIDENTAL_MALFORMED_MORE_CONSISTENT = "incidental_malformed_json_more_consistent"
    INDETERMINATE = "indeterminate"
    VALID_RESPONSE = "valid_response"


class HD2ParserFailureDiagnostic(BaseModel):
    """Safe sidecar receipt; never contains response or prompt content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    diagnostic_version: str = DIAGNOSTIC_VERSION
    status: str = "diagnosed_offline"
    study_id: str
    run_id: str
    scenario_id: str
    condition: str
    agent_id: str | None = None
    round_id: int | None = Field(default=None, ge=0)
    fingerprint: str = Field(min_length=64, max_length=64)
    request_attempt_count: int = Field(ge=1)
    ledger_terminal_status: str
    finish_reason: str
    finish_reason_available: bool
    completion_tokens: int | None = Field(default=None, ge=0)
    configured_max_tokens: int = Field(gt=0)
    completion_token_limit_reached: bool | None
    parser_error_category: str
    response_format_mode: str
    response_schema_name: str
    assessment: ParserFailureAssessment
    reason_codes: tuple[str, ...]
    checkpoint_entry_count: int = Field(ge=0)
    failed_fingerprint_checkpointed: bool
    config_sha256: str
    protocol_sha256: str
    batch_record_sha256: str
    request_ledger_sha256: str
    checkpoint_sha256: str
    original_artifacts_preserved: bool = True
    network_used_for_diagnostic: bool = False
    automatic_retry_performed: bool = False
    parser_recovery_enabled: bool = False
    development_only: bool = True
    pilot_only: bool = True
    not_paper_result: bool = True
    no_causal_conclusion: bool = True
    private_truth_exposed: bool = False


class HD2ParserDiagnosticError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def classify_parser_failure(
    *,
    parser_error_category: str | None,
    finish_reason: str | None,
    completion_tokens: int | None,
    configured_max_tokens: int,
    parser_valid: bool = False,
) -> tuple[ParserFailureAssessment, tuple[str, ...]]:
    """Classify only from safe transport/parser facts, never response content."""
    if parser_valid:
        return ParserFailureAssessment.VALID_RESPONSE, ("strict_json_valid",)
    reasons: list[str] = []
    at_limit = completion_tokens is not None and completion_tokens == configured_max_tokens
    if at_limit:
        reasons.append("completion_tokens_equal_configured_max")
    if finish_reason == "length":
        reasons.append("finish_reason_length")
    elif finish_reason in {None, FINISH_REASON_NOT_PERSISTED}:
        reasons.append("finish_reason_unavailable")
    elif finish_reason:
        reasons.append("finish_reason_non_length")
    if parser_error_category:
        reasons.append(f"parser_{parser_error_category}")
    if parser_error_category == "malformed_json" and (finish_reason == "length" or at_limit):
        return ParserFailureAssessment.TRUNCATION_MORE_CONSISTENT, tuple(reasons)
    if parser_error_category == "malformed_json" and finish_reason == "stop" and not at_limit:
        return ParserFailureAssessment.INCIDENTAL_MALFORMED_MORE_CONSISTENT, tuple(reasons)
    return ParserFailureAssessment.INDETERMINATE, tuple(reasons)


def diagnose_persisted_failure(
    *,
    result_root: str | Path,
    fingerprint: str,
    config_path: str | Path = HD2_CONFIG_RELATIVE,
) -> HD2ParserFailureDiagnostic:
    """Read only safe persisted records and reconstruct no response content."""
    root = Path(result_root).resolve()
    config_file = Path(config_path).resolve()
    config, _ = load_hd2_config(config_file)
    batch_path = root / "pilot_batch_record.json"
    if not batch_path.exists():
        raise HD2ParserDiagnosticError("batch_record_missing")
    try:
        batch = HD2PilotBatchRecord.model_validate_json(batch_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HD2ParserDiagnosticError("batch_record_invalid") from exc
    if batch.study_id != config.study_id or batch.config_sha256 != sha256_file(config_file):
        raise HD2ParserDiagnosticError("batch_config_mismatch")
    failed_states = [state for state in batch.runs if state.status is HD2PilotRunStatus.FAILED]
    matches: list[tuple[object, object, Path]] = []
    for state in failed_states:
        ledger_path = root / state.run_id / "request_ledger.jsonl"
        if not ledger_path.exists():
            continue
        for entry in RequestLedger(ledger_path).entries():
            if entry.fingerprint == fingerprint and entry.status is RequestLedgerStatus.COMPLETED:
                matches.append((state, entry, ledger_path))
    if len(matches) != 1:
        raise HD2ParserDiagnosticError("failed_fingerprint_not_unique")
    state, terminal, ledger_path = matches[0]
    if state.error_code not in {
        "malformed_json", "top_level_type", "missing_field", "extra_field",
        "field_type", "invalid_stance", "unavailable_content_id",
        "unavailable_evidence_id", "share_field",
    }:
        raise HD2ParserDiagnosticError("failure_is_not_parser_invalid")
    checkpoint_path = root / state.run_id / "agent_checkpoint.json"
    if not checkpoint_path.exists():
        raise HD2ParserDiagnosticError("checkpoint_missing")
    try:
        checkpoint = CascadeAgentCheckpoint.model_validate_json(checkpoint_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HD2ParserDiagnosticError("checkpoint_invalid") from exc
    checkpointed = any(entry.fingerprint == fingerprint for entry in checkpoint.entries)
    if checkpointed:
        raise HD2ParserDiagnosticError("failed_fingerprint_was_checkpointed")
    at_limit = None if terminal.completion_tokens is None else terminal.completion_tokens == COMPATIBILITY_MAX_TOKENS
    assessment, reasons = classify_parser_failure(
        parser_error_category=state.error_code,
        finish_reason=FINISH_REASON_NOT_PERSISTED,
        completion_tokens=terminal.completion_tokens,
        configured_max_tokens=COMPATIBILITY_MAX_TOKENS,
    )
    return HD2ParserFailureDiagnostic(
        study_id=batch.study_id,
        run_id=state.run_id,
        scenario_id=state.scenario_id,
        condition=state.condition,
        agent_id=terminal.agent_id,
        round_id=terminal.round_id,
        fingerprint=fingerprint,
        request_attempt_count=terminal.attempt_count,
        ledger_terminal_status=terminal.status.value,
        finish_reason=FINISH_REASON_NOT_PERSISTED,
        finish_reason_available=False,
        completion_tokens=terminal.completion_tokens,
        configured_max_tokens=COMPATIBILITY_MAX_TOKENS,
        completion_token_limit_reached=at_limit,
        parser_error_category=state.error_code,
        response_format_mode=config.response_format,
        response_schema_name=config.response_schema_name,
        assessment=assessment,
        reason_codes=reasons,
        checkpoint_entry_count=len(checkpoint.entries),
        failed_fingerprint_checkpointed=False,
        config_sha256=sha256_file(config_file),
        protocol_sha256=sha256_file(_ROOT / HD2_PROTOCOL_RELATIVE),
        batch_record_sha256=sha256_file(batch_path),
        request_ledger_sha256=sha256_file(ledger_path),
        checkpoint_sha256=sha256_file(checkpoint_path),
    )


def write_diagnostic_receipt(receipt: HD2ParserFailureDiagnostic, output_path: str | Path) -> str:
    path = Path(output_path).resolve()
    if path.exists():
        try:
            existing = HD2ParserFailureDiagnostic.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HD2ParserDiagnosticError("diagnostic_receipt_invalid") from exc
        if existing != receipt:
            raise HD2ParserDiagnosticError("diagnostic_receipt_mismatch")
        return sha256_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)
    return sha256_file(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline H-D.2 parser failure diagnostics")
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--config", default=HD2_CONFIG_RELATIVE)
    parser.add_argument("--write-receipt", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        receipt = diagnose_persisted_failure(
            result_root=args.result_root,
            fingerprint=args.fingerprint,
            config_path=args.config,
        )
        output_sha = None
        if args.write_receipt:
            output = args.output or str(Path(args.result_root) / DEFAULT_DIAGNOSTIC_FILENAME)
            output_sha = write_diagnostic_receipt(receipt, output)
        payload = receipt.model_dump(mode="json")
        payload["diagnostic_receipt_sha256"] = output_sha
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        return 0
    except HD2ParserDiagnosticError as exc:
        print(json.dumps({
            "status": "blocked",
            "error_code": exc.code,
            "network": "disabled",
            "provider_constructed": False,
            "results_modified": False,
        }, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DIAGNOSTIC_FILENAME",
    "DIAGNOSTIC_VERSION",
    "FINISH_REASON_NOT_PERSISTED",
    "HD2ParserDiagnosticError",
    "HD2ParserFailureDiagnostic",
    "ParserFailureAssessment",
    "classify_parser_failure",
    "diagnose_persisted_failure",
    "main",
    "write_diagnostic_receipt",
]
