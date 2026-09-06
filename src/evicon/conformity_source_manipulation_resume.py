"""Narrow timeout resume for the v1 source-manipulation qualification gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections import Counter
from pathlib import Path
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_identification import IdentificationError, safe_json, sha256_file
from .conformity_identification_network_gates import (
    MANIPULATION_SCHEMA,
    MANIPULATION_SCHEMA_NAME,
    MANIPULATION_OUTPUT,
    _provider,
)
from .conformity_identification_qualification import (
    DEFAULT_QUALIFICATION_APPROVAL,
    approval_sha256,
    load_qualification_approval,
)
from .conformity_source_manipulation import (
    DEFAULT_MANIPULATION_CONFIG,
    parse_manipulation_response,
    render_manipulation_request,
)
from .conformity_source_manipulation_smoke import build_cases
from .llm_contract import LLMProvider
from .request_ledger import LedgeredProvider, RequestLedger, RequestLedgerStatus, request_fingerprint_facts

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AMENDMENT = (
    "configs/provenance_cascade/identification/"
    "conformity_source_manipulation_timeout_resume.v1.toml"
)
DEFAULT_LEDGER = "outputs/conformity-source-manipulation-v1/request_ledger.jsonl"
DEFAULT_RECEIPT = "outputs/conformity-source-manipulation-v1/source_manipulation_receipt.json"


class TimeoutResumeAmendment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    amendment_id: Literal["evicon-conformity-source-manipulation-timeout-resume-v1"]
    amendment_version: Literal["conformity_source_manipulation_timeout_resume.v1"]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    accepted_on: str
    original_gate_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_gate_runner_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resume_runner_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    initial_ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    failed_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    failed_error_code: Literal["timeout"]
    completed_before_failure: Literal[6]
    unique_requests_before_failure: Literal[7]
    transport_attempts_before_failure: Literal[7]
    remaining_new_requests: Literal[5]
    logical_request_cap: Literal[12]
    completion_reservation_cap: Literal[1536]
    max_additional_attempts_for_failed_fingerprint: Literal[1]
    parser_recovery_enabled: Literal[False]
    completed_fingerprint_replay_allowed: Literal[False]
    prompt_or_schema_changed: Literal[False]
    network_execution_authorized: Literal[False]
    confirm_append_only_ledger: bool
    confirm_completed_requests_not_replayed: bool
    confirm_timeout_only_resume: bool
    confirm_no_result_merge: bool

    @model_validator(mode="after")
    def accepted_complete(self) -> "TimeoutResumeAmendment":
        if self.acceptance_status == "accepted" and (
            not self.accepted_by.strip() or not self.accepted_on.strip()
            or not all((
                self.confirm_append_only_ledger,
                self.confirm_completed_requests_not_replayed,
                self.confirm_timeout_only_resume,
                self.confirm_no_result_merge,
            ))
        ):
            raise ValueError("accepted timeout resume amendment incomplete")
        return self


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def _sha(path: str | Path) -> str:
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def load_amendment(path: str | Path = DEFAULT_AMENDMENT) -> TimeoutResumeAmendment:
    try:
        amendment = TimeoutResumeAmendment.model_validate(
            tomllib.loads(_resolve(path).read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise IdentificationError("timeout_resume_amendment_invalid") from exc
    if amendment.original_gate_config_sha256 != sha256_file(DEFAULT_MANIPULATION_CONFIG):
        raise IdentificationError("timeout_resume_gate_config_hash_mismatch")
    if amendment.original_gate_runner_sha256 != sha256_file("src/evicon/conformity_identification_network_gates.py"):
        raise IdentificationError("timeout_resume_gate_runner_hash_mismatch")
    if amendment.resume_runner_sha256 != sha256_file("src/evicon/conformity_source_manipulation_resume.py"):
        raise IdentificationError("timeout_resume_runner_hash_mismatch")
    if amendment.qualification_approval_sha256 != approval_sha256():
        raise IdentificationError("timeout_resume_approval_hash_mismatch")
    return amendment


def audit_initial_failure(
    amendment: TimeoutResumeAmendment,
    *,
    ledger_path: str | Path = DEFAULT_LEDGER,
    require_initial_hash: bool = True,
) -> dict[str, object]:
    ledger_file = _resolve(ledger_path)
    if require_initial_hash and _sha(ledger_file) != amendment.initial_ledger_sha256:
        raise IdentificationError("timeout_resume_ledger_hash_mismatch")
    ledger = RequestLedger(ledger_file)
    entries = ledger.entries()
    started = [item for item in entries if item.status is RequestLedgerStatus.STARTED]
    terminals = [item for item in entries if item.status in {RequestLedgerStatus.COMPLETED, RequestLedgerStatus.FAILED}]
    completed = [item for item in terminals if item.status is RequestLedgerStatus.COMPLETED]
    failed = [item for item in terminals if item.status is RequestLedgerStatus.FAILED]
    if (
        len({item.fingerprint for item in started}) != amendment.unique_requests_before_failure
        or len(started) != amendment.transport_attempts_before_failure
        or len(completed) != amendment.completed_before_failure
        or len(failed) != 1
        or failed[0].fingerprint != amendment.failed_fingerprint
        or failed[0].error_code != amendment.failed_error_code
    ):
        raise IdentificationError("timeout_resume_ledger_state_mismatch")
    return {
        "completed_count": len(completed), "failed_count": len(failed),
        "unique_logical_request_count": len({item.fingerprint for item in started}),
        "transport_attempt_count": len(started),
        "failed_fingerprint": amendment.failed_fingerprint,
    }


def _request_for(context):
    request = render_manipulation_request(context)
    return request.model_copy(update={"metadata": {
        **request.metadata, "protocol": context.case_id,
        "agent_id": "source-structure-check", "round_id": 0,
    }})


def run_resume(
    provider: LLMProvider,
    *,
    amendment_path: str | Path = DEFAULT_AMENDMENT,
    ledger_path: str | Path = DEFAULT_LEDGER,
    receipt_path: str | Path = DEFAULT_RECEIPT,
) -> dict[str, object]:
    amendment = load_amendment(amendment_path)
    if amendment.acceptance_status != "accepted":
        raise IdentificationError("timeout_resume_human_approval_required")
    load_qualification_approval()
    audit_initial_failure(amendment, ledger_path=ledger_path)
    receipt = _resolve(receipt_path)
    if receipt.exists():
        raise IdentificationError("source_manipulation_receipt_exists")
    ledger = RequestLedger(_resolve(ledger_path))
    wrapped = LedgeredProvider(
        provider, ledger=ledger, request_cap=amendment.logical_request_cap,
        completion_reservation_cap=amendment.completion_reservation_cap,
        resume=True,
    )
    expected_roots: Counter[str] = Counter()
    retried_timeout = False
    for context in build_cases():
        request = _request_for(context)
        facts = request_fingerprint_facts(request)
        entries = ledger.entries()
        terminal = [
            item for item in entries
            if item.fingerprint == facts["fingerprint"]
            and item.status in {RequestLedgerStatus.COMPLETED, RequestLedgerStatus.FAILED}
        ]
        latest = max(terminal, key=lambda item: item.attempt_count) if terminal else None
        if latest is not None and latest.status is RequestLedgerStatus.COMPLETED:
            # The original synchronous runner starting the next case proves the
            # preceding response passed its strict parser; no response is replayed.
            expected_roots[context.projection.value] += len({
                item.source_root_id for item in context.public_root_assignments
                if item.source_root_id is not None
            })
            continue
        if latest is not None and latest.status is RequestLedgerStatus.FAILED:
            if latest.fingerprint != amendment.failed_fingerprint or latest.error_code != "timeout" or retried_timeout:
                raise IdentificationError("timeout_resume_scope_violation")
            retried_timeout = True
        response = wrapped.complete(request)
        parsed = parse_manipulation_response(response.content, context)
        expected_roots[context.projection.value] += parsed.visible_source_root_count

    summary = ledger.summary(request_cap=12, completion_reservation_cap=1536)
    entries = ledger.entries()
    latest_by_fingerprint = {}
    for item in entries:
        if item.status in {RequestLedgerStatus.COMPLETED, RequestLedgerStatus.FAILED}:
            current = latest_by_fingerprint.get(item.fingerprint)
            if current is None or item.attempt_count > current.attempt_count:
                latest_by_fingerprint[item.fingerprint] = item
    final_completed = sum(item.status is RequestLedgerStatus.COMPLETED for item in latest_by_fingerprint.values())
    final_failed = sum(item.status is RequestLedgerStatus.FAILED for item in latest_by_fingerprint.values())
    if summary.unique_logical_request_count != 12 or final_completed != 12 or final_failed != 0:
        raise IdentificationError("timeout_resume_incomplete")
    if summary.transport_attempt_count != 13 or not retried_timeout:
        raise IdentificationError("timeout_resume_attempt_accounting_mismatch")
    payload = {
        "status": "completed", "gate_id": "evicon-conformity-source-manipulation-v1",
        "technical_amendment_id": amendment.amendment_id,
        "amendment_sha256": _sha(amendment_path),
        "qualification_approval_sha256": amendment.qualification_approval_sha256,
        "config_sha256": amendment.original_gate_config_sha256,
        "logical_request_count": 12, "transport_attempt_count": 13,
        "timeout_resume_transport_attempt_count": 1,
        "completed_fingerprint_replay_count": 0,
        "root_count_totals": dict(sorted(expected_roots.items())),
        "behavior_effect_estimated": False, "joined_to_behavior_study": False,
        "network_used": True, "private_truth_exposed": False,
        "not_paper_result": True, "no_causal_conclusion": True,
    }
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2)
    receipt.write_text(serialized, encoding="utf-8")
    try:
        safe_receipt_path = str(receipt.relative_to(_ROOT))
    except ValueError:
        safe_receipt_path = str(receipt)
    return {**payload, "receipt_path": safe_receipt_path}


def preflight(amendment_path: str | Path = DEFAULT_AMENDMENT) -> dict[str, object]:
    try:
        amendment = load_amendment(amendment_path)
        state = audit_initial_failure(amendment)
        reasons = [] if amendment.acceptance_status == "accepted" else ["timeout_resume_human_approval_required"]
        reasons.append("timeout_resume_network_authorization_required")
        return {
            "status": "timeout_resume_offline_ready",
            "ready_for_network": False, "blocking_reasons": reasons,
            "amendment_status": amendment.acceptance_status,
            "amendment_sha256": _sha(amendment_path), **state,
            "resume_runner_sha256": amendment.resume_runner_sha256,
            "remaining_new_request_count": 5,
            "maximum_additional_transport_attempt_count": 6,
            "logical_request_cap": 12, "completion_reservation_cap": 1536,
            "network": "disabled", "provider_constructed": False,
            "api_key_read": False, "results_written": False,
            "private_truth_exposed": False, "not_paper_result": True,
            "no_causal_conclusion": True,
        }
    except Exception as exc:
        return {
            "status": "blocked", "ready_for_network": False,
            "blocking_reasons": [getattr(exc, "code", "timeout_resume_preflight_failed")],
            "network": "disabled", "provider_constructed": False,
            "api_key_read": False, "results_written": False,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resume the one timeout in the source-manipulation gate")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-resume", action="store_true")
    parser.add_argument("--confirm-request-cap", type=int)
    parser.add_argument("--confirm-completion-reservation-cap", type=int)
    args = parser.parse_args(argv)
    if not args.allow_network:
        payload = preflight()
    elif not args.confirm_resume or args.confirm_request_cap != 12 or args.confirm_completion_reservation_cap != 1536:
        payload = {
            "status": "blocked", "blocking_reasons": ["explicit_resume_cap_confirmation_required"],
            "network": "disabled", "provider_constructed": False,
        }
    else:
        try:
            amendment = load_amendment()
            if amendment.acceptance_status != "accepted":
                raise IdentificationError("timeout_resume_human_approval_required")
            provider = _provider(
                environment=None, max_tokens=128, temperature=0.0, seed=20261100,
                schema_name=MANIPULATION_SCHEMA_NAME, schema=MANIPULATION_SCHEMA,
            )
            payload = run_resume(provider)
        except Exception as exc:
            payload = {
                "status": "failed", "error_code": getattr(exc, "code", "timeout_resume_failed"),
                "network": "enabled", "private_truth_exposed": False,
            }
    print(safe_json(payload))
    return 0 if payload.get("status") in {"completed", "timeout_resume_offline_ready"} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["TimeoutResumeAmendment", "audit_initial_failure", "load_amendment", "preflight", "run_resume"]
