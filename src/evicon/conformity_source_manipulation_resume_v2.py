"""Narrow v2 resume after a local provider-configuration failure.

This module is intentionally separate from the accepted v1 timeout amendment.
It preserves every existing ledger line and permits one new transport attempt only
after the caller has constructed a usable Provider outside the ledger path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections import Counter
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_identification import IdentificationError, safe_json, sha256_file
from .conformity_identification_network_gates import (
    MANIPULATION_SCHEMA,
    MANIPULATION_SCHEMA_NAME,
    _provider,
)
from .conformity_identification_qualification import approval_sha256, load_qualification_approval
from .conformity_source_manipulation import DEFAULT_MANIPULATION_CONFIG, parse_manipulation_response, render_manipulation_request
from .conformity_source_manipulation_resume import DEFAULT_AMENDMENT as V1_AMENDMENT
from .conformity_source_manipulation_smoke import build_cases
from .llm_contract import LLMProvider
from .request_ledger import LedgeredProvider, RequestLedger, RequestLedgerStatus, request_fingerprint_facts

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AMENDMENT = "configs/provenance_cascade/identification/conformity_source_manipulation_timeout_resume.v2.toml"
DEFAULT_LEDGER = "outputs/conformity-source-manipulation-v1/request_ledger.jsonl"
DEFAULT_RECEIPT = "outputs/conformity-source-manipulation-v1/source_manipulation_resume_v2_receipt.json"


class TimeoutResumeV2Amendment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    amendment_id: Literal["evicon-conformity-source-manipulation-timeout-resume-v2"]
    amendment_version: Literal["conformity_source_manipulation_timeout_resume.v2"]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    accepted_on: str
    prior_amendment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_gate_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resume_runner_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pre_resume_ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    failed_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    initial_failure_error_code: Literal["timeout"]
    local_configuration_error_code: Literal["missing_base_url"]
    completed_before_resume: Literal[6]
    unique_requests_before_resume: Literal[7]
    transport_attempts_before_resume: Literal[8]
    remaining_new_requests: Literal[5]
    logical_request_cap: Literal[12]
    completion_reservation_cap: Literal[1536]
    max_new_transport_attempts_for_failed_fingerprint: Literal[1]
    parser_recovery_enabled: Literal[False]
    prompt_or_schema_changed: Literal[False]
    network_execution_authorized: Literal[False]
    confirm_append_only_ledger: bool
    confirm_completed_fingerprints_not_replayed: bool
    confirm_local_configuration_failure_not_treated_as_http: bool
    confirm_timeout_scope_only: bool

    @model_validator(mode="after")
    def accepted_is_complete(self) -> "TimeoutResumeV2Amendment":
        if self.acceptance_status == "accepted" and (
            not self.accepted_by.strip() or not self.accepted_on.strip() or not all((
                self.confirm_append_only_ledger,
                self.confirm_completed_fingerprints_not_replayed,
                self.confirm_local_configuration_failure_not_treated_as_http,
                self.confirm_timeout_scope_only,
            ))
        ):
            raise ValueError("accepted timeout resume v2 amendment incomplete")
        return self


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def _sha(path: str | Path) -> str:
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def load_amendment(path: str | Path = DEFAULT_AMENDMENT) -> TimeoutResumeV2Amendment:
    try:
        amendment = TimeoutResumeV2Amendment.model_validate(tomllib.loads(_resolve(path).read_text(encoding="utf-8")))
    except Exception as exc:
        raise IdentificationError("timeout_resume_v2_amendment_invalid") from exc
    if amendment.prior_amendment_sha256 != _sha(V1_AMENDMENT):
        raise IdentificationError("timeout_resume_v2_prior_amendment_hash_mismatch")
    if amendment.original_gate_config_sha256 != sha256_file(DEFAULT_MANIPULATION_CONFIG):
        raise IdentificationError("timeout_resume_v2_gate_config_hash_mismatch")
    if amendment.qualification_approval_sha256 != approval_sha256():
        raise IdentificationError("timeout_resume_v2_qualification_approval_hash_mismatch")
    if amendment.resume_runner_sha256 != sha256_file("src/evicon/conformity_source_manipulation_resume_v2.py"):
        raise IdentificationError("timeout_resume_v2_runner_hash_mismatch")
    return amendment


def audit_pre_resume_ledger(amendment: TimeoutResumeV2Amendment, *, ledger_path: str | Path = DEFAULT_LEDGER) -> dict[str, object]:
    ledger_file = _resolve(ledger_path)
    if _sha(ledger_file) != amendment.pre_resume_ledger_sha256:
        raise IdentificationError("timeout_resume_v2_ledger_hash_mismatch")
    entries = RequestLedger(ledger_file).entries()
    started = [item for item in entries if item.status is RequestLedgerStatus.STARTED]
    terminal = [item for item in entries if item.status in {RequestLedgerStatus.COMPLETED, RequestLedgerStatus.FAILED}]
    target = sorted((item for item in terminal if item.fingerprint == amendment.failed_fingerprint), key=lambda item: item.attempt_count)
    completed = [item for item in terminal if item.status is RequestLedgerStatus.COMPLETED]
    if (
        len(started) != amendment.transport_attempts_before_resume
        or len({item.fingerprint for item in started}) != amendment.unique_requests_before_resume
        or len(completed) != amendment.completed_before_resume
        or [(item.status.value, item.error_code) for item in target] != [
            ("failed", amendment.initial_failure_error_code),
            ("failed", amendment.local_configuration_error_code),
        ]
    ):
        raise IdentificationError("timeout_resume_v2_ledger_state_mismatch")
    return {
        "completed_count": len(completed),
        "unique_logical_request_count": len({item.fingerprint for item in started}),
        "transport_attempt_count": len(started),
        "failed_fingerprint": amendment.failed_fingerprint,
        "local_configuration_failure_count": 1,
        "prior_http_transport_attempt_count": 1,
    }


def _request_for(context):
    request = render_manipulation_request(context)
    return request.model_copy(update={"metadata": {
        **request.metadata, "protocol": context.case_id,
        "agent_id": "source-structure-check", "round_id": 0,
    }})


def run_resume(provider: LLMProvider, *, amendment_path: str | Path = DEFAULT_AMENDMENT,
               ledger_path: str | Path = DEFAULT_LEDGER, receipt_path: str | Path = DEFAULT_RECEIPT) -> dict[str, object]:
    amendment = load_amendment(amendment_path)
    if amendment.acceptance_status != "accepted":
        raise IdentificationError("timeout_resume_v2_human_approval_required")
    load_qualification_approval()
    audit_pre_resume_ledger(amendment, ledger_path=ledger_path)
    receipt = _resolve(receipt_path)
    if receipt.exists():
        raise IdentificationError("timeout_resume_v2_receipt_exists")
    ledger = RequestLedger(_resolve(ledger_path))
    wrapped = LedgeredProvider(provider, ledger=ledger, request_cap=12, completion_reservation_cap=1536, resume=True)
    retried_target = False
    roots: Counter[str] = Counter()
    for context in build_cases():
        request = _request_for(context)
        fingerprint = request_fingerprint_facts(request)["fingerprint"]
        terminal = [
            item for item in ledger.entries() if item.fingerprint == fingerprint
            and item.status in {RequestLedgerStatus.COMPLETED, RequestLedgerStatus.FAILED}
        ]
        latest = max(terminal, key=lambda item: item.attempt_count) if terminal else None
        if latest is not None and latest.status is RequestLedgerStatus.COMPLETED:
            continue
        if latest is not None:
            if fingerprint != amendment.failed_fingerprint or latest.error_code != amendment.local_configuration_error_code or retried_target:
                raise IdentificationError("timeout_resume_v2_scope_violation")
            retried_target = True
        response = wrapped.complete(request)
        parsed = parse_manipulation_response(response.content, context)
        roots[context.projection.value] += parsed.visible_source_root_count
    entries = ledger.entries()
    latest_by_fingerprint = {}
    for item in entries:
        if item.status in {RequestLedgerStatus.COMPLETED, RequestLedgerStatus.FAILED}:
            old = latest_by_fingerprint.get(item.fingerprint)
            if old is None or item.attempt_count > old.attempt_count:
                latest_by_fingerprint[item.fingerprint] = item
    started = [item for item in entries if item.status is RequestLedgerStatus.STARTED]
    if len({item.fingerprint for item in started}) != 12 or len(started) != 14 or not retried_target:
        raise IdentificationError("timeout_resume_v2_attempt_accounting_mismatch")
    if any(item.status is not RequestLedgerStatus.COMPLETED for item in latest_by_fingerprint.values()):
        raise IdentificationError("timeout_resume_v2_incomplete")
    payload = {
        "status": "completed", "gate_id": "evicon-conformity-source-manipulation-v1",
        "technical_amendment_id": amendment.amendment_id, "amendment_sha256": _sha(amendment_path),
        "config_sha256": amendment.original_gate_config_sha256,
        "qualification_approval_sha256": amendment.qualification_approval_sha256,
        "logical_request_count": 12, "transport_attempt_count": 14,
        "http_transport_retry_count": 1, "local_configuration_failure_count": 1,
        "completed_fingerprint_replay_count": 0, "root_count_totals": dict(sorted(roots.items())),
        "behavior_effect_estimated": False, "joined_to_behavior_study": False,
        "network_used": True, "private_truth_exposed": False,
        "not_paper_result": True, "no_causal_conclusion": True,
    }
    receipt.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2), encoding="utf-8")
    try:
        safe_receipt_path = str(receipt.relative_to(_ROOT))
    except ValueError:
        safe_receipt_path = str(receipt)
    return {**payload, "receipt_path": safe_receipt_path}


def preflight(amendment_path: str | Path = DEFAULT_AMENDMENT) -> dict[str, object]:
    try:
        amendment = load_amendment(amendment_path)
        state = audit_pre_resume_ledger(amendment)
        reasons = [] if amendment.acceptance_status == "accepted" else ["timeout_resume_v2_human_approval_required"]
        reasons.append("timeout_resume_v2_network_authorization_required")
        return {"status": "timeout_resume_v2_offline_ready", "ready_for_network": False,
                "blocking_reasons": reasons, "amendment_status": amendment.acceptance_status,
                "amendment_sha256": _sha(amendment_path), **state, "remaining_new_request_count": 5,
                "maximum_new_transport_attempt_count": 6, "logical_request_cap": 12,
                "completion_reservation_cap": 1536, "network": "disabled", "provider_constructed": False,
                "api_key_read": False, "results_written": False, "private_truth_exposed": False,
                "not_paper_result": True, "no_causal_conclusion": True}
    except Exception as exc:
        return {"status": "blocked", "ready_for_network": False,
                "blocking_reasons": [getattr(exc, "code", "timeout_resume_v2_preflight_failed")],
                "network": "disabled", "provider_constructed": False, "api_key_read": False,
                "results_written": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-time v2 resume after local configuration failure")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-resume", action="store_true")
    parser.add_argument("--confirm-request-cap", type=int)
    parser.add_argument("--confirm-completion-reservation-cap", type=int)
    args = parser.parse_args(argv)
    if not args.allow_network:
        payload = preflight()
    elif not args.confirm_resume or args.confirm_request_cap != 12 or args.confirm_completion_reservation_cap != 1536:
        payload = {"status": "blocked", "blocking_reasons": ["explicit_resume_cap_confirmation_required"],
                   "network": "disabled", "provider_constructed": False}
    else:
        try:
            amendment = load_amendment()
            if amendment.acceptance_status != "accepted":
                raise IdentificationError("timeout_resume_v2_human_approval_required")
            # Construct the provider before touching the ledger: missing local
            # configuration cannot consume the sole authorized HTTP retry.
            provider = _provider(environment=None, max_tokens=128, temperature=0.0, seed=20261100,
                                 schema_name=MANIPULATION_SCHEMA_NAME, schema=MANIPULATION_SCHEMA)
            payload = run_resume(provider)
        except Exception as exc:
            payload = {"status": "failed", "error_code": getattr(exc, "code", "timeout_resume_v2_failed"),
                       "network": "enabled", "private_truth_exposed": False}
    print(safe_json(payload))
    return 0 if payload.get("status") in {"completed", "timeout_resume_v2_offline_ready"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
