"""One-shot, timeout-only resume gate for the immutable H-G.2.3.2 batch."""
from __future__ import annotations

import argparse
import json
import os
import tomllib
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cascade_agent_protocol_hg232 import HG232_RESPONSE_JSON_SCHEMA, HG232_SCHEMA_NAME
from .cascade_real_agent_runner import CascadeRealAgentRunError
from .llm_contract import LLMProvider, LLMProviderError, LLMRequest, LLMResponse, ProviderErrorCode
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ResponseFormatMode
from .provenance_cascade_hg2 import sha256_file
from .provenance_cascade_hg21_calibration import HG21ExecutionSummary, HG21RunSpec, _atomic, _path, _safety
from .provenance_cascade_hg232 import DEFAULT_APPROVAL as DEFAULT_DESIGN_APPROVAL, DEFAULT_COMPATIBILITY_RECEIPT, HG232Approval
from .provenance_cascade_hg232_calibration import DEFAULT_EXECUTION_APPROVAL
from .provenance_cascade_hg232_compatibility_receipt import HG232CompatibilityReceipt
from .provenance_cascade_hg232_resume import _adapter_runner, _raw_config, _safe_batch_state
from .request_ledger import RequestLedger, RequestLedgerStatus, request_fingerprint_facts

RESUME_VERSION = "provenance_cascade_hg232_timeout_resume.v3"
ROOT = "results/provenance-cascade-hg232-adoption-identifiability-v1"
BATCH = f"{ROOT}/calibration_batch_record.json"
FAILED_RUN_ID = "hg232-cascade-hg1-true-minority-correction-20261031-provenance_aware_controller"
FAILED_FINGERPRINT = "5f2a4650dad79abe1a2ff28f57092228f892d5f9d09248278eced91c1b1c747f"
INITIAL_BATCH_SHA256 = "9c7fc4313fad9833c6bb464e7e78f387aeb0458798eec016ab0c8dd86fcb5273"
HISTORICAL_RUNNER_SHA256 = "041f85a0d8889377b6bb63a0237b5d212e93081b9cce7f3ab6cd4ad24f30a06f"
HISTORICAL_PROBE_RUNNER_SHA256 = "12e602a98d892b25a3fe6068dc5f9c1c5174de24024a8fd74be68c6843bd98b1"
RUNNER_PATH = "src/evicon/provenance_cascade_hg232_calibration.py"
PROBE_RUNNER_PATH = "src/evicon/provenance_cascade_hg232_stability_probe.py"
PROBE_RECEIPT = "results/provenance-cascade-hg232-protocol-stability-probe-v1/protocol_stability_probe_receipt.json"
PROBE_RECEIPT_SHA256 = "5a1f1397b728797ef2e1c4a19bfaad3144eb2fd53538b7c1bd1e08df5987286a"
DEFAULT_RECEIPT = "outputs/study-locks/provenance_cascade_hg232_timeout_resume_amendment_v3.json"
DEFAULT_RESUME_APPROVAL = "configs/provenance_cascade/pilot/provenance_cascade_hg232_timeout_resume_approval_v3.toml"
RESUME_HELPER_PATH = "src/evicon/provenance_cascade_hg232_resume.py"
AMENDMENT_DOCUMENT_PATH = "docs/provenance-cascade-hg232-timeout-resume-amendment.md"
RESUME_HELPER_SHA256 = "00a0c78c61713db259811a75088764b1179f591b47312159f351972f4d4de238"


class HG232TimeoutResumeError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HG232TimeoutResumeReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    amendment_id: Literal["provenance-cascade-hg232-timeout-resume-amendment-v3"]
    status: Literal["offline_validated_pending_network_resume_authorization"]
    resume_version: Literal["provenance_cascade_hg232_timeout_resume.v3"]
    config_sha256: str
    protocol_sha256: str
    execution_approval_sha256: str
    compatibility_receipt_sha256: str
    probe_receipt_sha256: Literal[PROBE_RECEIPT_SHA256]
    historical_runner_sha256: Literal[HISTORICAL_RUNNER_SHA256]
    historical_probe_runner_sha256: Literal[HISTORICAL_PROBE_RUNNER_SHA256]
    resume_adapter_sha256: str
    resume_helper_sha256: Literal[RESUME_HELPER_SHA256]
    amendment_document_sha256: str
    initial_batch_record_sha256: Literal[INITIAL_BATCH_SHA256]
    failed_run_id: Literal[FAILED_RUN_ID]
    failed_fingerprint: Literal[FAILED_FINGERPRINT]
    failure_code: Literal["timeout"]
    failure_attempt_count: Literal[1]
    failed_run_checkpoint_count: Literal[8]
    completed_run_count: Literal[7]
    failed_run_count: Literal[1]
    planned_run_count: Literal[8]
    completed_replay_passed_count: Literal[7]
    unique_logical_request_count: Literal[135]
    completed_logical_request_count: Literal[134]
    failed_logical_request_count: Literal[1]
    transport_attempt_count: Literal[135]
    remaining_new_logical_request_count: Literal[153]
    max_timeout_retry_count: Literal[1]
    total_logical_request_cap: Literal[288]
    completion_reservation_cap: Literal[589824]
    prompt_schema_model_seed_condition_unchanged: Literal[True]
    metrics_controller_unchanged: Literal[True]
    completed_fingerprint_replay_forbidden: Literal[True]
    parser_recovery_enabled: Literal[False]
    append_only_ledger_required: Literal[True]
    one_shot_initial_state_lock: Literal[True]
    results_modified: Literal[False]
    network: Literal["disabled"]
    private_truth_exposed: Literal[False]
    development_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]


class HG232TimeoutResumeApproval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    approval_id: Literal["provenance-cascade-hg232-timeout-resume-execution-v3"]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    amendment_receipt_sha256: str
    resume_adapter_sha256: str
    resume_helper_sha256: Literal[RESUME_HELPER_SHA256]
    amendment_document_sha256: str
    initial_batch_record_sha256: Literal[INITIAL_BATCH_SHA256]
    failed_fingerprint: Literal[FAILED_FINGERPRINT]
    total_logical_request_cap: Literal[288]
    completion_reservation_cap: Literal[589824]
    confirm_timeout_only: bool
    confirm_completed_fingerprints_not_replayed: bool
    confirm_append_only_ledger: bool
    confirm_one_shot_initial_state: bool
    confirm_no_parser_recovery: bool
    network_resume_authorized: bool

    @model_validator(mode="after")
    def accepted(self) -> "HG232TimeoutResumeApproval":
        flags = (
            self.confirm_timeout_only,
            self.confirm_completed_fingerprints_not_replayed,
            self.confirm_append_only_ledger,
            self.confirm_one_shot_initial_state,
            self.confirm_no_parser_recovery,
            self.network_resume_authorized,
        )
        if self.acceptance_status == "accepted" and (not self.accepted_by.strip() or not all(flags)):
            raise ValueError("timeout resume approval incomplete")
        return self


def _load_json(path: str | Path) -> Any:
    try:
        return json.loads(_path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise HG232TimeoutResumeError("timeout_resume_input_invalid") from exc


def _initial_audit() -> dict[str, Any]:
    if sha256_file(RESUME_HELPER_PATH) != RESUME_HELPER_SHA256:
        raise HG232TimeoutResumeError("resume_helper_hash_mismatch")
    if sha256_file(RUNNER_PATH) != HISTORICAL_RUNNER_SHA256 or sha256_file(PROBE_RUNNER_PATH) != HISTORICAL_PROBE_RUNNER_SHA256:
        raise HG232TimeoutResumeError("historical_runner_hash_mismatch")
    batch_path = _path(BATCH)
    if sha256_file(batch_path) != INITIAL_BATCH_SHA256:
        raise HG232TimeoutResumeError("timeout_resume_initial_batch_hash_mismatch")
    state = _safe_batch_state(_path(ROOT))
    expected = {"status":"failed","failure_code":"timeout","completed":7,"failed":1,"planned":8,"unique":135,"attempts":135,"passed":7,"partial":8}
    if state != expected:
        raise HG232TimeoutResumeError("timeout_resume_initial_state_mismatch")
    run_dir = _path(ROOT) / FAILED_RUN_ID
    ledger = RequestLedger(run_dir / "request_ledger.jsonl")
    entries = ledger.entries()
    terminals = [entry for entry in entries if entry.status in {RequestLedgerStatus.COMPLETED, RequestLedgerStatus.FAILED}]
    failures = [entry for entry in terminals if entry.status is RequestLedgerStatus.FAILED]
    if len(failures) != 1:
        raise HG232TimeoutResumeError("timeout_resume_failure_count_mismatch")
    failure = failures[0]
    if failure.fingerprint != FAILED_FINGERPRINT or failure.error_code != "timeout" or failure.attempt_count != 1:
        raise HG232TimeoutResumeError("timeout_resume_failure_binding_mismatch")
    checkpoint = _load_json(run_dir / "agent_checkpoint.json")
    checkpoint_fingerprints = {entry["fingerprint"] for entry in checkpoint.get("entries", [])}
    if len(checkpoint_fingerprints) != 8 or FAILED_FINGERPRINT in checkpoint_fingerprints:
        raise HG232TimeoutResumeError("timeout_resume_checkpoint_mismatch")
    completed_fingerprints = {entry.fingerprint for entry in terminals if entry.status is RequestLedgerStatus.COMPLETED}
    if len(completed_fingerprints) != 8 or completed_fingerprints != checkpoint_fingerprints:
        raise HG232TimeoutResumeError("timeout_resume_completed_checkpoint_mismatch")
    config, config_path = _raw_config()
    design = HG232Approval.model_validate(tomllib.loads(_path(DEFAULT_DESIGN_APPROVAL).read_text(encoding="utf-8")))
    compatibility = HG232CompatibilityReceipt.model_validate_json(_path(DEFAULT_COMPATIBILITY_RECEIPT).read_text(encoding="utf-8"))
    if sha256_file(DEFAULT_COMPATIBILITY_RECEIPT) != design.compatibility_receipt_sha256 or compatibility.status != "completed":
        raise HG232TimeoutResumeError("timeout_resume_compatibility_mismatch")
    if sha256_file(PROBE_RECEIPT) != PROBE_RECEIPT_SHA256:
        raise HG232TimeoutResumeError("timeout_resume_probe_receipt_mismatch")
    return {
        "config": config,
        "config_path": config_path,
        "compatibility": compatibility,
        "completed_fingerprints": completed_fingerprints,
    }


def build_receipt() -> HG232TimeoutResumeReceipt:
    audit = _initial_audit()
    config = audit["config"]
    return HG232TimeoutResumeReceipt(
        amendment_id="provenance-cascade-hg232-timeout-resume-amendment-v3",
        status="offline_validated_pending_network_resume_authorization",
        resume_version=RESUME_VERSION,
        config_sha256=sha256_file(audit["config_path"]),
        protocol_sha256=config.protocol_sha256,
        execution_approval_sha256=sha256_file(DEFAULT_EXECUTION_APPROVAL),
        compatibility_receipt_sha256=sha256_file(DEFAULT_COMPATIBILITY_RECEIPT),
        probe_receipt_sha256=PROBE_RECEIPT_SHA256,
        historical_runner_sha256=HISTORICAL_RUNNER_SHA256,
        historical_probe_runner_sha256=HISTORICAL_PROBE_RUNNER_SHA256,
        resume_adapter_sha256=sha256_file(__file__),
        resume_helper_sha256=RESUME_HELPER_SHA256,
        amendment_document_sha256=sha256_file(AMENDMENT_DOCUMENT_PATH),
        initial_batch_record_sha256=INITIAL_BATCH_SHA256,
        failed_run_id=FAILED_RUN_ID,
        failed_fingerprint=FAILED_FINGERPRINT,
        failure_code="timeout",
        failure_attempt_count=1,
        failed_run_checkpoint_count=8,
        completed_run_count=7,
        failed_run_count=1,
        planned_run_count=8,
        completed_replay_passed_count=7,
        unique_logical_request_count=135,
        completed_logical_request_count=134,
        failed_logical_request_count=1,
        transport_attempt_count=135,
        remaining_new_logical_request_count=153,
        max_timeout_retry_count=1,
        total_logical_request_cap=288,
        completion_reservation_cap=589824,
        prompt_schema_model_seed_condition_unchanged=True,
        metrics_controller_unchanged=True,
        completed_fingerprint_replay_forbidden=True,
        parser_recovery_enabled=False,
        append_only_ledger_required=True,
        one_shot_initial_state_lock=True,
        results_modified=False,
        network="disabled",
        private_truth_exposed=False,
        development_only=True,
        not_paper_result=True,
        no_causal_conclusion=True,
    )


def write_receipt() -> tuple[HG232TimeoutResumeReceipt, str]:
    receipt = build_receipt()
    target = _path(DEFAULT_RECEIPT)
    content = receipt.model_dump_json(indent=2) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != content:
        raise HG232TimeoutResumeError("timeout_resume_receipt_conflict")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return receipt, sha256_file(target)


class TimeoutResumeFingerprintGuard:
    """Permit the failed timeout fingerprint once, then only unseen requests."""
    def __init__(self, provider: LLMProvider, *, preexisting_fingerprints: set[str]) -> None:
        self.provider = provider
        self.preexisting = frozenset(preexisting_fingerprints)
        self.calls = 0
        self.timeout_retry_consumed = False

    def complete(self, request: LLMRequest) -> LLMResponse:
        fingerprint = str(request_fingerprint_facts(request)["fingerprint"])
        if self.calls == 0:
            if fingerprint != FAILED_FINGERPRINT:
                raise LLMProviderError(ProviderErrorCode.INVALID_REQUEST, "timeout_resume_first_fingerprint_mismatch")
            self.timeout_retry_consumed = True
        elif fingerprint in self.preexisting or fingerprint == FAILED_FINGERPRINT:
            raise LLMProviderError(ProviderErrorCode.INVALID_REQUEST, "timeout_resume_preexisting_fingerprint_rejected")
        self.calls += 1
        return self.provider.complete(request)


def preflight() -> dict[str, object]:
    reasons: list[str] = []
    try:
        receipt, receipt_sha = write_receipt()
        approval_path = _path(DEFAULT_RESUME_APPROVAL)
        approval = HG232TimeoutResumeApproval.model_validate(tomllib.loads(approval_path.read_text(encoding="utf-8")))
        expected = {
            "amendment_receipt_sha256": receipt_sha,
            "resume_adapter_sha256": sha256_file(__file__),
            "initial_batch_record_sha256": INITIAL_BATCH_SHA256,
            "failed_fingerprint": FAILED_FINGERPRINT,
        }
        if any(getattr(approval, key) != value for key, value in expected.items()):
            reasons.append("timeout_resume_approval_binding_mismatch")
        if approval.acceptance_status != "accepted" or not approval.network_resume_authorized:
            reasons.append("timeout_resume_network_authorization_required")
        return {
            "status": "ready_for_timeout_resume" if not reasons else "blocked",
            "ready_for_timeout_resume": not reasons,
            "blocking_reasons": sorted(set(reasons)),
            "amendment_receipt_sha256": receipt_sha,
            "resume_adapter_sha256": sha256_file(__file__),
            "initial_batch_record_sha256": INITIAL_BATCH_SHA256,
            "failed_fingerprint": FAILED_FINGERPRINT,
            "completed_run_count": 7,
            "partial_run_completed_request_count": 8,
            "unique_logical_request_count": 135,
            "remaining_new_logical_request_count": 153,
            "total_logical_request_cap": 288,
            "completion_reservation_cap": 589824,
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_modified": False,
            "private_truth_exposed": False,
            "not_paper_result": True,
            "no_causal_conclusion": True,
        }
    except Exception as exc:
        return {
            "status":"blocked",
            "ready_for_timeout_resume":False,
            "blocking_reasons":[getattr(exc,"code","timeout_resume_preflight_failed")],
            "network":"disabled",
            "provider_constructed":False,
            "api_key_read":False,
            "results_modified":False,
        }


def execute_resume(
    *,
    allow_network: bool = False,
    confirm_resume: bool = False,
    confirm_request_cap: int | None = None,
    confirm_completion_reservation_cap: int | None = None,
    environment: Mapping[str, str] | None = None,
) -> HG21ExecutionSummary:
    checks = (
        (allow_network, "allow_network_required"),
        (confirm_resume, "confirm_resume_required"),
        (confirm_request_cap == 288, "confirm_request_cap_must_equal_288"),
        (confirm_completion_reservation_cap == 589824, "confirm_completion_reservation_cap_must_equal_589824"),
    )
    for valid, code in checks:
        if not valid:
            return HG21ExecutionSummary(status="blocked", error_code=code, safety=_safety())
    gate = preflight()
    if not gate["ready_for_timeout_resume"]:
        return HG21ExecutionSummary(status="blocked", error_code=gate["blocking_reasons"][0], safety=_safety())
    source = os.environ if environment is None else environment
    if not all(source.get(key) for key in ("EVICON_LLM_BASE_URL", "EVICON_LLM_MODEL", "EVICON_LLM_API_KEY")):
        return HG21ExecutionSummary(status="blocked", error_code="provider_environment_incomplete", safety=_safety())
    audit = _initial_audit()
    compatibility = audit["compatibility"]
    if source["EVICON_LLM_MODEL"] != compatibility.model:
        return HG21ExecutionSummary(status="blocked", error_code="provider_model_mismatch", safety=_safety())
    runner = _adapter_runner(audit["config"], audit["config_path"])
    base = ProviderConfig.from_env(allow_network=True, environment=source)
    constructed = 0

    def factory(spec: HG21RunSpec):
        nonlocal constructed
        constructed += 1
        settings = base.model_copy(update={
            "timeout_seconds": 15.0,
            "max_retries": audit["config"].calibration_max_retries,
            "temperature": audit["config"].temperature,
            "max_tokens": audit["config"].agent_max_tokens,
            "seed": spec.seed,
            "reasoning_effort": None,
            "response_format": ResponseFormatMode.JSON_SCHEMA,
            "response_schema_name": HG232_SCHEMA_NAME,
            "response_schema": HG232_RESPONSE_JSON_SCHEMA,
        })
        provider = OpenAICompatibleProvider(settings, environment=source)
        if spec.run_id == FAILED_RUN_ID:
            return TimeoutResumeFingerprintGuard(provider, preexisting_fingerprints=audit["completed_fingerprints"])
        return provider

    try:
        records, receipt = runner.run_all(
            provider_factory=factory,
            root=_path(ROOT),
            model_name=compatibility.model,
            resume=True,
            write_receipt=False,
            network="disabled",
        )
        final_receipt = receipt.model_copy(update={"network":"enabled", "results_written":True})
        target = _path(ROOT) / "calibration_receipt.json"
        if target.exists():
            raise HG232TimeoutResumeError("timeout_resume_calibration_receipt_exists")
        _atomic(target, final_receipt)
    except (CascadeRealAgentRunError, HG232TimeoutResumeError) as exc:
        return HG21ExecutionSummary(
            status="failed",
            error_code=getattr(exc,"code","timeout_resume_failed"),
            provider_constructed_count=constructed,
            network="enabled",
            results_written=True,
            safety=_safety(constructed > 0),
        )
    return HG21ExecutionSummary(
        status="completed",
        completed_run_count=len(records),
        logical_request_count=final_receipt.logical_request_count,
        transport_attempt_count=final_receipt.transport_attempt_count,
        actual_prompt_token_count=final_receipt.actual_prompt_token_count,
        actual_completion_token_count=final_receipt.actual_completion_token_count,
        actual_total_token_count=final_receipt.actual_total_token_count,
        directive_applied_count=final_receipt.directive_applied_count,
        replay_statuses=final_receipt.replay_statuses,
        receipt_path=str(target),
        provider_constructed_count=constructed,
        network="enabled",
        results_written=True,
        ready_for_network_authorization=True,
        safety=_safety(True),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="H-G.2.3.2 one-shot timeout resume v3")
    parser.add_argument("--mode", choices=("prepare", "preflight", "resume"), default="preflight")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-resume", action="store_true")
    parser.add_argument("--confirm-request-cap", type=int)
    parser.add_argument("--confirm-completion-reservation-cap", type=int)
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        _, digest = write_receipt()
        payload = {"status":"prepared", "amendment_receipt_sha256":digest, "network":"disabled", "results_modified":False}
    elif args.mode == "preflight":
        payload = preflight()
    else:
        payload = execute_resume(
            allow_network=args.allow_network,
            confirm_resume=args.confirm_resume,
            confirm_request_cap=args.confirm_request_cap,
            confirm_completion_reservation_cap=args.confirm_completion_reservation_cap,
        ).model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if payload.get("status") in {"prepared", "ready_for_timeout_resume", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
