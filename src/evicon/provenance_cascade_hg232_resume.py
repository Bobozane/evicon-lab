"""Narrow audited timeout-resume adapter for the locked H-G.2.3.2 runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tomllib
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cascade_agent_protocol_hg2 import HG2DecisionTask
from .cascade_agent_protocol_hg232 import HG232_RESPONSE_JSON_SCHEMA, HG232_SCHEMA_NAME
from .cascade_controller import CascadeControllerPolicyLoader
from .cascade_hg1_replay import HG1CascadeProtocolRunner, HG1OutcomeReplayValidator
from .cascade_real_agent_runner import CascadeRealAgentRunError, CascadeRealAgentRunRecord, CascadeRealAgentRunner
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ResponseFormatMode
from .provenance_cascade_hg1 import _claim_selector, _post_round_events
from .provenance_cascade_hg2 import load_hg2_config, sha256_file
from .provenance_cascade_hg21_calibration import (
    HG21ExecutionSummary, HG21RunSpec, _path, _safety,
)
from .provenance_cascade_hg232 import (
    DEFAULT_APPROVAL, DEFAULT_COMPATIBILITY_RECEIPT, DEFAULT_CONFIG,
    HG232Approval, HG232Config,
)
from .provenance_cascade_hg232_calibration import (
    CONTROLLER_PATH, DEFAULT_EXECUTION_APPROVAL, PARENT_SCENARIO_CONFIG,
    POLICY_PATH, REPLAY_PATH, HG232CalibrationRunner,
)
from .provenance_cascade_hg232_compatibility_receipt import validate_receipt
from .provenance_cascade_hg232_stability_contract import HG232ProbeReceipt
from .provenance_cascade_preregistration import CascadeCondition
from .request_ledger import RequestLedger

RESUME_VERSION = "provenance_cascade_hg232_timeout_resume.v2"
PRE_RESUME_RUNNER_SHA256 = "041f85a0d8889377b6bb63a0237b5d212e93081b9cce7f3ab6cd4ad24f30a06f"
RESUME_RUNNER_SHA256 = "bae45f306d940a17103ed1f964b6023b9974ceff4039b1ababb33e83367ad035"
LOCKED_PROBE_RECEIPT_SHA256 = "5a1f1397b728797ef2e1c4a19bfaad3144eb2fd53538b7c1bd1e08df5987286a"
LOCKED_PROBE_RECEIPT_PATH = "results/provenance-cascade-hg232-protocol-stability-probe-v1/protocol_stability_probe_receipt.json"
LOCKED_BATCH_SHA256 = "9c7fc4313fad9833c6bb464e7e78f387aeb0458798eec016ab0c8dd86fcb5273"
DEFAULT_ROOT = "results/provenance-cascade-hg232-adoption-identifiability-v1"
DEFAULT_AMENDMENT_RECEIPT = "outputs/study-locks/provenance_cascade_hg232_timeout_resume_amendment_v2.json"
DEFAULT_RESUME_APPROVAL = "configs/provenance_cascade/pilot/provenance_cascade_hg232_timeout_resume_approval_v2.toml"
ORIGINAL_RUNNER_PATH = "src/evicon/provenance_cascade_hg232_calibration.py"


class HG232ResumeError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HG232ResumeAmendment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    amendment_id: Literal["provenance-cascade-hg232-timeout-resume-amendment-v2"]
    status: Literal["offline_validated_pending_exact_hash_approval"]
    resume_version: Literal["provenance_cascade_hg232_timeout_resume.v2"]
    config_sha256: str
    protocol_sha256: str
    execution_approval_sha256: str
    compatibility_receipt_sha256: str
    probe_receipt_sha256: str
    pre_resume_runner_sha256: str
    resume_runner_sha256: str
    resume_adapter_sha256: str
    initial_batch_record_sha256: str
    initial_batch_status: Literal["failed"]
    initial_failure_code: Literal["timeout"]
    completed_run_count: Literal[7]
    failed_run_count: Literal[1]
    planned_run_count: Literal[8]
    completed_replay_passed_count: Literal[7]
    unique_logical_request_count: Literal[135]
    transport_attempt_count: Literal[135]
    partial_run_completed_request_count: Literal[8]
    total_logical_request_cap: Literal[288]
    completion_reservation_cap: Literal[589824]
    changes_preflight_only: Literal[True]
    request_execution_behavior_unchanged: Literal[True]
    prompt_unchanged: Literal[True]
    schema_unchanged: Literal[True]
    model_seed_condition_unchanged: Literal[True]
    metrics_unchanged: Literal[True]
    completed_fingerprints_replayed: Literal[False]
    parser_recovery_enabled: Literal[False]
    network: Literal["disabled"]
    results_modified: Literal[False]
    private_truth_exposed: Literal[False]
    development_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]


class HG232ResumeApproval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    approval_id: Literal["provenance-cascade-hg232-timeout-resume-execution-v2"]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    amendment_receipt_sha256: str
    pre_resume_runner_sha256: str
    resume_runner_sha256: str
    resume_adapter_sha256: str
    initial_batch_record_sha256: str
    total_logical_request_cap: Literal[288]
    completion_reservation_cap: Literal[589824]
    confirm_timeout_only: bool
    confirm_completed_runs_not_replayed: bool
    confirm_append_only_ledger: bool
    confirm_no_parser_recovery: bool
    confirm_no_overwrite: bool
    network_resume_authorized: bool

    @model_validator(mode="after")
    def accepted(self) -> "HG232ResumeApproval":
        confirmations = (
            self.confirm_timeout_only, self.confirm_completed_runs_not_replayed,
            self.confirm_append_only_ledger, self.confirm_no_parser_recovery,
            self.confirm_no_overwrite, self.network_resume_authorized,
        )
        if self.acceptance_status == "accepted" and (not self.accepted_by.strip() or not all(confirmations)):
            raise ValueError("resume approval incomplete")
        return self


def _raw_config() -> tuple[HG232Config, Path]:
    path = _path(DEFAULT_CONFIG)
    try:
        config = HG232Config.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise HG232ResumeError("resume_config_invalid") from exc
    bindings = (
        (config.parent_config_path, config.parent_config_sha256),
        (config.parent_protocol_path, config.parent_protocol_sha256),
        (config.parent_approval_path, config.parent_approval_sha256),
        (config.parent_amendment_receipt_path, config.parent_amendment_receipt_sha256),
        (config.amendment_path, config.amendment_sha256),
        (config.protocol_path, config.protocol_sha256),
        (config.strict_parser_path, config.strict_parser_sha256),
        (config.parent_failure_diagnostic_path, config.parent_failure_diagnostic_sha256),
    )
    if any(not _path(name).is_file() or sha256_file(_path(name)) != digest for name, digest in bindings):
        raise HG232ResumeError("resume_config_binding_mismatch")
    return config, path


def _safe_batch_state(root: Path) -> dict[str, int | str]:
    batch_path = root / "calibration_batch_record.json"
    try:
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HG232ResumeError("resume_batch_record_invalid") from exc
    counts = {status: sum(item.get("status") == status for item in batch.get("runs", [])) for status in ("completed", "failed", "planned")}
    unique = attempts = passed = 0
    partial = 0
    base, scenarios, _ = load_hg2_config(_path(PARENT_SCENARIO_CONFIG))
    for item in batch.get("runs", []):
        run_dir = root / item["run_id"]
        ledger_path = run_dir / "request_ledger.jsonl"
        if ledger_path.exists():
            summary = RequestLedger(ledger_path).summary(request_cap=18, completion_reservation_cap=36864)
            unique += summary.unique_logical_request_count
            attempts += summary.transport_attempt_count
            if item.get("status") == "failed":
                partial = summary.completed_count
        record_path = run_dir / "run_record.json"
        if record_path.exists():
            record = CascadeRealAgentRunRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
            replay = HG1OutcomeReplayValidator.validate(
                scenarios[item["scenario_id"]].graph, record.exposure_ledger,
                record.outcome_ledger, record.application_ledger, record.round_contexts,
            )
            passed += replay.status.value == "passed"
    return {
        "status": batch.get("status"), "failure_code": batch.get("failure_code"),
        "completed": counts["completed"], "failed": counts["failed"], "planned": counts["planned"],
        "unique": unique, "attempts": attempts, "passed": passed, "partial": partial,
    }


def _adapter_runner(config: HG232Config, config_path: Path) -> HG232CalibrationRunner:
    runner = HG232CalibrationRunner.__new__(HG232CalibrationRunner)
    runner.config_lock, runner.config_path = config, config_path
    runner.base_config, runner.scenarios, runner.schedules = load_hg2_config(_path(PARENT_SCENARIO_CONFIG))
    runner.policy = CascadeControllerPolicyLoader.load(_path(POLICY_PATH))
    runner.protocol_runner = HG1CascadeProtocolRunner()
    runner.runner = CascadeRealAgentRunner(base_runner=runner.protocol_runner, policy_config=runner.policy)
    runner.tasks = {
        item.scenario_id: HG2DecisionTask.model_validate(item.model_dump(exclude={"scenario_id"}))
        for item in runner.base_config.decision_tasks
    }
    runner.runs = tuple(
        HG21RunSpec(
            run_id=f"hg232-{scenario_id}-{config.seed}-{condition.value}",
            matched_group_id=f"hg232-{scenario_id}-{config.seed}",
            scenario_id=scenario_id, seed=config.seed, condition=condition,
            completion_reservation=36864,
        )
        for scenario_id in runner.base_config.scenario_ids
        for condition in runner.base_config.conditions
    )
    return runner


def prepare_amendment() -> tuple[HG232ResumeAmendment, str]:
    config, config_path = _raw_config()
    root = _path(DEFAULT_ROOT)
    if not root.is_dir() or sha256_file(ORIGINAL_RUNNER_PATH) != RESUME_RUNNER_SHA256:
        raise HG232ResumeError("resume_locked_input_mismatch")
    if sha256_file(root / "calibration_batch_record.json") != LOCKED_BATCH_SHA256:
        raise HG232ResumeError("resume_initial_batch_hash_mismatch")
    state = _safe_batch_state(root)
    expected = {"status":"failed","failure_code":"timeout","completed":7,"failed":1,"planned":8,"unique":135,"attempts":135,"passed":7,"partial":8}
    if state != expected:
        raise HG232ResumeError("resume_initial_state_mismatch")
    design = HG232Approval.model_validate(tomllib.loads(_path(DEFAULT_APPROVAL).read_text(encoding="utf-8")))
    compatibility = validate_receipt(DEFAULT_COMPATIBILITY_RECEIPT, expected_hash=design.compatibility_receipt_sha256, allow_existing_output=True)
    probe_path = _path(LOCKED_PROBE_RECEIPT_PATH)
    if not probe_path.is_file() or sha256_file(probe_path) != LOCKED_PROBE_RECEIPT_SHA256:
        raise HG232ResumeError("resume_probe_receipt_mismatch")
    probe = HG232ProbeReceipt.model_validate_json(probe_path.read_text(encoding="utf-8"))
    if probe.status != "passed" or probe.network != "enabled":
        raise HG232ResumeError("resume_probe_receipt_invalid")
    amendment = HG232ResumeAmendment(
        amendment_id="provenance-cascade-hg232-timeout-resume-amendment-v2",
        status="offline_validated_pending_exact_hash_approval", resume_version=RESUME_VERSION,
        config_sha256=sha256_file(config_path), protocol_sha256=config.protocol_sha256,
        execution_approval_sha256=sha256_file(DEFAULT_EXECUTION_APPROVAL),
        compatibility_receipt_sha256=sha256_file(DEFAULT_COMPATIBILITY_RECEIPT),
        probe_receipt_sha256=LOCKED_PROBE_RECEIPT_SHA256,
        pre_resume_runner_sha256=PRE_RESUME_RUNNER_SHA256, resume_runner_sha256=RESUME_RUNNER_SHA256, resume_adapter_sha256=sha256_file(__file__),
        initial_batch_record_sha256=LOCKED_BATCH_SHA256, initial_batch_status="failed",
        initial_failure_code="timeout", completed_run_count=7, failed_run_count=1, planned_run_count=8,
        completed_replay_passed_count=7, unique_logical_request_count=135, transport_attempt_count=135,
        partial_run_completed_request_count=8, total_logical_request_cap=288,
        completion_reservation_cap=589824, changes_preflight_only=True, request_execution_behavior_unchanged=True,
        prompt_unchanged=True, schema_unchanged=True, model_seed_condition_unchanged=True,
        metrics_unchanged=True, completed_fingerprints_replayed=False, parser_recovery_enabled=False,
        network="disabled", results_modified=False, private_truth_exposed=False,
        development_only=True, not_paper_result=True, no_causal_conclusion=True,
    )
    target = _path(DEFAULT_AMENDMENT_RECEIPT)
    serialized = amendment.model_dump_json(indent=2) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != serialized:
        raise HG232ResumeError("resume_amendment_receipt_conflict")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(serialized, encoding="utf-8")
    return amendment, sha256_file(target)


def preflight() -> dict[str, object]:
    reasons: list[str] = []
    try:
        amendment, amendment_sha = prepare_amendment()
        approval_path = _path(DEFAULT_RESUME_APPROVAL)
        approval = HG232ResumeApproval.model_validate(tomllib.loads(approval_path.read_text(encoding="utf-8")))
        expected = {
            "amendment_receipt_sha256": amendment_sha,
            "pre_resume_runner_sha256": PRE_RESUME_RUNNER_SHA256,
            "resume_runner_sha256": RESUME_RUNNER_SHA256,
            "resume_adapter_sha256": sha256_file(__file__),
            "initial_batch_record_sha256": LOCKED_BATCH_SHA256,
        }
        if any(getattr(approval, key) != value for key, value in expected.items()):
            reasons.append("timeout_resume_approval_binding_mismatch")
        if approval.acceptance_status != "accepted" or not approval.network_resume_authorized:
            reasons.append("timeout_resume_authorization_required")
        return {
            "status": "ready_for_timeout_resume" if not reasons else "blocked",
            "ready_for_timeout_resume": not reasons,
            "blocking_reasons": sorted(set(reasons)),
            "amendment_receipt_sha256": amendment_sha,
            "resume_adapter_sha256": sha256_file(__file__),
            "initial_batch_record_sha256": LOCKED_BATCH_SHA256,
            "completed_run_count": 7, "partial_run_completed_request_count": 8,
            "unique_logical_request_count": 135, "remaining_logical_request_count": 153,
            "total_logical_request_cap": 288, "completion_reservation_cap": 589824,
            "network": "disabled", "provider_constructed": False, "api_key_read": False,
            "results_modified": False, "private_truth_exposed": False,
            "not_paper_result": True, "no_causal_conclusion": True,
        }
    except Exception as exc:
        return {"status":"blocked","ready_for_timeout_resume":False,"blocking_reasons":[getattr(exc,"code","timeout_resume_preflight_failed")],"network":"disabled","provider_constructed":False,"api_key_read":False,"results_modified":False}


def execute_resume(
    *, allow_network: bool = False, confirm_resume: bool = False,
    confirm_request_cap: int | None = None,
    confirm_completion_reservation_cap: int | None = None,
    environment: Mapping[str, str] | None = None,
) -> HG21ExecutionSummary:
    checks = (
        (allow_network,"allow_network_required"),(confirm_resume,"confirm_resume_required"),
        (confirm_request_cap==288,"confirm_request_cap_must_equal_288"),
        (confirm_completion_reservation_cap==589824,"confirm_completion_reservation_cap_must_equal_589824"),
    )
    for valid, code in checks:
        if not valid: return HG21ExecutionSummary(status="blocked",error_code=code,safety=_safety())
    gate = preflight()
    if not gate["ready_for_timeout_resume"]:
        return HG21ExecutionSummary(status="blocked",error_code=gate["blocking_reasons"][0],safety=_safety())
    source = os.environ if environment is None else environment
    if not all(source.get(name) for name in ("EVICON_LLM_BASE_URL","EVICON_LLM_MODEL","EVICON_LLM_API_KEY")):
        return HG21ExecutionSummary(status="blocked",error_code="provider_environment_incomplete",safety=_safety())
    config, config_path = _raw_config()
    design = HG232Approval.model_validate(tomllib.loads(_path(DEFAULT_APPROVAL).read_text(encoding="utf-8")))
    compatibility = validate_receipt(DEFAULT_COMPATIBILITY_RECEIPT, expected_hash=design.compatibility_receipt_sha256, allow_existing_output=True)
    if source["EVICON_LLM_MODEL"] != compatibility.model:
        return HG21ExecutionSummary(status="blocked",error_code="provider_model_mismatch",safety=_safety())
    runner, constructed = _adapter_runner(config, config_path), 0
    base = ProviderConfig.from_env(allow_network=True, environment=source)
    def factory(spec: HG21RunSpec):
        nonlocal constructed
        constructed += 1
        settings = base.model_copy(update={
            "timeout_seconds":15.0,"max_retries":config.calibration_max_retries,
            "temperature":config.temperature,"max_tokens":config.agent_max_tokens,"seed":spec.seed,
            "reasoning_effort":None,"response_format":ResponseFormatMode.JSON_SCHEMA,
            "response_schema_name":HG232_SCHEMA_NAME,"response_schema":HG232_RESPONSE_JSON_SCHEMA,
        })
        return OpenAICompatibleProvider(settings,environment=source)
    root = _path(DEFAULT_ROOT)
    try:
        records, receipt = runner.run_all(provider_factory=factory,root=root,model_name=compatibility.model,resume=True,write_receipt=True,network="enabled")
    except CascadeRealAgentRunError as exc:
        return HG21ExecutionSummary(status="failed",error_code=exc.code,provider_constructed_count=constructed,network="enabled",results_written=True,safety=_safety(constructed>0))
    return HG21ExecutionSummary(
        status="completed",completed_run_count=len(records),logical_request_count=receipt.logical_request_count,
        transport_attempt_count=receipt.transport_attempt_count,actual_prompt_token_count=receipt.actual_prompt_token_count,
        actual_completion_token_count=receipt.actual_completion_token_count,actual_total_token_count=receipt.actual_total_token_count,
        directive_applied_count=receipt.directive_applied_count,replay_statuses=receipt.replay_statuses,
        receipt_path=str(root/"calibration_receipt.json"),provider_constructed_count=constructed,
        network="enabled",results_written=True,ready_for_network_authorization=True,safety=_safety(True),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description="H-G.2.3.2 audited timeout resume")
    parser.add_argument("--mode",choices=("prepare","preflight","resume"),default="preflight")
    parser.add_argument("--allow-network",action="store_true")
    parser.add_argument("--confirm-resume",action="store_true")
    parser.add_argument("--confirm-request-cap",type=int)
    parser.add_argument("--confirm-completion-reservation-cap",type=int)
    args=parser.parse_args(argv)
    if args.mode=="prepare":
        amendment,digest=prepare_amendment(); payload={"status":"prepared","amendment_receipt_sha256":digest,"network":"disabled","results_modified":False}
    elif args.mode=="preflight": payload=preflight()
    else: payload=execute_resume(allow_network=args.allow_network,confirm_resume=args.confirm_resume,confirm_request_cap=args.confirm_request_cap,confirm_completion_reservation_cap=args.confirm_completion_reservation_cap).model_dump(mode="json")
    print(json.dumps(payload,ensure_ascii=True,sort_keys=True))
    return 0 if payload.get("status") in {"prepared","ready_for_timeout_resume","completed"} else 1


if __name__=="__main__": raise SystemExit(main())
