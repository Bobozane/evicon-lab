"""Read-only final integrity audit for the completed H-G.2.3.2 calibration."""
from __future__ import annotations

import argparse
import json
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .cascade_hg1_replay import HG1OutcomeReplayValidator
from .cascade_outcome_replay import CascadeOutcomeReplayStatus
from .cascade_real_agent_runner import CascadeAgentCheckpoint, CascadeRealAgentRunRecord
from .provenance_cascade_hg2 import load_hg2_config, sha256_file
from .provenance_cascade_hg21_calibration import (
    CalibrationStatus, HG21BatchRecord, HG21CalibrationReceipt, HG21RunSpec, _hash_json,
)
from .provenance_cascade_hg232 import DEFAULT_CONFIG, load_config
from .provenance_cascade_hg232_calibration import load_hg232_decisions
from .provenance_cascade_hg232_timeout_resume_v4 import (
    DEFAULT_RECEIPT as TECHNICAL_AMENDMENT_PATH,
    DEFAULT_RESUME_APPROVAL, FAILED_FINGERPRINT, HG232TimeoutResumeApproval,
    HG232TimeoutResumeReceipt,
)
from .provenance_cascade_preregistration import CascadeCondition
from .request_ledger import RequestLedger, RequestLedgerStatus

_ROOT = Path(__file__).resolve().parents[2]
PILOT_ROOT = _ROOT / "results/provenance-cascade-hg232-adoption-identifiability-v1"
BATCH_PATH = PILOT_ROOT / "calibration_batch_record.json"
CALIBRATION_RECEIPT_PATH = PILOT_ROOT / "calibration_receipt.json"
V4_RUNNER_PATH = _ROOT / "src/evicon/provenance_cascade_hg232_timeout_resume_v4.py"
AUDIT_RECEIPT_PATH = _ROOT / "outputs/study-locks/provenance_cascade_hg232_final_integrity_audit.json"
BASE_MATERIAL_CONFIG = _ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg2_calibration.v1.toml"
AUDIT_VERSION = "provenance_cascade_hg232_final_integrity_audit.v1"
EXPECTED_V4_RUNNER_SHA256 = "601f9d9a7ab8c411ba7a571fa674b51013965b1c2abb3ad1c38f1be54ac81e38"
EXPECTED_TECHNICAL_AMENDMENT_SHA256 = "6876796a0696773e19b4fb208ad654fccb7d4d419d140a6677ef200e9a5754de"
_FORBIDDEN = {
    "ground_truth_label", "source_independence_label", "system_prompt", "user_prompt",
    "prompt", "raw_response", "provider_metadata", "api_key", "authorization",
}


class HG232IntegrityAuditError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HG232FinalIntegrityAuditReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    receipt_id: Literal["provenance-cascade-hg232-final-integrity-audit-v1"]
    audit_version: Literal["provenance_cascade_hg232_final_integrity_audit.v1"]
    status: Literal["passed_development_evaluator_analysis_allowed"]
    study_id: str
    config_sha256: str
    protocol_sha256: str
    template_sha256: str
    controller_sha256: str
    replay_sha256: str
    v4_resume_runner_sha256: Literal[EXPECTED_V4_RUNNER_SHA256]
    technical_amendment_sha256: Literal[EXPECTED_TECHNICAL_AMENDMENT_SHA256]
    timeout_resume_approval_sha256: str
    final_batch_record_sha256: str
    calibration_receipt_sha256: str
    run_record_sha256: dict[str, str]
    run_count: Literal[16]
    matched_group_count: Literal[4]
    unique_logical_request_count: Literal[288]
    completed_logical_request_count: Literal[288]
    transport_attempt_count: Literal[289]
    failed_transport_attempt_count: Literal[1]
    recovered_timeout_fingerprint_count: Literal[1]
    approved_timeout_fingerprint: Literal[FAILED_FINGERPRINT]
    approved_fingerprint_transport_attempt_count: Literal[2]
    duplicate_completed_fingerprint_count: Literal[0]
    orphaned_started_attempt_count: Literal[0]
    behavior_decision_count: Literal[288]
    checkpoint_entry_count: Literal[288]
    cascade_replay_passed_count: Literal[16]
    application_replay_passed_count: Literal[16]
    outcome_replay_passed_count: Literal[16]
    actual_prompt_token_count: Literal[215539]
    actual_completion_token_count: Literal[87634]
    actual_total_token_count: Literal[303173]
    prompt_or_output_contract_changed_by_resume: Literal[False]
    completed_fingerprint_replayed: Literal[False]
    parser_recovery_enabled: Literal[False]
    engineering_integrity_passed: Literal[True]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    private_truth_loaded: Literal[False]
    network_used_for_audit: Literal[False]


def _require(value: bool, code: str) -> None:
    if not value:
        raise HG232IntegrityAuditError(code)


def _safe(value: object) -> None:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True).lower()
    _require(not any(f'"{key}"' in payload for key in _FORBIDDEN), "sensitive_public_artifact_field")


def _read_model(path: Path, model: type[BaseModel], code: str) -> BaseModel:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HG232IntegrityAuditError(code) from exc


def _specs(scenario_ids: tuple[str, ...], seed: int) -> tuple[HG21RunSpec, ...]:
    return tuple(
        HG21RunSpec(
            run_id=f"hg232-{scenario_id}-{seed}-{condition.value}",
            matched_group_id=f"hg232-{scenario_id}-{seed}",
            scenario_id=scenario_id, seed=seed, condition=condition,
            completion_reservation=36864,
        )
        for scenario_id in scenario_ids for condition in CascadeCondition
    )


def _run_binding(spec: HG21RunSpec, *, config_sha256: str, protocol_sha256: str,
                 template_version: str, model_name: str, max_tokens: int, temperature: float) -> str:
    return _hash_json({
        "config_sha256": config_sha256, "protocol_sha256": protocol_sha256,
        "template_version": template_version, "scenario_id": spec.scenario_id,
        "seed": spec.seed, "condition": spec.condition.value, "run_id": spec.run_id,
        "model_name": model_name, "max_tokens": max_tokens, "temperature": temperature,
    })


def audit_hg232_integrity() -> HG232FinalIntegrityAuditReceipt:
    config, config_path = load_config(DEFAULT_CONFIG, allow_existing_output=True)
    base, scenarios, _ = load_hg2_config(BASE_MATERIAL_CONFIG)
    specs = _specs(tuple(base.scenario_ids), config.seed)
    receipt = _read_model(CALIBRATION_RECEIPT_PATH, HG21CalibrationReceipt, "calibration_receipt_invalid")
    batch = _read_model(BATCH_PATH, HG21BatchRecord, "batch_record_invalid")
    amendment_path = _ROOT / TECHNICAL_AMENDMENT_PATH
    amendment = _read_model(amendment_path, HG232TimeoutResumeReceipt, "technical_amendment_invalid")
    approval_path = _ROOT / DEFAULT_RESUME_APPROVAL
    approval = HG232TimeoutResumeApproval.model_validate(tomllib.loads(approval_path.read_text(encoding="utf-8")))
    assert isinstance(receipt, HG21CalibrationReceipt) and isinstance(batch, HG21BatchRecord)
    assert isinstance(amendment, HG232TimeoutResumeReceipt)
    _require(sha256_file(V4_RUNNER_PATH) == EXPECTED_V4_RUNNER_SHA256, "v4_runner_hash_mismatch")
    _require(sha256_file(amendment_path) == EXPECTED_TECHNICAL_AMENDMENT_SHA256, "technical_amendment_hash_mismatch")
    _require(approval.acceptance_status == "accepted" and approval.network_resume_authorized, "timeout_resume_approval_invalid")
    _require(approval.amendment_receipt_sha256 == EXPECTED_TECHNICAL_AMENDMENT_SHA256, "timeout_resume_approval_binding_mismatch")
    expected = {
        "config_sha256": sha256_file(config_path), "protocol_sha256": sha256_file(_ROOT / config.protocol_path),
        "template_sha256": sha256_file(_ROOT / config.protocol_path),
        "controller_sha256": sha256_file(_ROOT / "src/evicon/cascade_controller_hg1.py"),
        "replay_sha256": sha256_file(_ROOT / "src/evicon/cascade_hg1_replay.py"),
    }
    for field, digest in expected.items():
        _require(getattr(receipt, field) == digest, f"receipt_{field}_mismatch")
        if hasattr(batch, field):
            _require(getattr(batch, field) == digest, f"batch_{field}_mismatch")
    _require(receipt.status == "eligibility_calibration_completed" and batch.status is CalibrationStatus.COMPLETED, "calibration_incomplete")
    _require(receipt.run_count == receipt.completed_run_count == len(batch.runs) == len(specs) == 16, "run_count_mismatch")
    _require(receipt.matched_group_count == 4 and receipt.replay_passed_count == 16, "matched_or_replay_count_mismatch")

    states = {item.run_id: item for item in batch.runs}
    _require(len(states) == 16 and set(states) == {item.run_id for item in specs}, "run_set_incomplete")
    global_groups: dict[str, list] = defaultdict(list)
    run_hashes: dict[str, str] = {}
    decision_count = checkpoint_count = starts_total = failed_total = duplicate_completed = orphaned = 0
    completed_global: set[str] = set()
    records: list[CascadeRealAgentRunRecord] = []
    approved_attempts = 0
    for spec in specs:
        state = states[spec.run_id]
        global_groups[spec.matched_group_id].append(spec.condition.value)
        _require(state.status is CalibrationStatus.COMPLETED and state.error_code is None, "run_state_incomplete")
        _require(state.logical_request_count == 18 and state.replay_status == "passed", "run_state_count_or_replay_mismatch")
        run_dir = PILOT_ROOT / spec.run_id
        record_path = run_dir / "run_record.json"
        _require(sha256_file(record_path) == state.run_record_sha256, "run_record_hash_mismatch")
        record = _read_model(record_path, CascadeRealAgentRunRecord, "run_record_invalid")
        assert isinstance(record, CascadeRealAgentRunRecord)
        _safe(record.model_dump(mode="json"))
        _require(record.run_id == spec.run_id and record.scenario_id == spec.scenario_id and record.seed == spec.seed and record.condition is spec.condition, "run_coordinate_mismatch")
        _require(record.status == "completed" and record.error_code is None, "run_record_incomplete")
        _require(record.exposure_ledger is not None and record.application_ledger is not None and record.outcome_ledger is not None, "run_record_ledger_missing")
        replay = HG1OutcomeReplayValidator.validate(scenarios[spec.scenario_id].graph, record.exposure_ledger, record.outcome_ledger, record.application_ledger, record.round_contexts)
        _require(replay.status is CascadeOutcomeReplayStatus.PASSED, "outcome_replay_not_passed")
        _require(replay.cascade_replay.status.value == "passed", "cascade_replay_not_passed")
        _require(replay.application_replay.status.value == "passed", "application_replay_not_passed")

        entries = RequestLedger(run_dir / "request_ledger.jsonl").entries()
        _safe([item.model_dump(mode="json") for item in entries])
        by_fingerprint: dict[str, list] = defaultdict(list)
        for item in entries:
            by_fingerprint[item.fingerprint].append(item)
        _require(len(by_fingerprint) == 18, "logical_request_count_mismatch")
        local_completed: set[str] = set()
        local_starts = 0
        for fingerprint, rows in by_fingerprint.items():
            started = [item for item in rows if item.status is RequestLedgerStatus.STARTED]
            terminal = [item for item in rows if item.status is not RequestLedgerStatus.STARTED]
            attempts = {item.attempt_count for item in started}
            _require(attempts == set(range(1, len(started) + 1)), "attempt_sequence_invalid")
            _require(len(terminal) == len(started) and {item.attempt_count for item in terminal} == attempts, "orphaned_started_attempt")
            completed = [item for item in terminal if item.status is RequestLedgerStatus.COMPLETED]
            failed = [item for item in terminal if item.status is RequestLedgerStatus.FAILED]
            duplicate_completed += max(0, len(completed) - 1)
            _require(len(completed) == 1 and completed[0].attempt_count == max(attempts), "logical_request_not_finally_completed")
            if fingerprint == FAILED_FINGERPRINT:
                approved_attempts = len(started)
                _require(len(started) == 2 and len(failed) == 1 and failed[0].error_code == "timeout" and failed[0].attempt_count == 1 and completed[0].attempt_count == 2, "approved_timeout_recovery_mismatch")
            else:
                _require(len(started) == 1 and not failed, "unapproved_transport_recovery")
            local_starts += len(started)
            failed_total += len(failed)
            local_completed.add(fingerprint)
        _require(not (completed_global & local_completed), "cross_run_fingerprint_reuse")
        completed_global.update(local_completed)
        starts_total += local_starts
        _require(local_starts == state.transport_attempt_count, "run_transport_attempt_count_mismatch")
        binding = _run_binding(spec, config_sha256=expected["config_sha256"], protocol_sha256=config.protocol_sha256,
            template_version=config.template_version, model_name=batch.model_name, max_tokens=config.agent_max_tokens,
            temperature=config.temperature)
        decisions = load_hg232_decisions(run_dir / "behavior_decisions.jsonl", binding)
        checkpoint = _read_model(run_dir / "agent_checkpoint.json", CascadeAgentCheckpoint, "checkpoint_invalid")
        assert isinstance(checkpoint, CascadeAgentCheckpoint)
        _require(len(decisions) == len(checkpoint.entries) == 18, "decision_checkpoint_incomplete")
        _require({item.fingerprint for item in decisions} == local_completed, "decision_fingerprint_mismatch")
        _require({item.fingerprint for item in checkpoint.entries} == local_completed, "checkpoint_fingerprint_mismatch")
        decision_count += len(decisions); checkpoint_count += len(checkpoint.entries)
        run_hashes[spec.run_id] = sha256_file(record_path); records.append(record)

    required_conditions = {item.value for item in CascadeCondition}
    _require(len(global_groups) == 4 and all(set(items) == required_conditions and len(items) == 4 for items in global_groups.values()), "matched_group_incomplete")
    _require(len(completed_global) == 288 and starts_total == 289 and failed_total == 1, "aggregate_request_count_mismatch")
    _require(approved_attempts == 2 and duplicate_completed == 0 and orphaned == 0, "completed_fingerprint_replayed")
    _require(receipt.logical_request_count == 288 and receipt.transport_attempt_count == 289, "receipt_request_count_mismatch")
    _require(receipt.actual_prompt_token_count == 215539 and receipt.actual_completion_token_count == 87634 and receipt.actual_total_token_count == 303173, "receipt_token_count_mismatch")
    _require(_hash_json([(item.run_id, item.exposure_ledger_sha256, item.application_ledger_sha256, item.outcome_ledger_sha256) for item in records]) == receipt.ledger_hash, "receipt_run_aggregate_hash_mismatch")

    result = HG232FinalIntegrityAuditReceipt(
        receipt_id="provenance-cascade-hg232-final-integrity-audit-v1", audit_version=AUDIT_VERSION,
        status="passed_development_evaluator_analysis_allowed", study_id=config.study_id,
        config_sha256=expected["config_sha256"], protocol_sha256=expected["protocol_sha256"], template_sha256=expected["template_sha256"],
        controller_sha256=expected["controller_sha256"], replay_sha256=expected["replay_sha256"],
        v4_resume_runner_sha256=EXPECTED_V4_RUNNER_SHA256, technical_amendment_sha256=EXPECTED_TECHNICAL_AMENDMENT_SHA256,
        timeout_resume_approval_sha256=sha256_file(approval_path), final_batch_record_sha256=sha256_file(BATCH_PATH),
        calibration_receipt_sha256=sha256_file(CALIBRATION_RECEIPT_PATH), run_record_sha256=run_hashes,
        run_count=16, matched_group_count=4, unique_logical_request_count=288, completed_logical_request_count=288,
        transport_attempt_count=289, failed_transport_attempt_count=1, recovered_timeout_fingerprint_count=1,
        approved_timeout_fingerprint=FAILED_FINGERPRINT, approved_fingerprint_transport_attempt_count=2,
        duplicate_completed_fingerprint_count=0, orphaned_started_attempt_count=0,
        behavior_decision_count=decision_count, checkpoint_entry_count=checkpoint_count,
        cascade_replay_passed_count=16, application_replay_passed_count=16, outcome_replay_passed_count=16,
        actual_prompt_token_count=215539, actual_completion_token_count=87634, actual_total_token_count=303173,
        prompt_or_output_contract_changed_by_resume=False, completed_fingerprint_replayed=False,
        parser_recovery_enabled=False, engineering_integrity_passed=True, development_only=True,
        calibration_only=True, not_paper_result=True, no_causal_conclusion=True,
        private_truth_loaded=False, network_used_for_audit=False,
    )
    _safe(result.model_dump(mode="json"))
    return result


def write_audit_receipt(path: str | Path = AUDIT_RECEIPT_PATH) -> HG232FinalIntegrityAuditReceipt:
    receipt = audit_hg232_integrity(); destination = Path(path)
    if destination.exists():
        existing = HG232FinalIntegrityAuditReceipt.model_validate_json(destination.read_text(encoding="utf-8"))
        if existing != receipt:
            raise HG232IntegrityAuditError("integrity_audit_receipt_binding_mismatch")
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(receipt.model_dump_json(indent=2) + "\n", encoding="utf-8"); temporary.replace(destination)
    return receipt


def safe_summary(receipt: HG232FinalIntegrityAuditReceipt, path: Path | None = None) -> dict[str, object]:
    return {"status": receipt.status, "run_count": receipt.run_count, "matched_group_count": receipt.matched_group_count,
        "unique_logical_request_count": receipt.unique_logical_request_count, "transport_attempt_count": receipt.transport_attempt_count,
        "approved_fingerprint_transport_attempt_count": receipt.approved_fingerprint_transport_attempt_count,
        "three_layer_replay_passed_count": receipt.outcome_replay_passed_count,
        "receipt_sha256": sha256_file(path) if path and path.exists() else None, "network": "disabled",
        "private_truth_loaded": False, "not_paper_result": True, "no_causal_conclusion": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only final H-G.2.3.2 integrity audit.")
    parser.add_argument("--write-receipt", action="store_true"); args = parser.parse_args(argv)
    try:
        receipt = write_audit_receipt() if args.write_receipt else audit_hg232_integrity()
        print(json.dumps(safe_summary(receipt, AUDIT_RECEIPT_PATH if args.write_receipt else None), sort_keys=True)); return 0
    except HG232IntegrityAuditError as exc:
        print(json.dumps({"status":"blocked","error_code":exc.code,"network":"disabled","private_truth_loaded":False}, sort_keys=True)); return 1


if __name__ == "__main__":
    raise SystemExit(main())
