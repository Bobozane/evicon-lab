"""H-G.2.1 technical execution amendment v2 with a locked Provider model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, model_validator

from .cascade_agent_protocol_hg21 import HG21_RESPONSE_JSON_SCHEMA, HG21_SCHEMA_NAME
from .cascade_hg1_replay import HG1OutcomeReplayValidator
from .cascade_real_agent_runner import CascadeRealAgentRunError, CascadeRealAgentRunRecord
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ResponseFormatMode
from .provenance_cascade_hg2 import HG2FakeProvider, sha256_file
from .provenance_cascade_hg21 import DEFAULT_APPROVAL, DEFAULT_COMPATIBILITY_RECEIPT, HG21Approval
from .provenance_cascade_hg21_calibration import (
    CalibrationStatus,
    HG21BatchRecord,
    HG21CalibrationReceipt,
    HG21EligibilityCalibrationRunner,
    HG21ExecutionApproval,
    HG21ExecutionSummary,
    HG21RunState,
    _atomic,
    _hash_json,
    load_decisions,
    _path,
    _safety,
)
from .provenance_cascade_hg21_compatibility_receipt import validate_receipt
from .request_ledger import RequestLedger

DEFAULT_CONFIG = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg21_calibration_execution_v2.toml"
DEFAULT_EXECUTION_APPROVAL = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg21_calibration_execution_v2_approval.toml"
DEFAULT_AMENDMENT_RECEIPT = "outputs/study-locks/provenance_cascade_hg21_execution_environment_amendment_v2.json"
PARENT_CONFIG = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg21_calibration.v1.toml"


class HG21ExecutionV2Error(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HG21ExecutionV2Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    study_id: Literal["evicon-provenance-cascade-hg21-eligibility-calibration-execution-v2"]
    config_version: Literal["provenance_cascade_hg21_execution_environment.v2"]
    status: Literal["pending_exact_hash_approval"]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    parent_config_path: str
    parent_config_sha256: str
    protocol_path: str
    protocol_sha256: str
    amendment_path: str
    amendment_sha256: str
    compatibility_receipt_path: str
    compatibility_receipt_sha256: str
    required_model_name: Literal["gpt-5.6-luna"]
    failed_v1_batch_record_path: str
    failed_v1_batch_record_sha256: str
    failed_v1_request_ledger_path: str
    failed_v1_request_ledger_sha256: str
    failed_v1_model_name: Literal["gemini-3.6-flash"]
    failed_v1_error_code: Literal["http_client_error"]
    failed_v1_completed_run_count: Literal[0]
    failed_v1_must_not_resume: Literal[True]
    prompt_semantics_changed: Literal[False]
    scenario_design_changed: Literal[False]
    controller_rules_changed: Literal[False]
    metric_definitions_changed: Literal[False]
    response_format: Literal["json_schema"]
    response_schema_name: Literal["cascade_agent_epistemic_behavior_response_v1_provider_subset"]
    seed: Literal[20261021]
    scenario_count: Literal[4]
    condition_count: Literal[4]
    agent_count: Literal[6]
    round_count: Literal[3]
    run_count: Literal[16]
    matched_group_count: Literal[4]
    logical_requests_per_run: Literal[18]
    request_cap: Literal[288]
    agent_max_tokens: Literal[1024]
    completion_reservation_cap: Literal[294912]
    temperature: Literal[0.2]
    calibration_max_retries: Literal[0]
    calibration_timeout_seconds: Literal[15]
    run_id_namespace: Literal["hg21v2"]
    output_root: Literal["results/provenance-cascade-hg21-eligibility-calibration-v2"]
    full_pilot_authorized: Literal[False]

    @model_validator(mode="after")
    def fixed_scope(self) -> "HG21ExecutionV2Config":
        if self.request_cap != self.run_count * self.logical_requests_per_run:
            raise ValueError("request cap mismatch")
        if self.completion_reservation_cap != self.request_cap * self.agent_max_tokens:
            raise ValueError("completion reservation mismatch")
        return self


class HG21ExecutionV2AmendmentReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    receipt_id: Literal["provenance-cascade-hg21-execution-environment-amendment-v2"]
    status: Literal["offline_validated_pending_exact_hash_approval"]
    config_sha256: str
    protocol_sha256: str
    runner_sha256: str
    amendment_sha256: str
    compatibility_receipt_sha256: str
    failed_v1_batch_record_sha256: str
    failed_v1_request_ledger_sha256: str
    failed_v1_preserved: Literal[True]
    failed_v1_must_not_resume: Literal[True]
    required_model_name: Literal["gpt-5.6-luna"]
    run_id_namespace: Literal["hg21v2"]
    output_root: Literal["results/provenance-cascade-hg21-eligibility-calibration-v2"]
    run_count: Literal[16]
    matched_group_count: Literal[4]
    logical_request_cap: Literal[288]
    completion_reservation_cap: Literal[294912]
    prompt_semantics_changed: Literal[False]
    scenario_design_changed: Literal[False]
    controller_rules_changed: Literal[False]
    metric_definitions_changed: Literal[False]
    network: Literal["disabled"]
    results_written: Literal[False]
    private_truth_exposed: Literal[False]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]


def load_config(path: str | Path = DEFAULT_CONFIG) -> tuple[HG21ExecutionV2Config, Path]:
    config_path = _path(path)
    try:
        config = HG21ExecutionV2Config.model_validate(tomllib.loads(config_path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise HG21ExecutionV2Error("execution_v2_config_invalid") from exc
    bindings = (
        (config.parent_config_path, config.parent_config_sha256),
        (config.protocol_path, config.protocol_sha256),
        (config.amendment_path, config.amendment_sha256),
        (config.compatibility_receipt_path, config.compatibility_receipt_sha256),
        (config.failed_v1_batch_record_path, config.failed_v1_batch_record_sha256),
        (config.failed_v1_request_ledger_path, config.failed_v1_request_ledger_sha256),
    )
    if any(sha256_file(_path(file_path)) != digest for file_path, digest in bindings):
        raise HG21ExecutionV2Error("execution_v2_binding_hash_mismatch")
    receipt = validate_receipt(config.compatibility_receipt_path, expected_hash=config.compatibility_receipt_sha256)
    if receipt.model != config.required_model_name:
        raise HG21ExecutionV2Error("compatibility_model_mismatch")
    return config, config_path


class HG21ExecutionV2Runner(HG21EligibilityCalibrationRunner):
    def __init__(self, config_path: str | Path = DEFAULT_CONFIG) -> None:
        technical, technical_path = load_config(config_path)
        super().__init__(PARENT_CONFIG)
        self.technical = technical
        self.config_path = technical_path
        self.amendment = self.amendment.model_copy(update={
            "study_id": technical.study_id,
            "output_root": technical.output_root,
        })
        self.runs = tuple(
            item.model_copy(update={
                "run_id": item.run_id.replace("hg21-", "hg21v2-", 1),
                "matched_group_id": item.matched_group_id.replace("hg21-", "hg21v2-", 1),
            })
            for item in self.runs
        )


    def _batch_binding_v2(self, root: Path, model_name: str, execution_approval_sha: str) -> str:
        return _hash_json({
            "config_sha256": sha256_file(self.config_path),
            "protocol_sha256": self.technical.protocol_sha256,
            "design_approval_sha256": sha256_file(DEFAULT_APPROVAL),
            "execution_approval_sha256": execution_approval_sha,
            "compatibility_receipt_sha256": self.technical.compatibility_receipt_sha256,
            "model": model_name,
            "output_root": str(root),
            "run_ids": [spec.run_id for spec in self.runs],
        })

    def run_all(
        self,
        *,
        provider_factory: Any,
        root: str | Path | None,
        model_name: str,
        resume: bool = False,
        write_receipt: bool = True,
        network: str = "enabled",
    ) -> tuple[list[CascadeRealAgentRunRecord], HG21CalibrationReceipt]:
        design = HG21Approval.model_validate(tomllib.loads(_path(DEFAULT_APPROVAL).read_text(encoding="utf-8")))
        validate_receipt(DEFAULT_COMPATIBILITY_RECEIPT, expected_hash=design.compatibility_receipt_sha256)
        execution_path = _path(DEFAULT_EXECUTION_APPROVAL)
        HG21ExecutionApproval.model_validate(tomllib.loads(execution_path.read_text(encoding="utf-8")))
        root_path = _path(self.technical.output_root) if root is None else Path(root).resolve()
        binding = self._batch_binding_v2(root_path, model_name, sha256_file(execution_path))
        batch_path = root_path / "calibration_batch_record.json"
        receipt_path = root_path / "calibration_receipt.json"
        if root_path.exists():
            if not resume:
                raise CascadeRealAgentRunError("output_root_exists")
            try:
                batch = HG21BatchRecord.model_validate_json(batch_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise CascadeRealAgentRunError("resume_batch_record_invalid") from exc
            if batch.binding_sha256 != binding or batch.model_name != model_name or batch.output_root != str(root_path):
                raise CascadeRealAgentRunError("resume_binding_mismatch")
        else:
            if resume:
                raise CascadeRealAgentRunError("resume_output_missing")
            root_path.mkdir(parents=True, exist_ok=False)
            batch = HG21BatchRecord(
                study_id=self.technical.study_id,
                status=CalibrationStatus.RUNNING,
                binding_sha256=binding,
                config_sha256=sha256_file(self.config_path),
                protocol_sha256=self.technical.protocol_sha256,
                controller_sha256=sha256_file("src/evicon/cascade_controller_hg1.py"),
                replay_sha256=sha256_file("src/evicon/cascade_hg1_replay.py"),
                amendment_receipt_sha256=sha256_file(DEFAULT_AMENDMENT_RECEIPT),
                design_approval_sha256=sha256_file(DEFAULT_APPROVAL),
                execution_approval_sha256=sha256_file(execution_path),
                compatibility_receipt_sha256=sha256_file(DEFAULT_COMPATIBILITY_RECEIPT),
                model_name=model_name,
                output_root=str(root_path),
                runs=tuple(
                    HG21RunState(
                        run_id=spec.run_id,
                        scenario_id=spec.scenario_id,
                        seed=spec.seed,
                        condition=spec.condition.value,
                    )
                    for spec in self.runs
                ),
            )
            _atomic(batch_path, batch)
        records: list[CascadeRealAgentRunRecord] = []
        all_decisions = []
        for spec in self.runs:
            state = next(item for item in batch.runs if item.run_id == spec.run_id)
            run_dir = root_path / spec.run_id
            record_path = run_dir / "run_record.json"
            if record_path.exists():
                if not resume:
                    raise CascadeRealAgentRunError("run_output_exists")
                try:
                    record = CascadeRealAgentRunRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
                    replay = HG1OutcomeReplayValidator.validate(
                        self.scenarios[spec.scenario_id].graph,
                        record.exposure_ledger,
                        record.outcome_ledger,
                        record.application_ledger,
                        record.round_contexts,
                    )
                    if record.run_id != spec.run_id or replay.status.value != "passed":
                        raise ValueError("binding")
                except Exception as exc:
                    raise CascadeRealAgentRunError("run_record_binding_mismatch") from exc
                decisions = load_decisions(run_dir / "behavior_decisions.jsonl", self._run_binding(spec, model_name))
                if len(decisions) != 18:
                    raise CascadeRealAgentRunError("behavior_decision_checkpoint_incomplete")
                records.append(record.model_copy(update={"replay": replay}))
                all_decisions.extend(decisions)
                continue
            batch = self._replace(
                batch,
                state.model_copy(update={"status": CalibrationStatus.RUNNING, "error_code": None}),
                CalibrationStatus.RUNNING,
                None,
            )
            _atomic(batch_path, batch)
            try:
                provider = provider_factory(spec)
                record, decisions = self.run_one(
                    spec,
                    provider=provider,
                    root=root_path,
                    model_name=model_name,
                    resume=resume,
                )
                if record.replay is None or record.replay.status.value != "passed":
                    raise CascadeRealAgentRunError("run_replay_incomplete")
            except CascadeRealAgentRunError as exc:
                batch = self._replace(
                    batch,
                    state.model_copy(update={"status": CalibrationStatus.FAILED, "error_code": exc.code}),
                    CalibrationStatus.FAILED,
                    exc.code,
                )
                _atomic(batch_path, batch)
                raise
            except Exception:
                batch = self._replace(
                    batch,
                    state.model_copy(update={"status": CalibrationStatus.FAILED, "error_code": "provider_construction_failed"}),
                    CalibrationStatus.FAILED,
                    "provider_construction_failed",
                )
                _atomic(batch_path, batch)
                raise CascadeRealAgentRunError("provider_construction_failed") from None
            safe = record.model_copy(update={"request_ledger_path": "request_ledger.jsonl"})
            if record_path.exists():
                raise CascadeRealAgentRunError("run_record_exists")
            _atomic(record_path, safe)
            records.append(safe)
            all_decisions.extend(decisions)
            summary = RequestLedger(run_dir / "request_ledger.jsonl").summary(
                request_cap=18,
                completion_reservation_cap=18432,
            )
            completed = state.model_copy(update={
                "status": CalibrationStatus.COMPLETED,
                "logical_request_count": summary.unique_logical_request_count,
                "transport_attempt_count": summary.transport_attempt_count,
                "directive_applied_count": safe.directive_applied_count,
                "behavior_observation_count": len(decisions),
                "replay_status": safe.replay.status.value,
                "run_record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
                "error_code": None,
            })
            batch = self._replace(batch, completed, CalibrationStatus.RUNNING, None)
            _atomic(batch_path, batch)
        if len(records) != 16 or len(all_decisions) != 288:
            raise CascadeRealAgentRunError("calibration_incomplete")
        totals = self._ledger_totals(root_path)
        if totals["logical_request_count"] != 288:
            raise CascadeRealAgentRunError("logical_request_count_mismatch")
        receipt = HG21CalibrationReceipt(
            study_id=self.technical.study_id,
            config_sha256=sha256_file(self.config_path),
            protocol_sha256=self.technical.protocol_sha256,
            template_sha256=self.amendment.parent_protocol_sha256,
            controller_sha256=sha256_file("src/evicon/cascade_controller_hg1.py"),
            replay_sha256=sha256_file("src/evicon/cascade_hg1_replay.py"),
            amendment_receipt_sha256=sha256_file(DEFAULT_AMENDMENT_RECEIPT),
            design_approval_sha256=sha256_file(DEFAULT_APPROVAL),
            execution_approval_sha256=sha256_file(execution_path),
            compatibility_receipt_sha256=sha256_file(DEFAULT_COMPATIBILITY_RECEIPT),
            run_count=16,
            completed_run_count=16,
            matched_group_count=4,
            logical_request_count=totals["logical_request_count"],
            provider_call_count=totals["provider_call_count"],
            transport_attempt_count=totals["transport_attempt_count"],
            actual_prompt_token_count=totals["actual_prompt_token_count"],
            actual_completion_token_count=totals["actual_completion_token_count"],
            actual_total_token_count=totals["actual_total_token_count"],
            replay_passed_count=16,
            replay_statuses={record.run_id: record.replay.status.value for record in records if record.replay},
            directive_applied_count=sum(record.directive_applied_count for record in records),
            behavior_observation_count=len(all_decisions),
            round0_non_defer_behavior_count=sum(
                item.round_id == 0 and item.behavioral_decision != "defer_action" for item in all_decisions
            ),
            share_decision_count=sum(item.share_requested for item in all_decisions),
            request_cap=288,
            completion_reservation_cap=294912,
            ledger_hash=_hash_json([
                (record.run_id, record.exposure_ledger_sha256, record.application_ledger_sha256, record.outcome_ledger_sha256)
                for record in records
            ]),
            network=network,
            results_written=write_receipt,
        )
        if write_receipt:
            if receipt_path.exists():
                existing = HG21CalibrationReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
                if existing != receipt:
                    raise CascadeRealAgentRunError("receipt_binding_mismatch")
                receipt = existing
            else:
                _atomic(receipt_path, receipt)
        batch = batch.model_copy(update={"status": CalibrationStatus.COMPLETED, "failure_code": None})
        _atomic(batch_path, batch)
        return records, receipt


def _validate_lock() -> tuple[HG21ExecutionV2Config, Path, HG21ExecutionApproval, HG21ExecutionV2AmendmentReceipt]:
    config, config_path = load_config()
    approval_path, receipt_path = _path(DEFAULT_EXECUTION_APPROVAL), _path(DEFAULT_AMENDMENT_RECEIPT)
    try:
        approval = HG21ExecutionApproval.model_validate(tomllib.loads(approval_path.read_text(encoding="utf-8")))
        receipt = HG21ExecutionV2AmendmentReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HG21ExecutionV2Error("execution_v2_governance_invalid") from exc
    expected_receipt = {
        "config_sha256": sha256_file(config_path),
        "protocol_sha256": config.protocol_sha256,
        "runner_sha256": sha256_file(__file__),
        "amendment_sha256": config.amendment_sha256,
        "compatibility_receipt_sha256": config.compatibility_receipt_sha256,
        "failed_v1_batch_record_sha256": config.failed_v1_batch_record_sha256,
        "failed_v1_request_ledger_sha256": config.failed_v1_request_ledger_sha256,
    }
    if any(getattr(receipt, key) != value for key, value in expected_receipt.items()):
        raise HG21ExecutionV2Error("execution_v2_amendment_receipt_mismatch")
    expected_approval = {
        "config_sha256": sha256_file(config_path),
        "protocol_sha256": config.protocol_sha256,
        "design_approval_sha256": sha256_file(DEFAULT_APPROVAL),
        "compatibility_receipt_sha256": config.compatibility_receipt_sha256,
        "runner_sha256": sha256_file(__file__),
        "run_count": 16,
        "matched_group_count": 4,
        "logical_request_cap": 288,
        "completion_reservation_cap": 294912,
    }
    if any(getattr(approval, key) != value for key, value in expected_approval.items()):
        raise HG21ExecutionV2Error("execution_v2_approval_binding_mismatch")
    return config, config_path, approval, receipt


def final_preflight() -> dict[str, object]:
    reasons: list[str] = []
    try:
        config, config_path, approval, receipt = _validate_lock()
        if approval.acceptance_status != "accepted":
            reasons.append("technical_execution_approval_required")
        if not approval.network_execution_authorized:
            reasons.append("calibration_network_authorization_required")
        if _path(config.output_root).exists():
            reasons.append("output_root_exists")
        return {
            "status": "ready_for_network_authorization" if not reasons else "blocked",
            "ready_for_network_authorization": not reasons,
            "blocking_reasons": sorted(set(reasons)),
            "required_model_name": config.required_model_name,
            "config_sha256": sha256_file(config_path),
            "protocol_sha256": config.protocol_sha256,
            "amendment_receipt_sha256": sha256_file(DEFAULT_AMENDMENT_RECEIPT),
            "execution_approval_sha256": sha256_file(DEFAULT_EXECUTION_APPROVAL),
            "failed_v1_preserved": receipt.failed_v1_preserved,
            "failed_v1_must_not_resume": receipt.failed_v1_must_not_resume,
            "run_count": 16,
            "matched_group_count": 4,
            "logical_request_cap": 288,
            "completion_reservation_cap": 294912,
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "private_truth_exposed": False,
            "development_only": True,
            "calibration_only": True,
            "not_paper_result": True,
            "no_causal_conclusion": True,
            "full_pilot_authorized": False,
        }
    except Exception as exc:
        return {
            "status": "blocked",
            "ready_for_network_authorization": False,
            "blocking_reasons": [getattr(exc, "code", "execution_v2_preflight_failed")],
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "private_truth_exposed": False,
        }


def run_fake_smoke() -> dict[str, object]:
    runner = HG21ExecutionV2Runner()
    with tempfile.TemporaryDirectory(prefix="evicon-hg21-v2-") as temporary:
        root = Path(temporary) / "calibration"
        records, receipt = runner.run_all(
            provider_factory=lambda _spec: HG2FakeProvider(),
            root=root,
            model_name=runner.technical.required_model_name,
            resume=False,
            write_receipt=False,
            network="disabled",
        )
        _, resumed = runner.run_all(
            provider_factory=lambda _spec: (_ for _ in ()).throw(AssertionError("completed run replayed")),
            root=root,
            model_name=runner.technical.required_model_name,
            resume=True,
            write_receipt=False,
            network="disabled",
        )
    return {
        "status": "fake_smoke_passed",
        "run_count": len(records),
        "matched_group_count": receipt.matched_group_count,
        "logical_request_count": receipt.logical_request_count,
        "provider_call_count": receipt.provider_call_count,
        "resume_additional_provider_call_count": resumed.provider_call_count - receipt.provider_call_count,
        "replay_passed_count": receipt.replay_passed_count,
        "directive_applied_count": receipt.directive_applied_count,
        "behavior_observation_count": receipt.behavior_observation_count,
        "required_model_name": runner.technical.required_model_name,
        "network": "disabled",
        "results_written": False,
        "private_truth_exposed": False,
        "effectiveness_claimed": False,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def execute_real(
    *,
    allow_network: bool = False,
    confirm_run: bool = False,
    confirm_request_cap: int | None = None,
    confirm_completion_reservation_cap: int | None = None,
    resume: bool = False,
    environment: Mapping[str, str] | None = None,
    provider_factory_override: Any | None = None,
) -> HG21ExecutionSummary:
    checks = (
        (allow_network, "allow_network_required"),
        (confirm_run, "confirm_run_required"),
        (confirm_request_cap == 288, "confirm_request_cap_must_equal_288"),
        (confirm_completion_reservation_cap == 294912, "confirm_completion_reservation_cap_must_equal_294912"),
    )
    for valid, code in checks:
        if not valid:
            return HG21ExecutionSummary(status="blocked", error_code=code, safety=_safety())
    gate = final_preflight()
    if not gate["ready_for_network_authorization"] and not (resume and gate["blocking_reasons"] == ["output_root_exists"]):
        return HG21ExecutionSummary(status="blocked", error_code=gate["blocking_reasons"][0], safety=_safety())
    source = os.environ if environment is None else environment
    if not all(source.get(name) for name in ("EVICON_LLM_BASE_URL", "EVICON_LLM_MODEL", "EVICON_LLM_API_KEY")):
        return HG21ExecutionSummary(status="blocked", error_code="provider_environment_incomplete", safety=_safety())
    config, _, _, _ = _validate_lock()
    if str(source["EVICON_LLM_MODEL"]) != config.required_model_name:
        return HG21ExecutionSummary(status="blocked", error_code="provider_model_mismatch", safety=_safety())
    runner, constructed = HG21ExecutionV2Runner(), 0
    base = None if provider_factory_override is not None else ProviderConfig.from_env(allow_network=True, environment=source)

    def factory(spec):
        nonlocal constructed
        constructed += 1
        if provider_factory_override is not None:
            return provider_factory_override(spec)
        settings = base.model_copy(update={
            "timeout_seconds": 15.0,
            "max_retries": 0,
            "temperature": 0.2,
            "max_tokens": 1024,
            "seed": spec.seed,
            "reasoning_effort": None,
            "response_format": ResponseFormatMode.JSON_SCHEMA,
            "response_schema_name": HG21_SCHEMA_NAME,
            "response_schema": HG21_RESPONSE_JSON_SCHEMA,
        })
        return OpenAICompatibleProvider(settings, environment=source)

    root = _path(config.output_root)
    try:
        records, receipt = runner.run_all(
            provider_factory=factory,
            root=None,
            model_name=config.required_model_name,
            resume=resume,
            write_receipt=True,
            network="enabled" if provider_factory_override is None else "disabled",
        )
    except CascadeRealAgentRunError as exc:
        return HG21ExecutionSummary(
            status="failed",
            error_code=exc.code,
            provider_constructed_count=constructed,
            network="enabled" if provider_factory_override is None else "disabled",
            results_written=root.exists(),
            safety=_safety(constructed > 0),
        )
    return HG21ExecutionSummary(
        status="completed",
        completed_run_count=len(records),
        logical_request_count=receipt.logical_request_count,
        transport_attempt_count=receipt.transport_attempt_count,
        actual_prompt_token_count=receipt.actual_prompt_token_count,
        actual_completion_token_count=receipt.actual_completion_token_count,
        actual_total_token_count=receipt.actual_total_token_count,
        directive_applied_count=receipt.directive_applied_count,
        behavior_observation_count=receipt.behavior_observation_count,
        replay_statuses=receipt.replay_statuses,
        receipt_path=str(root / "calibration_receipt.json"),
        provider_constructed_count=constructed,
        network="enabled" if provider_factory_override is None else "disabled",
        results_written=True,
        ready_for_network_authorization=True,
        safety=_safety(constructed > 0),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="H-G.2.1 execution-environment amendment v2")
    parser.add_argument("--mode", choices=("preflight", "fake-smoke", "real-calibration"), default="preflight")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--confirm-request-cap", type=int)
    parser.add_argument("--confirm-completion-reservation-cap", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "preflight":
        payload = final_preflight()
    elif args.mode == "fake-smoke":
        payload = run_fake_smoke()
    else:
        payload = execute_real(
            allow_network=args.allow_network,
            confirm_run=args.confirm_run,
            confirm_request_cap=args.confirm_request_cap,
            confirm_completion_reservation_cap=args.confirm_completion_reservation_cap,
            resume=args.resume,
        ).model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if payload.get("status") in {"completed", "fake_smoke_passed", "ready_for_network_authorization"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
