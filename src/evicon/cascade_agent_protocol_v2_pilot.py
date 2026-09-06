"""Isolated H-E pilot boundary for the H-D.2 strict Agent protocol.

This module is deliberately a sidecar.  It reuses the existing public
CascadeRealAgentRunner and request ledger, but binds them to the immutable
H-D.2 contract and never enables parser recovery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import tomllib
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .cascade_agent_protocol_v2 import (
    CascadeAgentProtocolV2Runtime,
    HD2Config,
    HD2ConfigError,
    HD2RunSpec,
    V2_SCHEMA_NAME,
    V2_TEMPLATE_VERSION,
    load_hd2_config,
)
from .cascade_agent_protocol_v2_smoke import V2FakeProvider
from .cascade_agent_protocol_v2_receipt import CompatibilityReceiptError, validate_receipt
from .cascade_protocol import CascadeScenarioLoader
from .cascade_real_agent_runner import CascadeRealAgentRunError, CascadeRealAgentRunRecord, CascadeRealAgentRunner
from .llm_contract import LLMProvider
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ResponseFormatMode
from .provenance_cascade_amendment import load_hd_config
from .request_ledger import RequestLedger

_ROOT = Path(__file__).resolve().parents[2]
HD2_CONFIG_RELATIVE = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd2.v1.toml"
HD2_PROTOCOL_RELATIVE = "src/evicon/cascade_agent_protocol_v2.py"
# The versioned template is defined by the protocol module; bind both explicitly
# so a template drift cannot pass the final gate under the same protocol hash.
HD2_TEMPLATE_RELATIVE = HD2_PROTOCOL_RELATIVE
HD2_OUTPUT_ROOT = "results/provenance-cascade-pilot-hd2-v1"
COMPATIBILITY_RECEIPT_RELATIVE = "outputs/study-locks/provenance_cascade_agent_v2_compatibility_receipt.json"
APPROVAL_VERSION = "provenance_cascade_hd2_researcher_approval.v1"


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HDEApproval(BaseModel):
    """Strict, researcher-owned design approval; separate CLI flags authorize network use."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: str = Field(min_length=1)
    approval_version: str = APPROVAL_VERSION
    acceptance_status: str
    accepted_by: str | None = None
    accepted_on: str | None = None
    acceptance_scope: str
    config_path: str
    config_sha256: str
    protocol_path: str
    protocol_sha256: str
    template_path: str = HD2_TEMPLATE_RELATIVE
    template_sha256: str = ""
    compatibility_receipt_path: str | None = None
    compatibility_receipt_sha256: str | None = None
    confirm_hd2_config: bool = False
    confirm_protocol: bool = False
    confirm_template_hash: bool = False
    confirm_response_format: bool = False
    confirm_compatibility_receipt: bool = False
    confirm_scope_and_budget: bool = False
    confirm_run_scope: bool = False
    confirm_logical_request_scope: bool = False
    confirm_completion_reservation: bool = False
    confirm_retry_timeout: bool = False
    confirm_output_root: bool = False
    confirm_no_overwrite: bool = False
    confirm_append_only_ledger: bool = False
    confirm_resume: bool = False
    confirm_legacy_exclusion: bool = False
    network_execution_authorized: bool = False

    @model_validator(mode="after")
    def accepted_fields(self) -> "HDEApproval":
        if self.acceptance_status not in {"pending", "accepted"}:
            raise ValueError("acceptance_status_invalid")
        if self.acceptance_status == "accepted":
            if not self.accepted_by or not self.accepted_on:
                raise ValueError("accepted_identity_missing")
            try:
                date.fromisoformat(self.accepted_on)
            except ValueError as exc:
                raise ValueError("accepted_on_invalid") from exc
        for value in (self.config_sha256, self.protocol_sha256, self.template_sha256):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
                raise ValueError("approval_hash_invalid")
        if self.compatibility_receipt_path and self.compatibility_receipt_sha256:
            if len(self.compatibility_receipt_sha256) != 64 or any(
                char not in "0123456789abcdef" for char in self.compatibility_receipt_sha256.lower()
            ):
                raise ValueError("compatibility_hash_invalid")
        return self

    @property
    def ready(self) -> bool:
        return self.acceptance_status == "accepted" and all(
            (
                self.accepted_by,
                self.accepted_on,
                self.confirm_hd2_config,
                self.confirm_protocol,
                self.confirm_template_hash,
                self.confirm_response_format,
                self.confirm_compatibility_receipt,
                self.confirm_scope_and_budget,
                self.confirm_run_scope,
                self.confirm_logical_request_scope,
                self.confirm_completion_reservation,
                self.confirm_retry_timeout,
                self.confirm_output_root,
                self.confirm_no_overwrite,
                self.confirm_append_only_ledger,
                self.confirm_resume,
                self.confirm_legacy_exclusion,
                self.compatibility_receipt_path,
                self.compatibility_receipt_sha256,
            )
        )


class HDEPilotReceipt(BaseModel):
    """Safe receipt emitted only after every registered run passes replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    study_id: str
    config_sha256: str
    protocol_sha256: str
    template_sha256: str = ""
    template_version: str = V2_TEMPLATE_VERSION
    response_format: str = "json_schema"
    schema_name: str = V2_SCHEMA_NAME
    run_count: int = Field(ge=0)
    completed_run_count: int = Field(default=0, ge=0)
    failed_run_count: int = Field(default=0, ge=0)
    matched_group_count: int = Field(ge=0)
    logical_request_count: int = Field(ge=0)
    provider_call_count: int = Field(ge=0)
    transport_attempt_count: int = Field(default=0, ge=0)
    actual_prompt_token_count: int | None = Field(default=None, ge=0)
    actual_completion_token_count: int | None = Field(default=None, ge=0)
    actual_total_token_count: int | None = Field(default=None, ge=0)
    replay_passed_count: int = Field(ge=0)
    replay_statuses: dict[str, str] = Field(default_factory=dict)
    directive_applied_count: int = Field(ge=0)
    request_cap: int = Field(gt=0)
    completion_reservation_cap: int = Field(gt=0)
    ledger_hash: str
    network: str
    results_written: bool
    old_pilot_touched: bool
    private_truth_exposed: bool
    pilot_only: bool = True
    parser_recovery_enabled: bool = False
    safety: dict[str, bool]


class HD2PilotRunStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class HD2PilotRunState(BaseModel):
    """Content-free state for one registered run coordinate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    scenario_id: str
    seed: int
    condition: str
    status: HD2PilotRunStatus = HD2PilotRunStatus.PLANNED
    logical_request_count: int = Field(default=0, ge=0)
    transport_attempt_count: int = Field(default=0, ge=0)
    actual_prompt_token_count: int | None = Field(default=None, ge=0)
    actual_completion_token_count: int | None = Field(default=None, ge=0)
    actual_total_token_count: int | None = Field(default=None, ge=0)
    directive_applied_count: int = Field(default=0, ge=0)
    cascade_replay_status: str | None = None
    application_replay_status: str | None = None
    outcome_replay_status: str | None = None
    run_record_sha256: str | None = None
    error_code: str | None = None


class HD2PilotBatchRecord(BaseModel):
    """Safe resumable batch state; request ledgers remain append-only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_version: str = "provenance_cascade_hd2_execution.v1"
    study_id: str
    status: HD2PilotRunStatus
    binding_sha256: str = Field(min_length=64, max_length=64)
    config_sha256: str = Field(min_length=64, max_length=64)
    protocol_sha256: str = Field(min_length=64, max_length=64)
    template_sha256: str = Field(min_length=64, max_length=64)
    approval_sha256: str | None = None
    compatibility_receipt_sha256: str | None = None
    model_name: str
    output_root: str
    runs: tuple[HD2PilotRunState, ...]
    failure_code: str | None = None
    development_only: bool = True
    pilot_only: bool = True
    not_paper_result: bool = True
    no_causal_conclusion: bool = True
    private_truth_exposed: bool = False


class HD2PilotExecutionSummary(BaseModel):
    """Safe CLI result for completed, failed, or blocked H-D.2 execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    error_code: str | None = None
    completed_run_count: int = Field(default=0, ge=0)
    failed_run_count: int = Field(default=0, ge=0)
    logical_request_count: int = Field(default=0, ge=0)
    transport_attempt_count: int = Field(default=0, ge=0)
    actual_prompt_token_count: int | None = Field(default=None, ge=0)
    actual_completion_token_count: int | None = Field(default=None, ge=0)
    actual_total_token_count: int | None = Field(default=None, ge=0)
    directive_applied_count: int = Field(default=0, ge=0)
    replay_statuses: dict[str, str] = Field(default_factory=dict)
    receipt_path: str | None = None
    provider_constructed_count: int = Field(default=0, ge=0)
    network: str
    results_written: bool
    safety: dict[str, bool]

def load_approval(path: str | Path) -> HDEApproval:
    try:
        payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        return HDEApproval.model_validate(payload)
    except (OSError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise ValueError("approval_invalid") from exc


def _load_scenarios(config_path: Path, config: HD2Config) -> dict[str, Any]:
    parent_path = (config_path.parent / config.parent_config_path).resolve()
    parent, _ = load_hd_config(parent_path)
    materials = {item.scenario_id: item for item in parent.scenario_materials}
    scenarios: dict[str, Any] = {}
    for scenario_id in config.scenario_ids:
        material = materials[scenario_id]
        scenario_path = parent_path.parent / material.config_path
        scenarios[scenario_id] = CascadeScenarioLoader.load(scenario_path)
    return scenarios


def _receipt_path_value(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (_ROOT / candidate).resolve()


def _safe_compatibility_ok(path: Path, expected_hash: str | None) -> tuple[bool, str]:
    """Validate only the safe, bound H-D.3.2 receipt fields."""
    try:
        validate_receipt(
            path,
            expected_hash=expected_hash,
            config_path=_ROOT / HD2_CONFIG_RELATIVE,
            protocol_path=_ROOT / HD2_PROTOCOL_RELATIVE,
        )
        return True, ""
    except CompatibilityReceiptError as exc:
        return False, exc.code


def final_preflight(config_path: str | Path, approval_path: str | Path, *, resume: bool = False) -> dict[str, object]:
    """Offline final gate. It never reads environment variables or constructs a Provider."""
    config_file = Path(config_path).resolve()
    approval_file = Path(approval_path).resolve()
    result: dict[str, object] = {
        "status": "blocked",
        "network": "disabled",
        "provider_constructed": False,
        "api_key_read": False,
        "transport_called": False,
        "results_written": False,
        "ready_for_network_authorization": False,
        "blocking_reasons": [],
        "safety": {
            "development_only": True,
            "pilot_only": True,
            "not_paper_result": True,
            "no_causal_conclusion": True,
            "private_truth_exposed": False,
            "network_disabled": True,
            "old_pilot_touched": False,
        },
    }
    reasons: list[str] = []
    if config_file != (_ROOT / HD2_CONFIG_RELATIVE).resolve():
        reasons.append("config_path_not_hd2")
    try:
        config, _ = load_hd2_config(config_file)
    except HD2ConfigError as exc:
        result["blocking_reasons"] = [exc.code]
        return result
    result.update({
        "study_id": config.study_id,
        "config_sha256": sha256_file(config_file),
        "protocol_sha256": sha256_file(_ROOT / HD2_PROTOCOL_RELATIVE),
        "template_sha256": sha256_file(_ROOT / HD2_TEMPLATE_RELATIVE),
        "run_count": len(config.runs),
        "matched_group_count": len({item.matched_group_id for item in config.runs}),
        "logical_request_count": config.request_cap,
        "completion_reservation_cap": config.completion_reservation_cap,
        "response_format": config.response_format,
        "schema_name": config.response_schema_name,
        "output_root": config.output_root,
    })
    try:
        approval = load_approval(approval_file)
    except ValueError as exc:
        result["blocking_reasons"] = [str(exc)]
        return result
    expected_config_hash = sha256_file(config_file)
    expected_protocol_hash = sha256_file(_ROOT / HD2_PROTOCOL_RELATIVE)
    if approval.config_path != HD2_CONFIG_RELATIVE or approval.config_sha256.lower() != expected_config_hash:
        reasons.append("config_hash_mismatch")
    if approval.protocol_path != HD2_PROTOCOL_RELATIVE or approval.protocol_sha256.lower() != expected_protocol_hash:
        reasons.append("protocol_hash_mismatch")
    expected_template_hash = sha256_file(_ROOT / HD2_TEMPLATE_RELATIVE)
    if approval.template_path != HD2_TEMPLATE_RELATIVE or approval.template_sha256.lower() != expected_template_hash:
        reasons.append("template_hash_mismatch")
    if config.response_format != "json_schema" or config.response_schema_name != V2_SCHEMA_NAME:
        reasons.append("response_format_mismatch")
    if config.output_root != HD2_OUTPUT_ROOT or "002" in config.output_root or "003" in config.output_root or "calibration" in config.output_root:
        reasons.append("output_scope_invalid")
    output_path = (_ROOT / config.output_root).resolve()
    result["resume_requested"] = resume
    if output_path.exists() and not resume:
        reasons.append("output_root_exists")
    elif output_path.exists() and resume:
        result["resume_binding_validation_required"] = True
    if approval.acceptance_status != "accepted" or not approval.ready:
        reasons.append("approval_pending")
    if approval.compatibility_receipt_path:
        receipt_path = _receipt_path_value(approval.compatibility_receipt_path)
        compatible, code = _safe_compatibility_ok(receipt_path, approval.compatibility_receipt_sha256)
        if not compatible:
            reasons.append(code)
        else:
            result["compatibility_receipt_sha256"] = sha256_file(receipt_path)
            result["compatibility_module_version"] = "cascade_agent_protocol_v2_compatibility.v1"
            result["compatibility_max_tokens"] = 256
            result["compatibility_temperature"] = 0.2
            result["compatibility_seed"] = 20260911
            result["compatibility_max_retries"] = 0
    else:
        reasons.append("compatibility_receipt_missing")
    if len(config.runs) != 48 or len({r.matched_group_id for r in config.runs}) != 12:
        reasons.append("run_plan_incomplete")
    if config.request_cap != 864 or config.completion_reservation_cap != 221184:
        reasons.append("budget_scope_mismatch")
    result["blocking_reasons"] = sorted(set(reasons))
    if not reasons:
        result["status"] = "ready_for_real_pilot"
        result["ready_for_network_authorization"] = True
    return result


class HD2PilotRunner:
    """Run one H-D.2 coordinate with the strict runtime and no parser recovery."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()
        self.config, _ = load_hd2_config(self.config_path)
        self.scenarios = _load_scenarios(self.config_path, self.config)
        self.base_runner = CascadeRealAgentRunner()

    def run_one(
        self,
        spec: HD2RunSpec,
        *,
        provider: LLMProvider,
        root: str | Path,
        model_name: str,
        resume: bool = False,
    ) -> CascadeRealAgentRunRecord:
        if spec not in self.config.runs:
            raise ValueError("run_spec_not_registered")
        root_path = Path(root).resolve()
        ledger_path = root_path / spec.run_id / "request_ledger.jsonl"
        checkpoint_path = root_path / spec.run_id / "agent_checkpoint.json"
        return self.base_runner.run_scenario(
            self.scenarios[spec.scenario_id], spec.seed, spec.condition,
            provider=provider, run_id=spec.run_id, ledger_path=ledger_path,
            model_name=model_name, agent_temperature=0.2, agent_max_tokens=256,
            request_cap=spec.expected_provider_requests,
            completion_reservation_cap=spec.completion_reservation,
            resume=resume, checkpoint_path=checkpoint_path,
            recovery_fingerprints=None, recovery_kind=None, max_recovery_attempts=0,
            runtime=CascadeAgentProtocolV2Runtime(),
        )


    def run_all(
        self,
        *,
        provider_factory: Any,
        root: str | Path | None = None,
        model_name: str,
        resume: bool = False,
        write_receipt: bool = True,
        execution_binding_sha256: str | None = None,
        approval_sha256: str | None = None,
        compatibility_receipt_sha256: str | None = None,
        network: str = "enabled",
    ) -> tuple[list[CascadeRealAgentRunRecord], HDEPilotReceipt]:
        """Run all registered coordinates with safe, resumable batch state.

        Provider construction is deferred until an incomplete run is about to
        execute. Completed run records are validated and loaded without
        invoking ``provider_factory``. Parser-invalid responses remain terminal
        because no recovery fingerprints are ever supplied.
        """
        root_path = _resolve_run_root(root, self.config.output_root)
        expected_root = (_ROOT / HD2_OUTPUT_ROOT).resolve()
        if root is None and root_path != expected_root:
            raise CascadeRealAgentRunError("output_root_invalid")
        binding = execution_binding_sha256 or _execution_binding(
            config_path=self.config_path,
            model_name=model_name,
            output_root=root_path,
            approval_sha256=approval_sha256,
            compatibility_receipt_sha256=compatibility_receipt_sha256,
        )
        batch_path = root_path / "pilot_batch_record.json"
        receipt_path = root_path / "pilot_receipt.json"
        if root_path.exists():
            if not resume:
                raise CascadeRealAgentRunError("output_root_exists")
            batch = _load_batch_record(batch_path)
            _validate_batch_binding(batch, self.config, binding=binding, model_name=model_name, output_root=root_path)
        else:
            if resume:
                raise CascadeRealAgentRunError("resume_output_missing")
            root_path.mkdir(parents=True, exist_ok=False)
            batch = _initial_batch_record(
                self.config,
                config_path=self.config_path,
                binding=binding,
                model_name=model_name,
                output_root=root_path,
                approval_sha256=approval_sha256,
                compatibility_receipt_sha256=compatibility_receipt_sha256,
            )
            _atomic_model_json(batch_path, batch)

        records: list[CascadeRealAgentRunRecord] = []
        for spec in self.config.runs:
            state = next(item for item in batch.runs if item.run_id == spec.run_id)
            run_dir = root_path / spec.run_id
            run_record_path = run_dir / "run_record.json"
            if run_record_path.exists():
                if not resume and state.status is not HD2PilotRunStatus.COMPLETED:
                    raise CascadeRealAgentRunError("run_output_exists")
                record = _load_completed_run_record(run_record_path, spec)
                records.append(record)
                updated = _state_from_record(spec, record, run_dir)
                batch = _replace_run_state(batch, updated, status=HD2PilotRunStatus.RUNNING, failure_code=None)
                _atomic_model_json(batch_path, batch)
                continue
            if state.status is HD2PilotRunStatus.COMPLETED:
                raise CascadeRealAgentRunError("completed_run_record_missing")
            if run_dir.exists() and not resume and any(run_dir.iterdir()):
                raise CascadeRealAgentRunError("run_output_exists")
            running = state.model_copy(update={"status": HD2PilotRunStatus.RUNNING, "error_code": None})
            batch = _replace_run_state(batch, running, status=HD2PilotRunStatus.RUNNING, failure_code=None)
            _atomic_model_json(batch_path, batch)
            try:
                provider = provider_factory(spec)
            except Exception:
                failed = state.model_copy(update={"status": HD2PilotRunStatus.FAILED, "error_code": "provider_construction_failed"})
                batch = _replace_run_state(batch, failed, status=HD2PilotRunStatus.FAILED, failure_code="provider_construction_failed")
                _atomic_model_json(batch_path, batch)
                raise CascadeRealAgentRunError("provider_construction_failed") from None
            try:
                record = self.run_one(
                    spec,
                    provider=provider,
                    root=root_path,
                    model_name=model_name,
                    resume=resume,
                )
                if record.replay is None or record.replay.status.value != "passed":
                    raise CascadeRealAgentRunError("run_replay_incomplete")
            except CascadeRealAgentRunError as exc:
                failed = state.model_copy(update={"status": HD2PilotRunStatus.FAILED, "error_code": exc.code})
                batch = _replace_run_state(batch, failed, status=HD2PilotRunStatus.FAILED, failure_code=exc.code)
                _atomic_model_json(batch_path, batch)
                raise
            except Exception:
                failed = state.model_copy(update={"status": HD2PilotRunStatus.FAILED, "error_code": "pilot_execution_failed"})
                batch = _replace_run_state(batch, failed, status=HD2PilotRunStatus.FAILED, failure_code="pilot_execution_failed")
                _atomic_model_json(batch_path, batch)
                raise CascadeRealAgentRunError("pilot_execution_failed") from None
            safe_record = record.model_copy(update={"request_ledger_path": "request_ledger.jsonl"})
            if run_record_path.exists():
                raise CascadeRealAgentRunError("run_record_exists")
            _atomic_model_json(run_record_path, safe_record)
            records.append(safe_record)
            completed = _state_from_record(spec, safe_record, run_dir)
            batch = _replace_run_state(batch, completed, status=HD2PilotRunStatus.RUNNING, failure_code=None)
            _atomic_model_json(batch_path, batch)

        replay_passed = sum(record.replay is not None and record.replay.status.value == "passed" for record in records)
        if len(records) != 48 or replay_passed != 48 or any(item.status is not HD2PilotRunStatus.COMPLETED for item in batch.runs):
            raise CascadeRealAgentRunError("batch_replay_incomplete")
        aggregates = _aggregate_ledgers(root_path, self.config)
        replay_statuses = {record.run_id: record.replay.status.value if record.replay else "missing" for record in records}
        receipt = HDEPilotReceipt(
            status="completed",
            study_id=self.config.study_id,
            config_sha256=sha256_file(self.config_path),
            protocol_sha256=sha256_file(_ROOT / HD2_PROTOCOL_RELATIVE),
            template_sha256=sha256_file(_ROOT / HD2_TEMPLATE_RELATIVE),
            run_count=len(records),
            completed_run_count=len(records),
            failed_run_count=0,
            matched_group_count=len({spec.matched_group_id for spec in self.config.runs}),
            logical_request_count=aggregates["logical_request_count"],
            provider_call_count=aggregates["provider_call_count"],
            transport_attempt_count=aggregates["transport_attempt_count"],
            actual_prompt_token_count=aggregates["actual_prompt_token_count"],
            actual_completion_token_count=aggregates["actual_completion_token_count"],
            actual_total_token_count=aggregates["actual_total_token_count"],
            replay_passed_count=replay_passed,
            replay_statuses=replay_statuses,
            directive_applied_count=sum(record.directive_applied_count for record in records),
            request_cap=self.config.request_cap,
            completion_reservation_cap=self.config.completion_reservation_cap,
            ledger_hash=_receipt_hash(records),
            network=network,
            results_written=write_receipt,
            old_pilot_touched=False,
            private_truth_exposed=False,
            pilot_only=True,
            parser_recovery_enabled=False,
            safety={
                "development_only": True,
                "pilot_only": True,
                "not_paper_result": True,
                "no_causal_conclusion": True,
                "private_truth_exposed": False,
                "network_disabled": network != "enabled",
            },
        )
        if write_receipt:
            if receipt_path.exists():
                if not resume:
                    raise CascadeRealAgentRunError("receipt_exists")
                try:
                    existing = HDEPilotReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise CascadeRealAgentRunError("receipt_invalid") from exc
                if existing.model_dump(mode="json") != receipt.model_dump(mode="json"):
                    raise CascadeRealAgentRunError("receipt_binding_mismatch")
                receipt = existing
            else:
                _atomic_model_json(receipt_path, receipt)
        batch = batch.model_copy(update={"status": HD2PilotRunStatus.COMPLETED, "failure_code": None})
        _atomic_model_json(batch_path, batch)
        return records, receipt


def _resolve_run_root(root: str | Path | None, configured: str) -> Path:
    candidate = Path(root) if root is not None else Path(configured)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def _execution_binding(
    *,
    config_path: Path,
    model_name: str,
    output_root: Path,
    approval_sha256: str | None,
    compatibility_receipt_sha256: str | None,
) -> str:
    payload = {
        "config_sha256": sha256_file(config_path),
        "protocol_sha256": sha256_file(_ROOT / HD2_PROTOCOL_RELATIVE),
        "template_sha256": sha256_file(_ROOT / HD2_TEMPLATE_RELATIVE),
        "approval_sha256": approval_sha256,
        "compatibility_receipt_sha256": compatibility_receipt_sha256,
        "model_name": model_name,
        "output_root": str(output_root),
        "parser_recovery_enabled": False,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _initial_batch_record(
    config: HD2Config,
    *,
    config_path: Path,
    binding: str,
    model_name: str,
    output_root: Path,
    approval_sha256: str | None,
    compatibility_receipt_sha256: str | None,
) -> HD2PilotBatchRecord:
    return HD2PilotBatchRecord(
        study_id=config.study_id,
        status=HD2PilotRunStatus.RUNNING,
        binding_sha256=binding,
        config_sha256=sha256_file(config_path),
        protocol_sha256=sha256_file(_ROOT / HD2_PROTOCOL_RELATIVE),
        template_sha256=sha256_file(_ROOT / HD2_TEMPLATE_RELATIVE),
        approval_sha256=approval_sha256,
        compatibility_receipt_sha256=compatibility_receipt_sha256,
        model_name=model_name,
        output_root=str(output_root),
        runs=tuple(
            HD2PilotRunState(
                run_id=spec.run_id,
                scenario_id=spec.scenario_id,
                seed=spec.seed,
                condition=spec.condition.value,
            )
            for spec in config.runs
        ),
    )


def _atomic_model_json(path: Path, value: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_batch_record(path: Path) -> HD2PilotBatchRecord:
    if not path.exists():
        raise CascadeRealAgentRunError("resume_batch_record_missing")
    try:
        return HD2PilotBatchRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CascadeRealAgentRunError("batch_record_invalid") from exc


def _validate_batch_binding(
    batch: HD2PilotBatchRecord,
    config: HD2Config,
    *,
    binding: str,
    model_name: str,
    output_root: Path,
) -> None:
    expected_ids = tuple(spec.run_id for spec in config.runs)
    if batch.binding_sha256 != binding:
        raise CascadeRealAgentRunError("resume_binding_mismatch")
    if batch.model_name != model_name or batch.output_root != str(output_root):
        raise CascadeRealAgentRunError("resume_binding_mismatch")
    if tuple(state.run_id for state in batch.runs) != expected_ids:
        raise CascadeRealAgentRunError("resume_run_plan_mismatch")
    if batch.config_sha256 != sha256_file(_ROOT / HD2_CONFIG_RELATIVE):
        raise CascadeRealAgentRunError("resume_config_hash_mismatch")
    if batch.protocol_sha256 != sha256_file(_ROOT / HD2_PROTOCOL_RELATIVE) or batch.template_sha256 != sha256_file(_ROOT / HD2_TEMPLATE_RELATIVE):
        raise CascadeRealAgentRunError("resume_protocol_hash_mismatch")


def _replace_run_state(
    batch: HD2PilotBatchRecord,
    state: HD2PilotRunState,
    *,
    status: HD2PilotRunStatus,
    failure_code: str | None,
) -> HD2PilotBatchRecord:
    runs = tuple(state if item.run_id == state.run_id else item for item in batch.runs)
    return batch.model_copy(update={"runs": runs, "status": status, "failure_code": failure_code})


def _load_completed_run_record(path: Path, spec: HD2RunSpec) -> CascadeRealAgentRunRecord:
    try:
        record = CascadeRealAgentRunRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CascadeRealAgentRunError("run_record_invalid") from exc
    if (
        record.run_id != spec.run_id
        or record.scenario_id != spec.scenario_id
        or record.seed != spec.seed
        or record.condition != spec.condition
        or record.replay is None
        or record.replay.status.value != "passed"
    ):
        raise CascadeRealAgentRunError("run_record_binding_mismatch")
    return record


def _state_from_record(spec: HD2RunSpec, record: CascadeRealAgentRunRecord, run_dir: Path) -> HD2PilotRunState:
    summary = RequestLedger(run_dir / "request_ledger.jsonl").summary(
        request_cap=spec.expected_provider_requests,
        completion_reservation_cap=spec.completion_reservation,
    )
    replay = record.replay
    if replay is None:
        raise CascadeRealAgentRunError("run_replay_incomplete")
    return HD2PilotRunState(
        run_id=spec.run_id,
        scenario_id=spec.scenario_id,
        seed=spec.seed,
        condition=spec.condition.value,
        status=HD2PilotRunStatus.COMPLETED,
        logical_request_count=summary.unique_logical_request_count,
        transport_attempt_count=summary.transport_attempt_count,
        actual_prompt_token_count=summary.actual_prompt_token_count,
        actual_completion_token_count=summary.actual_completion_token_count,
        actual_total_token_count=summary.actual_total_token_count,
        directive_applied_count=record.directive_applied_count,
        cascade_replay_status=replay.cascade_replay.status.value,
        application_replay_status=replay.application_replay.status.value,
        outcome_replay_status=replay.status.value,
        run_record_sha256=sha256_file(run_dir / "run_record.json"),
        error_code=None,
    )


def _aggregate_ledgers(root: Path, config: HD2Config) -> dict[str, int | None]:
    summaries = []
    for spec in config.runs:
        ledger_path = root / spec.run_id / "request_ledger.jsonl"
        if ledger_path.exists():
            summaries.append(RequestLedger(ledger_path).summary(
                request_cap=spec.expected_provider_requests,
                completion_reservation_cap=spec.completion_reservation,
            ))
    completed_summaries = [summary for summary in summaries if summary.completed_count]

    def total(field: str) -> int | None:
        values = [getattr(summary, field) for summary in completed_summaries]
        if not values:
            return None
        return sum(values) if all(value is not None for value in values) else None

    return {
        "logical_request_count": sum(summary.unique_logical_request_count for summary in summaries),
        "provider_call_count": sum(summary.provider_call_count for summary in summaries),
        "transport_attempt_count": sum(summary.transport_attempt_count for summary in summaries),
        "actual_prompt_token_count": total("actual_prompt_token_count"),
        "actual_completion_token_count": total("actual_completion_token_count"),
        "actual_total_token_count": total("actual_total_token_count"),
    }

def _receipt_hash(records: list[CascadeRealAgentRunRecord]) -> str:
    payload = [(r.run_id, r.exposure_ledger_sha256, r.application_ledger_sha256, r.outcome_ledger_sha256) for r in records]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run_fake_smoke(config_path: str | Path) -> dict[str, object]:
    """Exercise the exact 48-run ``run_all`` path below a temporary root."""
    pilot = HD2PilotRunner(config_path)
    with tempfile.TemporaryDirectory(prefix="evicon-hd2-v1-fake-") as temporary:
        records, receipt = pilot.run_all(
            provider_factory=lambda _spec: V2FakeProvider(),
            root=Path(temporary) / "pilot",
            model_name="hd2-v2-fake",
            resume=False,
            write_receipt=False,
            network="disabled",
        )
    status = "fake_smoke_passed" if (
        len(records) == 48
        and receipt.completed_run_count == 48
        and receipt.logical_request_count == 864
        and receipt.replay_passed_count == 48
    ) else "fake_smoke_failed"
    return receipt.model_copy(update={"status": status}).model_dump(mode="json")


def _blocked_summary(code: str) -> HD2PilotExecutionSummary:
    return HD2PilotExecutionSummary(
        status="blocked",
        error_code=code,
        network="disabled",
        results_written=False,
        safety={
            "development_only": True,
            "pilot_only": True,
            "not_paper_result": True,
            "no_causal_conclusion": True,
            "private_truth_exposed": False,
            "provider_constructed": False,
        },
    )


def _confirmation_error(
    *,
    allow_network: bool,
    confirm_run: bool,
    confirm_request_cap: int | None,
    confirm_completion_reservation_cap: int | None,
) -> str | None:
    if not allow_network:
        return "allow_network_required"
    if not confirm_run:
        return "confirm_run_required"
    if confirm_request_cap != 864:
        return "confirm_request_cap_must_equal_864"
    if confirm_completion_reservation_cap != 221184:
        return "confirm_completion_reservation_cap_must_equal_221184"
    return None


def _summary_from_root(
    *,
    root: Path,
    config: HD2Config,
    status: str,
    error_code: str | None,
    provider_constructed_count: int,
    network: str,
) -> HD2PilotExecutionSummary:
    batch_path = root / "pilot_batch_record.json"
    batch = _load_batch_record(batch_path) if batch_path.exists() else None
    aggregates = _aggregate_ledgers(root, config) if root.exists() else {
        "logical_request_count": 0,
        "transport_attempt_count": 0,
        "actual_prompt_token_count": None,
        "actual_completion_token_count": None,
        "actual_total_token_count": None,
    }
    states = batch.runs if batch is not None else ()
    replay_statuses = {
        state.run_id: state.outcome_replay_status or state.status.value
        for state in states
    }
    receipt_path = root / "pilot_receipt.json"
    return HD2PilotExecutionSummary(
        status=status,
        error_code=error_code,
        completed_run_count=sum(state.status is HD2PilotRunStatus.COMPLETED for state in states),
        failed_run_count=sum(state.status is HD2PilotRunStatus.FAILED for state in states),
        logical_request_count=int(aggregates["logical_request_count"] or 0),
        transport_attempt_count=int(aggregates["transport_attempt_count"] or 0),
        actual_prompt_token_count=aggregates["actual_prompt_token_count"],
        actual_completion_token_count=aggregates["actual_completion_token_count"],
        actual_total_token_count=aggregates["actual_total_token_count"],
        directive_applied_count=sum(state.directive_applied_count for state in states),
        replay_statuses=replay_statuses,
        receipt_path=str(receipt_path) if receipt_path.exists() else None,
        provider_constructed_count=provider_constructed_count,
        network=network,
        results_written=root.exists(),
        safety={
            "development_only": True,
            "pilot_only": True,
            "not_paper_result": True,
            "no_causal_conclusion": True,
            "private_truth_exposed": False,
            "parser_recovery_enabled": False,
            "old_pilot_touched": False,
        },
    )


def execute_hd2_real_pilot(
    *,
    config_path: str | Path = HD2_CONFIG_RELATIVE,
    approval_path: str | Path = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd2_approval_template.toml",
    allow_network: bool = False,
    confirm_run: bool = False,
    confirm_request_cap: int | None = None,
    confirm_completion_reservation_cap: int | None = None,
    resume: bool = False,
    provider_factory: Any | None = None,
    test_output_root: str | Path | None = None,
    test_model_name: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> HD2PilotExecutionSummary:
    """Execute the real-pilot path after every immutable gate has passed.

    ``provider_factory`` and ``test_output_root`` are dependency-injection
    hooks for offline tests only and are intentionally absent from the CLI.
    """
    confirmation = _confirmation_error(
        allow_network=allow_network,
        confirm_run=confirm_run,
        confirm_request_cap=confirm_request_cap,
        confirm_completion_reservation_cap=confirm_completion_reservation_cap,
    )
    if confirmation:
        return _blocked_summary(confirmation)
    gate = final_preflight(config_path, approval_path, resume=resume or test_output_root is not None)
    if gate.get("status") != "ready_for_real_pilot":
        reasons = gate.get("blocking_reasons") or ["final_preflight_blocked"]
        return _blocked_summary(str(reasons[0]))
    try:
        config_file = Path(config_path).resolve()
        approval_file = Path(approval_path).resolve()
        pilot = HD2PilotRunner(config_file)
        approval = load_approval(approval_file)
    except (HD2ConfigError, ValueError, OSError):
        return _blocked_summary("execution_inputs_invalid")
    if test_output_root is not None and provider_factory is None:
        return _blocked_summary("test_output_requires_injected_provider")
    config = pilot.config
    root = _resolve_run_root(test_output_root, config.output_root)
    source = os.environ if environment is None else environment
    injected = provider_factory is not None
    if injected:
        model_name = test_model_name or "hd2-v2-fake"
        network = "disabled"
    else:
        if not all(source.get(name) for name in ("EVICON_LLM_BASE_URL", "EVICON_LLM_MODEL", "EVICON_LLM_API_KEY")):
            return _blocked_summary("provider_environment_incomplete")
        model_name = str(source["EVICON_LLM_MODEL"])
        network = "enabled"
    approval_hash = sha256_file(approval_file)
    compatibility_hash = approval.compatibility_receipt_sha256
    binding = _execution_binding(
        config_path=config_file,
        model_name=model_name,
        output_root=root,
        approval_sha256=approval_hash,
        compatibility_receipt_sha256=compatibility_hash,
    )
    try:
        if root.exists():
            if not resume:
                return _blocked_summary("output_root_exists")
            batch = _load_batch_record(root / "pilot_batch_record.json")
            _validate_batch_binding(batch, config, binding=binding, model_name=model_name, output_root=root)
        elif resume:
            return _blocked_summary("resume_output_missing")
    except CascadeRealAgentRunError as exc:
        return _blocked_summary(exc.code)

    constructed = 0
    if injected:
        supplied_factory = provider_factory

        def active_factory(spec: HD2RunSpec) -> LLMProvider:
            nonlocal constructed
            constructed += 1
            return supplied_factory(spec)
    else:
        try:
            base = ProviderConfig.from_env(allow_network=True, environment=source)
        except Exception:
            return _blocked_summary("provider_configuration_invalid")

        def active_factory(spec: HD2RunSpec) -> LLMProvider:
            nonlocal constructed
            settings = base.model_copy(update={
                "model_name": model_name,
                "timeout_seconds": 15.0,
                "max_retries": 1,
                "temperature": 0.2,
                "max_tokens": 256,
                "seed": spec.seed,
                "response_format": ResponseFormatMode.JSON_SCHEMA,
                "response_schema_name": V2_SCHEMA_NAME,
            })
            constructed += 1
            return OpenAICompatibleProvider(settings, environment=source)

    try:
        pilot.run_all(
            provider_factory=active_factory,
            root=root,
            model_name=model_name,
            resume=resume,
            write_receipt=True,
            execution_binding_sha256=binding,
            approval_sha256=approval_hash,
            compatibility_receipt_sha256=compatibility_hash,
            network=network,
        )
        return _summary_from_root(
            root=root,
            config=config,
            status="completed",
            error_code=None,
            provider_constructed_count=constructed,
            network=network,
        )
    except CascadeRealAgentRunError as exc:
        return _summary_from_root(
            root=root,
            config=config,
            status="failed",
            error_code=exc.code,
            provider_constructed_count=constructed,
            network=network,
        )
    except Exception:
        try:
            return _summary_from_root(
                root=root,
                config=config,
                status="failed",
                error_code="pilot_execution_failed",
                provider_constructed_count=constructed,
                network=network,
            )
        except Exception:
            return HD2PilotExecutionSummary(
                status="failed",
                error_code="pilot_execution_failed",
                provider_constructed_count=constructed,
                network=network,
                results_written=root.exists(),
                safety={
                    "development_only": True,
                    "pilot_only": True,
                    "not_paper_result": True,
                    "no_causal_conclusion": True,
                    "private_truth_exposed": False,
                    "parser_recovery_enabled": False,
                    "old_pilot_touched": False,
                },
            )


def main(
    argv: Sequence[str] | None = None,
    *,
    provider_factory: Any | None = None,
    test_output_root: str | Path | None = None,
    test_model_name: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="H-E isolated H-D.2 strict pilot runner")
    parser.add_argument("--config", default=HD2_CONFIG_RELATIVE)
    parser.add_argument("--approval", default="configs/provenance_cascade/pilot/provenance_cascade_pilot_hd2_approval_template.toml")
    parser.add_argument("--mode", choices=("fake-smoke", "real-pilot"), default="fake-smoke")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--confirm-request-cap", type=int)
    parser.add_argument("--confirm-completion-reservation-cap", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "fake-smoke":
        print(json.dumps(run_fake_smoke(args.config), ensure_ascii=True, sort_keys=True))
        return 0
    summary = execute_hd2_real_pilot(
        config_path=args.config,
        approval_path=args.approval,
        allow_network=args.allow_network,
        confirm_run=args.confirm_run,
        confirm_request_cap=args.confirm_request_cap,
        confirm_completion_reservation_cap=args.confirm_completion_reservation_cap,
        resume=args.resume,
        provider_factory=provider_factory,
        test_output_root=test_output_root,
        test_model_name=test_model_name,
        environment=environment,
    )
    print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))
    return 0 if summary.status == "completed" else 1


__all__ = [
    "HDEApproval",
    "HDEPilotReceipt",
    "HD2PilotBatchRecord",
    "HD2PilotExecutionSummary",
    "HD2PilotRunState",
    "HD2PilotRunStatus",
    "HD2PilotRunner",
    "execute_hd2_real_pilot",
    "final_preflight",
    "load_approval",
    "main",
    "run_fake_smoke",
    "sha256_file",
]


if __name__ == "__main__":
    raise SystemExit(main())
