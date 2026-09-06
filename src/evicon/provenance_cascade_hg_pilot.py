"""Isolated, opt-in H-G Pilot runner with offline-first gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .cascade_agent_protocol_hg import (
    HG_PROTOCOL_VERSION,
    HG_SCHEMA_NAME,
    HG_TEMPLATE_VERSION,
    CascadeAgentProtocolHGRuntime,
    build_identifiable_prompt_context,
)
from .cascade_controller import CascadeControllerPolicyLoader
from .cascade_hg_outcome_replay import HG_OUTCOME_REPLAY_VERSION, HGOutcomeReplayValidator
from .cascade_real_agent_runner import CascadeRealAgentRunError, CascadeRealAgentRunRecord, CascadeRealAgentRunner
from .llm_contract import LLMProvider
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ResponseFormatMode
from .provenance_cascade_identifiability import (
    HGApproval,
    HGFakeProvider,
    HGPilotConfig,
    HGPilotRunSpec,
    hg_preflight,
    load_hg_config,
    sha256_file,
)
from .request_ledger import RequestLedger

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_RELATIVE = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg.v1.toml"
DEFAULT_APPROVAL_RELATIVE = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg_approval_template.toml"
DEFAULT_PROTOCOL_RELATIVE = "src/evicon/cascade_agent_protocol_hg.py"
DEFAULT_OUTPUT_ROOT = "results/provenance-cascade-pilot-hg-v1"


class HGPilotRunStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class HGPilotRunState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    scenario_id: str
    seed: int
    condition: str
    status: HGPilotRunStatus = HGPilotRunStatus.PLANNED
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


class HGPilotBatchRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    batch_version: str = "provenance_cascade_hg_execution.v1"
    outcome_replay_contract_version: str = HG_OUTCOME_REPLAY_VERSION
    study_id: str
    status: HGPilotRunStatus
    binding_sha256: str = Field(min_length=64, max_length=64)
    config_sha256: str = Field(min_length=64, max_length=64)
    protocol_sha256: str = Field(min_length=64, max_length=64)
    template_sha256: str = Field(min_length=64, max_length=64)
    amendment_sha256: str = Field(min_length=64, max_length=64)
    approval_sha256: str
    compatibility_receipt_sha256: str
    model_name: str
    output_root: str
    runs: tuple[HGPilotRunState, ...]
    failure_code: str | None = None
    development_only: bool = True
    pilot_only: bool = True
    not_paper_result: bool = True
    no_causal_conclusion: bool = True
    private_truth_exposed: bool = False
    parser_recovery_enabled: bool = False


class HGPilotReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: str
    study_id: str
    config_sha256: str
    protocol_sha256: str
    template_sha256: str
    amendment_sha256: str
    approval_sha256: str
    compatibility_receipt_sha256: str
    protocol_version: str = HG_PROTOCOL_VERSION
    template_version: str = HG_TEMPLATE_VERSION
    response_format: str = "json_schema"
    schema_name: str = HG_SCHEMA_NAME
    outcome_replay_contract_version: str = HG_OUTCOME_REPLAY_VERSION
    run_count: int = Field(ge=0)
    completed_run_count: int = Field(ge=0)
    failed_run_count: int = Field(ge=0)
    matched_group_count: int = Field(ge=0)
    logical_request_count: int = Field(ge=0)
    provider_call_count: int = Field(ge=0)
    transport_attempt_count: int = Field(ge=0)
    actual_prompt_token_count: int | None = Field(default=None, ge=0)
    actual_completion_token_count: int | None = Field(default=None, ge=0)
    actual_total_token_count: int | None = Field(default=None, ge=0)
    replay_passed_count: int = Field(ge=0)
    replay_statuses: dict[str, str]
    directive_applied_count: int = Field(ge=0)
    request_cap: int = Field(gt=0)
    completion_reservation_cap: int = Field(gt=0)
    ledger_hash: str
    network: str
    results_written: bool
    legacy_assets_touched: bool = False
    private_truth_exposed: bool = False
    development_only: bool = True
    pilot_only: bool = True
    not_paper_result: bool = True
    no_causal_conclusion: bool = True
    parser_recovery_enabled: bool = False


class HGPilotExecutionSummary(BaseModel):
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
    network: str = "disabled"
    results_written: bool = False
    ready_for_real_pilot: bool = False
    safety: dict[str, bool] = Field(default_factory=dict)


def _rooted(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def _relative_to_config(config_path: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (config_path.parent / candidate).resolve()


def _atomic_model(path: Path, value: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_approval(path: Path) -> HGApproval:
    import tomllib
    try:
        return HGApproval.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise CascadeRealAgentRunError("approval_invalid") from exc


class HGPilotRunner:
    """Execute registered H-G coordinates without parser recovery."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_RELATIVE) -> None:
        self.config_path = _rooted(config_path)
        self.config, self.scenarios, _ = load_hg_config(self.config_path)
        policy_path = _relative_to_config(self.config_path, self.config.policy_path)
        self.policy = CascadeControllerPolicyLoader.load(policy_path)
        self.base_runner = CascadeRealAgentRunner(policy_config=self.policy)

    def run_one(
        self,
        spec: HGPilotRunSpec,
        *,
        provider: LLMProvider,
        root: str | Path,
        model_name: str,
        resume: bool = False,
    ) -> CascadeRealAgentRunRecord:
        if spec not in self.config.runs:
            raise CascadeRealAgentRunError("run_spec_not_registered")
        root_path = Path(root).resolve()
        run_dir = root_path / spec.run_id
        return self.base_runner.run_scenario(
            self.scenarios[spec.scenario_id],
            spec.seed,
            spec.condition,
            provider=provider,
            run_id=spec.run_id,
            ledger_path=run_dir / "request_ledger.jsonl",
            checkpoint_path=run_dir / "agent_checkpoint.json",
            model_name=model_name,
            agent_temperature=0.2,
            agent_max_tokens=self.config.agent_max_tokens,
            request_cap=spec.expected_provider_requests,
            completion_reservation_cap=spec.completion_reservation,
            resume=resume,
            recovery_fingerprints=None,
            recovery_kind=None,
            max_recovery_attempts=0,
            runtime=CascadeAgentProtocolHGRuntime(),
            context_builder=build_identifiable_prompt_context,
            outcome_replay_validator=HGOutcomeReplayValidator,
            policy_config=self.policy,
        )

    def run_all(
        self,
        *,
        provider_factory: Any,
        root: str | Path | None,
        model_name: str,
        approval_sha256: str,
        compatibility_receipt_sha256: str,
        amendment_sha256: str,
        resume: bool = False,
        write_receipt: bool = True,
        network: str = "enabled",
    ) -> tuple[list[CascadeRealAgentRunRecord], HGPilotReceipt]:
        root_path = _rooted(self.config.output_root) if root is None else Path(root).resolve()
        if root is None and root_path != _rooted(DEFAULT_OUTPUT_ROOT):
            raise CascadeRealAgentRunError("output_root_invalid")
        binding = _execution_binding(
            config=self.config,
            config_path=self.config_path,
            output_root=root_path,
            model_name=model_name,
            approval_sha256=approval_sha256,
            compatibility_receipt_sha256=compatibility_receipt_sha256,
            amendment_sha256=amendment_sha256,
        )
        batch_path = root_path / "pilot_batch_record.json"
        receipt_path = root_path / "pilot_receipt.json"
        if root_path.exists():
            if not resume:
                raise CascadeRealAgentRunError("output_root_exists")
            batch = _load_batch(batch_path)
            _validate_batch(batch, self.config, self.config_path, binding, root_path, model_name)
        else:
            if resume:
                raise CascadeRealAgentRunError("resume_output_missing")
            root_path.mkdir(parents=True, exist_ok=False)
            batch = _initial_batch(
                self.config,
                config_path=self.config_path,
                binding=binding,
                output_root=root_path,
                model_name=model_name,
                approval_sha256=approval_sha256,
                compatibility_receipt_sha256=compatibility_receipt_sha256,
                amendment_sha256=amendment_sha256,
            )
            _atomic_model(batch_path, batch)
        records: list[CascadeRealAgentRunRecord] = []
        for spec in self.config.runs:
            state = next(item for item in batch.runs if item.run_id == spec.run_id)
            run_dir = root_path / spec.run_id
            record_path = run_dir / "run_record.json"
            if record_path.exists():
                if not resume:
                    raise CascadeRealAgentRunError("run_output_exists")
                record = _load_run_record(record_path, spec, self.scenarios[spec.scenario_id])
                records.append(record)
                batch = _replace_state(batch, _state_from_record(spec, record, run_dir), HGPilotRunStatus.RUNNING, None)
                _atomic_model(batch_path, batch)
                continue
            if state.status is HGPilotRunStatus.COMPLETED:
                raise CascadeRealAgentRunError("completed_run_record_missing")
            if run_dir.exists() and any(run_dir.iterdir()) and not resume:
                raise CascadeRealAgentRunError("run_output_exists")
            batch = _replace_state(
                batch,
                state.model_copy(update={"status": HGPilotRunStatus.RUNNING, "error_code": None}),
                HGPilotRunStatus.RUNNING,
                None,
            )
            _atomic_model(batch_path, batch)
            try:
                provider = provider_factory(spec)
            except Exception:
                failed = state.model_copy(update={"status": HGPilotRunStatus.FAILED, "error_code": "provider_construction_failed"})
                batch = _replace_state(batch, failed, HGPilotRunStatus.FAILED, "provider_construction_failed")
                _atomic_model(batch_path, batch)
                raise CascadeRealAgentRunError("provider_construction_failed") from None
            try:
                record = self.run_one(spec, provider=provider, root=root_path, model_name=model_name, resume=resume)
                if record.replay is None or record.replay.status.value != "passed":
                    raise CascadeRealAgentRunError("run_replay_incomplete")
            except CascadeRealAgentRunError as exc:
                failed = state.model_copy(update={"status": HGPilotRunStatus.FAILED, "error_code": exc.code})
                batch = _replace_state(batch, failed, HGPilotRunStatus.FAILED, exc.code)
                _atomic_model(batch_path, batch)
                raise
            safe_record = record.model_copy(update={"request_ledger_path": "request_ledger.jsonl"})
            if record_path.exists():
                raise CascadeRealAgentRunError("run_record_exists")
            _atomic_model(record_path, safe_record)
            records.append(safe_record)
            batch = _replace_state(batch, _state_from_record(spec, safe_record, run_dir), HGPilotRunStatus.RUNNING, None)
            _atomic_model(batch_path, batch)
        replay_passed = sum(record.replay is not None and record.replay.status.value == "passed" for record in records)
        if len(records) != 48 or replay_passed != 48 or any(item.status is not HGPilotRunStatus.COMPLETED for item in batch.runs):
            raise CascadeRealAgentRunError("batch_replay_incomplete")
        aggregates = _aggregate_ledgers(root_path, self.config)
        receipt = HGPilotReceipt(
            status="completed",
            study_id=self.config.study_id,
            config_sha256=sha256_file(self.config_path),
            protocol_sha256=sha256_file(_rooted(DEFAULT_PROTOCOL_RELATIVE)),
            template_sha256=sha256_file(_rooted(DEFAULT_PROTOCOL_RELATIVE)),
            amendment_sha256=amendment_sha256,
            approval_sha256=approval_sha256,
            compatibility_receipt_sha256=compatibility_receipt_sha256,
            run_count=len(records),
            completed_run_count=len(records),
            failed_run_count=0,
            matched_group_count=len({spec.matched_group_id for spec in self.config.runs}),
            logical_request_count=int(aggregates["logical_request_count"] or 0),
            provider_call_count=int(aggregates["provider_call_count"] or 0),
            transport_attempt_count=int(aggregates["transport_attempt_count"] or 0),
            actual_prompt_token_count=aggregates["actual_prompt_token_count"],
            actual_completion_token_count=aggregates["actual_completion_token_count"],
            actual_total_token_count=aggregates["actual_total_token_count"],
            replay_passed_count=replay_passed,
            replay_statuses={record.run_id: record.replay.status.value if record.replay else "missing" for record in records},
            directive_applied_count=sum(record.directive_applied_count for record in records),
            request_cap=self.config.request_cap,
            completion_reservation_cap=self.config.completion_reservation_cap,
            ledger_hash=_records_hash(records),
            network=network,
            results_written=write_receipt,
        )
        if write_receipt:
            if receipt_path.exists():
                if not resume:
                    raise CascadeRealAgentRunError("receipt_exists")
                existing = HGPilotReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
                if existing.model_dump(mode="json") != receipt.model_dump(mode="json"):
                    raise CascadeRealAgentRunError("receipt_binding_mismatch")
                receipt = existing
            else:
                _atomic_model(receipt_path, receipt)
        batch = batch.model_copy(update={"status": HGPilotRunStatus.COMPLETED, "failure_code": None})
        _atomic_model(batch_path, batch)
        return records, receipt


def _execution_binding(
    *,
    config: HGPilotConfig,
    config_path: Path,
    output_root: Path,
    model_name: str,
    approval_sha256: str,
    compatibility_receipt_sha256: str,
    amendment_sha256: str,
) -> str:
    payload = {
        "config_sha256": sha256_file(config_path),
        "protocol_sha256": sha256_file(_rooted(DEFAULT_PROTOCOL_RELATIVE)),
        "template_sha256": sha256_file(_rooted(DEFAULT_PROTOCOL_RELATIVE)),
        "amendment_sha256": amendment_sha256,
        "approval_sha256": approval_sha256,
        "compatibility_receipt_sha256": compatibility_receipt_sha256,
        "model_name": model_name,
        "output_root": str(output_root),
        "run_ids": [item.run_id for item in config.runs],
        "parser_recovery_enabled": False,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _initial_batch(
    config: HGPilotConfig,
    *,
    config_path: Path,
    binding: str,
    output_root: Path,
    model_name: str,
    approval_sha256: str,
    compatibility_receipt_sha256: str,
    amendment_sha256: str,
) -> HGPilotBatchRecord:
    return HGPilotBatchRecord(
        study_id=config.study_id,
        status=HGPilotRunStatus.RUNNING,
        binding_sha256=binding,
        config_sha256=sha256_file(config_path),
        protocol_sha256=sha256_file(_rooted(DEFAULT_PROTOCOL_RELATIVE)),
        template_sha256=sha256_file(_rooted(DEFAULT_PROTOCOL_RELATIVE)),
        amendment_sha256=amendment_sha256,
        approval_sha256=approval_sha256,
        compatibility_receipt_sha256=compatibility_receipt_sha256,
        model_name=model_name,
        output_root=str(output_root),
        runs=tuple(HGPilotRunState(
            run_id=spec.run_id,
            scenario_id=spec.scenario_id,
            seed=spec.seed,
            condition=spec.condition.value,
        ) for spec in config.runs),
    )


def _load_batch(path: Path) -> HGPilotBatchRecord:
    if not path.is_file():
        raise CascadeRealAgentRunError("resume_batch_record_missing")
    try:
        return HGPilotBatchRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CascadeRealAgentRunError("batch_record_invalid") from exc


def _validate_batch(batch: HGPilotBatchRecord, config: HGPilotConfig, config_path: Path, binding: str, root: Path, model: str) -> None:
    if batch.binding_sha256 != binding or batch.model_name != model or batch.output_root != str(root):
        raise CascadeRealAgentRunError("resume_binding_mismatch")
    if tuple(item.run_id for item in batch.runs) != tuple(item.run_id for item in config.runs):
        raise CascadeRealAgentRunError("resume_run_plan_mismatch")
    if batch.config_sha256 != sha256_file(config_path):
        raise CascadeRealAgentRunError("resume_config_hash_mismatch")
    protocol_hash = sha256_file(_rooted(DEFAULT_PROTOCOL_RELATIVE))
    if batch.protocol_sha256 != protocol_hash or batch.template_sha256 != protocol_hash:
        raise CascadeRealAgentRunError("resume_protocol_hash_mismatch")


def _replace_state(batch: HGPilotBatchRecord, state: HGPilotRunState, status: HGPilotRunStatus, code: str | None) -> HGPilotBatchRecord:
    return batch.model_copy(update={
        "runs": tuple(state if item.run_id == state.run_id else item for item in batch.runs),
        "status": status,
        "failure_code": code,
    })


def _load_run_record(path: Path, spec: HGPilotRunSpec, scenario: Any) -> CascadeRealAgentRunRecord:
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
    if (
        record.exposure_ledger is None
        or record.application_ledger is None
        or record.outcome_ledger is None
    ):
        raise CascadeRealAgentRunError("run_record_replay_sidecar_missing")
    try:
        replay = HGOutcomeReplayValidator.validate(
            scenario.graph,
            record.exposure_ledger,
            record.outcome_ledger,
            record.application_ledger,
            record.round_contexts,
        )
    except Exception as exc:
        raise CascadeRealAgentRunError("run_record_hg_replay_invalid") from exc
    return record.model_copy(update={"replay": replay})


def _state_from_record(spec: HGPilotRunSpec, record: CascadeRealAgentRunRecord, run_dir: Path) -> HGPilotRunState:
    summary = RequestLedger(run_dir / "request_ledger.jsonl").summary(
        request_cap=spec.expected_provider_requests,
        completion_reservation_cap=spec.completion_reservation,
    )
    replay = record.replay
    if replay is None:
        raise CascadeRealAgentRunError("run_replay_incomplete")
    return HGPilotRunState(
        run_id=spec.run_id,
        scenario_id=spec.scenario_id,
        seed=spec.seed,
        condition=spec.condition.value,
        status=HGPilotRunStatus.COMPLETED,
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
    )


def _aggregate_ledgers(root: Path, config: HGPilotConfig) -> dict[str, int | None]:
    summaries = [RequestLedger(root / spec.run_id / "request_ledger.jsonl").summary(
        request_cap=spec.expected_provider_requests,
        completion_reservation_cap=spec.completion_reservation,
    ) for spec in config.runs if (root / spec.run_id / "request_ledger.jsonl").exists()]
    completed = [summary for summary in summaries if summary.completed_count]

    def total(field: str) -> int | None:
        values = [getattr(summary, field) for summary in completed]
        return sum(values) if values and all(value is not None for value in values) else None

    return {
        "logical_request_count": sum(item.unique_logical_request_count for item in summaries),
        "provider_call_count": sum(item.provider_call_count for item in summaries),
        "transport_attempt_count": sum(item.transport_attempt_count for item in summaries),
        "actual_prompt_token_count": total("actual_prompt_token_count"),
        "actual_completion_token_count": total("actual_completion_token_count"),
        "actual_total_token_count": total("actual_total_token_count"),
    }


def _records_hash(records: list[CascadeRealAgentRunRecord]) -> str:
    payload = [(item.run_id, item.exposure_ledger_sha256, item.application_ledger_sha256, item.outcome_ledger_sha256) for item in records]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run_hg_pilot_fake_smoke(config_path: str | Path = DEFAULT_CONFIG_RELATIVE) -> dict[str, object]:
    runner = HGPilotRunner(config_path)
    with tempfile.TemporaryDirectory(prefix="evicon-hg-pilot-fake-") as temporary:
        records, receipt = runner.run_all(
            provider_factory=lambda _spec: HGFakeProvider(),
            root=Path(temporary) / "pilot",
            model_name="hg-identifiability-fake",
            approval_sha256="a" * 64,
            compatibility_receipt_sha256="b" * 64,
            amendment_sha256="f840925cd46a27741c2245e011acf68725b4654c85a2a9ce899d69cb4b34e49a",
            write_receipt=False,
            network="disabled",
        )
    return {
        "status": "fake_smoke_passed" if len(records) == 48 and receipt.logical_request_count == 864 and receipt.replay_passed_count == 48 else "fake_smoke_failed",
        "run_count": len(records),
        "matched_group_count": receipt.matched_group_count,
        "logical_request_count": receipt.logical_request_count,
        "provider_call_count": receipt.provider_call_count,
        "transport_attempt_count": receipt.transport_attempt_count,
        "replay_passed_count": receipt.replay_passed_count,
        "directive_applied_count": receipt.directive_applied_count,
        "request_cap": receipt.request_cap,
        "completion_reservation_cap": receipt.completion_reservation_cap,
        "network": "disabled",
        "results_written": False,
        "private_truth_exposed": False,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def _safety(*, provider_constructed: bool = False) -> dict[str, bool]:
    return {
        "development_only": True,
        "pilot_only": True,
        "not_paper_result": True,
        "no_causal_conclusion": True,
        "private_truth_exposed": False,
        "provider_constructed": provider_constructed,
        "parser_recovery_enabled": False,
    }


def _blocked(code: str) -> HGPilotExecutionSummary:
    return HGPilotExecutionSummary(
        status="blocked",
        error_code=code,
        network="disabled",
        ready_for_real_pilot=False,
        safety=_safety(),
    )


def _partial_failure_summary(
    *,
    root: Path,
    config: HGPilotConfig,
    error_code: str,
    provider_constructed_count: int,
) -> HGPilotExecutionSummary:
    completed_run_count = 0
    failed_run_count = 0
    directive_applied_count = 0
    replay_statuses: dict[str, str] = {}
    try:
        batch = _load_batch(root / "pilot_batch_record.json")
        completed_run_count = sum(item.status is HGPilotRunStatus.COMPLETED for item in batch.runs)
        failed_run_count = sum(item.status is HGPilotRunStatus.FAILED for item in batch.runs)
        directive_applied_count = sum(item.directive_applied_count for item in batch.runs)
        replay_statuses = {
            item.run_id: item.outcome_replay_status or "missing"
            for item in batch.runs
            if item.status in {HGPilotRunStatus.COMPLETED, HGPilotRunStatus.FAILED}
        }
        aggregates = _aggregate_ledgers(root, config)
    except Exception:
        aggregates = {
            "logical_request_count": 0,
            "transport_attempt_count": 0,
            "actual_prompt_token_count": None,
            "actual_completion_token_count": None,
            "actual_total_token_count": None,
        }
    return HGPilotExecutionSummary(
        status="failed",
        error_code=error_code,
        completed_run_count=completed_run_count,
        failed_run_count=failed_run_count,
        logical_request_count=int(aggregates.get("logical_request_count") or 0),
        transport_attempt_count=int(aggregates.get("transport_attempt_count") or 0),
        actual_prompt_token_count=aggregates.get("actual_prompt_token_count"),
        actual_completion_token_count=aggregates.get("actual_completion_token_count"),
        actual_total_token_count=aggregates.get("actual_total_token_count"),
        directive_applied_count=directive_applied_count,
        replay_statuses=replay_statuses,
        provider_constructed_count=provider_constructed_count,
        network="enabled",
        results_written=root.exists(),
        ready_for_real_pilot=False,
        safety=_safety(provider_constructed=provider_constructed_count > 0),
    )


def execute_hg_real_pilot(
    *,
    config_path: str | Path = DEFAULT_CONFIG_RELATIVE,
    approval_path: str | Path = DEFAULT_APPROVAL_RELATIVE,
    allow_network: bool = False,
    confirm_run: bool = False,
    confirm_request_cap: int | None = None,
    confirm_completion_reservation_cap: int | None = None,
    resume: bool = False,
    environment: Mapping[str, str] | None = None,
) -> HGPilotExecutionSummary:
    if not allow_network:
        return _blocked("allow_network_required")
    if not confirm_run:
        return _blocked("confirm_run_required")
    if confirm_request_cap != 864:
        return _blocked("confirm_request_cap_must_equal_864")
    if confirm_completion_reservation_cap != 442368:
        return _blocked("confirm_completion_reservation_cap_must_equal_442368")
    gate = hg_preflight(config_path, approval_path_override=approval_path, allow_existing_output=resume)
    if not gate.get("ready_for_real_pilot"):
        return _blocked(str((gate.get("blocking_reasons") or ["final_preflight_blocked"])[0]))
    source = os.environ if environment is None else environment
    if not all(source.get(name) for name in ("EVICON_LLM_BASE_URL", "EVICON_LLM_MODEL", "EVICON_LLM_API_KEY")):
        return _blocked("provider_environment_incomplete")
    config_path_resolved = _rooted(config_path)
    approval_path_resolved = _rooted(approval_path)
    approval = _load_approval(approval_path_resolved)
    runner = HGPilotRunner(config_path_resolved)
    root = _rooted(runner.config.output_root)
    if root.exists() and not resume:
        return _blocked("output_root_exists")
    model_name = str(source["EVICON_LLM_MODEL"])
    base = ProviderConfig.from_env(allow_network=True, environment=source)
    constructed = 0

    def provider_factory(spec: HGPilotRunSpec) -> LLMProvider:
        nonlocal constructed
        constructed += 1
        settings = base.model_copy(update={
            "model_name": model_name,
            "timeout_seconds": 15.0,
            "max_retries": 1,
            "temperature": 0.2,
            "max_tokens": 512,
            "seed": spec.seed,
            "reasoning_effort": None,
            "response_format": ResponseFormatMode.JSON_SCHEMA,
            "response_schema_name": HG_SCHEMA_NAME,
        })
        return OpenAICompatibleProvider(settings, environment=source)
    try:
        records, receipt = runner.run_all(
            provider_factory=provider_factory,
            root=None,
            model_name=model_name,
            approval_sha256=sha256_file(approval_path_resolved),
            compatibility_receipt_sha256=approval.compatibility_receipt_sha256,
            amendment_sha256=approval.amendment_sha256,
            resume=resume,
            write_receipt=True,
            network="enabled",
        )
    except CascadeRealAgentRunError as exc:
        return _partial_failure_summary(
            root=root,
            config=runner.config,
            error_code=exc.code,
            provider_constructed_count=constructed,
        )
    return HGPilotExecutionSummary(
        status="completed",
        completed_run_count=len(records),
        logical_request_count=receipt.logical_request_count,
        transport_attempt_count=receipt.transport_attempt_count,
        actual_prompt_token_count=receipt.actual_prompt_token_count,
        actual_completion_token_count=receipt.actual_completion_token_count,
        actual_total_token_count=receipt.actual_total_token_count,
        directive_applied_count=receipt.directive_applied_count,
        replay_statuses=receipt.replay_statuses,
        receipt_path=str(root / "pilot_receipt.json"),
        provider_constructed_count=constructed,
        network="enabled",
        results_written=True,
        ready_for_real_pilot=True,
        safety=_safety(provider_constructed=constructed > 0),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="H-G public-identifiable provenance-cascade Pilot")
    parser.add_argument("--config", default=DEFAULT_CONFIG_RELATIVE)
    parser.add_argument("--approval", default=DEFAULT_APPROVAL_RELATIVE)
    parser.add_argument("--mode", choices=("preflight", "fake-smoke", "real-pilot"), default="preflight")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--confirm-request-cap", type=int)
    parser.add_argument("--confirm-completion-reservation-cap", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "preflight":
        payload = hg_preflight(args.config, approval_path_override=args.approval, allow_existing_output=args.resume)
    elif args.mode == "fake-smoke":
        payload = run_hg_pilot_fake_smoke(args.config)
    else:
        payload = execute_hg_real_pilot(
            config_path=args.config,
            approval_path=args.approval,
            allow_network=args.allow_network,
            confirm_run=args.confirm_run,
            confirm_request_cap=args.confirm_request_cap,
            confirm_completion_reservation_cap=args.confirm_completion_reservation_cap,
            resume=args.resume,
        ).model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if payload.get("status") in {"completed", "fake_smoke_passed", "ready_for_real_pilot"} else 1


__all__ = [
    "HGPilotBatchRecord",
    "HGPilotExecutionSummary",
    "HGPilotReceipt",
    "HGPilotRunState",
    "HGPilotRunStatus",
    "HGPilotRunner",
    "execute_hg_real_pilot",
    "run_hg_pilot_fake_smoke",
]

if __name__ == "__main__":
    raise SystemExit(main())
