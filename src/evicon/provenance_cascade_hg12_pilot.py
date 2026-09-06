"""Isolated opt-in H-G.1.2 Pilot runner with offline-first gates."""
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

from .cascade_agent_protocol_hg11 import (
    HG11_PROTOCOL_VERSION, HG11_SCHEMA_NAME, HG11_TEMPLATE_VERSION,
    CascadeAgentProtocolHG11Runtime, build_identifiable_prompt_context,
)
from .cascade_controller import CascadeControllerPolicyLoader
from .cascade_controller_hg1 import propose_hg1, validate_hg1_proposal
from .cascade_hg1_replay import HG1CascadeProtocolRunner, HG1OutcomeReplayValidator
from .cascade_real_agent_runner import CascadeRealAgentRunError, CascadeRealAgentRunRecord, CascadeRealAgentRunner
from .llm_contract import LLMProvider
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ResponseFormatMode
from .provenance_cascade_hg1 import HG1FakeProvider, _claim_selector, _post_round_events
from .provenance_cascade_hg12_design import DEFAULT_CONFIG, load_hg12_design
from .provenance_cascade_hg12_governance import (
    DEFAULT_APPROVAL, DEFAULT_COMPATIBILITY_RECEIPT, final_preflight,
    load_approval, sha256_file, validate_receipt,
)
from .provenance_cascade_preregistration import CascadeCondition
from .request_ledger import RequestLedger

_ROOT = Path(__file__).resolve().parents[2]


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


class RunStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class HG12RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    matched_group_id: str
    scenario_id: str
    seed: int
    condition: CascadeCondition
    expected_provider_requests: int = 18
    completion_reservation: int = 18432


class HG12RunState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    scenario_id: str
    seed: int
    condition: str
    status: RunStatus = RunStatus.PLANNED
    logical_request_count: int = Field(default=0, ge=0)
    transport_attempt_count: int = Field(default=0, ge=0)
    directive_applied_count: int = Field(default=0, ge=0)
    replay_status: str | None = None
    run_record_sha256: str | None = None
    error_code: str | None = None


class HG12BatchRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    batch_version: str = "provenance_cascade_hg12_execution.v1"
    study_id: str
    status: RunStatus
    binding_sha256: str
    config_sha256: str
    protocol_sha256: str
    controller_sha256: str
    replay_sha256: str
    amendment_sha256: str
    amendment_receipt_sha256: str
    approval_sha256: str
    compatibility_receipt_sha256: str
    model_name: str
    output_root: str
    runs: tuple[HG12RunState, ...]
    failure_code: str | None = None
    development_only: bool = True
    pilot_only: bool = True
    not_paper_result: bool = True
    no_causal_conclusion: bool = True
    private_truth_exposed: bool = False
    parser_recovery_enabled: bool = False


class HG12PilotReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: str = "completed"
    study_id: str
    config_sha256: str
    protocol_sha256: str
    template_sha256: str
    controller_sha256: str
    replay_sha256: str
    amendment_sha256: str
    amendment_receipt_sha256: str
    approval_sha256: str
    compatibility_receipt_sha256: str
    protocol_version: str = HG11_PROTOCOL_VERSION
    template_version: str = HG11_TEMPLATE_VERSION
    response_format: str = "json_schema"
    schema_name: str = HG11_SCHEMA_NAME
    run_count: int
    completed_run_count: int
    matched_group_count: int
    logical_request_count: int
    provider_call_count: int
    transport_attempt_count: int
    actual_prompt_token_count: int | None
    actual_completion_token_count: int | None
    actual_total_token_count: int | None
    replay_passed_count: int
    replay_statuses: dict[str, str]
    directive_applied_count: int
    request_cap: int
    completion_reservation_cap: int
    ledger_hash: str
    network: str
    results_written: bool
    development_only: bool = True
    pilot_only: bool = True
    not_paper_result: bool = True
    no_causal_conclusion: bool = True
    private_truth_exposed: bool = False
    parser_recovery_enabled: bool = False


class HG12ExecutionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: str
    error_code: str | None = None
    completed_run_count: int = 0
    failed_run_count: int = 0
    logical_request_count: int = 0
    transport_attempt_count: int = 0
    actual_prompt_token_count: int | None = None
    actual_completion_token_count: int | None = None
    actual_total_token_count: int | None = None
    directive_applied_count: int = 0
    replay_statuses: dict[str, str] = Field(default_factory=dict)
    receipt_path: str | None = None
    provider_constructed_count: int = 0
    network: str = "disabled"
    results_written: bool = False
    ready_for_real_pilot: bool = False
    safety: dict[str, bool] = Field(default_factory=dict)


def _atomic(path: Path, value: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def _specs(config) -> tuple[HG12RunSpec, ...]:
    return tuple(
        HG12RunSpec(
            run_id=f"hg12-{material.scenario_id}-{seed}-{condition.value}",
            matched_group_id=f"hg12-{material.scenario_id}-{seed}",
            scenario_id=material.scenario_id, seed=seed, condition=condition,
        )
        for material in config.schedule_materials
        for seed in config.seeds
        for condition in config.conditions
    )


class HG12PilotRunner:
    def __init__(self, config_path: str | Path = DEFAULT_CONFIG) -> None:
        self.config_path = _path(config_path)
        self.config, self.scenarios, self.truths, self.schedules = load_hg12_design(self.config_path)
        self.runs = _specs(self.config)
        self.policy = CascadeControllerPolicyLoader.load(_path(self.config.policy_path))
        self.protocol_runner = HG1CascadeProtocolRunner()
        self.runner = CascadeRealAgentRunner(base_runner=self.protocol_runner, policy_config=self.policy)

    def run_one(self, spec: HG12RunSpec, *, provider: LLMProvider, root: Path, model_name: str, resume: bool = False):
        if spec not in self.runs:
            raise CascadeRealAgentRunError("run_spec_not_registered")
        schedule = self.schedules[spec.scenario_id]
        run_dir = root / spec.run_id
        return self.runner.run_scenario(
            self.scenarios[spec.scenario_id], spec.seed, spec.condition,
            provider=provider, run_id=spec.run_id,
            ledger_path=run_dir / "request_ledger.jsonl",
            checkpoint_path=run_dir / "agent_checkpoint.json",
            model_name=model_name, agent_temperature=0.2,
            agent_max_tokens=1024, request_cap=18,
            completion_reservation_cap=18432, resume=resume,
            recovery_fingerprints=None, recovery_kind=None, max_recovery_attempts=0,
            runtime=CascadeAgentProtocolHG11Runtime(),
            context_builder=build_identifiable_prompt_context,
            outcome_replay_validator=HG1OutcomeReplayValidator,
            initial_events_builder=self.protocol_runner.preload_initial_events,
            claim_selector=lambda scenario, snapshot, round_id: _claim_selector(schedule, scenario, snapshot, round_id),
            proposal_builder=propose_hg1, proposal_validator=validate_hg1_proposal,
            post_round_events_builder=lambda scenario, round_id, events: _post_round_events(scenario, schedule, round_id, events),
            actor_schedule_provider=lambda _scenario, round_id: schedule.propagation_slots if round_id == 0 else (),
            policy_config=self.policy,
        )

    def run_all(
        self, *, provider_factory: Any, root: str | Path | None, model_name: str,
        resume: bool = False, write_receipt: bool = True, network: str = "enabled",
    ) -> tuple[list[CascadeRealAgentRunRecord], HG12PilotReceipt]:
        approval = load_approval()
        receipt = validate_receipt(expected_hash=approval.compatibility_receipt_sha256)
        root_path = _path(self.config.output_root) if root is None else Path(root).resolve()
        binding = self._binding(root_path, model_name, approval.compatibility_receipt_sha256)
        batch_path = root_path / "pilot_batch_record.json"
        receipt_path = root_path / "pilot_receipt.json"
        if root_path.exists():
            if not resume:
                raise CascadeRealAgentRunError("output_root_exists")
            batch = self._load_batch(batch_path)
            self._validate_batch(batch, binding, root_path, model_name)
        else:
            if resume:
                raise CascadeRealAgentRunError("resume_output_missing")
            root_path.mkdir(parents=True, exist_ok=False)
            batch = HG12BatchRecord(
                study_id=self.config.study_id, status=RunStatus.RUNNING,
                binding_sha256=binding, config_sha256=sha256_file(self.config_path),
                protocol_sha256=self.config.protocol_sha256,
                controller_sha256=self.config.controller_sha256,
                replay_sha256=self.config.replay_sha256,
                amendment_sha256=self.config.amendment_sha256,
                amendment_receipt_sha256=approval.amendment_receipt_sha256,
                approval_sha256=sha256_file(DEFAULT_APPROVAL),
                compatibility_receipt_sha256=sha256_file(DEFAULT_COMPATIBILITY_RECEIPT),
                model_name=model_name, output_root=str(root_path),
                runs=tuple(HG12RunState(
                    run_id=spec.run_id, scenario_id=spec.scenario_id,
                    seed=spec.seed, condition=spec.condition.value,
                ) for spec in self.runs),
            )
            _atomic(batch_path, batch)
        records: list[CascadeRealAgentRunRecord] = []
        for spec in self.runs:
            state = next(item for item in batch.runs if item.run_id == spec.run_id)
            run_dir = root_path / spec.run_id
            record_path = run_dir / "run_record.json"
            if record_path.exists():
                if not resume:
                    raise CascadeRealAgentRunError("run_output_exists")
                record = self._load_record(record_path, spec)
                records.append(record)
                batch = self._replace(batch, self._state(spec, record, run_dir), RunStatus.RUNNING, None)
                _atomic(batch_path, batch)
                continue
            if state.status is RunStatus.COMPLETED:
                raise CascadeRealAgentRunError("completed_run_record_missing")
            batch = self._replace(batch, state.model_copy(update={"status": RunStatus.RUNNING, "error_code": None}), RunStatus.RUNNING, None)
            _atomic(batch_path, batch)
            try:
                provider = provider_factory(spec)
                record = self.run_one(spec, provider=provider, root=root_path, model_name=model_name, resume=resume)
                if record.replay is None or record.replay.status.value != "passed":
                    raise CascadeRealAgentRunError("run_replay_incomplete")
            except CascadeRealAgentRunError as exc:
                failed = state.model_copy(update={"status": RunStatus.FAILED, "error_code": exc.code})
                batch = self._replace(batch, failed, RunStatus.FAILED, exc.code)
                _atomic(batch_path, batch)
                raise
            except Exception:
                failed = state.model_copy(update={"status": RunStatus.FAILED, "error_code": "provider_construction_failed"})
                batch = self._replace(batch, failed, RunStatus.FAILED, "provider_construction_failed")
                _atomic(batch_path, batch)
                raise CascadeRealAgentRunError("provider_construction_failed") from None
            safe = record.model_copy(update={"request_ledger_path": "request_ledger.jsonl"})
            if record_path.exists():
                raise CascadeRealAgentRunError("run_record_exists")
            _atomic(record_path, safe)
            records.append(safe)
            batch = self._replace(batch, self._state(spec, safe, run_dir), RunStatus.RUNNING, None)
            _atomic(batch_path, batch)
        if len(records) != 48 or any(item.status is not RunStatus.COMPLETED for item in batch.runs):
            raise CascadeRealAgentRunError("batch_incomplete")
        totals = self._ledger_totals(root_path)
        pilot_receipt = HG12PilotReceipt(
            study_id=self.config.study_id, config_sha256=sha256_file(self.config_path),
            protocol_sha256=self.config.protocol_sha256, template_sha256=self.config.protocol_sha256,
            controller_sha256=self.config.controller_sha256, replay_sha256=self.config.replay_sha256,
            amendment_sha256=self.config.amendment_sha256,
            amendment_receipt_sha256=approval.amendment_receipt_sha256,
            approval_sha256=sha256_file(DEFAULT_APPROVAL),
            compatibility_receipt_sha256=sha256_file(DEFAULT_COMPATIBILITY_RECEIPT),
            run_count=48, completed_run_count=48, matched_group_count=12,
            logical_request_count=int(totals["logical_request_count"] or 0),
            provider_call_count=int(totals["provider_call_count"] or 0),
            transport_attempt_count=int(totals["transport_attempt_count"] or 0),
            actual_prompt_token_count=totals["actual_prompt_token_count"],
            actual_completion_token_count=totals["actual_completion_token_count"],
            actual_total_token_count=totals["actual_total_token_count"],
            replay_passed_count=48,
            replay_statuses={record.run_id: record.replay.status.value for record in records if record.replay},
            directive_applied_count=sum(record.directive_applied_count for record in records),
            request_cap=864, completion_reservation_cap=884736,
            ledger_hash=hashlib.sha256(json.dumps([
                (record.run_id, record.exposure_ledger_sha256, record.application_ledger_sha256, record.outcome_ledger_sha256)
                for record in records
            ], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            network=network, results_written=write_receipt,
        )
        if write_receipt:
            if receipt_path.exists():
                existing = HG12PilotReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
                if existing != pilot_receipt:
                    raise CascadeRealAgentRunError("receipt_binding_mismatch")
                pilot_receipt = existing
            else:
                _atomic(receipt_path, pilot_receipt)
        batch = batch.model_copy(update={"status": RunStatus.COMPLETED, "failure_code": None})
        _atomic(batch_path, batch)
        return records, pilot_receipt

    def _binding(self, root: Path, model: str, compatibility_sha: str) -> str:
        data = {
            "config_sha256": sha256_file(self.config_path), "protocol_sha256": self.config.protocol_sha256,
            "controller_sha256": self.config.controller_sha256, "replay_sha256": self.config.replay_sha256,
            "amendment_sha256": self.config.amendment_sha256, "approval_sha256": sha256_file(DEFAULT_APPROVAL),
            "compatibility_receipt_sha256": compatibility_sha, "model": model,
            "output_root": str(root), "run_ids": [spec.run_id for spec in self.runs],
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @staticmethod
    def _load_batch(path: Path) -> HG12BatchRecord:
        try:
            return HG12BatchRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CascadeRealAgentRunError("resume_batch_record_invalid") from exc

    def _validate_batch(self, batch: HG12BatchRecord, binding: str, root: Path, model: str) -> None:
        if batch.binding_sha256 != binding or batch.output_root != str(root) or batch.model_name != model:
            raise CascadeRealAgentRunError("resume_binding_mismatch")
        if tuple(item.run_id for item in batch.runs) != tuple(spec.run_id for spec in self.runs):
            raise CascadeRealAgentRunError("resume_run_plan_mismatch")
        if batch.config_sha256 != sha256_file(self.config_path):
            raise CascadeRealAgentRunError("resume_config_hash_mismatch")

    def _load_record(self, path: Path, spec: HG12RunSpec) -> CascadeRealAgentRunRecord:
        try:
            record = CascadeRealAgentRunRecord.model_validate_json(path.read_text(encoding="utf-8"))
            if record.run_id != spec.run_id or record.scenario_id != spec.scenario_id or record.seed != spec.seed or record.condition != spec.condition:
                raise ValueError("binding")
            replay = HG1OutcomeReplayValidator.validate(
                self.scenarios[spec.scenario_id].graph, record.exposure_ledger,
                record.outcome_ledger, record.application_ledger, record.round_contexts,
            )
            if replay.status.value != "passed":
                raise ValueError("replay")
            return record.model_copy(update={"replay": replay})
        except Exception as exc:
            raise CascadeRealAgentRunError("run_record_binding_mismatch") from exc

    @staticmethod
    def _replace(batch: HG12BatchRecord, state: HG12RunState, status: RunStatus, code: str | None):
        return batch.model_copy(update={
            "runs": tuple(state if item.run_id == state.run_id else item for item in batch.runs),
            "status": status, "failure_code": code,
        })

    @staticmethod
    def _state(spec: HG12RunSpec, record: CascadeRealAgentRunRecord, run_dir: Path) -> HG12RunState:
        summary = RequestLedger(run_dir / "request_ledger.jsonl").summary(
            request_cap=18, completion_reservation_cap=18432,
        )
        return HG12RunState(
            run_id=spec.run_id, scenario_id=spec.scenario_id, seed=spec.seed,
            condition=spec.condition.value, status=RunStatus.COMPLETED,
            logical_request_count=summary.unique_logical_request_count,
            transport_attempt_count=summary.transport_attempt_count,
            directive_applied_count=record.directive_applied_count,
            replay_status=record.replay.status.value if record.replay else None,
            run_record_sha256=hashlib.sha256((run_dir / "run_record.json").read_bytes()).hexdigest(),
        )

    def _ledger_totals(self, root: Path) -> dict[str, int | None]:
        summaries = [RequestLedger(root / spec.run_id / "request_ledger.jsonl").summary(
            request_cap=18, completion_reservation_cap=18432,
        ) for spec in self.runs]
        def total(field: str):
            values = [getattr(item, field) for item in summaries]
            return sum(values) if all(value is not None for value in values) else None
        return {
            "logical_request_count": sum(item.unique_logical_request_count for item in summaries),
            "provider_call_count": sum(item.provider_call_count for item in summaries),
            "transport_attempt_count": sum(item.transport_attempt_count for item in summaries),
            "actual_prompt_token_count": total("actual_prompt_token_count"),
            "actual_completion_token_count": total("actual_completion_token_count"),
            "actual_total_token_count": total("actual_total_token_count"),
        }


def run_fake_smoke() -> dict[str, object]:
    runner = HG12PilotRunner()
    with tempfile.TemporaryDirectory(prefix="evicon-hg12-pilot-") as temporary:
        records, receipt = runner.run_all(
            provider_factory=lambda _spec: HG1FakeProvider(), root=Path(temporary) / "pilot",
            model_name="hg12-fake", write_receipt=False, network="disabled",
        )
    return {
        "status": "fake_smoke_passed", "run_count": len(records),
        "matched_group_count": receipt.matched_group_count,
        "logical_request_count": receipt.logical_request_count,
        "provider_call_count": receipt.provider_call_count,
        "transport_attempt_count": receipt.transport_attempt_count,
        "replay_passed_count": receipt.replay_passed_count,
        "directive_applied_count": receipt.directive_applied_count,
        "completion_reservation_cap": receipt.completion_reservation_cap,
        "network": "disabled", "results_written": False,
        "private_truth_exposed": False, "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def _safety(constructed: bool = False) -> dict[str, bool]:
    return {
        "development_only": True, "pilot_only": True, "not_paper_result": True,
        "no_causal_conclusion": True, "private_truth_exposed": False,
        "provider_constructed": constructed, "parser_recovery_enabled": False,
    }


def execute_real(
    *, allow_network: bool = False, confirm_run: bool = False,
    confirm_request_cap: int | None = None,
    confirm_completion_reservation_cap: int | None = None,
    resume: bool = False, environment: Mapping[str, str] | None = None,
) -> HG12ExecutionSummary:
    checks = (
        (allow_network, "allow_network_required"),
        (confirm_run, "confirm_run_required"),
        (confirm_request_cap == 864, "confirm_request_cap_must_equal_864"),
        (confirm_completion_reservation_cap == 884736, "confirm_completion_reservation_cap_must_equal_884736"),
    )
    for valid, code in checks:
        if not valid:
            return HG12ExecutionSummary(status="blocked", error_code=code, safety=_safety())
    gate = final_preflight()
    if not gate["ready_for_real_pilot"] and not (resume and gate["blocking_reasons"] == ["output_root_exists"]):
        return HG12ExecutionSummary(status="blocked", error_code=gate["blocking_reasons"][0], safety=_safety())
    source = os.environ if environment is None else environment
    if not all(source.get(name) for name in ("EVICON_LLM_BASE_URL", "EVICON_LLM_MODEL", "EVICON_LLM_API_KEY")):
        return HG12ExecutionSummary(status="blocked", error_code="provider_environment_incomplete", safety=_safety())
    runner = HG12PilotRunner()
    root = _path(runner.config.output_root)
    model = str(source["EVICON_LLM_MODEL"])
    if root.exists() and resume:
        approval = load_approval()
        batch = runner._load_batch(root / "pilot_batch_record.json")
        runner._validate_batch(batch, runner._binding(root, model, approval.compatibility_receipt_sha256), root, model)
    base = ProviderConfig.from_env(allow_network=True, environment=source)
    constructed = 0
    def factory(spec: HG12RunSpec):
        nonlocal constructed
        constructed += 1
        settings = base.model_copy(update={
            "timeout_seconds": 15.0, "max_retries": 1, "temperature": 0.2,
            "max_tokens": 1024, "seed": spec.seed, "reasoning_effort": None,
            "response_format": ResponseFormatMode.JSON_SCHEMA,
            "response_schema_name": HG11_SCHEMA_NAME,
        })
        return OpenAICompatibleProvider(settings, environment=source)
    try:
        records, receipt = runner.run_all(
            provider_factory=factory, root=None, model_name=model,
            resume=resume, write_receipt=True, network="enabled",
        )
    except CascadeRealAgentRunError as exc:
        return HG12ExecutionSummary(
            status="failed", error_code=exc.code, provider_constructed_count=constructed,
            network="enabled", results_written=root.exists(), safety=_safety(constructed > 0),
        )
    return HG12ExecutionSummary(
        status="completed", completed_run_count=len(records),
        logical_request_count=receipt.logical_request_count,
        transport_attempt_count=receipt.transport_attempt_count,
        actual_prompt_token_count=receipt.actual_prompt_token_count,
        actual_completion_token_count=receipt.actual_completion_token_count,
        actual_total_token_count=receipt.actual_total_token_count,
        directive_applied_count=receipt.directive_applied_count,
        replay_statuses=receipt.replay_statuses,
        receipt_path=str(root / "pilot_receipt.json"),
        provider_constructed_count=constructed, network="enabled",
        results_written=True, ready_for_real_pilot=True, safety=_safety(True),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="H-G.1.2 provenance-cascade Pilot")
    parser.add_argument("--mode", choices=("preflight", "fake-smoke", "real-pilot"), default="preflight")
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
            allow_network=args.allow_network, confirm_run=args.confirm_run,
            confirm_request_cap=args.confirm_request_cap,
            confirm_completion_reservation_cap=args.confirm_completion_reservation_cap,
            resume=args.resume,
        ).model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if payload.get("status") in {"completed", "fake_smoke_passed", "ready_for_real_pilot"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
