"""Isolated H-G.2.1 real eligibility-calibration runner and offline gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import tomllib
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cascade_agent_protocol_hg2 import HG2DecisionTask, HG2PromptContext, HG2RuntimeResult
from .cascade_agent_protocol_hg21 import (
    HG21_PROTOCOL_VERSION, HG21_RESPONSE_JSON_SCHEMA, HG21_SCHEMA_NAME,
    HG21_TEMPLATE_VERSION, HG21Runtime,
)
from .cascade_agent_protocol_hg import build_identifiable_prompt_context
from .cascade_agent_runtime import CascadeAgentRuntimeStatus
from .cascade_controller import CascadeControllerPolicyLoader
from .cascade_controller_hg1 import propose_hg1, validate_hg1_proposal
from .cascade_hg1_replay import HG1CascadeProtocolRunner, HG1OutcomeReplayValidator
from .cascade_real_agent_runner import CascadeRealAgentRunError, CascadeRealAgentRunRecord, CascadeRealAgentRunner
from .llm_contract import LLMProvider
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ResponseFormatMode
from .provenance_cascade_hg1 import _claim_selector, _post_round_events
from .provenance_cascade_hg2 import HG2FakeProvider, load_hg2_config, sha256_file
from .provenance_cascade_hg21 import (
    DEFAULT_AMENDMENT_RECEIPT, DEFAULT_APPROVAL, DEFAULT_COMPATIBILITY_RECEIPT,
    DEFAULT_CONFIG, HG21Approval, load_config,
)
from .provenance_cascade_hg21_compatibility_receipt import validate_receipt
from .provenance_cascade_preregistration import CascadeCondition
from .request_ledger import RequestLedger, request_fingerprint_facts

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXECUTION_APPROVAL = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg21_execution_approval.toml"
POLICY_PATH = "configs/provenance_cascade/pilot/amendments/hg1_identifiability_v1/cascade_controller_policy.hg1.v1.toml"
CONTROLLER_PATH = "src/evicon/cascade_controller_hg1.py"
REPLAY_PATH = "src/evicon/cascade_hg1_replay.py"
PARENT_CONFIG = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg2_calibration.v1.toml"


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def _hash_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


class CalibrationStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class HG21RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    matched_group_id: str
    scenario_id: str
    seed: int
    condition: CascadeCondition
    expected_provider_requests: int = 18
    completion_reservation: int = 18432


class HG21RunState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    scenario_id: str
    seed: int
    condition: str
    status: CalibrationStatus = CalibrationStatus.PLANNED
    logical_request_count: int = Field(default=0, ge=0)
    transport_attempt_count: int = Field(default=0, ge=0)
    directive_applied_count: int = Field(default=0, ge=0)
    behavior_observation_count: int = Field(default=0, ge=0)
    replay_status: str | None = None
    run_record_sha256: str | None = None
    error_code: str | None = None


class HG21BatchRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    batch_version: str = "provenance_cascade_hg21_eligibility_calibration.v1"
    study_id: str
    status: CalibrationStatus
    binding_sha256: str
    config_sha256: str
    protocol_sha256: str
    controller_sha256: str
    replay_sha256: str
    amendment_receipt_sha256: str
    design_approval_sha256: str
    execution_approval_sha256: str
    compatibility_receipt_sha256: str
    model_name: str
    output_root: str
    runs: tuple[HG21RunState, ...]
    failure_code: str | None = None
    development_only: bool = True
    calibration_only: bool = True
    not_paper_result: bool = True
    no_causal_conclusion: bool = True
    private_truth_exposed: bool = False
    parser_recovery_enabled: bool = False


class HG21DecisionRecord(BaseModel):
    """Append-only parsed decision audit with no prompts or model text."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    binding_sha256: str = Field(min_length=64, max_length=64)
    fingerprint: str = Field(min_length=64, max_length=64)
    agent_id: str
    round_id: int = Field(ge=0)
    epistemic_stance: str
    behavioral_decision: str
    content_id_count: int = Field(ge=0)
    evidence_id_count: int = Field(ge=0)
    share_requested: bool


class HG21CalibrationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: str = "eligibility_calibration_completed"
    study_id: str
    config_sha256: str
    protocol_sha256: str
    template_sha256: str
    controller_sha256: str
    replay_sha256: str
    amendment_receipt_sha256: str
    design_approval_sha256: str
    execution_approval_sha256: str
    compatibility_receipt_sha256: str
    protocol_version: str = HG21_PROTOCOL_VERSION
    template_version: str = HG21_TEMPLATE_VERSION
    response_format: str = "json_schema"
    schema_name: str = HG21_SCHEMA_NAME
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
    behavior_observation_count: int
    round0_non_defer_behavior_count: int
    share_decision_count: int
    request_cap: int
    completion_reservation_cap: int
    ledger_hash: str
    network: str
    results_written: bool
    eligibility_only: bool = True
    effectiveness_claimed: bool = False
    full_pilot_authorized: bool = False
    development_only: bool = True
    calibration_only: bool = True
    not_paper_result: bool = True
    no_causal_conclusion: bool = True
    private_truth_exposed: bool = False
    parser_recovery_enabled: bool = False


class HG21ExecutionSummary(BaseModel):
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
    behavior_observation_count: int = 0
    replay_statuses: dict[str, str] = Field(default_factory=dict)
    receipt_path: str | None = None
    provider_constructed_count: int = 0
    network: str = "disabled"
    results_written: bool = False
    ready_for_network_authorization: bool = False
    safety: dict[str, bool] = Field(default_factory=dict)


class HG21ExecutionApproval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    approval_id: str
    acceptance_status: str
    accepted_by: str
    config_sha256: str
    protocol_sha256: str
    design_approval_sha256: str
    compatibility_receipt_sha256: str
    runner_sha256: str
    run_count: int
    matched_group_count: int
    logical_request_cap: int
    completion_reservation_cap: int
    confirm_one_seed_eligibility_only: bool
    confirm_no_effectiveness_claim: bool
    confirm_append_only_ledger: bool
    confirm_resume_and_no_overwrite: bool
    confirm_parser_invalid_stops: bool
    network_execution_authorized: bool

    @model_validator(mode="after")
    def accepted(self) -> "HG21ExecutionApproval":
        confirmations = (
            self.confirm_one_seed_eligibility_only, self.confirm_no_effectiveness_claim,
            self.confirm_append_only_ledger, self.confirm_resume_and_no_overwrite,
            self.confirm_parser_invalid_stops,
        )
        if self.acceptance_status == "accepted" and (not self.accepted_by.strip() or not all(confirmations)):
            raise ValueError("execution approval incomplete")
        return self


class RecordingHG21Runtime(HG21Runtime):
    def __init__(self, path: Path, binding_sha256: str) -> None:
        super().__init__()
        self.path = path
        self.binding_sha256 = binding_sha256

    def execute(self, context, provider, *, request_metadata: Mapping[str, Any] | None = None) -> HG2RuntimeResult:
        result = super().execute(context, provider, request_metadata=request_metadata)
        if result.status is not CascadeAgentRuntimeStatus.COMPLETED or result.response is None:
            return result
        request = self.render_request(context)
        if request_metadata:
            allowed = {key: value for key, value in request_metadata.items() if key in {"protocol", "condition", "phase", "run_id", "seed", "matched_group_id"}}
            request = request.model_copy(update={"metadata": {**request.metadata, **allowed}})
        fingerprint = str(request_fingerprint_facts(request)["fingerprint"])
        existing = {entry.fingerprint: entry for entry in load_decisions(self.path, self.binding_sha256)}
        record = HG21DecisionRecord(
            binding_sha256=self.binding_sha256, fingerprint=fingerprint,
            agent_id=context.public_context.agent_id, round_id=context.public_context.round_id,
            epistemic_stance=result.response.epistemic_stance.value,
            behavioral_decision=result.response.behavioral_decision.value,
            content_id_count=len(result.response.content_ids_used),
            evidence_id_count=len(result.response.evidence_ids_used),
            share_requested=result.response.share_content_id is not None,
        )
        if fingerprint in existing:
            if existing[fingerprint] != record:
                raise CascadeRealAgentRunError("decision_checkpoint_fingerprint_conflict")
            return result
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json())
            handle.write("\n")
        return result


def load_decisions(path: Path, binding: str) -> tuple[HG21DecisionRecord, ...]:
    if not path.exists():
        return ()
    try:
        records = tuple(HG21DecisionRecord.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    except Exception as exc:
        raise CascadeRealAgentRunError("decision_checkpoint_invalid") from exc
    if any(record.binding_sha256 != binding for record in records):
        raise CascadeRealAgentRunError("decision_checkpoint_binding_mismatch")
    if len({record.fingerprint for record in records}) != len(records):
        raise CascadeRealAgentRunError("decision_checkpoint_duplicate")
    return records


def _specs(config) -> tuple[HG21RunSpec, ...]:
    return tuple(
        HG21RunSpec(
            run_id=f"hg21-{scenario_id}-{config.seed}-{condition.value}",
            matched_group_id=f"hg21-{scenario_id}-{config.seed}",
            scenario_id=scenario_id, seed=config.seed, condition=condition,
        )
        for scenario_id in config.scenario_ids
        for condition in config.conditions
    )


class HG21EligibilityCalibrationRunner:
    def __init__(self, config_path: str | Path = DEFAULT_CONFIG) -> None:
        self.amendment, self.config_path = load_config(config_path)
        self.config, self.scenarios, self.schedules = load_hg2_config(_path(PARENT_CONFIG))
        self.runs = _specs(self.config)
        self.policy = CascadeControllerPolicyLoader.load(_path(POLICY_PATH))
        self.protocol_runner = HG1CascadeProtocolRunner()
        self.runner = CascadeRealAgentRunner(base_runner=self.protocol_runner, policy_config=self.policy)
        self.tasks = {item.scenario_id: HG2DecisionTask.model_validate(item.model_dump(exclude={"scenario_id"})) for item in self.config.decision_tasks}

    def _run_binding(self, spec: HG21RunSpec, model_name: str) -> str:
        return _hash_json({
            "config_sha256": sha256_file(self.config_path), "protocol_sha256": self.amendment.protocol_sha256,
            "scenario_id": spec.scenario_id, "seed": spec.seed, "condition": spec.condition.value,
            "run_id": spec.run_id, "model_name": model_name, "max_tokens": self.amendment.agent_max_tokens,
            "temperature": self.amendment.temperature,
        })

    def run_one(self, spec: HG21RunSpec, *, provider: LLMProvider, root: Path, model_name: str, resume: bool = False) -> tuple[CascadeRealAgentRunRecord, tuple[HG21DecisionRecord, ...]]:
        if spec not in self.runs:
            raise CascadeRealAgentRunError("run_spec_not_registered")
        schedule = self.schedules[spec.scenario_id]
        run_dir = root / spec.run_id
        decision_path = run_dir / "behavior_decisions.jsonl"
        binding = self._run_binding(spec, model_name)
        runtime = RecordingHG21Runtime(decision_path, binding)

        def context_builder(view, *, claim_id, directive, runtime_config):
            public = build_identifiable_prompt_context(view, claim_id=claim_id, directive=directive, runtime_config=runtime_config)
            return HG2PromptContext(public_context=public, role=self.config.roles[view.agent_id], decision_task=self.tasks[view.scenario_id])

        record = self.runner.run_scenario(
            self.scenarios[spec.scenario_id], spec.seed, spec.condition,
            provider=provider, run_id=spec.run_id,
            ledger_path=run_dir / "request_ledger.jsonl",
            checkpoint_path=run_dir / "agent_checkpoint.json",
            model_name=model_name, agent_temperature=self.amendment.temperature,
            agent_max_tokens=self.amendment.agent_max_tokens,
            request_cap=spec.expected_provider_requests,
            completion_reservation_cap=spec.completion_reservation,
            resume=resume, recovery_fingerprints=None, recovery_kind=None, max_recovery_attempts=0,
            runtime=runtime, context_builder=context_builder,
            outcome_replay_validator=HG1OutcomeReplayValidator,
            initial_events_builder=self.protocol_runner.preload_initial_events,
            claim_selector=lambda scenario, snapshot, round_id: _claim_selector(schedule, scenario, snapshot, round_id),
            proposal_builder=propose_hg1, proposal_validator=validate_hg1_proposal,
            post_round_events_builder=lambda scenario, round_id, events: _post_round_events(scenario, schedule, round_id, events),
            actor_schedule_provider=lambda _scenario, round_id: schedule.propagation_slots if round_id == 0 else (),
            policy_config=self.policy,
        )
        decisions = load_decisions(decision_path, binding)
        if len(decisions) != spec.expected_provider_requests:
            raise CascadeRealAgentRunError("behavior_decision_checkpoint_incomplete")
        return record, decisions

    def _batch_binding(self, root: Path, model: str, execution_approval_sha: str) -> str:
        return _hash_json({
            "config_sha256": sha256_file(self.config_path), "protocol_sha256": self.amendment.protocol_sha256,
            "design_approval_sha256": sha256_file(DEFAULT_APPROVAL),
            "execution_approval_sha256": execution_approval_sha,
            "compatibility_receipt_sha256": sha256_file(DEFAULT_COMPATIBILITY_RECEIPT),
            "model": model, "output_root": str(root), "run_ids": [spec.run_id for spec in self.runs],
        })

    def run_all(self, *, provider_factory: Any, root: str | Path | None, model_name: str, resume: bool = False, write_receipt: bool = True, network: str = "enabled") -> tuple[list[CascadeRealAgentRunRecord], HG21CalibrationReceipt]:
        design = HG21Approval.model_validate(tomllib.loads(_path(DEFAULT_APPROVAL).read_text(encoding="utf-8")))
        validate_receipt(DEFAULT_COMPATIBILITY_RECEIPT, expected_hash=design.compatibility_receipt_sha256)
        execution_path = _path(DEFAULT_EXECUTION_APPROVAL)
        execution = HG21ExecutionApproval.model_validate(tomllib.loads(execution_path.read_text(encoding="utf-8")))
        root_path = _path(self.amendment.output_root) if root is None else Path(root).resolve()
        binding = self._batch_binding(root_path, model_name, sha256_file(execution_path))
        batch_path, receipt_path = root_path / "calibration_batch_record.json", root_path / "calibration_receipt.json"
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
                study_id=self.amendment.study_id, status=CalibrationStatus.RUNNING,
                binding_sha256=binding, config_sha256=sha256_file(self.config_path),
                protocol_sha256=self.amendment.protocol_sha256,
                controller_sha256=sha256_file(CONTROLLER_PATH), replay_sha256=sha256_file(REPLAY_PATH),
                amendment_receipt_sha256=sha256_file(DEFAULT_AMENDMENT_RECEIPT),
                design_approval_sha256=sha256_file(DEFAULT_APPROVAL), execution_approval_sha256=sha256_file(execution_path),
                compatibility_receipt_sha256=sha256_file(DEFAULT_COMPATIBILITY_RECEIPT),
                model_name=model_name, output_root=str(root_path),
                runs=tuple(HG21RunState(run_id=s.run_id, scenario_id=s.scenario_id, seed=s.seed, condition=s.condition.value) for s in self.runs),
            )
            _atomic(batch_path, batch)
        records: list[CascadeRealAgentRunRecord] = []
        all_decisions: list[HG21DecisionRecord] = []
        for spec in self.runs:
            state = next(item for item in batch.runs if item.run_id == spec.run_id)
            run_dir, record_path = root_path / spec.run_id, root_path / spec.run_id / "run_record.json"
            if record_path.exists():
                if not resume:
                    raise CascadeRealAgentRunError("run_output_exists")
                try:
                    record = CascadeRealAgentRunRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
                    replay = HG1OutcomeReplayValidator.validate(
                        self.scenarios[spec.scenario_id].graph, record.exposure_ledger,
                        record.outcome_ledger, record.application_ledger, record.round_contexts,
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
            batch = self._replace(batch, state.model_copy(update={"status": CalibrationStatus.RUNNING, "error_code": None}), CalibrationStatus.RUNNING, None)
            _atomic(batch_path, batch)
            try:
                provider = provider_factory(spec)
                record, decisions = self.run_one(spec, provider=provider, root=root_path, model_name=model_name, resume=resume)
                if record.replay is None or record.replay.status.value != "passed":
                    raise CascadeRealAgentRunError("run_replay_incomplete")
            except CascadeRealAgentRunError as exc:
                batch = self._replace(batch, state.model_copy(update={"status": CalibrationStatus.FAILED, "error_code": exc.code}), CalibrationStatus.FAILED, exc.code)
                _atomic(batch_path, batch)
                raise
            except Exception:
                batch = self._replace(batch, state.model_copy(update={"status": CalibrationStatus.FAILED, "error_code": "provider_construction_failed"}), CalibrationStatus.FAILED, "provider_construction_failed")
                _atomic(batch_path, batch)
                raise CascadeRealAgentRunError("provider_construction_failed") from None
            safe = record.model_copy(update={"request_ledger_path": "request_ledger.jsonl"})
            if record_path.exists():
                raise CascadeRealAgentRunError("run_record_exists")
            _atomic(record_path, safe)
            records.append(safe)
            all_decisions.extend(decisions)
            summary = RequestLedger(run_dir / "request_ledger.jsonl").summary(request_cap=18, completion_reservation_cap=18432)
            completed = state.model_copy(update={
                "status": CalibrationStatus.COMPLETED, "logical_request_count": summary.unique_logical_request_count,
                "transport_attempt_count": summary.transport_attempt_count,
                "directive_applied_count": safe.directive_applied_count,
                "behavior_observation_count": len(decisions), "replay_status": safe.replay.status.value,
                "run_record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(), "error_code": None,
            })
            batch = self._replace(batch, completed, CalibrationStatus.RUNNING, None)
            _atomic(batch_path, batch)
        if len(records) != 16 or len(all_decisions) != 288:
            raise CascadeRealAgentRunError("calibration_incomplete")
        totals = self._ledger_totals(root_path)
        if totals["logical_request_count"] != 288:
            raise CascadeRealAgentRunError("logical_request_count_mismatch")
        receipt = HG21CalibrationReceipt(
            study_id=self.amendment.study_id, config_sha256=sha256_file(self.config_path),
            protocol_sha256=self.amendment.protocol_sha256, template_sha256=self.amendment.parent_protocol_sha256,
            controller_sha256=sha256_file(CONTROLLER_PATH), replay_sha256=sha256_file(REPLAY_PATH),
            amendment_receipt_sha256=sha256_file(DEFAULT_AMENDMENT_RECEIPT),
            design_approval_sha256=sha256_file(DEFAULT_APPROVAL), execution_approval_sha256=sha256_file(execution_path),
            compatibility_receipt_sha256=sha256_file(DEFAULT_COMPATIBILITY_RECEIPT),
            run_count=16, completed_run_count=16, matched_group_count=4,
            logical_request_count=totals["logical_request_count"], provider_call_count=totals["provider_call_count"],
            transport_attempt_count=totals["transport_attempt_count"],
            actual_prompt_token_count=totals["actual_prompt_token_count"], actual_completion_token_count=totals["actual_completion_token_count"],
            actual_total_token_count=totals["actual_total_token_count"], replay_passed_count=16,
            replay_statuses={record.run_id: record.replay.status.value for record in records if record.replay},
            directive_applied_count=sum(record.directive_applied_count for record in records), behavior_observation_count=len(all_decisions),
            round0_non_defer_behavior_count=sum(d.round_id == 0 and d.behavioral_decision != "defer_action" for d in all_decisions),
            share_decision_count=sum(d.share_requested for d in all_decisions), request_cap=288,
            completion_reservation_cap=294912,
            ledger_hash=_hash_json([(r.run_id, r.exposure_ledger_sha256, r.application_ledger_sha256, r.outcome_ledger_sha256) for r in records]),
            network=network, results_written=write_receipt,
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

    @staticmethod
    def _replace(batch: HG21BatchRecord, state: HG21RunState, status: CalibrationStatus, code: str | None) -> HG21BatchRecord:
        return batch.model_copy(update={
            "runs": tuple(state if item.run_id == state.run_id else item for item in batch.runs),
            "status": status, "failure_code": code,
        })

    def _ledger_totals(self, root: Path) -> dict[str, int | None]:
        summaries = [RequestLedger(root / spec.run_id / "request_ledger.jsonl").summary(request_cap=18, completion_reservation_cap=18432) for spec in self.runs]
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


def _safety(constructed: bool = False) -> dict[str, bool]:
    return {
        "development_only": True, "calibration_only": True, "not_paper_result": True,
        "no_causal_conclusion": True, "private_truth_exposed": False,
        "provider_constructed": constructed, "parser_recovery_enabled": False,
        "effectiveness_claimed": False, "full_pilot_authorized": False,
    }


def final_preflight() -> dict[str, object]:
    reasons: list[str] = []
    try:
        amendment, config_path = load_config()
        design = HG21Approval.model_validate(tomllib.loads(_path(DEFAULT_APPROVAL).read_text(encoding="utf-8")))
        validate_receipt(DEFAULT_COMPATIBILITY_RECEIPT, expected_hash=design.compatibility_receipt_sha256)
        execution_path = _path(DEFAULT_EXECUTION_APPROVAL)
        if not execution_path.is_file():
            reasons.append("calibration_network_authorization_required")
            execution = None
        else:
            execution = HG21ExecutionApproval.model_validate(tomllib.loads(execution_path.read_text(encoding="utf-8")))
            expected = {
                "config_sha256": sha256_file(config_path), "protocol_sha256": amendment.protocol_sha256,
                "design_approval_sha256": sha256_file(DEFAULT_APPROVAL),
                "compatibility_receipt_sha256": sha256_file(DEFAULT_COMPATIBILITY_RECEIPT),
                "runner_sha256": sha256_file(__file__),
            }
            if any(getattr(execution, key) != value for key, value in expected.items()):
                reasons.append("execution_approval_binding_mismatch")
            if execution.acceptance_status != "accepted" or not execution.network_execution_authorized:
                reasons.append("calibration_network_authorization_required")
            if (execution.run_count, execution.matched_group_count, execution.logical_request_cap, execution.completion_reservation_cap) != (16, 4, 288, 294912):
                reasons.append("execution_approval_scope_mismatch")
        if _path(amendment.output_root).exists():
            reasons.append("output_root_exists")
        return {
            "status": "ready_for_network_authorization" if not reasons else "blocked",
            "ready_for_network_authorization": not reasons,
            "blocking_reasons": sorted(set(reasons)), "config_sha256": sha256_file(config_path),
            "protocol_sha256": amendment.protocol_sha256, "design_approval_sha256": sha256_file(DEFAULT_APPROVAL),
            "compatibility_receipt_sha256": sha256_file(DEFAULT_COMPATIBILITY_RECEIPT),
            "execution_approval_sha256": sha256_file(DEFAULT_EXECUTION_APPROVAL) if _path(DEFAULT_EXECUTION_APPROVAL).exists() else None,
            "runner_sha256": sha256_file(__file__), "run_count": 16, "matched_group_count": 4,
            "logical_request_cap": 288, "completion_reservation_cap": 294912,
            "network": "disabled", "provider_constructed": False, "api_key_read": False,
            "results_written": False, "private_truth_exposed": False,
            "development_only": True, "calibration_only": True, "not_paper_result": True,
            "no_causal_conclusion": True, "full_pilot_authorized": False,
        }
    except Exception as exc:
        return {
            "status": "blocked", "ready_for_network_authorization": False,
            "blocking_reasons": [getattr(exc, "code", "calibration_preflight_failed")],
            "network": "disabled", "provider_constructed": False, "api_key_read": False,
            "results_written": False, "private_truth_exposed": False,
        }


def run_fake_smoke() -> dict[str, object]:
    runner = HG21EligibilityCalibrationRunner()
    with tempfile.TemporaryDirectory(prefix="evicon-hg21-calibration-") as temporary:
        root = Path(temporary) / "calibration"
        records, receipt = runner.run_all(
            provider_factory=lambda _spec: HG2FakeProvider(), root=root,
            model_name="hg21-fake", resume=False, write_receipt=False, network="disabled",
        )
        first_calls = receipt.provider_call_count
        records_resumed, resumed_receipt = runner.run_all(
            provider_factory=lambda _spec: (_ for _ in ()).throw(AssertionError("completed run replayed")),
            root=root, model_name="hg21-fake", resume=True, write_receipt=False, network="disabled",
        )
    return {
        "status": "fake_smoke_passed", "run_count": len(records), "resumed_run_count": len(records_resumed),
        "matched_group_count": receipt.matched_group_count, "logical_request_count": receipt.logical_request_count,
        "provider_call_count": receipt.provider_call_count, "resume_additional_provider_call_count": resumed_receipt.provider_call_count - first_calls,
        "transport_attempt_count": receipt.transport_attempt_count, "replay_passed_count": receipt.replay_passed_count,
        "directive_applied_count": receipt.directive_applied_count, "behavior_observation_count": receipt.behavior_observation_count,
        "round0_non_defer_behavior_count": receipt.round0_non_defer_behavior_count,
        "share_decision_count": receipt.share_decision_count,
        "completion_reservation_cap": receipt.completion_reservation_cap,
        "network": "disabled", "results_written": False, "private_truth_exposed": False,
        "effectiveness_claimed": False, "not_paper_result": True, "no_causal_conclusion": True,
    }


def execute_real(*, allow_network: bool = False, confirm_run: bool = False, confirm_request_cap: int | None = None, confirm_completion_reservation_cap: int | None = None, resume: bool = False, environment: Mapping[str, str] | None = None) -> HG21ExecutionSummary:
    checks = (
        (allow_network, "allow_network_required"), (confirm_run, "confirm_run_required"),
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
    runner = HG21EligibilityCalibrationRunner()
    model, constructed = str(source["EVICON_LLM_MODEL"]), 0
    base = ProviderConfig.from_env(allow_network=True, environment=source)
    def factory(spec: HG21RunSpec):
        nonlocal constructed
        constructed += 1
        settings = base.model_copy(update={
            "timeout_seconds": 15.0, "max_retries": 0, "temperature": 0.2,
            "max_tokens": 1024, "seed": spec.seed, "reasoning_effort": None,
            "response_format": ResponseFormatMode.JSON_SCHEMA,
            "response_schema_name": HG21_SCHEMA_NAME, "response_schema": HG21_RESPONSE_JSON_SCHEMA,
        })
        return OpenAICompatibleProvider(settings, environment=source)
    root = _path(runner.amendment.output_root)
    try:
        records, receipt = runner.run_all(provider_factory=factory, root=None, model_name=model, resume=resume, write_receipt=True, network="enabled")
    except CascadeRealAgentRunError as exc:
        return HG21ExecutionSummary(status="failed", error_code=exc.code, provider_constructed_count=constructed, network="enabled", results_written=root.exists(), safety=_safety(constructed > 0))
    return HG21ExecutionSummary(
        status="completed", completed_run_count=len(records), logical_request_count=receipt.logical_request_count,
        transport_attempt_count=receipt.transport_attempt_count, actual_prompt_token_count=receipt.actual_prompt_token_count,
        actual_completion_token_count=receipt.actual_completion_token_count, actual_total_token_count=receipt.actual_total_token_count,
        directive_applied_count=receipt.directive_applied_count, behavior_observation_count=receipt.behavior_observation_count,
        replay_statuses=receipt.replay_statuses, receipt_path=str(root / "calibration_receipt.json"),
        provider_constructed_count=constructed, network="enabled", results_written=True,
        ready_for_network_authorization=True, safety=_safety(True),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="H-G.2.1 real eligibility calibration")
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
            allow_network=args.allow_network, confirm_run=args.confirm_run,
            confirm_request_cap=args.confirm_request_cap,
            confirm_completion_reservation_cap=args.confirm_completion_reservation_cap,
            resume=args.resume,
        ).model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if payload.get("status") in {"completed", "fake_smoke_passed", "ready_for_network_authorization"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
