"""Explicitly confirmed execution for the protocol-blind v2 confirmatory plan.

The default command path is offline.  This module composes existing real-Agent
and contextual-probe runtimes only after a caller has explicitly passed every
network and budget confirmation.  It stores safe accounting receipts, never
prompts or private probe content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .agent_runtime import AgentRequestSettings, AgentRuntime
from .batch_experiment import (
    BatchExperimentManifest,
    BatchExperimentRecord,
    BatchExperimentStatus,
    BatchMatchedGroupRecord,
    BatchRunRecord,
    BatchRunSpec,
    BatchRunStatus,
    BatchRuntimeParameters,
    expected_artifact_paths,
)
from .contextual_probe_runtime import (
    CONTEXTUAL_PROBE_TEMPLATE_VERSION,
    ContextualProbeError,
    ContextualProbeRequestSettings,
    ContextualProbeRuntime,
)
from .evaluation import OfflineEvaluator, write_evaluation_report
from .evaluation.models import (
    EvaluationCondition,
    EvaluationManifest,
    EvaluationMetricOptions,
    EvaluationRunEntry,
    MetricSuite,
)
from .freeze_wvs7_probe_set import EXPECTED_VARIABLE_IDS
from .frozen_wvs7_manifest import load_frozen_manifest, sha256_file
from .measured_protocol_experiment import MeasuredProtocolExperimentRunner, MeasurementExperimentStatus
from .models import ProbeResult, RunConfig
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ReasoningEffort
from .probe_loader import ProbeSetLoader
from .replay import validate_replay
from .request_ledger import (
    LedgeredProvider,
    RequestLedger,
    RequestLedgerEntry,
    RequestLedgerStatus,
    RequestLedgerSummary,
)
from .scenario_loader import ScenarioLoader
from .wvs7_baseline_pilot import _validate_frozen_probe_set
from .wvs7_confirmatory_plan_lock import ConfirmatoryPlanLockReceipt
from .wvs7_confirmatory_study import (
    EXPECTED_CONDITIONS,
    EXPECTED_PRIMARY_METRICS,
    EXPECTED_V2_PROBE_SHA256,
    EXPECTED_V2_PROBE_SET_ID,
    EXPECTED_V2_TEMPLATE,
    ConfirmatoryGateReport,
    ConfirmatoryStudyError,
    ConfirmatoryStudyManifest,
    load_confirmatory_manifest,
    validate_confirmatory_manifest,
)
from .wvs7_test_scenario import TestScenarioAuthoringError, load_authoring, load_review, validate_test_scenario
from .private_probe_recovery import PrivateProbeRecoveryCheckpoint, ProbeRecoveryPhase


_ROOT = Path(__file__).resolve().parents[2]
CONFIRMATORY_REQUEST_CAP = 1_152
CONFIRMATORY_COMPLETION_RESERVATION_CAP = 153_600
CONFIRMATORY_TOTAL_TOKEN_REFERENCE = 571_806
REQUESTS_PER_RUN = 96
COMPLETION_RESERVATION_PER_RUN = 12_800
CONFIRMATORY_BATCH_ID = "wvs7-protocol-blind-v2-confirmatory-execution"
_LOCK_PATH = _ROOT / "outputs/study-locks/wvs7_protocol_blind_v2_confirmatory_plan_lock.json"
PARSER_RECOVERY_STRATEGY_VERSION = "parser_invalid_probe.v1"
PARSER_INVALID_ERROR_CODES = frozenset({
    ContextualProbeError.PARSER_FAILURE.value,
    ContextualProbeError.MALFORMED_JSON.value,
    ContextualProbeError.INVALID_SCHEMA.value,
    ContextualProbeError.EMPTY_CHOICE.value,
    ContextualProbeError.FORBIDDEN_CONTROL_FIELD.value,
    ContextualProbeError.INVALID_CHOICE.value,
    ContextualProbeError.PRIVATE_CONTENT.value,
})


class WVS7ConfirmatoryRealError(ValueError):
    """Stable safe failure for confirmatory preflight or orchestration."""


@dataclass(frozen=True)
class PreparedWVS7ConfirmatoryRealExperiment:
    """Validated local inputs. No Provider is constructed by this object."""

    manifest: ConfirmatoryStudyManifest
    gate_report: ConfirmatoryGateReport
    plan_lock: ConfirmatoryPlanLockReceipt
    scenario: object
    probe_set: object
    batch_manifest: BatchExperimentManifest
    plan_sha256: str
    plan_lock_sha256: str
    frozen_probe_set_path: Path
    scenario_path: Path


class ConfirmatoryRealPreflightReport(BaseModel):
    """Content-free local readiness information for the fixed 12-run plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ready", "blocked"]
    study_id: str
    batch_id: str
    seed_count: int = Field(ge=0)
    condition_count: int = Field(ge=0)
    run_count: int = Field(ge=0)
    request_cap: int = Field(gt=0)
    completion_reservation_cap: int = Field(gt=0)
    estimated_total_token_reference: int = Field(gt=0)
    frozen_probe_set_sha256: str
    test_scenario_sha256: str
    output_paths_available: bool
    blocking_reasons: list[str] = Field(default_factory=list)


class ConfirmatoryBatchReceipt(BaseModel):
    """Safe terminal receipt; source contents and provider metadata are omitted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_id: str
    status: Literal["confirmatory"]
    confirmatory: Literal[True]
    not_causal_conclusion: Literal[True]
    statistical_analysis_deferred: Literal[True]
    pilot_version: Literal["protocol_blind_v2"]
    v1_002_excluded_from_v2_main_analysis: Literal[True]
    v2_003_calibration_excluded_from_confirmatory_statistics: Literal[True]
    completed_run_count: int = Field(ge=0)
    failed_run_count: int = Field(ge=0)
    matched_group_count: int = Field(ge=0)
    replay_statuses: dict[str, str]
    plan_sha256: str
    plan_lock_sha256: str
    frozen_probe_set_sha256: str
    test_scenario_sha256: str
    request_ledger: RequestLedgerSummary
    initial_final_report_path: str
    completeness_audit_path: str


class ParserRecoveryAmendment(BaseModel):
    """Safe, single-coordinate amendment for a parser-invalid probe transport."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    amendment_version: Literal["confirmatory_parser_recovery.v1"] = "confirmatory_parser_recovery.v1"
    batch_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    probe_id: str = Field(min_length=1)
    fingerprint: str = Field(min_length=64, max_length=64)
    original_error_code: str = Field(min_length=1)
    recovery_strategy_version: Literal["parser_invalid_probe.v1"] = PARSER_RECOVERY_STRATEGY_VERSION
    allowed_extra_attempts: Literal[1] = 1
    recovery_transport_attempt_count: int = Field(default=0, ge=0, le=1)
    status: Literal["authorized", "completed", "failed"] = "authorized"
    plan_sha256: str = Field(min_length=64, max_length=64)
    plan_lock_sha256: str = Field(min_length=64, max_length=64)
    scenario_sha256: str = Field(min_length=64, max_length=64)
    frozen_probe_set_sha256: str = Field(min_length=64, max_length=64)
    error_code: str | None = None


class _ParserRecoveryAuthorization:
    def __init__(self, amendment: ParserRecoveryAmendment, path: Path) -> None:
        self.amendment = amendment
        self.path = path


class ConfirmatoryExecutionSummary(BaseModel):
    """In-memory terminal summary without public turns or private measurement data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["confirmatory", "failed"]
    batch_id: str
    completed_run_count: int = Field(ge=0)
    failed_run_count: int = Field(ge=0)
    matched_groups_complete: bool
    receipt_path: str | None = None
    error_code: str | None = None


def _rooted(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else _ROOT / value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_file_hash(path: Path, code: str) -> str:
    try:
        return sha256_file(path)
    except Exception as exc:
        raise WVS7ConfirmatoryRealError(code) from exc


def _load_plan_lock(path: Path) -> ConfirmatoryPlanLockReceipt:
    try:
        return ConfirmatoryPlanLockReceipt.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise WVS7ConfirmatoryRealError("confirmatory_plan_lock_invalid") from exc


def _validate_plan_lock(
    lock: ConfirmatoryPlanLockReceipt,
    *,
    plan_file: Path,
    manifest: ConfirmatoryStudyManifest,
) -> None:
    expected = {
        "confirmatory_manifest": _safe_file_hash(plan_file, "confirmatory_plan_missing"),
        "test_scenario": _safe_file_hash(_rooted(manifest.test_scenario_path), "test_scenario_missing"),
        "frozen_probe_set": _safe_file_hash(_rooted(manifest.frozen_probe_set_path), "frozen_probe_set_missing"),
    }
    if (
        lock.status != "locked"
        or lock.study_id != manifest.study_id
        or lock.test_scenario_id != manifest.test_scenario_ids[0]
        or lock.probe_set_id != EXPECTED_V2_PROBE_SET_ID
        or lock.frozen_probe_set_sha256 != manifest.frozen_probe_set_sha256
        or lock.seeds != manifest.seeds
        or lock.conditions != [condition.value for condition in EXPECTED_CONDITIONS]
        or lock.run_count != 12
        or lock.matched_group_count != 3
        or lock.logical_requests_per_matched_group != 384
        or lock.total_logical_requests != CONFIRMATORY_REQUEST_CAP
        or lock.completion_reservation_per_matched_group != 51200
        or lock.total_completion_reservation != CONFIRMATORY_COMPLETION_RESERVATION_CAP
        or lock.observed_total_token_reference != CONFIRMATORY_TOTAL_TOKEN_REFERENCE
        or lock.primary_metrics != list(EXPECTED_PRIMARY_METRICS)
        or lock.input_file_sha256.get("confirmatory_manifest") != expected["confirmatory_manifest"]
        or lock.input_file_sha256.get("test_scenario") != expected["test_scenario"]
        or lock.input_file_sha256.get("frozen_probe_set") != expected["frozen_probe_set"]
    ):
        raise WVS7ConfirmatoryRealError("confirmatory_plan_lock_mismatch")


def _validate_execution_contract(manifest: ConfirmatoryStudyManifest) -> None:
    if (
        len(manifest.runs) != 12
        or len(manifest.seeds) != 3
        or tuple(manifest.conditions) != EXPECTED_CONDITIONS
        or tuple(manifest.primary_metrics) != EXPECTED_PRIMARY_METRICS
        or 20260819 in manifest.seeds
        or manifest.total_logical_requests != CONFIRMATORY_REQUEST_CAP
        or manifest.total_completion_reservation != CONFIRMATORY_COMPLETION_RESERVATION_CAP
        or manifest.observational_total_token_reference != CONFIRMATORY_TOTAL_TOKEN_REFERENCE
    ):
        raise WVS7ConfirmatoryRealError("confirmatory_execution_plan_mismatch")
    if any(
        run.probe_template_version != EXPECTED_V2_TEMPLATE
        or run.run_id in {"wvs7-real-baseline-pilot-seed-002", "wvs7-real-baseline-pilot-seed-003"}
        for run in manifest.runs
    ):
        raise WVS7ConfirmatoryRealError("confirmatory_run_contract_mismatch")


def _build_batch_manifest(
    manifest: ConfirmatoryStudyManifest,
    *,
    output_root: Path,
) -> BatchExperimentManifest:
    runtime = BatchRuntimeParameters(
        agent_temperature=0.2,
        agent_max_tokens=256,
        probe_temperature=0.0,
        probe_max_tokens=128,
    )
    specs = [
        BatchRunSpec(
            run_id=run.run_id,
            matched_group_id=run.matched_group_id,
            condition=run.condition,
            scenario_id=run.scenario_id,
            model_name=run.model_name,
            probe_model_name=run.model_name,
            agent_ids=run.agent_ids,
            seed=run.seed,
            probe_seed=run.seed,
            max_rounds=run.max_rounds,
            probe_set_id=run.probe_set_id,
            probe_item_ids=[f"wvs7-english-core-{variable_id}" for variable_id in EXPECTED_VARIABLE_IDS],
            is_holdout=False,
            pre_target_round=0,
            post_target_round=run.max_rounds - 1,
            runtime_parameters=runtime,
            output_dir=str(output_root),
            artifacts=expected_artifact_paths(str(output_root), run.run_id),
        )
        for run in manifest.runs
    ]
    return BatchExperimentManifest(
        batch_id=CONFIRMATORY_BATCH_ID,
        scenario_id=manifest.test_scenario_ids[0],
        model_name=manifest.model_name,
        probe_model_name=manifest.model_name,
        agent_ids=manifest.agent_ids,
        max_rounds=manifest.max_rounds,
        probe_set_id=manifest.probe_set_id,
        probe_item_ids=[f"wvs7-english-core-{variable_id}" for variable_id in EXPECTED_VARIABLE_IDS],
        is_holdout=False,
        seeds=list(manifest.seeds),
        conditions=list(manifest.conditions),
        runs=specs,
        run_order=[run.run_id for run in manifest.runs],
        batch_output_dir=str(output_root / "batches"),
        version=manifest.preregistration_version,
        metadata={
            "confirmatory": True,
            "not_causal_conclusion": True,
            "statistical_analysis_deferred": True,
            "pilot_version": "protocol_blind_v2",
            "frozen_probe_set_sha256": manifest.frozen_probe_set_sha256,
            "test_scenario_sha256": manifest.test_scenario_sha256,
        },
    )


def prepare_wvs7_confirmatory_real_experiment(
    config_path: str | Path,
    *,
    plan_lock_path: str | Path = _LOCK_PATH,
    output_root: str | Path | None = None,
) -> PreparedWVS7ConfirmatoryRealExperiment:
    """Validate immutable local inputs. This function never constructs a Provider."""
    plan_file = _rooted(config_path)
    try:
        manifest = load_confirmatory_manifest(plan_file)
    except ConfirmatoryStudyError as exc:
        raise WVS7ConfirmatoryRealError("confirmatory_manifest_invalid") from exc
    gate = validate_confirmatory_manifest(manifest, verify_frozen_file=True)
    if gate.status != "ready_for_confirmatory":
        raise WVS7ConfirmatoryRealError("confirmatory_manifest_blocked")
    _validate_execution_contract(manifest)
    lock_file = _rooted(plan_lock_path)
    lock = _load_plan_lock(lock_file)
    _validate_plan_lock(lock, plan_file=plan_file, manifest=manifest)
    try:
        authoring = load_authoring(_rooted(manifest.test_scenario_authoring_path))
        review = load_review(_rooted(manifest.test_scenario_review_path))
        review_report = validate_test_scenario(authoring, review, require_review=True)
    except (TestScenarioAuthoringError, OSError, ValueError) as exc:
        raise WVS7ConfirmatoryRealError("confirmatory_test_scenario_review_invalid") from exc
    if review_report.status != "ready" or review.status != "accepted":
        raise WVS7ConfirmatoryRealError("confirmatory_test_scenario_review_not_accepted")
    scenario_path = _rooted(manifest.test_scenario_path)
    frozen_path = _rooted(manifest.frozen_probe_set_path)
    if _safe_file_hash(scenario_path, "test_scenario_missing") != manifest.test_scenario_sha256:
        raise WVS7ConfirmatoryRealError("test_scenario_hash_mismatch")
    if _safe_file_hash(frozen_path, "frozen_probe_set_missing") != EXPECTED_V2_PROBE_SHA256:
        raise WVS7ConfirmatoryRealError("frozen_probe_set_hash_mismatch")
    try:
        frozen_manifest = load_frozen_manifest(frozen_path.with_name("english_core_wvs7_23_frozen_manifest.json"))
        scenario = ScenarioLoader.load(scenario_path)
        probe_set = ProbeSetLoader.load(frozen_path)
        # Reuse the existing frozen contract verifier, through its public data contract.
        validation_config = type("FrozenConfig", (), {
            "frozen_probe_set_sha256": manifest.frozen_probe_set_sha256,
            "probe_set_id": manifest.probe_set_id,
        })()
        _validate_frozen_probe_set(validation_config, frozen_path, frozen_manifest)
    except Exception as exc:
        raise WVS7ConfirmatoryRealError("confirmatory_execution_inputs_invalid") from exc
    if (
        scenario.scenario_id != manifest.test_scenario_ids[0]
        or [agent.agent_id for agent in scenario.agents] != manifest.agent_ids
        or scenario.max_rounds != manifest.max_rounds
        or probe_set.probe_set_id != EXPECTED_V2_PROBE_SET_ID
        or len(probe_set.items_for_holdout(False)) != 23
    ):
        raise WVS7ConfirmatoryRealError("confirmatory_execution_inputs_invalid")
    root = Path(output_root) if output_root is not None else _ROOT / "results"
    batch_manifest = _build_batch_manifest(manifest, output_root=root)
    return PreparedWVS7ConfirmatoryRealExperiment(
        manifest=manifest,
        gate_report=gate,
        plan_lock=lock,
        scenario=scenario,
        probe_set=probe_set,
        batch_manifest=batch_manifest,
        plan_sha256=_safe_file_hash(plan_file, "confirmatory_plan_missing"),
        plan_lock_sha256=_safe_file_hash(lock_file, "confirmatory_plan_lock_missing"),
        frozen_probe_set_path=frozen_path,
        scenario_path=scenario_path,
    )


def preflight_wvs7_confirmatory_real_experiment(
    config_path: str | Path,
    *,
    plan_lock_path: str | Path = _LOCK_PATH,
    output_root: str | Path | None = None,
    resume: bool = False,
) -> tuple[PreparedWVS7ConfirmatoryRealExperiment, ConfirmatoryRealPreflightReport]:
    """Run all local safety checks without reading provider environment variables."""
    prepared = prepare_wvs7_confirmatory_real_experiment(
        config_path, plan_lock_path=plan_lock_path, output_root=output_root,
    )
    batch_dir = Path(prepared.batch_manifest.batch_output_dir) / prepared.batch_manifest.batch_id
    run_paths = [Path(spec.artifacts.run_directory) for spec in prepared.batch_manifest.runs]
    available = not batch_dir.exists() and not any(path.exists() for path in run_paths)
    ready = batch_dir.exists() if resume else available
    report = ConfirmatoryRealPreflightReport(
        status="ready" if ready else "blocked",
        study_id=prepared.manifest.study_id,
        batch_id=prepared.batch_manifest.batch_id,
        seed_count=len(prepared.manifest.seeds),
        condition_count=len(prepared.manifest.conditions),
        run_count=len(prepared.manifest.runs),
        request_cap=CONFIRMATORY_REQUEST_CAP,
        completion_reservation_cap=CONFIRMATORY_COMPLETION_RESERVATION_CAP,
        estimated_total_token_reference=CONFIRMATORY_TOTAL_TOKEN_REFERENCE,
        frozen_probe_set_sha256=prepared.manifest.frozen_probe_set_sha256,
        test_scenario_sha256=prepared.manifest.test_scenario_sha256,
        output_paths_available=available,
        blocking_reasons=[] if ready else (["resume_batch_not_found"] if resume else ["confirmatory_output_already_exists"]),
    )
    return prepared, report


class WVS7ConfirmatoryRealRunner:
    """Serial, resumable twelve-run executor under immutable plan and hash binding."""

    def __init__(
        self,
        prepared: PreparedWVS7ConfirmatoryRealExperiment,
        *,
        agent_provider: object,
        probe_provider: object,
        request_interval_seconds: float = 0.0,
    ) -> None:
        if request_interval_seconds < 0:
            raise WVS7ConfirmatoryRealError("confirmatory_request_interval_invalid")
        self.prepared = prepared
        self.manifest = prepared.batch_manifest
        self._agent_provider = agent_provider
        self._probe_provider = probe_provider
        self._request_interval_seconds = request_interval_seconds

    @property
    def batch_directory(self) -> Path:
        return Path(self.manifest.batch_output_dir) / self.manifest.batch_id

    def run(self, *, resume: bool = False) -> ConfirmatoryExecutionSummary:
        records = self._load_or_initialize_records(resume=resume)
        stop = False
        for spec in self._ordered_specs():
            current = next(record for record in records if record.run_id == spec.run_id)
            if current.status is BatchRunStatus.COMPLETED:
                continue
            if current.status is BatchRunStatus.FAILED and not resume:
                stop = True
                break
            if stop:
                break
            records = self._replace(records, spec.run_id, BatchRunStatus.RUNNING)
            self._write_record(self._record(records, BatchExperimentStatus.RUNNING, error_code=None))
            try:
                result, replay_code = self._run_condition(
                    spec,
                    resume=resume and current.status is BatchRunStatus.FAILED,
                    previous_record=current,
                )
            except (WVS7ConfirmatoryRealError, OSError, ValidationError, ValueError) as exc:
                code = _safe_error_code(exc)
                records = self._replace(records, spec.run_id, BatchRunStatus.FAILED, error_code=code)
                self._write_record(self._record(records, BatchExperimentStatus.FAILED, error_code=code))
                stop = True
                break
            if result.status is not MeasurementExperimentStatus.COMPLETED or replay_code != "passed":
                code = result.error_code or replay_code or result.status.value
                records = self._replace(records, spec.run_id, BatchRunStatus.FAILED, measurement_status=result.status.value, error_code=code)
                self._write_record(self._record(records, BatchExperimentStatus.FAILED, error_code=code))
                stop = True
                break
            records = self._replace(records, spec.run_id, BatchRunStatus.COMPLETED, measurement_status=result.status.value)
            self._write_record(self._record(records, BatchExperimentStatus.RUNNING, error_code=None))

        completed = all(record.status is BatchRunStatus.COMPLETED for record in records)
        status = BatchExperimentStatus.COMPLETED if completed else BatchExperimentStatus.FAILED
        record = self._record(records, status, error_code=None if completed else "incomplete_matched_group")
        self._write_record(record)
        if not completed:
            return ConfirmatoryExecutionSummary(
                status="failed", batch_id=self.manifest.batch_id,
                completed_run_count=record.completed_run_count, failed_run_count=record.failed_run_count,
                matched_groups_complete=all(group.is_complete for group in record.matched_groups),
                error_code=record.error_code,
            )
        replays = self._validate_completed_outputs()
        report_path = self._evaluate_completed_batch()
        audit_path = self._write_completeness_audit(record, replays)
        receipt = self._write_receipt(record, replays, report_path, audit_path)
        return ConfirmatoryExecutionSummary(
            status="confirmatory", batch_id=self.manifest.batch_id,
            completed_run_count=record.completed_run_count, failed_run_count=record.failed_run_count,
            matched_groups_complete=True, receipt_path=str(self._receipt_path()),
        )

    def _run_condition(
        self,
        spec: BatchRunSpec,
        *,
        resume: bool,
        previous_record: BatchRunRecord,
    ) -> tuple[object, str | None]:
        ledger = RequestLedger(self.batch_directory / "request_ledgers" / f"{spec.run_id}.jsonl")
        recovery = self._authorize_parser_recovery(spec, previous_record, ledger, resume=resume)
        recovery_fingerprints = set() if recovery is None else {recovery.amendment.fingerprint}
        agent_provider = LedgeredProvider(
            self._agent_provider, ledger=ledger, request_cap=REQUESTS_PER_RUN,
            completion_reservation_cap=COMPLETION_RESERVATION_PER_RUN, resume=resume,
            min_interval_seconds=self._request_interval_seconds, sleeper=time.sleep,
        )
        probe_provider = LedgeredProvider(
            self._probe_provider, ledger=ledger, request_cap=REQUESTS_PER_RUN,
            completion_reservation_cap=COMPLETION_RESERVATION_PER_RUN, resume=resume,
            recovery_fingerprints=recovery_fingerprints,
            recovery_kind=PARSER_RECOVERY_STRATEGY_VERSION if recovery is not None else None,
            max_recovery_attempts=1,
            min_interval_seconds=self._request_interval_seconds, sleeper=time.sleep,
        )
        runner = MeasuredProtocolExperimentRunner(
            RunConfig(
                run_id=spec.run_id, scenario_id=spec.scenario_id, model_name=spec.model_name,
                protocol=spec.condition, agent_count=len(spec.agent_ids), max_rounds=spec.max_rounds,
                seed=spec.seed, intervention_budget=0.0, output_dir=spec.output_dir,
            ),
            scenario=self.prepared.scenario, probe_set=self.prepared.probe_set,
            agent_runtime=AgentRuntime(AgentRequestSettings(
                model_name=spec.model_name, temperature=spec.runtime_parameters.agent_temperature,
                max_tokens=spec.runtime_parameters.agent_max_tokens, seed=spec.seed,
            )),
            contextual_probe_runtime=ContextualProbeRuntime(ContextualProbeRequestSettings(
                model_name=spec.probe_model_name, temperature=spec.runtime_parameters.probe_temperature,
                max_tokens=spec.runtime_parameters.probe_max_tokens, seed=spec.probe_seed,
            )),
            agent_provider=agent_provider, probe_provider=probe_provider,
            probe_seed=spec.probe_seed, pre_target_round=0, post_target_round=spec.max_rounds - 1,
            execute_holdout_probe=False,
            recovery_checkpoint_dir=self.batch_directory / "private_probe_recovery" / spec.run_id,
        )
        try:
            result = runner.run(resume=resume)
        except Exception:
            if recovery is not None:
                self._update_recovery_amendment(recovery, status="failed", error_code="condition_execution_failed")
            return _failed_measurement_result(), "condition_execution_failed"
        if result.status is not MeasurementExperimentStatus.COMPLETED:
            if recovery is not None:
                self._update_recovery_amendment(
                    recovery,
                    status="failed",
                    error_code=result.error_code or result.status.value,
                    attempted=True,
                )
            return result, result.error_code or result.status.value
        if recovery is not None:
            self._update_recovery_amendment(recovery, status="completed", attempted=True)
        try:
            replay = validate_replay(spec.artifacts.events_path, spec.artifacts.run_record_path)
        except Exception:
            return result, "replay_validation_failed"
        return result, replay.validation

    def _authorize_parser_recovery(
        self,
        spec: BatchRunSpec,
        previous_record: BatchRunRecord,
        ledger: RequestLedger,
        *,
        resume: bool,
    ) -> _ParserRecoveryAuthorization | None:
        if not resume or previous_record.status is not BatchRunStatus.FAILED:
            return None
        if previous_record.error_code not in PARSER_INVALID_ERROR_CODES:
            return None
        if previous_record.measurement_status != MeasurementExperimentStatus.POST_PROBE_FAILED.value:
            raise WVS7ConfirmatoryRealError("parser_recovery_requires_post_probe_failure")
        checkpoint_path = (
            self.batch_directory / "private_probe_recovery" / spec.run_id / "post_responses.json"
        )
        try:
            checkpoint = PrivateProbeRecoveryCheckpoint.model_validate_json(
                checkpoint_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError) as exc:
            raise WVS7ConfirmatoryRealError("parser_recovery_checkpoint_invalid") from exc
        expected_probe_ids = set(spec.probe_item_ids)
        expected_coordinates = [
            (agent_id, probe_id)
            for agent_id in spec.agent_ids
            for probe_id in spec.probe_item_ids
        ]
        observed_coordinates = {(item.agent_id, item.probe_id) for item in checkpoint.responses}
        missing = [coordinate for coordinate in expected_coordinates if coordinate not in observed_coordinates]
        if (
            checkpoint.run_id != spec.run_id
            or checkpoint.scenario_id != spec.scenario_id
            or checkpoint.protocol.value != spec.condition.value
            or checkpoint.probe_set_id != spec.probe_set_id
            or checkpoint.phase is not ProbeRecoveryPhase.POST
            or checkpoint.target_round != spec.max_rounds - 1
            or set(checkpoint.agent_ids) != set(spec.agent_ids)
            or set(checkpoint.probe_item_ids) != expected_probe_ids
            or not missing
        ):
            raise WVS7ConfirmatoryRealError("parser_recovery_coordinate_not_unique")
        agent_id, probe_id = missing[0]
        entries = ledger.entries()
        candidates = [
            entry
            for entry in entries
            if entry.status is RequestLedgerStatus.COMPLETED
            and entry.phase == "probe"
            and entry.template_version == CONTEXTUAL_PROBE_TEMPLATE_VERSION
            and entry.condition == spec.condition.value
            and entry.agent_id == agent_id
            and entry.probe_id == probe_id
            and entry.round_id == spec.max_rounds - 1
        ]
        fingerprints = {entry.fingerprint for entry in candidates}
        if len(fingerprints) != 1:
            raise WVS7ConfirmatoryRealError("parser_recovery_fingerprint_not_authorized")
        fingerprint = next(iter(fingerprints))
        later_completed = [
            entry
            for entry in entries
            if entry.status is RequestLedgerStatus.COMPLETED
            and entry.phase == "probe"
            and entry.template_version == CONTEXTUAL_PROBE_TEMPLATE_VERSION
            and entry.condition == spec.condition.value
            and entry.round_id == spec.max_rounds - 1
            and entry.agent_id is not None
            and entry.probe_id is not None
            and (entry.agent_id, entry.probe_id) in set(missing[1:])
        ]
        if later_completed:
            raise WVS7ConfirmatoryRealError("parser_recovery_fingerprint_not_authorized")
        matching = [entry for entry in entries if entry.fingerprint == fingerprint]
        if (
            sum(entry.status is RequestLedgerStatus.STARTED for entry in matching) != 1
            or sum(entry.status is RequestLedgerStatus.COMPLETED for entry in matching) != 1
            or any(entry.recovery_kind is not None for entry in matching)
            or any(entry.attempt_count != 1 for entry in matching)
        ):
            raise WVS7ConfirmatoryRealError("parser_recovery_fingerprint_not_authorized")
        amendment_path = self.batch_directory / "confirmatory_parser_recovery_amendment.json"
        if amendment_path.exists():
            try:
                existing = ParserRecoveryAmendment.model_validate_json(amendment_path.read_text(encoding="utf-8"))
            except (OSError, ValidationError, ValueError) as exc:
                raise WVS7ConfirmatoryRealError("parser_recovery_amendment_invalid") from exc
            if existing.recovery_transport_attempt_count >= 1:
                raise WVS7ConfirmatoryRealError("parser_recovery_attempt_limit_reached")
        amendment = ParserRecoveryAmendment(
            batch_id=self.manifest.batch_id,
            run_id=spec.run_id,
            condition=spec.condition.value,
            agent_id=agent_id,
            round_id=spec.max_rounds - 1,
            probe_id=probe_id,
            fingerprint=fingerprint,
            original_error_code=previous_record.error_code or "invalid_schema",
            plan_sha256=self.prepared.plan_sha256,
            plan_lock_sha256=self.prepared.plan_lock_sha256,
            scenario_sha256=self.prepared.manifest.test_scenario_sha256,
            frozen_probe_set_sha256=self.prepared.manifest.frozen_probe_set_sha256,
        )
        if amendment_path.exists():
            existing = ParserRecoveryAmendment.model_validate_json(amendment_path.read_text(encoding="utf-8"))
            immutable = amendment.model_dump(mode="json", exclude={"status", "recovery_transport_attempt_count", "error_code"})
            existing_immutable = existing.model_dump(mode="json", exclude={"status", "recovery_transport_attempt_count", "error_code"})
            if existing_immutable != immutable or existing.recovery_transport_attempt_count >= 1:
                raise WVS7ConfirmatoryRealError("parser_recovery_amendment_mismatch")
            amendment = existing
        else:
            amendment_path.parent.mkdir(parents=True, exist_ok=True)
            amendment_path.write_text(amendment.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return _ParserRecoveryAuthorization(amendment, amendment_path)

    @staticmethod
    def _update_recovery_amendment(
        authorization: _ParserRecoveryAuthorization,
        *,
        status: Literal["authorized", "completed", "failed"],
        error_code: str | None = None,
        attempted: bool = False,
    ) -> None:
        amendment = authorization.amendment.model_copy(
            update={
                "status": status,
                "recovery_transport_attempt_count": 1 if attempted else authorization.amendment.recovery_transport_attempt_count,
                "error_code": error_code,
            }
        )
        temporary = authorization.path.with_suffix(".tmp")
        temporary.write_text(amendment.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(authorization.path)


    def _load_or_initialize_records(self, *, resume: bool) -> list[BatchRunRecord]:
        manifest_path = self.batch_directory / "batch_manifest.json"
        record_path = self.batch_directory / "batch_record.json"
        binding_path = self.batch_directory / "confirmatory_execution_binding.json"
        if self.batch_directory.exists():
            if not resume:
                raise FileExistsError("refusing to overwrite existing confirmatory batch directory")
            try:
                stored = BatchExperimentManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
                record = BatchExperimentRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
                binding = json.loads(binding_path.read_text(encoding="utf-8"))
            except (OSError, ValidationError, ValueError, json.JSONDecodeError) as exc:
                raise WVS7ConfirmatoryRealError("confirmatory_resume_artifacts_invalid") from exc
            if stored.model_dump(mode="json") != self.manifest.model_dump(mode="json"):
                raise WVS7ConfirmatoryRealError("confirmatory_resume_manifest_mismatch")
            if binding != self._binding_payload():
                raise WVS7ConfirmatoryRealError("confirmatory_resume_fingerprint_mismatch")
            return list(record.runs)
        for spec in self.manifest.runs:
            if Path(spec.artifacts.run_directory).exists():
                raise FileExistsError("refusing to overwrite an existing confirmatory run directory")
        self.batch_directory.mkdir(parents=True, exist_ok=False)
        manifest_path.write_text(self.manifest.model_dump_json(indent=2), encoding="utf-8")
        binding_path.write_text(json.dumps(self._binding_payload(), ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        records = [
            BatchRunRecord(run_id=spec.run_id, matched_group_id=spec.matched_group_id,
                           condition=spec.condition, status=BatchRunStatus.PLANNED, artifacts=spec.artifacts)
            for spec in self.manifest.runs
        ]
        self._write_record(self._record(records, BatchExperimentStatus.PLANNED, error_code=None))
        return records

    def _binding_payload(self) -> dict[str, object]:
        return {
            "plan_sha256": self.prepared.plan_sha256,
            "plan_lock_sha256": self.prepared.plan_lock_sha256,
            "frozen_probe_set_sha256": self.prepared.manifest.frozen_probe_set_sha256,
            "test_scenario_sha256": self.prepared.manifest.test_scenario_sha256,
            "probe_template_version": EXPECTED_V2_TEMPLATE,
        }

    def _ordered_specs(self) -> list[BatchRunSpec]:
        by_id = {spec.run_id: spec for spec in self.manifest.runs}
        return [by_id[run_id] for run_id in self.manifest.run_order]

    @staticmethod
    def _replace(records: list[BatchRunRecord], run_id: str, status: BatchRunStatus, *, measurement_status: str | None = None, error_code: str | None = None) -> list[BatchRunRecord]:
        return [
            record.model_copy(update={"status": status, "measurement_status": measurement_status, "error_code": error_code})
            if record.run_id == run_id else record
            for record in records
        ]

    def _record(self, records: list[BatchRunRecord], status: BatchExperimentStatus, *, error_code: str | None) -> BatchExperimentRecord:
        record_by_id = {record.run_id: record for record in records}
        groups: list[BatchMatchedGroupRecord] = []
        for seed in self.manifest.seeds:
            specs = [spec for spec in self._ordered_specs() if spec.seed == seed]
            group_records = [record_by_id[spec.run_id] for spec in specs]
            groups.append(BatchMatchedGroupRecord(
                matched_group_id=specs[0].matched_group_id,
                seed=seed,
                run_ids=[item.run_id for item in group_records],
                run_statuses=[item.status for item in group_records],
                is_complete=all(item.status is BatchRunStatus.COMPLETED for item in group_records),
            ))
        return BatchExperimentRecord(
            batch_id=self.manifest.batch_id,
            manifest_sha256=_canonical_sha256(self.manifest.model_dump(mode="json")),
            status=status, runs=records, matched_groups=groups,
            completed_run_count=sum(item.status is BatchRunStatus.COMPLETED for item in records),
            failed_run_count=sum(item.status is BatchRunStatus.FAILED for item in records),
            error_code=error_code,
        )

    def _write_record(self, record: BatchExperimentRecord) -> None:
        (self.batch_directory / "batch_record.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")

    def _validate_completed_outputs(self) -> dict[str, str]:
        statuses: dict[str, str] = {}
        for spec in self.manifest.runs:
            replay = validate_replay(spec.artifacts.events_path, spec.artifacts.run_record_path)
            if replay.validation != "passed":
                raise WVS7ConfirmatoryRealError("confirmatory_replay_validation_failed")
            self._validate_profiles(spec)
            self._validate_run_ledger(spec)
            statuses[spec.run_id] = replay.validation
        if _safe_file_hash(self.prepared.frozen_probe_set_path, "frozen_probe_set_missing") != self.prepared.manifest.frozen_probe_set_sha256:
            raise WVS7ConfirmatoryRealError("frozen_probe_set_changed_after_confirmatory_run")
        if _safe_file_hash(self.prepared.scenario_path, "test_scenario_missing") != self.prepared.manifest.test_scenario_sha256:
            raise WVS7ConfirmatoryRealError("test_scenario_changed_after_confirmatory_run")
        return statuses

    @staticmethod
    def _validate_profiles(spec: BatchRunSpec) -> None:
        for path in (Path(spec.artifacts.pre_probe_results_path), Path(spec.artifacts.post_probe_results_path)):
            try:
                rows = [ProbeResult.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            except (OSError, ValidationError, ValueError) as exc:
                raise WVS7ConfirmatoryRealError("confirmatory_probe_profiles_invalid") from exc
            if len(rows) != 2 or any(len(row.responses) != 23 for row in rows):
                raise WVS7ConfirmatoryRealError("confirmatory_probe_profiles_invalid")

    def _validate_run_ledger(self, spec: BatchRunSpec) -> None:
        ledger = RequestLedger(self.batch_directory / "request_ledgers" / f"{spec.run_id}.jsonl")
        entries = ledger.entries()
        started = {entry.fingerprint for entry in entries if entry.status.value == "started"}
        terminal = {entry.fingerprint for entry in entries if entry.status.value == "completed"}
        if len(started) != REQUESTS_PER_RUN or started != terminal:
            raise WVS7ConfirmatoryRealError("confirmatory_run_ledger_incomplete")
        probes = [entry for entry in entries if entry.phase == "probe"]
        if not probes or any(entry.template_version != EXPECTED_V2_TEMPLATE for entry in probes):
            raise WVS7ConfirmatoryRealError("confirmatory_probe_template_mismatch")
        recovery_entries = [entry for entry in entries if entry.recovery_kind == PARSER_RECOVERY_STRATEGY_VERSION]
        if len([entry for entry in recovery_entries if entry.status is RequestLedgerStatus.STARTED]) > 1:
            raise WVS7ConfirmatoryRealError("confirmatory_parser_recovery_attempt_limit")
        if any(entry.phase != "probe" for entry in recovery_entries):
            raise WVS7ConfirmatoryRealError("confirmatory_parser_recovery_scope_invalid")

    def _evaluation_manifest(self) -> EvaluationManifest:
        entries: list[EvaluationRunEntry] = []
        for spec in self.manifest.runs:
            shared = {
                "run_id": spec.run_id,
                "condition": EvaluationCondition(spec.condition.value),
                "protocol": spec.condition,
                "run_record_path": str(Path(spec.artifacts.run_record_path).resolve()),
                "events_path": str(Path(spec.artifacts.events_path).resolve()),
                "seed": spec.seed,
                "model_name": spec.model_name,
                "counterfactual_group_id": spec.matched_group_id,
            }
            entries.extend([
                EvaluationRunEntry(**shared, entry_id=f"{spec.run_id}::initial", role="initial", probe_result_path=str(Path(spec.artifacts.pre_probe_results_path).resolve())),
                EvaluationRunEntry(**shared, entry_id=f"{spec.run_id}::final", role="final", probe_result_path=str(Path(spec.artifacts.post_probe_results_path).resolve())),
            ])
        return EvaluationManifest(
            evaluation_id=f"{self.manifest.batch_id}-initial-final-v2",
            scenario_id=self.manifest.scenario_id,
            metric_suite=[MetricSuite.BASELINE_DIVERSITY, MetricSuite.REPRESENTATION, MetricSuite.COUNTERFACTUAL, MetricSuite.AUDIT],
            runs=entries,
            probe_sets=[self.manifest.probe_set_id],
            metric_options=EvaluationMetricOptions(coverage_threshold=0.5, strict=True),
            metadata={
                "confirmatory": True,
                "not_causal_conclusion": True,
                "statistical_analysis_deferred": True,
                "pilot_version": "protocol_blind_v2",
                "measurement_roles": "explicit_initial_and_final",
            },
        )

    def _evaluate_completed_batch(self) -> Path:
        manifest_path = self.batch_directory / "offline_evaluation_manifest_initial_final_v2.json"
        report_path = self.batch_directory / "offline_evaluation_report_initial_final_v2.json"
        if manifest_path.exists() != report_path.exists():
            raise WVS7ConfirmatoryRealError("confirmatory_evaluation_artifacts_inconsistent")
        if report_path.exists():
            return report_path
        evaluation = self._evaluation_manifest()
        manifest_path.write_text(evaluation.model_dump_json(indent=2), encoding="utf-8")
        report = OfflineEvaluator(evaluation, manifest_directory=self.batch_directory).evaluate()
        write_evaluation_report(report, report_path)
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            count = sum(isinstance(metric, dict) and metric.get("metric_name") == "profile_drift" for metric in payload["metrics"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise WVS7ConfirmatoryRealError("confirmatory_evaluation_report_invalid") from exc
        if count != 12:
            raise WVS7ConfirmatoryRealError("confirmatory_profile_drift_count_mismatch")
        return report_path

    def _ledger_summary(self) -> RequestLedgerSummary:
        summaries = [
            RequestLedger(self.batch_directory / "request_ledgers" / f"{spec.run_id}.jsonl").summary(
                request_cap=REQUESTS_PER_RUN, completion_reservation_cap=COMPLETION_RESERVATION_PER_RUN,
            )
            for spec in self.manifest.runs
        ]
        known_prompt = all(item.actual_prompt_token_count is not None for item in summaries)
        known_completion = all(item.actual_completion_token_count is not None for item in summaries)
        known_total = all(item.actual_total_token_count is not None for item in summaries)
        return RequestLedgerSummary(
            request_cap=CONFIRMATORY_REQUEST_CAP,
            completion_reservation_cap=CONFIRMATORY_COMPLETION_RESERVATION_CAP,
            provider_call_count=sum(item.provider_call_count for item in summaries),
            unique_logical_request_count=sum(item.unique_logical_request_count for item in summaries),
            transport_attempt_count=sum(item.transport_attempt_count for item in summaries),
            parser_recovery_transport_attempt_count=sum(item.parser_recovery_transport_attempt_count for item in summaries),
            completion_reserved_token_count=sum(item.completion_reserved_token_count for item in summaries),
            actual_prompt_token_count=sum(item.actual_prompt_token_count or 0 for item in summaries) if known_prompt else None,
            actual_completion_token_count=sum(item.actual_completion_token_count or 0 for item in summaries) if known_completion else None,
            actual_total_token_count=sum(item.actual_total_token_count or 0 for item in summaries) if known_total else None,
            completed_count=sum(item.completed_count for item in summaries),
            failed_count=sum(item.failed_count for item in summaries),
            unknown_usage_count=sum(item.unknown_usage_count for item in summaries),
        )

    def _write_completeness_audit(self, record: BatchExperimentRecord, replays: dict[str, str]) -> Path:
        path = self.batch_directory / "confirmatory_completeness_audit.json"
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("status") != "confirmatory" or existing.get("replay_statuses") != replays:
                    raise WVS7ConfirmatoryRealError("confirmatory_completeness_audit_mismatch")
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise WVS7ConfirmatoryRealError("confirmatory_completeness_audit_invalid") from exc
            return path
        payload = {
            "status": "confirmatory",
            "confirmatory": True,
            "not_causal_conclusion": True,
            "statistical_analysis_deferred": True,
            "completed_run_count": record.completed_run_count,
            "failed_run_count": record.failed_run_count,
            "matched_group_count": len(record.matched_groups),
            "matched_groups_complete": all(group.is_complete for group in record.matched_groups),
            "replay_statuses": replays,
            "frozen_probe_set_sha256": self.prepared.manifest.frozen_probe_set_sha256,
            "test_scenario_sha256": self.prepared.manifest.test_scenario_sha256,
        }
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _receipt_path(self) -> Path:
        return self.batch_directory / "confirmatory_batch_receipt.json"

    def _write_receipt(self, record: BatchExperimentRecord, replays: dict[str, str], report_path: Path, audit_path: Path) -> ConfirmatoryBatchReceipt:
        path = self._receipt_path()
        if path.exists():
            try:
                return ConfirmatoryBatchReceipt.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValidationError, ValueError) as exc:
                raise WVS7ConfirmatoryRealError("confirmatory_batch_receipt_invalid") from exc
        receipt = ConfirmatoryBatchReceipt(
            batch_id=self.manifest.batch_id,
            status="confirmatory", confirmatory=True, not_causal_conclusion=True,
            statistical_analysis_deferred=True, pilot_version="protocol_blind_v2",
            v1_002_excluded_from_v2_main_analysis=True,
            v2_003_calibration_excluded_from_confirmatory_statistics=True,
            completed_run_count=record.completed_run_count, failed_run_count=record.failed_run_count,
            matched_group_count=len(record.matched_groups), replay_statuses=replays,
            plan_sha256=self.prepared.plan_sha256, plan_lock_sha256=self.prepared.plan_lock_sha256,
            frozen_probe_set_sha256=self.prepared.manifest.frozen_probe_set_sha256,
            test_scenario_sha256=self.prepared.manifest.test_scenario_sha256,
            request_ledger=self._ledger_summary(), initial_final_report_path=str(report_path), completeness_audit_path=str(audit_path),
        )
        path.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
        return receipt


def _failed_measurement_result() -> object:
    """Tiny object matching the terminal fields used by the batch loop."""
    return type("FailedMeasurement", (), {"status": MeasurementExperimentStatus.AGENT_RUN_FAILED, "error_code": "condition_execution_failed"})()


def _confirmation_reasons(arguments: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    if not arguments.allow_network:
        reasons.append("allow_network_required")
    if not arguments.confirm_run:
        reasons.append("confirm_run_required")
    if arguments.confirm_request_cap != CONFIRMATORY_REQUEST_CAP:
        reasons.append("confirm_request_cap_must_equal_1152")
    if arguments.confirm_completion_reservation_cap != CONFIRMATORY_COMPLETION_RESERVATION_CAP:
        reasons.append("confirm_completion_reservation_cap_must_equal_153600")
    if arguments.confirm_estimated_total_token_reference != CONFIRMATORY_TOTAL_TOKEN_REFERENCE:
        reasons.append("confirm_estimated_total_token_reference_must_equal_571806")
    return reasons


def _safe_error_code(error: BaseException) -> str:
    message = str(error)
    known = (
        "confirmatory_manifest_invalid", "confirmatory_manifest_blocked", "confirmatory_plan_lock_invalid",
        "confirmatory_plan_lock_mismatch", "confirmatory_execution_plan_mismatch", "confirmatory_run_contract_mismatch",
        "confirmatory_test_scenario_review_invalid", "confirmatory_test_scenario_review_not_accepted",
        "test_scenario_hash_mismatch", "frozen_probe_set_hash_mismatch", "confirmatory_execution_inputs_invalid",
        "confirmatory_output_already_exists", "resume_batch_not_found", "confirmatory_resume_manifest_mismatch",
        "confirmatory_resume_fingerprint_mismatch", "confirmatory_replay_validation_failed",
        "parser_recovery_requires_post_probe_failure", "parser_recovery_checkpoint_invalid",
        "parser_recovery_coordinate_not_unique", "parser_recovery_fingerprint_not_authorized",
        "parser_recovery_amendment_invalid", "parser_recovery_amendment_mismatch",
        "parser_recovery_attempt_limit_reached", "parser_recovery_fingerprint_changed",
        "confirmatory_parser_recovery_attempt_limit",
        "confirmatory_parser_recovery_scope_invalid",
    )
    return next((code for code in known if code in message), "confirmatory_real_execution_blocked")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run only the explicitly confirmed protocol-blind v2 confirmatory plan.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--confirm-request-cap", type=int)
    parser.add_argument("--confirm-completion-reservation-cap", type=int)
    parser.add_argument("--confirm-estimated-total-token-reference", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--request-interval-seconds", type=float, default=0.0)
    arguments = parser.parse_args(argv)
    try:
        prepared, report = preflight_wvs7_confirmatory_real_experiment(arguments.config, resume=arguments.resume)
    except (WVS7ConfirmatoryRealError, OSError, ValidationError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "blocking_reasons": [_safe_error_code(exc)], "provider_call_count": 0}, sort_keys=True))
        return 1
    if not arguments.allow_network:
        print(json.dumps({"provider_call_count": 0, "status": "network_disabled"}, sort_keys=True))
        return 0
    reasons = _confirmation_reasons(arguments)
    if reasons:
        print(json.dumps({"status": "blocked", "blocking_reasons": reasons, "provider_call_count": 0}, sort_keys=True))
        return 1
    if report.status != "ready" and not arguments.resume:
        print(json.dumps({"status": "blocked", "blocking_reasons": report.blocking_reasons, "provider_call_count": 0}, sort_keys=True))
        return 1
    try:
        if report.status != "ready":
            print(json.dumps({"status": "blocked", "blocking_reasons": report.blocking_reasons, "provider_call_count": 0}, sort_keys=True))
            return 1
        if not all(os.environ.get(name) for name in ("EVICON_LLM_BASE_URL", "EVICON_LLM_API_KEY", "EVICON_LLM_MODEL")):
            print(json.dumps({"status": "blocked", "blocking_reasons": ["provider_environment_incomplete"], "provider_call_count": 0}, sort_keys=True))
            return 1
        base = ProviderConfig.from_env(allow_network=True)
        if base.model_name != prepared.manifest.model_name:
            print(json.dumps({"status": "blocked", "blocking_reasons": ["environment_model_does_not_match_confirmatory_plan"], "provider_call_count": 0}, sort_keys=True))
            return 1
        if arguments.max_retries < 0:
            raise WVS7ConfirmatoryRealError("confirmatory_retry_configuration_invalid")
        agent_config = base.model_copy(update={
            "temperature": 0.2, "max_tokens": 256, "max_retries": arguments.max_retries,
            "reasoning_effort": ReasoningEffort.NONE,
        })
        probe_config = base.model_copy(update={
            "temperature": 0.0, "max_tokens": 128, "max_retries": arguments.max_retries,
            "reasoning_effort": ReasoningEffort.NONE,
        })
        summary = WVS7ConfirmatoryRealRunner(
            prepared, agent_provider=OpenAICompatibleProvider(agent_config), probe_provider=OpenAICompatibleProvider(probe_config),
            request_interval_seconds=arguments.request_interval_seconds,
        ).run(resume=arguments.resume)
        print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))
        return 0 if summary.status == "confirmatory" else 1
    except (WVS7ConfirmatoryRealError, OSError, ValidationError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "blocking_reasons": [_safe_error_code(exc)], "provider_call_count": 0}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIRMATORY_BATCH_ID",
    "CONFIRMATORY_COMPLETION_RESERVATION_CAP",
    "CONFIRMATORY_REQUEST_CAP",
    "CONFIRMATORY_TOTAL_TOKEN_REFERENCE",
    "ConfirmatoryBatchReceipt",
    "ConfirmatoryExecutionSummary",
    "ParserRecoveryAmendment",
    "PARSER_RECOVERY_STRATEGY_VERSION",
    "ConfirmatoryRealPreflightReport",
    "PreparedWVS7ConfirmatoryRealExperiment",
    "WVS7ConfirmatoryRealError",
    "WVS7ConfirmatoryRealRunner",
    "prepare_wvs7_confirmatory_real_experiment",
    "preflight_wvs7_confirmatory_real_experiment",
]
