"""Explicitly confirmed, budgeted single-seed real baseline pilot orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path

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
)
from .contextual_probe_runtime import ContextualProbeRequestSettings, ContextualProbeRuntime
from .evaluation import OfflineEvaluator, write_evaluation_report
from .evaluation.models import EvaluationCondition, EvaluationManifest, EvaluationMetricOptions, EvaluationRunEntry, MetricSuite
from .frozen_wvs7_manifest import sha256_file
from .measured_protocol_experiment import MeasuredProtocolExperimentRunner, MeasurementExperimentStatus
from .models import ProbeResult
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ReasoningEffort
from .replay import validate_replay
from .request_ledger import LedgeredProvider, RequestLedger, RequestLedgerSummary
from .wvs7_baseline_pilot import PreparedWVS7BaselinePilot, WVS7BaselinePilotError, prepare_wvs7_baseline_pilot
from .wvs7_real_preflight import (
    REAL_REQUEST_CAP,
    REAL_COMPLETION_RESERVATION_CAP,
    WVS7RealPreflightError,
    build_real_batch_manifest,
    preflight_wvs7_real_pilot,
)


class WVS7RealPilotError(ValueError):
    """Stable real-pilot orchestration error without provider or source content."""


class WVS7RealPilotSummary(BaseModel):
    """Safe terminal receipt. It never contains prompts, answers, or provider metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_id: str = Field(min_length=1)
    status: str
    completed_run_count: int = Field(ge=0)
    failed_run_count: int = Field(ge=0)
    matched_groups_complete: bool
    frozen_probe_set_sha256: str = Field(min_length=64, max_length=64)
    request_ledger: RequestLedgerSummary
    replay_statuses: dict[str, str] = Field(default_factory=dict)
    offline_metric_names: list[str] = Field(default_factory=list)
    pilot_only: bool = True
    single_seed: bool = True
    not_paper_result: bool = True
    no_causal_conclusion: bool = True
    error_code: str | None = None


class WVS7RealPilotRunner:
    """Serial four-condition runner with no Mediator or online control integration."""

    def __init__(
        self,
        prepared: PreparedWVS7BaselinePilot,
        *,
        batch_run_id: str,
        model_name: str,
        agent_provider: object,
        probe_provider: object,
        max_retries: int,
        request_interval_seconds: float = 0.0,
    ) -> None:
        self.prepared = prepared
        self.manifest = build_real_batch_manifest(prepared, batch_run_id=batch_run_id, model_name=model_name)
        self.batch_run_id = batch_run_id
        self._agent_provider = agent_provider
        self._probe_provider = probe_provider
        self.max_retries = max_retries
        self.request_interval_seconds = request_interval_seconds
        if max_retries < 0 or request_interval_seconds < 0:
            raise WVS7RealPilotError("real_pilot_retry_or_interval_is_invalid")

    @property
    def batch_directory(self) -> Path:
        return Path(self.manifest.batch_output_dir) / self.batch_run_id

    @property
    def ledger(self) -> RequestLedger:
        return RequestLedger(self.batch_directory / "request_ledger.jsonl")

    def run(self, *, resume: bool = False) -> WVS7RealPilotSummary:
        records = self._load_or_initialize_records(resume=resume)
        ledger = self.ledger
        agent_provider = LedgeredProvider(
            self._agent_provider,
            ledger=ledger,
            request_cap=REAL_REQUEST_CAP,
            completion_reservation_cap=REAL_COMPLETION_RESERVATION_CAP,
            resume=resume, min_interval_seconds=self.request_interval_seconds, sleeper=time.sleep,
        )
        probe_provider = LedgeredProvider(
            self._probe_provider,
            ledger=ledger,
            request_cap=REAL_REQUEST_CAP,
            completion_reservation_cap=REAL_COMPLETION_RESERVATION_CAP,
            resume=resume, min_interval_seconds=self.request_interval_seconds, sleeper=time.sleep,
        )
        group_failed = any(record.status is BatchRunStatus.FAILED for record in records)
        for spec in self._ordered_specs():
            current = next(record for record in records if record.run_id == spec.run_id)
            if current.status is BatchRunStatus.COMPLETED:
                continue
            if group_failed and current.status is not BatchRunStatus.FAILED:
                continue
            if current.status is BatchRunStatus.FAILED and not resume:
                group_failed = True
                continue
            if Path(spec.artifacts.run_directory).exists():
                records = self._replace(records, spec.run_id, BatchRunStatus.FAILED, error_code="existing_failed_run_directory")
                self._write_record(self._record(records, BatchExperimentStatus.FAILED, error_code="existing_failed_run_directory"))
                group_failed = True
                continue

            records = self._replace(records, spec.run_id, BatchRunStatus.RUNNING)
            self._write_record(self._record(records, BatchExperimentStatus.RUNNING, error_code=None))
            result = self._run_condition(
                spec,
                agent_provider=agent_provider,
                probe_provider=probe_provider,
                resume=resume and current.status is BatchRunStatus.FAILED,
            )
            if result.status is not MeasurementExperimentStatus.COMPLETED:
                records = self._replace(
                    records, spec.run_id, BatchRunStatus.FAILED,
                    measurement_status=result.status.value, error_code=result.error_code or result.status.value,
                )
                self._write_record(self._record(
                    records,
                    BatchExperimentStatus.FAILED,
                    error_code=result.error_code or result.status.value,
                ))
                group_failed = True
                continue
            records = self._replace(records, spec.run_id, BatchRunStatus.COMPLETED, measurement_status=result.status.value)
            self._write_record(self._record(records, BatchExperimentStatus.RUNNING, error_code=None))
            # A resumed failed condition has now reached a safe completed
            # checkpoint, so the remaining explicitly ordered conditions may
            # continue. Any later failed record keeps the group stopped.
            group_failed = any(record.status is BatchRunStatus.FAILED for record in records)

        completed = all(record.status is BatchRunStatus.COMPLETED for record in records)
        batch_status = BatchExperimentStatus.COMPLETED if completed else BatchExperimentStatus.FAILED
        error_code = None if completed else "incomplete_matched_group"
        record = self._record(records, batch_status, error_code=error_code)
        self._write_record(record)
        if not completed:
            summary = self._summary(record, replay_statuses={}, metric_names=[], error_code=error_code)
            self._write_summary(summary)
            return summary

        replay_statuses = self._validate_completed_outputs()
        metric_names = self._evaluate_completed_batch()
        summary = self._summary(record, replay_statuses=replay_statuses, metric_names=metric_names, error_code=None)
        self._write_summary(summary)
        return summary

    def _run_condition(
        self,
        spec: BatchRunSpec,
        *,
        agent_provider: LedgeredProvider,
        probe_provider: LedgeredProvider,
        resume: bool,
    ):
        agent_runtime = AgentRuntime(AgentRequestSettings(
            model_name=spec.model_name,
            temperature=spec.runtime_parameters.agent_temperature,
            max_tokens=spec.runtime_parameters.agent_max_tokens,
            seed=spec.seed,
        ))
        probe_runtime = ContextualProbeRuntime(ContextualProbeRequestSettings(
            model_name=spec.probe_model_name,
            temperature=spec.runtime_parameters.probe_temperature,
            max_tokens=spec.runtime_parameters.probe_max_tokens,
            seed=spec.probe_seed,
        ))
        from .models import RunConfig

        runner = MeasuredProtocolExperimentRunner(
            RunConfig(
                run_id=spec.run_id, scenario_id=spec.scenario_id, model_name=spec.model_name,
                protocol=spec.condition, agent_count=len(spec.agent_ids), max_rounds=spec.max_rounds,
                seed=spec.seed, intervention_budget=0.0, output_dir=spec.output_dir,
            ),
            scenario=self.prepared.scenario, probe_set=self.prepared.probe_set,
            agent_runtime=agent_runtime, contextual_probe_runtime=probe_runtime,
            agent_provider=agent_provider, probe_provider=probe_provider,
            probe_seed=spec.probe_seed, pre_target_round=spec.pre_target_round,
            post_target_round=spec.post_target_round, execute_holdout_probe=False,
            recovery_checkpoint_dir=self.batch_directory / "private_probe_recovery" / spec.run_id,
        )
        return runner.run(resume=resume)

    def _load_or_initialize_records(self, *, resume: bool) -> list[BatchRunRecord]:
        manifest_path = self.batch_directory / "batch_manifest.json"
        record_path = self.batch_directory / "batch_record.json"
        if self.batch_directory.exists():
            if not resume:
                raise FileExistsError("refusing to overwrite existing real pilot batch directory")
            try:
                stored_manifest = BatchExperimentManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
                record = BatchExperimentRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
            except (OSError, ValidationError, ValueError) as exc:
                raise WVS7RealPilotError("real_pilot_recovery_artifacts_are_invalid") from exc
            if stored_manifest.model_dump(mode="json") != self.manifest.model_dump(mode="json"):
                raise WVS7RealPilotError("real_pilot_manifest_does_not_match_resume_request")
            return list(record.runs)

        for spec in self.manifest.runs:
            if Path(spec.artifacts.run_directory).exists():
                raise FileExistsError("refusing to overwrite an existing real pilot run directory")
        self.batch_directory.mkdir(parents=True, exist_ok=False)
        manifest_path.write_text(self.manifest.model_dump_json(indent=2), encoding="utf-8")
        records = [
            BatchRunRecord(
                run_id=spec.run_id, matched_group_id=spec.matched_group_id, condition=spec.condition,
                status=BatchRunStatus.PLANNED, artifacts=spec.artifacts,
            )
            for spec in self.manifest.runs
        ]
        self._write_record(self._record(records, BatchExperimentStatus.PLANNED, error_code=None))
        return records

    def _ordered_specs(self) -> list[BatchRunSpec]:
        by_id = {spec.run_id: spec for spec in self.manifest.runs}
        return [by_id[run_id] for run_id in self.manifest.run_order]

    @staticmethod
    def _replace(
        records: list[BatchRunRecord], run_id: str, status: BatchRunStatus,
        *, measurement_status: str | None = None, error_code: str | None = None,
    ) -> list[BatchRunRecord]:
        return [
            record.model_copy(update={"status": status, "measurement_status": measurement_status, "error_code": error_code})
            if record.run_id == run_id else record
            for record in records
        ]

    def _record(self, records: list[BatchRunRecord], status: BatchExperimentStatus, *, error_code: str | None) -> BatchExperimentRecord:
        group = BatchMatchedGroupRecord(
            matched_group_id=self.manifest.runs[0].matched_group_id,
            seed=self.manifest.seeds[0],
            run_ids=[record.run_id for record in records],
            run_statuses=[record.status for record in records],
            is_complete=all(record.status is BatchRunStatus.COMPLETED for record in records),
        )
        digest = hashlib.sha256(json.dumps(self.manifest.model_dump(mode="json"), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return BatchExperimentRecord(
            batch_id=self.batch_run_id, manifest_sha256=digest, status=status, runs=records,
            matched_groups=[group], completed_run_count=sum(record.status is BatchRunStatus.COMPLETED for record in records),
            failed_run_count=sum(record.status is BatchRunStatus.FAILED for record in records), error_code=error_code,
        )

    def _write_record(self, record: BatchExperimentRecord) -> None:
        (self.batch_directory / "batch_record.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")

    def _validate_completed_outputs(self) -> dict[str, str]:
        statuses: dict[str, str] = {}
        for spec in self.manifest.runs:
            report = validate_replay(spec.artifacts.events_path, spec.artifacts.run_record_path)
            if report.validation != "passed":
                raise WVS7RealPilotError("real_pilot_replay_validation_failed")
            self._validate_profiles(spec)
            statuses[spec.run_id] = report.validation
        if sha256_file(self.prepared.config.frozen_probe_set_path) != self.prepared.config.frozen_probe_set_sha256:
            raise WVS7RealPilotError("frozen_probe_set_changed_after_run")
        return statuses

    def _validate_profiles(self, spec: BatchRunSpec) -> None:
        for path in (Path(spec.artifacts.pre_probe_results_path), Path(spec.artifacts.post_probe_results_path)):
            try:
                rows = [ProbeResult.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            except (OSError, ValidationError, ValueError) as exc:
                raise WVS7RealPilotError("real_pilot_probe_profiles_are_invalid") from exc
            # Each profile aggregates the 23 item responses into the frozen
            # instrument dimensions, so profile vector length need not equal
            # item count. Item-level coverage is the explicit 23-response
            # check here.
            if len(rows) != 2 or any(len(row.responses) != 23 for row in rows):
                raise WVS7RealPilotError("real_pilot_probe_profiles_are_invalid")

    def _evaluate_completed_batch(self) -> list[str]:
        manifest_path = self.batch_directory / "offline_evaluation_manifest.json"
        report_path = self.batch_directory / "offline_evaluation_report.json"
        if report_path.exists() and manifest_path.exists():
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
                metrics = payload.get("metrics", [])
                if not isinstance(metrics, list):
                    raise ValueError("metrics must be a list")
                return [str(metric["metric_name"]) for metric in metrics if isinstance(metric, dict) and isinstance(metric.get("metric_name"), str)]
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise WVS7RealPilotError("real_pilot_evaluation_artifacts_are_invalid") from exc
        if report_path.exists() or manifest_path.exists():
            raise WVS7RealPilotError("refusing_to_overwrite_real_pilot_evaluation")
        evaluation_manifest = build_wvs7_evaluation_manifest(
            self.manifest,
            evaluation_id=f"{self.batch_run_id}-offline",
            coverage_threshold=self.prepared.config.coverage_threshold,
        )
        manifest_path.write_text(evaluation_manifest.model_dump_json(indent=2), encoding="utf-8")
        report = OfflineEvaluator(evaluation_manifest, manifest_directory=self.batch_directory).evaluate()
        write_evaluation_report(report, report_path)
        return [metric.metric_name for metric in report.metrics]

    def _summary(
        self, record: BatchExperimentRecord, *, replay_statuses: dict[str, str], metric_names: list[str], error_code: str | None,
    ) -> WVS7RealPilotSummary:
        return WVS7RealPilotSummary(
            batch_id=self.batch_run_id, status=record.status.value,
            completed_run_count=record.completed_run_count, failed_run_count=record.failed_run_count,
            matched_groups_complete=record.matched_groups[0].is_complete,
            frozen_probe_set_sha256=self.prepared.config.frozen_probe_set_sha256,
            request_ledger=self.ledger.summary(
                request_cap=REAL_REQUEST_CAP,
                completion_reservation_cap=REAL_COMPLETION_RESERVATION_CAP,
            ),
            replay_statuses=replay_statuses, offline_metric_names=metric_names, error_code=error_code,
        )

    def _write_summary(self, summary: WVS7RealPilotSummary) -> None:
        (self.batch_directory / "real_pilot_summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")


def build_wvs7_evaluation_manifest(
    batch_manifest: BatchExperimentManifest,
    *,
    evaluation_id: str,
    coverage_threshold: float,
) -> EvaluationManifest:
    """Bind each completed trajectory to explicit private pre and post measurements."""
    entries: list[EvaluationRunEntry] = []
    for spec in batch_manifest.runs:
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
            EvaluationRunEntry(
                **shared,
                entry_id=f"{spec.run_id}::initial",
                role="initial",
                probe_result_path=str(Path(spec.artifacts.pre_probe_results_path).resolve()),
            ),
            EvaluationRunEntry(
                **shared,
                entry_id=f"{spec.run_id}::final",
                role="final",
                probe_result_path=str(Path(spec.artifacts.post_probe_results_path).resolve()),
            ),
        ])
    return EvaluationManifest(
        evaluation_id=evaluation_id,
        scenario_id=batch_manifest.scenario_id,
        metric_suite=[
            MetricSuite.BASELINE_DIVERSITY,
            MetricSuite.REPRESENTATION,
            MetricSuite.COUNTERFACTUAL,
            MetricSuite.AUDIT,
        ],
        runs=entries,
        probe_sets=[batch_manifest.probe_set_id],
        metric_options=EvaluationMetricOptions(coverage_threshold=coverage_threshold, strict=True),
        metadata={
            "pilot_only": True,
            "single_seed": True,
            "not_paper_result": True,
            "no_causal_conclusion": True,
            "measurement_roles": "explicit_initial_and_final",
        },
    )

def _confirmation_reasons(arguments: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    if not arguments.allow_network:
        reasons.append("allow_network_required")
    if not arguments.confirm_run:
        reasons.append("confirm_run_required")
    if arguments.confirm_request_cap != REAL_REQUEST_CAP:
        reasons.append("confirm_request_cap_must_equal_384")
    confirmation = arguments.confirm_completion_reservation_cap
    if confirmation is None:
        confirmation = arguments.confirm_token_cap
    if confirmation != REAL_COMPLETION_RESERVATION_CAP:
        reasons.append("confirm_completion_reservation_cap_must_equal_51200")
    return reasons


def _safe_error_code(error: BaseException) -> str:
    message = str(error)
    for code in (
        "request_cap_reached", "completion_reservation_cap_reached", "completed_request_fingerprint_exists",
        "request_fingerprint_mismatch", "existing_failed_run_directory", "incomplete_matched_group",
        "real_pilot_replay_validation_failed", "real_pilot_probe_profiles_are_invalid",
    ):
        if code in message:
            return code
    return "real_pilot_failed"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one explicitly confirmed, bounded real WVS baseline pilot.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--confirm-request-cap", type=int)
    parser.add_argument("--confirm-completion-reservation-cap", type=int)
    parser.add_argument("--confirm-token-cap", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--request-interval-seconds", type=float, default=0.0)
    arguments = parser.parse_args(argv)
    reasons = _confirmation_reasons(arguments)
    if reasons:
        print(json.dumps({"status": "blocked", "blocking_reasons": reasons, "provider_call_count": 0}, sort_keys=True))
        return 1
    try:
        prepared, report = preflight_wvs7_real_pilot(
            arguments.config,
            run_id=arguments.run_id,
            resume=arguments.resume,
        )
        if report.status != "ready":
            print(json.dumps({"status": "blocked", "blocking_reasons": report.blocking_reasons, "provider_call_count": 0}, sort_keys=True))
            return 1
        if not os.environ.get("EVICON_LLM_MODEL") or not os.environ.get("EVICON_LLM_BASE_URL") or not os.environ.get("EVICON_LLM_API_KEY"):
            print(json.dumps({"status": "blocked", "blocking_reasons": ["provider_environment_incomplete"], "provider_call_count": 0}, sort_keys=True))
            return 1
        base = ProviderConfig.from_env(allow_network=True)
        model_name = base.model_name
        if not model_name:
            raise WVS7RealPilotError("environment_model_required")
        seed = prepared.config.seeds[0]
        agent_config = base.model_copy(update={
            "model_name": model_name, "seed": seed, "temperature": prepared.config.agent_temperature,
            "max_tokens": prepared.config.agent_max_tokens, "max_retries": arguments.max_retries,
            "reasoning_effort": ReasoningEffort.NONE,
        })
        probe_config = base.model_copy(update={
            "model_name": model_name, "seed": seed, "temperature": prepared.config.probe_temperature,
            "max_tokens": prepared.config.probe_max_tokens, "max_retries": arguments.max_retries,
            "reasoning_effort": ReasoningEffort.NONE,
        })
        runner = WVS7RealPilotRunner(
            prepared, batch_run_id=arguments.run_id, model_name=model_name,
            agent_provider=OpenAICompatibleProvider(agent_config), probe_provider=OpenAICompatibleProvider(probe_config),
            max_retries=arguments.max_retries, request_interval_seconds=arguments.request_interval_seconds,
        )
        summary = runner.run(resume=arguments.resume)
        print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))
        return 0 if summary.status == BatchExperimentStatus.COMPLETED.value else 1
    except (WVS7BaselinePilotError, WVS7RealPreflightError, WVS7RealPilotError, OSError, ValidationError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "blocking_reasons": [_safe_error_code(exc)]}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "WVS7RealPilotError",
    "WVS7RealPilotRunner",
    "WVS7RealPilotSummary",
    "build_wvs7_evaluation_manifest",
]
