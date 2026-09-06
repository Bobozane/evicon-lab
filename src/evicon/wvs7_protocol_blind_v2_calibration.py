"""Explicit execution gate for the preregistered WVS English protocol-blind v2 calibration.

The default CLI path is intentionally offline.  The injected runner below is
also usable with local fake providers for engineering tests; it never chooses a
protocol, agent, prompt, metric, or budget at runtime.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .batch_experiment import BatchExperimentStatus
from .frozen_wvs7_manifest import load_frozen_manifest, sha256_file
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ReasoningEffort
from .probe_loader import ProbeSetLoader
from .request_ledger import RequestLedger, RequestLedgerEntry, RequestLedgerStatus, RequestLedgerSummary
from .scenario_loader import ScenarioLoader
from .study_preregistration import (
    COMPLETION_RESERVATION_CAP,
    OBSERVATIONAL_V1_002_TOTAL_TOKENS,
    StudyPreregistration,
    StudyPreregistrationError,
    StudyPreregistrationLoader,
    StudyPreregistrationSummary,
    StudyStatus,
    load_and_validate_study_preregistration,
    validate_study_preregistration,
)
from .wvs7_baseline_pilot import (
    PreparedWVS7BaselinePilot,
    WVS7BaselinePilotConfig,
    _build_batch_manifest,
    _validate_frozen_probe_set,
    _validate_scenario,
)
from .wvs7_re_evaluate import WVS7ReevaluationError, re_evaluate_prepared_wvs7_batch
from .wvs7_real_pilot import WVS7RealPilotError, WVS7RealPilotRunner, WVS7RealPilotSummary


_ROOT = Path(__file__).resolve().parents[2]
_REQUEST_CAP = 384
_V2_TEMPLATE = "contextual_value_probe.v2"


class WVS7ProtocolBlindV2CalibrationError(ValueError):
    """Stable execution-gate error with no source, provider, or prompt contents."""


@dataclass(frozen=True)
class PreparedProtocolBlindV2Calibration:
    """Validated v2 declaration plus the existing runner's in-memory plan."""

    preregistration: StudyPreregistration
    preregistration_summary: StudyPreregistrationSummary
    baseline: PreparedWVS7BaselinePilot


class ProtocolBlindV2PreflightReport(BaseModel):
    """Safe local-only preflight result; it does not inspect environment variables."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ready", "blocked"]
    study_id: str
    run_id: str
    condition_count: int = Field(ge=0)
    seed_count: int = Field(ge=0)
    agent_count: int = Field(ge=0)
    rounds: int = Field(ge=0)
    request_cap: int = Field(gt=0)
    completion_reservation_cap: int = Field(gt=0)
    estimated_total_token_reference: int = Field(gt=0)
    output_paths_available: bool
    blocking_reasons: list[str] = Field(default_factory=list)


class ProtocolBlindV2CalibrationReceipt(BaseModel):
    """Content-free receipt for a completed v2 calibration matched group."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: str
    run_id: str
    status: Literal["calibration"]
    development_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    pilot_version: Literal["protocol_blind_v2"]
    v1_002_excluded_from_v2_main_analysis: Literal[True]
    condition_count: int = Field(ge=0)
    completed_run_count: int = Field(ge=0)
    matched_groups_complete: bool
    frozen_probe_set_sha256: str = Field(min_length=64, max_length=64)
    request_ledger: RequestLedgerSummary
    initial_final_report_path: str = Field(min_length=1)
    profile_drift_condition_count: int = Field(ge=0)
    total_token_reference: int = Field(gt=0)
    actual_total_token_delta_direction: Literal["below_reference", "at_reference", "above_reference", "unknown"]
    actual_total_token_delta_ratio: float | None = None


class ProtocolBlindV2ExecutionSummary(BaseModel):
    """Safe in-memory outcome of a fake or explicitly enabled real execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["calibration", "failed"]
    run_id: str
    completed_run_count: int = Field(ge=0)
    failed_run_count: int = Field(ge=0)
    matched_groups_complete: bool
    receipt_path: str | None = None
    error_code: str | None = None


def prepare_protocol_blind_v2_calibration(
    config_path: str | Path,
    *,
    test_output_root: Path | None = None,
) -> PreparedProtocolBlindV2Calibration:
    """Build the existing runner plan from the preregistration, without I/O outputs.

    ``test_output_root`` is dependency injection for local fake-provider tests;
    the CLI never supplies it and therefore always uses preregistered results
    locations.
    """
    try:
        study = StudyPreregistrationLoader.load(config_path)
        summary = validate_study_preregistration(study)
    except (StudyPreregistrationError, ValidationError, ValueError) as exc:
        raise WVS7ProtocolBlindV2CalibrationError("preregistration_invalid") from exc
    if study.status is not StudyStatus.CALIBRATION:
        raise WVS7ProtocolBlindV2CalibrationError("calibration_preregistration_not_runnable")

    config = _baseline_config_from_preregistration(study, test_output_root=test_output_root)
    try:
        scenario = ScenarioLoader.load(_project_path(study.scenario_path))
        probe_path = _project_path(study.frozen_probe_set_path)
        frozen_manifest = load_frozen_manifest(_project_path(study.frozen_manifest_path))
        _validate_frozen_probe_set(config, probe_path, frozen_manifest)
        probe_set = ProbeSetLoader.load(probe_path)
        _validate_scenario(config, scenario)
        batch_manifest = _build_batch_manifest(config, scenario, probe_set)
    except (OSError, ValueError) as exc:
        raise WVS7ProtocolBlindV2CalibrationError("preregistered_execution_inputs_invalid") from exc
    _validate_prepared_contract(study, config, batch_manifest)
    return PreparedProtocolBlindV2Calibration(study, summary, PreparedWVS7BaselinePilot(
        config=config,
        scenario=scenario,
        probe_set=probe_set,
        frozen_manifest=frozen_manifest,
        batch_manifest=batch_manifest,
    ))


def preflight_protocol_blind_v2_calibration(
    config_path: str | Path,
    *,
    resume: bool = False,
) -> tuple[PreparedProtocolBlindV2Calibration, ProtocolBlindV2PreflightReport]:
    """Check local artifacts and planned output paths without a provider or HTTP."""
    prepared = prepare_protocol_blind_v2_calibration(config_path)
    study = prepared.preregistration
    batch_dir = Path(study.batch_output_dir) / study.planned_calibration_run_id
    run_paths = [Path(study.run_output_dir) / f"{study.planned_calibration_run_id}-{condition.value}" for condition in study.conditions]
    batch_exists = batch_dir.exists()
    outputs_available = not batch_exists and not any(path.exists() for path in run_paths)
    ready = batch_exists if resume else outputs_available
    report = ProtocolBlindV2PreflightReport(
        status="ready" if ready else "blocked",
        study_id=study.study_id,
        run_id=study.planned_calibration_run_id,
        condition_count=len(study.conditions),
        seed_count=len(study.seeds),
        agent_count=len(study.agent_ids),
        rounds=study.max_rounds,
        request_cap=study.cost_plan.requests_per_matched_group,
        completion_reservation_cap=study.cost_plan.completion_reservation_cap,
        estimated_total_token_reference=study.cost_plan.estimated_total_tokens_per_matched_group,
        output_paths_available=outputs_available,
        blocking_reasons=[] if ready else (["resume_batch_not_found"] if resume else ["planned_output_already_exists"]),
    )
    return prepared, report


class WVS7ProtocolBlindV2CalibrationRunner:
    """Run the existing real baseline machinery under a validated v2 receipt gate."""

    def __init__(
        self,
        prepared: PreparedProtocolBlindV2Calibration,
        *,
        agent_provider: object,
        probe_provider: object,
    ) -> None:
        self.prepared = prepared
        study = prepared.preregistration
        self._runner = WVS7RealPilotRunner(
            prepared.baseline,
            batch_run_id=study.planned_calibration_run_id,
            model_name=study.model_name,
            agent_provider=agent_provider,
            probe_provider=probe_provider,
            max_retries=study.max_retries,
            request_interval_seconds=study.request_interval_seconds,
        )
        self._runner.manifest = self._runner.manifest.model_copy(update={
            "metadata": {
                **self._runner.manifest.metadata,
                "status": "calibration",
                "development_only": True,
                "not_paper_result": True,
                "no_causal_conclusion": True,
                "pilot_version": "protocol_blind_v2",
                "v1_002_excluded_from_v2_main_analysis": True,
            },
        })

    @property
    def batch_directory(self) -> Path:
        return self._runner.batch_directory

    def run(self, *, resume: bool = False) -> ProtocolBlindV2ExecutionSummary:
        """Delegate execution, then append only safe v2-specific audit receipts."""
        result = self._runner.run(resume=resume)
        if result.status != BatchExperimentStatus.COMPLETED.value:
            return ProtocolBlindV2ExecutionSummary(
                status="failed",
                run_id=self.prepared.preregistration.planned_calibration_run_id,
                completed_run_count=result.completed_run_count,
                failed_run_count=result.failed_run_count,
                matched_groups_complete=result.matched_groups_complete,
                error_code=result.error_code or "incomplete_matched_group",
            )
        receipt = self._load_or_create_receipt(result)
        return ProtocolBlindV2ExecutionSummary(
            status="calibration",
            run_id=receipt.run_id,
            completed_run_count=receipt.completed_run_count,
            failed_run_count=0,
            matched_groups_complete=receipt.matched_groups_complete,
            receipt_path=str(self._receipt_path()),
        )

    def _load_or_create_receipt(self, result: WVS7RealPilotSummary) -> ProtocolBlindV2CalibrationReceipt:
        receipt_path = self._receipt_path()
        if receipt_path.exists():
            try:
                receipt = ProtocolBlindV2CalibrationReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
            except (OSError, ValidationError, ValueError) as exc:
                raise WVS7ProtocolBlindV2CalibrationError("calibration_receipt_invalid") from exc
            if receipt.run_id != self.prepared.preregistration.planned_calibration_run_id:
                raise WVS7ProtocolBlindV2CalibrationError("calibration_receipt_run_id_mismatch")
            self._validate_completion(result)
            report_path, profile_count = self._ensure_initial_final_report()
            if receipt.initial_final_report_path != str(report_path) or receipt.profile_drift_condition_count != profile_count:
                raise WVS7ProtocolBlindV2CalibrationError("calibration_receipt_evaluation_mismatch")
            return receipt

        self._validate_completion(result)
        report_path, profile_count = self._ensure_initial_final_report()
        ledger_summary = self._runner.ledger.summary(
            request_cap=_REQUEST_CAP,
            completion_reservation_cap=COMPLETION_RESERVATION_CAP,
        )
        direction, ratio = _token_reference_delta(
            ledger_summary.actual_total_token_count,
            self.prepared.preregistration.cost_plan.estimated_total_tokens_per_matched_group,
        )
        receipt = ProtocolBlindV2CalibrationReceipt(
            study_id=self.prepared.preregistration.study_id,
            run_id=self.prepared.preregistration.planned_calibration_run_id,
            status="calibration",
            development_only=True,
            not_paper_result=True,
            no_causal_conclusion=True,
            pilot_version="protocol_blind_v2",
            v1_002_excluded_from_v2_main_analysis=True,
            condition_count=4,
            completed_run_count=result.completed_run_count,
            matched_groups_complete=result.matched_groups_complete,
            frozen_probe_set_sha256=self.prepared.preregistration.frozen_probe_set_sha256,
            request_ledger=ledger_summary,
            initial_final_report_path=str(report_path),
            profile_drift_condition_count=profile_count,
            total_token_reference=self.prepared.preregistration.cost_plan.estimated_total_tokens_per_matched_group,
            actual_total_token_delta_direction=direction,
            actual_total_token_delta_ratio=ratio,
        )
        receipt_path.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
        self._write_initial_final_calibration_report(receipt)
        return receipt

    def _validate_completion(self, result: WVS7RealPilotSummary) -> None:
        study = self.prepared.preregistration
        if result.completed_run_count != 4 or not result.matched_groups_complete or len(result.replay_statuses) != 4:
            raise WVS7ProtocolBlindV2CalibrationError("completed_matched_group_required")
        if set(result.replay_statuses.values()) != {"passed"}:
            raise WVS7ProtocolBlindV2CalibrationError("calibration_replay_validation_failed")
        if sha256_file(_project_path(study.frozen_probe_set_path)) != study.frozen_probe_set_sha256:
            raise WVS7ProtocolBlindV2CalibrationError("frozen_probe_set_changed_after_calibration")
        entries = self._runner.ledger.entries()
        logical_fingerprints = {entry.fingerprint for entry in entries if entry.status is RequestLedgerStatus.STARTED}
        if len(logical_fingerprints) != _REQUEST_CAP:
            raise WVS7ProtocolBlindV2CalibrationError("logical_request_count_mismatch")
        probe_entries = [entry for entry in entries if entry.phase == "probe"]
        if not probe_entries or any(entry.template_version != _V2_TEMPLATE for entry in probe_entries):
            raise WVS7ProtocolBlindV2CalibrationError("probe_template_version_mismatch")

    def _ensure_initial_final_report(self) -> tuple[Path, int]:
        manifest_path = self.batch_directory / "offline_evaluation_manifest_initial_final_v2.json"
        report_path = self.batch_directory / "offline_evaluation_report_initial_final_v2.json"
        if manifest_path.exists() != report_path.exists():
            raise WVS7ProtocolBlindV2CalibrationError("initial_final_report_artifacts_inconsistent")
        if not report_path.exists():
            try:
                result = re_evaluate_prepared_wvs7_batch(self.prepared.baseline, batch_id=self.prepared.preregistration.planned_calibration_run_id)
                report_path = Path(str(result["report_path"]))
            except (WVS7ReevaluationError, OSError, ValueError) as exc:
                raise WVS7ProtocolBlindV2CalibrationError("initial_final_reevaluation_failed") from exc
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            profile_count = sum(
                isinstance(metric, dict) and metric.get("metric_name") == "profile_drift"
                for metric in metrics
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise WVS7ProtocolBlindV2CalibrationError("initial_final_report_invalid") from exc
        if profile_count != 4:
            raise WVS7ProtocolBlindV2CalibrationError("profile_drift_condition_count_mismatch")
        return report_path, profile_count

    def _write_initial_final_calibration_report(self, receipt: ProtocolBlindV2CalibrationReceipt) -> None:
        """Write a flag-only report alongside the generic evaluator output."""
        path = self.batch_directory / "protocol_blind_v2_initial_final_report.json"
        if path.exists():
            raise WVS7ProtocolBlindV2CalibrationError("initial_final_calibration_report_already_exists")
        payload = {
            "status": "calibration",
            "development_only": True,
            "not_paper_result": True,
            "no_causal_conclusion": True,
            "pilot_version": "protocol_blind_v2",
            "v1_002_excluded_from_v2_main_analysis": True,
            "profile_drift_condition_count": receipt.profile_drift_condition_count,
            "source_report_sha256": sha256_file(receipt.initial_final_report_path),
        }
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")

    def _receipt_path(self) -> Path:
        return self.batch_directory / "protocol_blind_v2_calibration_receipt.json"


def _baseline_config_from_preregistration(
    study: StudyPreregistration,
    *,
    test_output_root: Path | None,
) -> WVS7BaselinePilotConfig:
    """Map fixed preregistration fields to the existing baseline runner schema."""
    run_output = str(test_output_root / "results") if test_output_root is not None else study.run_output_dir
    batch_output = str(test_output_root / "results" / "batches") if test_output_root is not None else study.batch_output_dir
    declarations = [
        {
            "run_id": f"{study.planned_calibration_run_id}-{condition.value}",
            "matched_group_id": f"{study.planned_calibration_run_id}-matched-group",
            "condition": condition.value,
            "seed": study.seeds[0],
        }
        for condition in study.conditions
    ]
    return WVS7BaselinePilotConfig.model_validate({
        "study_id": study.study_id,
        "version": study.preregistration_version,
        "scenario_path": study.scenario_path,
        "frozen_probe_set_path": study.frozen_probe_set_path,
        "frozen_manifest_path": study.frozen_manifest_path,
        "frozen_probe_set_sha256": study.frozen_probe_set_sha256,
        "probe_set_id": study.probe_set_id,
        "agent_ids": study.agent_ids,
        "max_rounds": study.max_rounds,
        "seeds": study.seeds,
        "model_name": study.model_name,
        "probe_model_name": study.model_name,
        "agent_temperature": study.agent_temperature,
        "agent_max_tokens": study.agent_max_tokens,
        "probe_temperature": study.probe_temperature,
        "probe_max_tokens": study.probe_max_tokens,
        "max_retries": study.max_retries,
        "coverage_threshold": study.coverage_threshold,
        "run_output_dir": run_output,
        "batch_output_dir": batch_output,
        "metadata": {
            "status": "calibration",
            "development_only": True,
            "not_paper_result": True,
            "no_causal_conclusion": True,
            "pilot_version": "protocol_blind_v2",
            "v1_002_excluded_from_v2_main_analysis": True,
        },
        "runs": declarations,
        "run_order": [declaration["run_id"] for declaration in declarations],
    })


def _validate_prepared_contract(
    study: StudyPreregistration,
    config: WVS7BaselinePilotConfig,
    batch_manifest: object,
) -> None:
    """Defensive binding check so no inherited baseline field can drift v2."""
    if (
        config.model_name != study.model_name
        or config.probe_model_name != study.model_name
        or config.agent_ids != study.agent_ids
        or config.seeds != study.seeds
        or config.max_rounds != study.max_rounds
        or config.agent_temperature != study.agent_temperature
        or config.agent_max_tokens != study.agent_max_tokens
        or config.probe_temperature != study.probe_temperature
        or config.probe_max_tokens != study.probe_max_tokens
        or len(batch_manifest.runs) != 4
        or [run.condition for run in batch_manifest.runs] != study.conditions
    ):
        raise WVS7ProtocolBlindV2CalibrationError("preregistered_execution_contract_mismatch")


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _ROOT / path


def _token_reference_delta(actual: int | None, reference: int) -> tuple[str, float | None]:
    if actual is None:
        return "unknown", None
    delta = (actual - reference) / reference
    if actual < reference:
        return "below_reference", delta
    if actual > reference:
        return "above_reference", delta
    return "at_reference", 0.0


def _confirmation_reasons(arguments: argparse.Namespace, study: StudyPreregistration) -> list[str]:
    reasons: list[str] = []
    if not arguments.allow_network:
        reasons.append("allow_network_required")
    if not arguments.confirm_run:
        reasons.append("confirm_run_required")
    if arguments.confirm_request_cap != study.cost_plan.requests_per_matched_group:
        reasons.append("confirm_request_cap_must_equal_384")
    if arguments.confirm_completion_reservation_cap != study.cost_plan.completion_reservation_cap:
        reasons.append("confirm_completion_reservation_cap_must_equal_51200")
    if arguments.confirm_estimated_total_token_reference != study.cost_plan.estimated_total_tokens_per_matched_group:
        reasons.append("confirm_estimated_total_token_reference_must_equal_190602")
    return reasons


def _safe_error_code(error: BaseException) -> str:
    text = str(error)
    for code in (
        "preregistration_invalid", "calibration_preregistration_not_runnable", "planned_output_already_exists",
        "resume_batch_not_found", "preregistered_execution_inputs_invalid", "logical_request_count_mismatch",
        "probe_template_version_mismatch", "profile_drift_condition_count_mismatch",
        "frozen_probe_set_changed_after_calibration",
    ):
        if code in text:
            return code
    return "protocol_blind_v2_calibration_blocked"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run only an explicitly confirmed WVS protocol-blind v2 calibration.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--confirm-request-cap", type=int)
    parser.add_argument("--confirm-completion-reservation-cap", type=int)
    parser.add_argument("--confirm-estimated-total-token-reference", type=int)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        summary = load_and_validate_study_preregistration(arguments.config)
        study = StudyPreregistrationLoader.load(arguments.config)
    except (StudyPreregistrationError, OSError, ValidationError, ValueError):
        print(json.dumps({"status": "blocked", "blocking_reasons": ["preregistration_validation_failed"], "provider_call_count": 0}, sort_keys=True))
        return 1
    if not arguments.allow_network:
        print(json.dumps({
            "status": "network_disabled",
            "study_id": summary.study_id,
            "run_id": study.planned_calibration_run_id,
            "provider_call_count": 0,
        }, sort_keys=True))
        return 0
    reasons = _confirmation_reasons(arguments, study)
    if reasons:
        print(json.dumps({"status": "blocked", "blocking_reasons": reasons, "provider_call_count": 0}, sort_keys=True))
        return 1
    try:
        prepared, preflight = preflight_protocol_blind_v2_calibration(arguments.config, resume=arguments.resume)
        if preflight.status != "ready":
            print(json.dumps({"status": "blocked", "blocking_reasons": preflight.blocking_reasons, "provider_call_count": 0}, sort_keys=True))
            return 1
        if not all(os.environ.get(name) for name in ("EVICON_LLM_BASE_URL", "EVICON_LLM_API_KEY", "EVICON_LLM_MODEL")):
            print(json.dumps({"status": "blocked", "blocking_reasons": ["provider_environment_incomplete"], "provider_call_count": 0}, sort_keys=True))
            return 1
        base = ProviderConfig.from_env(allow_network=True)
        if base.model_name != prepared.preregistration.model_name:
            print(json.dumps({"status": "blocked", "blocking_reasons": ["environment_model_does_not_match_preregistration"], "provider_call_count": 0}, sort_keys=True))
            return 1
        agent_config = base.model_copy(update={
            "temperature": prepared.preregistration.agent_temperature,
            "max_tokens": prepared.preregistration.agent_max_tokens,
            "max_retries": prepared.preregistration.max_retries,
            "reasoning_effort": ReasoningEffort.NONE,
            "seed": prepared.preregistration.seeds[0],
        })
        probe_config = base.model_copy(update={
            "temperature": prepared.preregistration.probe_temperature,
            "max_tokens": prepared.preregistration.probe_max_tokens,
            "max_retries": prepared.preregistration.max_retries,
            "reasoning_effort": ReasoningEffort.NONE,
            "seed": prepared.preregistration.seeds[0],
        })
        runner = WVS7ProtocolBlindV2CalibrationRunner(
            prepared,
            agent_provider=OpenAICompatibleProvider(agent_config),
            probe_provider=OpenAICompatibleProvider(probe_config),
        )
        result = runner.run(resume=arguments.resume)
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))
        return 0 if result.status == "calibration" else 1
    except (WVS7ProtocolBlindV2CalibrationError, WVS7RealPilotError, WVS7ReevaluationError, OSError, ValidationError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "blocking_reasons": [_safe_error_code(exc)], "provider_call_count": 0}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PreparedProtocolBlindV2Calibration",
    "ProtocolBlindV2CalibrationReceipt",
    "ProtocolBlindV2ExecutionSummary",
    "ProtocolBlindV2PreflightReport",
    "WVS7ProtocolBlindV2CalibrationError",
    "WVS7ProtocolBlindV2CalibrationRunner",
    "preflight_protocol_blind_v2_calibration",
    "prepare_protocol_blind_v2_calibration",
]
