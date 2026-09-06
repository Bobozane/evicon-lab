"""Evaluator-only offline analysis for the completed H-G provenance-cascade Pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .cascade_hg_compatibility_receipt import validate_receipt as validate_compatibility_receipt
from .cascade_hg_outcome_replay import HG_OUTCOME_REPLAY_VERSION, HGOutcomeReplayValidator
from .cascade_outcome_replay import CascadeOutcomeReplayStatus
from .cascade_protocol import CascadeScenarioLoader
from .cascade_real_agent_runner import CascadeRealAgentRunRecord
from .provenance_cascade import EvaluatorTruthLoader, validate_fixture_pair
from .provenance_cascade_hg_pilot import (
    HGPilotBatchRecord,
    HGPilotReceipt,
    HGPilotRunStatus,
    _records_hash,
)
from .provenance_cascade_hg_replay_amendment import HGReplayTechnicalAmendmentReceipt
from .provenance_cascade_identifiability import HGApproval, HGPilotConfig, sha256_file
from .provenance_cascade_pilot_analysis import (
    HFConditionMetricSummary,
    HFConditionOperations,
    HFMatchedDifference,
    HFMetricStatus,
    HFRunMetric,
    HFRunOperations,
    HFSafeInputHash,
    _assert_public_safe,
    _condition_operations,
    _condition_summaries,
    _evaluate_run,
    _matched_differences,
    _stable_sha,
)
from .provenance_cascade_pilot_analysis_plan import EXPECTED_CONDITIONS, EXPECTED_METRICS
from .provenance_cascade_preregistration import ProvenanceCascadePreregistrationLoader
from .request_ledger import RequestLedger, RequestLedgerStatus
from .provenance_cascade_hg_analysis_plan import (
    EXPECTED_HG_REPLAY_CONTRACT,
    HG_ANALYSIS_VERSION,
    HGAnalysisPlan,
    load_hg_analysis_plan,
)

_ROOT = Path(__file__).resolve().parents[2]


class HGEvaluatorAnalysisError(ValueError):
    """Stable, content-free analysis failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HGScenarioConditionMetricSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_id: str
    condition: str
    metric_name: str
    run_count: int = Field(ge=0)
    applicable_count: int = Field(ge=0)
    not_applicable_count: int = Field(ge=0)
    denominator_sum: int = Field(ge=0)
    numerator_sum: float | None = None
    pooled_value: float | None = None
    mean: float | None = None
    median: float | None = None
    standard_deviation: float | None = None


class HGMetricSeparationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    metric_name: str
    applicable_condition_count: int = Field(ge=0)
    condition_means: dict[str, float | None]
    range_across_condition_means: float | None = None
    condition_mean_difference_observed: bool
    interpretation: Literal["descriptive_only_no_threshold"] = "descriptive_only_no_threshold"


class HGAnalysisManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    analysis_id: str
    analysis_version: Literal["provenance_cascade_hg_pilot_analysis.v1"]
    study_id: str
    input_hashes: tuple[HFSafeInputHash, ...]
    run_record_sha256: dict[str, str]
    final_batch_record_sha256: str
    pre_resume_batch_record_sha256: str
    technical_replay_amendment_sha256: str
    outcome_replay_contract_version: str
    scenario_count: Literal[4]
    seed_count: Literal[3]
    condition_count: Literal[4]
    run_count: Literal[48]
    matched_group_count: Literal[12]
    logical_request_count: Literal[864]
    transport_attempt_count: Literal[864]
    primary_metrics: tuple[str, ...]
    h_d21_pilot_excluded: Literal[True]
    h_f_analysis_excluded: Literal[True]
    wvs_and_calibration_excluded: Literal[True]
    development_only: Literal[True]
    pilot_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    private_truth_exposed: Literal[False]
    network_used_for_analysis: Literal[False]


class HGAnalysisReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    analysis_id: str
    analysis_version: Literal["provenance_cascade_hg_pilot_analysis.v1"]
    data_integrity_status: Literal["passed"]
    effect_evidence_status: Literal["descriptive_pilot_only"]
    decision_status: Literal["thresholds_not_pre_registered"]
    decision_reason_codes: tuple[str, ...]
    run_metrics: tuple[HFRunMetric, ...]
    scenario_condition_summaries: tuple[HGScenarioConditionMetricSummary, ...]
    condition_summaries: tuple[HFConditionMetricSummary, ...]
    matched_differences: tuple[HFMatchedDifference, ...]
    separation_summaries: tuple[HGMetricSeparationSummary, ...]
    run_operations: tuple[HFRunOperations, ...]
    condition_operations: tuple[HFConditionOperations, ...]
    warnings: tuple[str, ...]
    development_only: Literal[True]
    pilot_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    private_truth_exposed: Literal[False]


class HGAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["completed", "validated", "blocked"]
    analysis_id: str
    run_count: int = Field(ge=0)
    matched_group_count: int = Field(ge=0)
    metric_count: int = Field(ge=0)
    decision_status: str | None = None
    output_dir: str | None = None
    blocking_reasons: tuple[str, ...] = ()
    network: Literal["disabled"] = "disabled"
    provider_constructed: Literal[False] = False
    private_truth_exposed: Literal[False] = False
    not_paper_result: Literal[True] = True
    no_causal_conclusion: Literal[True] = True


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (_ROOT / path).resolve()


def _relative_to_config(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _require(value: bool, code: str) -> None:
    if not value:
        raise HGEvaluatorAnalysisError(code)


def _validate_hash(path: Path, expected: str, code: str) -> None:
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise HGEvaluatorAnalysisError(code) from exc
    if actual != expected:
        raise HGEvaluatorAnalysisError(code)


def _load_json_model(path: Path, model: type[BaseModel], code: str) -> BaseModel:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError, ValueError) as exc:
        raise HGEvaluatorAnalysisError(code) from exc


def _scenario_material_bindings_sha(config: HGPilotConfig) -> str:
    payload = [
        {
            "scenario_id": item.scenario_id,
            "scenario_sha256": item.scenario_sha256,
            "graph_sha256": item.graph_sha256,
            "truth_sha256": item.truth_sha256,
        }
        for item in config.scenario_materials
    ]
    return _stable_sha(payload)


def _load_public_scenarios(config_path: Path, config: HGPilotConfig) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    for material in config.scenario_materials:
        scenario_path = _relative_to_config(config_path, material.scenario_path)
        graph_path = _relative_to_config(config_path, material.graph_path)
        _validate_hash(scenario_path, material.scenario_sha256, "scenario_hash_mismatch")
        _validate_hash(graph_path, material.graph_sha256, "graph_hash_mismatch")
        try:
            raw_graph = json.loads(graph_path.read_text(encoding="utf-8"))
            _assert_public_safe(raw_graph)
            scenario = CascadeScenarioLoader.load(scenario_path)
        except Exception as exc:
            raise HGEvaluatorAnalysisError("public_scenario_invalid") from exc
        _require(scenario.scenario_id == material.scenario_id, "scenario_coordinate_mismatch")
        scenarios[scenario.scenario_id] = scenario
    _require(len(scenarios) == 4, "scenario_set_incomplete")
    return scenarios


def _validate_coordinates(config: HGPilotConfig, batch: HGPilotBatchRecord) -> None:
    specs = {item.run_id: item for item in config.runs}
    states = {item.run_id: item for item in batch.runs}
    _require(len(specs) == len(config.runs) == 48, "run_set_incomplete")
    _require(len(states) == len(batch.runs) == 48 and set(states) == set(specs), "run_set_incomplete")
    groups: dict[str, set[Any]] = defaultdict(set)
    for spec in specs.values():
        state = states[spec.run_id]
        groups[spec.matched_group_id].add(spec.condition)
        _require(
            state.scenario_id == spec.scenario_id
            and state.seed == spec.seed
            and state.condition == spec.condition.value,
            "run_coordinate_mismatch",
        )
        _require(state.status is HGPilotRunStatus.COMPLETED and state.error_code is None, "run_not_completed")
        _require(state.logical_request_count == 18 and state.transport_attempt_count == 18, "run_request_audit_mismatch")
        _require(
            state.cascade_replay_status == "passed"
            and state.application_replay_status == "passed"
            and state.outcome_replay_status == "passed",
            "replay_not_passed",
        )
    _require(
        len(groups) == 12 and all(value == set(EXPECTED_CONDITIONS) for value in groups.values()),
        "matched_group_incomplete",
    )


def _scenario_condition_summaries(metrics: list[HFRunMetric], scenario_ids: tuple[str, ...]) -> tuple[HGScenarioConditionMetricSummary, ...]:
    output: list[HGScenarioConditionMetricSummary] = []
    for scenario_id in scenario_ids:
        for condition in EXPECTED_CONDITIONS:
            for metric_name in EXPECTED_METRICS:
                rows = [
                    item for item in metrics
                    if item.scenario_id == scenario_id
                    and item.condition is condition
                    and item.metric_name == metric_name
                ]
                values = [item.value for item in rows if item.status is HFMetricStatus.VALUE and item.value is not None]
                numerators = [item.numerator for item in rows if item.status is HFMetricStatus.VALUE and item.numerator is not None]
                denominators = [item.denominator for item in rows if item.status is HFMetricStatus.VALUE]
                numerator_sum = float(sum(numerators)) if numerators and len(numerators) == len(values) else None
                denominator_sum = sum(denominators)
                output.append(HGScenarioConditionMetricSummary(
                    scenario_id=scenario_id,
                    condition=condition.value,
                    metric_name=metric_name,
                    run_count=len(rows),
                    applicable_count=len(values),
                    not_applicable_count=len(rows) - len(values),
                    denominator_sum=denominator_sum,
                    numerator_sum=numerator_sum,
                    pooled_value=(numerator_sum / denominator_sum) if numerator_sum is not None and denominator_sum else None,
                    mean=float(statistics.mean(values)) if values else None,
                    median=float(statistics.median(values)) if values else None,
                    standard_deviation=float(statistics.pstdev(values)) if values else None,
                ))
    return tuple(output)


def _separation_summaries(summaries: tuple[HFConditionMetricSummary, ...]) -> tuple[HGMetricSeparationSummary, ...]:
    output: list[HGMetricSeparationSummary] = []
    for metric_name in EXPECTED_METRICS:
        means = {
            item.condition.value: item.mean
            for item in summaries
            if item.metric_name == metric_name
        }
        values = [value for value in means.values() if value is not None]
        observed_range = float(max(values) - min(values)) if values else None
        output.append(HGMetricSeparationSummary(
            metric_name=metric_name,
            applicable_condition_count=len(values),
            condition_means=means,
            range_across_condition_means=observed_range,
            condition_mean_difference_observed=bool(observed_range is not None and observed_range > 0.0),
        ))
    return tuple(output)


def analyze_hg_pilot(plan: HGAnalysisPlan) -> tuple[HGAnalysisManifest, HGAnalysisReport]:
    paths = {
        "config": _rooted(plan.config_path),
        "protocol": _rooted(plan.protocol_path),
        "policy": _rooted(plan.policy_path),
        "amendment": _rooted(plan.amendment_path),
        "amendment_receipt": _rooted(plan.amendment_receipt_path),
        "approval": _rooted(plan.approval_path),
        "compatibility_receipt": _rooted(plan.compatibility_receipt_path),
        "technical_replay_amendment": _rooted(plan.technical_replay_amendment_path),
        "old_validator": _rooted(plan.old_validator_path),
        "new_validator": _rooted(plan.new_validator_path),
        "pilot_receipt": _rooted(plan.pilot_receipt_path),
        "batch_record": _rooted(plan.batch_record_path),
        "preregistration": _rooted(plan.preregistration_path),
    }
    expected_hashes = {
        "config": plan.config_sha256,
        "protocol": plan.protocol_sha256,
        "policy": plan.policy_sha256,
        "amendment": plan.amendment_sha256,
        "amendment_receipt": plan.amendment_receipt_sha256,
        "approval": plan.approval_sha256,
        "compatibility_receipt": plan.compatibility_receipt_sha256,
        "technical_replay_amendment": plan.technical_replay_amendment_sha256,
        "old_validator": plan.old_validator_sha256,
        "new_validator": plan.new_validator_sha256,
        "pilot_receipt": plan.pilot_receipt_sha256,
        "batch_record": plan.batch_record_sha256,
        "preregistration": plan.preregistration_sha256,
    }
    for kind, path in paths.items():
        _validate_hash(path, expected_hashes[kind], f"{kind}_hash_mismatch")

    try:
        config = HGPilotConfig.model_validate(tomllib.loads(paths["config"].read_text(encoding="utf-8")))
    except Exception as exc:
        raise HGEvaluatorAnalysisError("hg_config_invalid") from exc
    _require(config.study_id == plan.study_id, "study_id_mismatch")
    _require(tuple(item.scenario_id for item in config.scenario_materials) == plan.scenario_ids, "scenario_plan_mismatch")
    _require(tuple(item.value for item in config.conditions) == plan.conditions and config.seeds == plan.seeds, "matched_plan_mismatch")
    _require(config.request_cap == 864 and config.completion_reservation_cap == 442368, "budget_binding_mismatch")
    _require(sha256_file(_relative_to_config(paths["config"], config.policy_path)) == plan.policy_sha256, "policy_hash_mismatch")

    try:
        approval = HGApproval.model_validate(tomllib.loads(paths["approval"].read_text(encoding="utf-8")))
    except Exception as exc:
        raise HGEvaluatorAnalysisError("approval_invalid") from exc
    _require(approval.ready and approval.acceptance_status == "accepted", "approval_not_accepted")
    _require(
        approval.config_sha256 == plan.config_sha256
        and approval.protocol_sha256 == plan.protocol_sha256
        and approval.amendment_sha256 == plan.amendment_sha256
        and approval.amendment_receipt_sha256 == plan.amendment_receipt_sha256
        and approval.compatibility_receipt_sha256 == plan.compatibility_receipt_sha256,
        "approval_binding_mismatch",
    )
    try:
        validate_compatibility_receipt(
            paths["compatibility_receipt"],
            expected_hash=plan.compatibility_receipt_sha256,
            config_path=paths["config"],
            protocol_path=paths["protocol"],
            amendment_sha256=plan.amendment_sha256,
        )
    except Exception as exc:
        raise HGEvaluatorAnalysisError("compatibility_receipt_invalid") from exc

    try:
        amendment_receipt = json.loads(paths["amendment_receipt"].read_text(encoding="utf-8"))
    except Exception as exc:
        raise HGEvaluatorAnalysisError("amendment_receipt_invalid") from exc
    _require(
        amendment_receipt.get("config_sha256") == plan.config_sha256
        and amendment_receipt.get("protocol_sha256") == plan.protocol_sha256
        and amendment_receipt.get("amendment_sha256") == plan.amendment_sha256
        and amendment_receipt.get("fake_replay_passed_count") == 48,
        "amendment_receipt_binding_mismatch",
    )

    technical = _load_json_model(
        paths["technical_replay_amendment"],
        HGReplayTechnicalAmendmentReceipt,
        "technical_replay_amendment_invalid",
    )
    assert isinstance(technical, HGReplayTechnicalAmendmentReceipt)
    _require(
        technical.config_sha256 == plan.config_sha256
        and technical.protocol_sha256 == plan.protocol_sha256
        and technical.amendment_sha256 == plan.amendment_sha256
        and technical.approval_sha256 == plan.approval_sha256
        and technical.compatibility_receipt_sha256 == plan.compatibility_receipt_sha256,
        "technical_replay_binding_mismatch",
    )
    _require(
        technical.old_validator_sha256 == plan.old_validator_sha256
        and technical.new_validator_sha256 == plan.new_validator_sha256
        and technical.outcome_replay_contract_version == plan.outcome_replay_contract_version,
        "technical_validator_binding_mismatch",
    )
    _require(
        technical.scenario_material_bindings_sha256 == _scenario_material_bindings_sha(config)
        and technical.completed_run_count == 12
        and technical.failed_run_logical_request_count == 18
        and technical.failed_run_completed_fingerprint_count == 18
        and technical.failed_run_resume_provider_calls_required == 0
        and technical.failed_run_cascade_replay_status == "passed"
        and technical.failed_run_application_replay_status == "passed"
        and technical.failed_run_outcome_replay_status == "passed"
        and technical.changes_replay_validation_only
        and not technical.existing_results_rewritten
        and technical.network == "disabled",
        "technical_replay_audit_invalid",
    )

    receipt = _load_json_model(paths["pilot_receipt"], HGPilotReceipt, "pilot_receipt_invalid")
    batch = _load_json_model(paths["batch_record"], HGPilotBatchRecord, "batch_record_invalid")
    assert isinstance(receipt, HGPilotReceipt) and isinstance(batch, HGPilotBatchRecord)
    _require(
        receipt.status == "completed"
        and receipt.run_count == receipt.completed_run_count == 48
        and receipt.failed_run_count == 0
        and receipt.matched_group_count == 12
        and receipt.replay_passed_count == 48,
        "pilot_receipt_incomplete",
    )
    _require(
        receipt.logical_request_count == receipt.provider_call_count == receipt.transport_attempt_count == 864
        and receipt.request_cap == 864
        and receipt.completion_reservation_cap == 442368,
        "pilot_request_audit_mismatch",
    )
    _require(
        receipt.config_sha256 == plan.config_sha256
        and receipt.protocol_sha256 == plan.protocol_sha256
        and receipt.template_sha256 == plan.protocol_sha256
        and receipt.amendment_sha256 == plan.amendment_sha256
        and receipt.approval_sha256 == plan.approval_sha256
        and receipt.compatibility_receipt_sha256 == plan.compatibility_receipt_sha256,
        "pilot_receipt_binding_mismatch",
    )
    _require(
        receipt.outcome_replay_contract_version == plan.outcome_replay_contract_version
        and batch.outcome_replay_contract_version == plan.outcome_replay_contract_version,
        "outcome_replay_contract_mismatch",
    )
    _require(
        not receipt.legacy_assets_touched
        and not receipt.private_truth_exposed
        and not receipt.parser_recovery_enabled
        and receipt.not_paper_result
        and receipt.no_causal_conclusion,
        "pilot_safety_binding_mismatch",
    )
    _require(
        batch.status is HGPilotRunStatus.COMPLETED
        and batch.failure_code is None
        and len(batch.runs) == 48
        and batch.config_sha256 == plan.config_sha256
        and batch.protocol_sha256 == plan.protocol_sha256
        and batch.template_sha256 == plan.protocol_sha256
        and batch.amendment_sha256 == plan.amendment_sha256
        and batch.approval_sha256 == plan.approval_sha256
        and batch.compatibility_receipt_sha256 == plan.compatibility_receipt_sha256,
        "batch_record_binding_mismatch",
    )
    _validate_coordinates(config, batch)

    prereg = ProvenanceCascadePreregistrationLoader.load(paths["preregistration"])
    _require(tuple(prereg.primary_metrics) == EXPECTED_METRICS, "preregistered_metric_suite_mismatch")
    _require(prereg.success_criterion.no_unregistered_thresholds, "preregistration_threshold_boundary_missing")
    scenarios = _load_public_scenarios(paths["config"], config)
    pilot_root = _rooted(plan.pilot_root)
    states = {item.run_id: item for item in batch.runs}
    public_records: list[tuple[Any, CascadeRealAgentRunRecord, list[Any]]] = []
    run_hashes: dict[str, str] = {}
    for spec in config.runs:
        state = states[spec.run_id]
        run_path = pilot_root / spec.run_id / "run_record.json"
        _validate_hash(run_path, state.run_record_sha256 or "", "run_record_hash_mismatch")
        try:
            raw = json.loads(run_path.read_text(encoding="utf-8"))
            _assert_public_safe(raw)
            record = CascadeRealAgentRunRecord.model_validate(raw)
        except Exception as exc:
            raise HGEvaluatorAnalysisError("run_record_invalid") from exc
        _require(
            record.run_id == spec.run_id
            and record.scenario_id == spec.scenario_id
            and record.seed == spec.seed
            and record.condition is spec.condition,
            "run_coordinate_mismatch",
        )
        _require(
            record.status == "completed"
            and record.error_code is None
            and record.exposure_ledger is not None
            and record.application_ledger is not None
            and record.outcome_ledger is not None
            and record.replay is not None,
            "run_record_incomplete",
        )
        _require(record.exposure_ledger_sha256 == _stable_sha(record.exposure_ledger.model_dump(mode="json")), "exposure_hash_mismatch")
        _require(record.application_ledger_sha256 == _stable_sha(record.application_ledger.model_dump(mode="json")), "application_hash_mismatch")
        _require(record.outcome_ledger_sha256 == record.outcome_ledger.sha256(), "outcome_hash_mismatch")
        replay = HGOutcomeReplayValidator.validate(
            scenarios[spec.scenario_id].graph,
            record.exposure_ledger,
            record.outcome_ledger,
            record.application_ledger,
            record.round_contexts,
        )
        _require(
            replay.status is CascadeOutcomeReplayStatus.PASSED
            and replay.cascade_replay.status.value == "passed"
            and replay.application_replay.status.value == "passed",
            "replay_not_passed",
        )
        entries = RequestLedger(pilot_root / spec.run_id / "request_ledger.jsonl").entries()
        started = [item for item in entries if item.status is RequestLedgerStatus.STARTED]
        completed = [item for item in entries if item.status is RequestLedgerStatus.COMPLETED]
        failed = [item for item in entries if item.status is RequestLedgerStatus.FAILED]
        _require(
            len(started) == len(completed) == 18
            and not failed
            and len({item.fingerprint for item in started}) == 18
            and {item.fingerprint for item in started} == {item.fingerprint for item in completed},
            "request_ledger_incomplete",
        )
        summary = RequestLedger(pilot_root / spec.run_id / "request_ledger.jsonl").summary(
            request_cap=spec.expected_provider_requests,
            completion_reservation_cap=spec.completion_reservation,
        )
        _require(
            summary.unique_logical_request_count == state.logical_request_count == 18
            and summary.transport_attempt_count == state.transport_attempt_count == 18
            and summary.actual_prompt_token_count == state.actual_prompt_token_count
            and summary.actual_completion_token_count == state.actual_completion_token_count
            and summary.actual_total_token_count == state.actual_total_token_count,
            "request_ledger_state_mismatch",
        )
        public_records.append((spec, record, entries))
        run_hashes[spec.run_id] = state.run_record_sha256 or ""

    _require(_records_hash([item[1] for item in public_records]) == receipt.ledger_hash, "pilot_record_aggregate_hash_mismatch")

    # Evaluator-private truth is loaded only after every public record, ledger, hash,
    # matched group, and replay layer has passed validation.
    truth_by_scenario: dict[str, Any] = {}
    truth_bindings = {item.scenario_id: item for item in plan.truth_fixtures}
    for material in config.scenario_materials:
        binding = truth_bindings.get(material.scenario_id)
        _require(binding is not None, "evaluator_fixture_binding_mismatch")
        _require(binding.sha256 == material.truth_sha256, "evaluator_fixture_hash_mismatch")
        truth_path = _rooted(binding.path)
        _require(truth_path == _relative_to_config(paths["config"], material.truth_path), "evaluator_fixture_path_mismatch")
        _validate_hash(truth_path, binding.sha256, "evaluator_fixture_hash_mismatch")
        try:
            truth = EvaluatorTruthLoader.load(truth_path)
            validate_fixture_pair(scenarios[material.scenario_id].graph, truth)
        except Exception as exc:
            raise HGEvaluatorAnalysisError("evaluator_fixture_binding_mismatch") from exc
        truth_by_scenario[material.scenario_id] = truth

    run_metrics: list[HFRunMetric] = []
    run_operations: list[HFRunOperations] = []
    for spec, record, entries in public_records:
        metrics, operations = _evaluate_run(
            spec,
            record,
            scenarios[spec.scenario_id].graph,
            truth_by_scenario[spec.scenario_id],
            entries,
        )
        run_metrics.extend(metrics)
        run_operations.append(operations)

    _require(len(run_metrics) == 48 * 9, "analysis_coordinate_incomplete")
    _require(sum(item.logical_request_count for item in run_operations) == 864, "logical_request_audit_mismatch")
    _require(sum(item.transport_attempt_count for item in run_operations) == 864, "transport_attempt_audit_mismatch")
    _require(sum(item.connection_failure_count for item in run_operations) == 0, "unexpected_connection_failure_terminal")
    _require(sum(item.connection_recovery_count for item in run_operations) == 0, "unexpected_connection_recovery")
    _require(sum(item.applied_count for item in run_operations) == receipt.directive_applied_count, "directive_count_mismatch")
    _require(sum(item.prompt_tokens or 0 for item in run_operations) == receipt.actual_prompt_token_count, "prompt_token_mismatch")
    _require(sum(item.completion_tokens or 0 for item in run_operations) == receipt.actual_completion_token_count, "completion_token_mismatch")
    _require(sum(item.total_tokens or 0 for item in run_operations) == receipt.actual_total_token_count, "total_token_mismatch")

    condition_summaries = _condition_summaries(run_metrics)
    scenario_condition_summaries = _scenario_condition_summaries(run_metrics, plan.scenario_ids)
    matched_differences = _matched_differences(run_metrics, plan.scenario_ids, plan.seeds)
    operations = _condition_operations(run_operations)
    separation = _separation_summaries(condition_summaries)
    warnings = {
        "three_seed_pilot_descriptive_only",
        "no_causal_conclusion",
        "thresholds_not_pre_registered",
        "manual_go_no_go_decision_required",
        "h_d21_non_identifiable_pilot_excluded",
        "h_f_legacy_analysis_excluded",
        "technical_replay_amendment_applied",
        "condition_mean_differences_are_not_significance_tests",
    }
    for metric in run_metrics:
        warnings.update(metric.warnings)

    safe_input_hashes = tuple(
        HFSafeInputHash(kind=kind, sha256=expected_hashes[kind])
        for kind in (
            "config", "protocol", "policy", "amendment", "amendment_receipt",
            "approval", "compatibility_receipt", "technical_replay_amendment",
            "old_validator", "new_validator", "pilot_receipt", "batch_record", "preregistration",
        )
    ) + tuple(
        HFSafeInputHash(kind=f"evaluator_fixture:{item.scenario_id}", sha256=item.sha256)
        for item in plan.truth_fixtures
    )
    manifest = HGAnalysisManifest(
        analysis_id=plan.analysis_id,
        analysis_version=HG_ANALYSIS_VERSION,
        study_id=plan.study_id,
        input_hashes=safe_input_hashes,
        run_record_sha256=run_hashes,
        final_batch_record_sha256=plan.batch_record_sha256,
        pre_resume_batch_record_sha256=technical.batch_record_sha256,
        technical_replay_amendment_sha256=plan.technical_replay_amendment_sha256,
        outcome_replay_contract_version=plan.outcome_replay_contract_version,
        scenario_count=4,
        seed_count=3,
        condition_count=4,
        run_count=48,
        matched_group_count=12,
        logical_request_count=864,
        transport_attempt_count=864,
        primary_metrics=EXPECTED_METRICS,
        h_d21_pilot_excluded=True,
        h_f_analysis_excluded=True,
        wvs_and_calibration_excluded=True,
        development_only=True,
        pilot_only=True,
        not_paper_result=True,
        no_causal_conclusion=True,
        private_truth_exposed=False,
        network_used_for_analysis=False,
    )
    report = HGAnalysisReport(
        analysis_id=plan.analysis_id,
        analysis_version=HG_ANALYSIS_VERSION,
        data_integrity_status="passed",
        effect_evidence_status="descriptive_pilot_only",
        decision_status="thresholds_not_pre_registered",
        decision_reason_codes=("thresholds_not_pre_registered", "manual_review_required"),
        run_metrics=tuple(run_metrics),
        scenario_condition_summaries=scenario_condition_summaries,
        condition_summaries=condition_summaries,
        matched_differences=matched_differences,
        separation_summaries=separation,
        run_operations=tuple(run_operations),
        condition_operations=operations,
        warnings=tuple(sorted(warnings)),
        development_only=True,
        pilot_only=True,
        not_paper_result=True,
        no_causal_conclusion=True,
        private_truth_exposed=False,
    )
    serialized = json.dumps(
        {"manifest": manifest.model_dump(mode="json"), "report": report.model_dump(mode="json")},
        ensure_ascii=True,
        sort_keys=True,
    ).lower()
    for forbidden in (
        "ground_truth_label", "source_independence_label", "system_prompt", "user_prompt",
        "provider_metadata", "api_key", "authorization", "model_response", "public_statement",
    ):
        _require(forbidden not in serialized, "analysis_output_sensitive_field_detected")
    return manifest, report


def _write_json(path: Path, value: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def run_hg_analysis(
    config_path: str | Path,
    *,
    write_outputs: bool,
    output_dir: str | Path | None = None,
) -> HGAnalysisResult:
    try:
        plan = load_hg_analysis_plan(config_path)
        manifest, report = analyze_hg_pilot(plan)
        destination = _rooted(output_dir or plan.output_dir)
        if write_outputs:
            if destination.exists():
                raise HGEvaluatorAnalysisError("analysis_output_exists")
            destination.mkdir(parents=True)
            _write_json(destination / "analysis_manifest.json", manifest)
            _write_json(destination / "analysis_report.json", report)
        return HGAnalysisResult(
            status="completed" if write_outputs else "validated",
            analysis_id=plan.analysis_id,
            run_count=48,
            matched_group_count=12,
            metric_count=len(report.run_metrics),
            decision_status=report.decision_status,
            output_dir=str(destination) if write_outputs else None,
        )
    except Exception as exc:
        code = exc.code if isinstance(exc, HGEvaluatorAnalysisError) else "hg_pilot_analysis_failed"
        return HGAnalysisResult(
            status="blocked",
            analysis_id="provenance-cascade-hg-pilot-analysis",
            run_count=0,
            matched_group_count=0,
            metric_count=0,
            blocking_reasons=(code,),
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline evaluator-only H-G Pilot analysis")
    parser.add_argument(
        "--config",
        default="configs/provenance_cascade/pilot/provenance_cascade_pilot_hg_analysis.v1.toml",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    result = run_hg_analysis(args.config, write_outputs=not args.validate_only)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))
    return 0 if result.status != "blocked" else 1


__all__ = [
    "HGAnalysisManifest", "HGAnalysisReport", "HGAnalysisResult",
    "HGEvaluatorAnalysisError", "HGMetricSeparationSummary",
    "HGScenarioConditionMetricSummary", "analyze_hg_pilot", "main", "run_hg_analysis",
]


if __name__ == "__main__":
    raise SystemExit(main())
