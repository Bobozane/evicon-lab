"""Read-only, evaluator-only analysis for the completed H-G.1.1 Pilot."""
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

from .cascade_hg1_replay import HG1_OUTCOME_REPLAY_VERSION, HG1OutcomeReplayValidator
from .cascade_outcome_replay import CascadeOutcomeReplayStatus
from .cascade_protocol import CascadeScenarioLoader
from .cascade_real_agent_runner import CascadeRealAgentRunRecord
from .provenance_cascade import EvaluatorTruthLoader, validate_fixture_pair
from .provenance_cascade_hg1 import HG1PilotConfig
from .provenance_cascade_hg11 import HG11Approval, HG11PilotConfig, HG11RunSpec, sha256_file
from .provenance_cascade_hg11_compatibility_receipt import validate_receipt as validate_compatibility_receipt
from .provenance_cascade_hg11_pilot import HG11PilotBatchRecord, HG11PilotReceipt, HG11PilotRunStatus, _records_hash
from .provenance_cascade_pilot_analysis import (
    HFConditionMetricSummary, HFConditionOperations, HFMatchedDifference, HFMetricStatus,
    HFRunMetric, HFRunOperations, HFSafeInputHash, _assert_public_safe,
    _condition_operations, _condition_summaries, _evaluate_run, _matched_differences, _stable_sha,
)
from .provenance_cascade_pilot_analysis_plan import EXPECTED_METRICS
from .provenance_cascade_preregistration import ProvenanceCascadePreregistrationLoader
from .request_ledger import RequestLedger, RequestLedgerStatus
from .provenance_cascade_hg11_analysis_plan import HG11_ANALYSIS_VERSION, HG11AnalysisPlan, load_hg11_analysis_plan

_ROOT = Path(__file__).resolve().parents[2]


class HG11EvaluatorAnalysisError(ValueError):
    """Stable, content-free evaluator analysis failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HG11ScenarioConditionMetricSummary(BaseModel):
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


class HG11MetricSeparationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    metric_name: str
    applicable_condition_count: int = Field(ge=0)
    condition_means: dict[str, float | None]
    range_across_condition_means: float | None = None
    condition_mean_difference_observed: bool
    interpretation: Literal["descriptive_only_no_threshold"] = "descriptive_only_no_threshold"


class HG11AnalysisManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    analysis_id: str
    analysis_version: Literal["provenance_cascade_hg11_pilot_analysis.v1"]
    study_id: str
    input_hashes: tuple[HFSafeInputHash, ...]
    run_record_sha256: dict[str, str]
    scenario_count: Literal[4]
    seed_count: Literal[3]
    condition_count: Literal[4]
    run_count: Literal[48]
    matched_group_count: Literal[12]
    logical_request_count: Literal[864]
    transport_attempt_count: int = Field(ge=864)
    primary_metrics: tuple[str, ...]
    h_g1_and_hd21_excluded: Literal[True]
    wvs_and_calibration_excluded: Literal[True]
    development_only: Literal[True]
    pilot_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    private_truth_exposed: Literal[False]
    network_used_for_analysis: Literal[False]


class HG11AnalysisReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    analysis_id: str
    analysis_version: Literal["provenance_cascade_hg11_pilot_analysis.v1"]
    data_integrity_status: Literal["passed"]
    effect_evidence_status: Literal["descriptive_pilot_only"]
    decision_status: Literal["thresholds_not_pre_registered"]
    decision_reason_codes: tuple[str, ...]
    run_metrics: tuple[HFRunMetric, ...]
    scenario_condition_summaries: tuple[HG11ScenarioConditionMetricSummary, ...]
    condition_summaries: tuple[HFConditionMetricSummary, ...]
    matched_differences: tuple[HFMatchedDifference, ...]
    separation_summaries: tuple[HG11MetricSeparationSummary, ...]
    run_operations: tuple[HFRunOperations, ...]
    condition_operations: tuple[HFConditionOperations, ...]
    warnings: tuple[str, ...]
    development_only: Literal[True]
    pilot_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    private_truth_exposed: Literal[False]


class HG11AnalysisResult(BaseModel):
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


def _require(value: bool, code: str) -> None:
    if not value:
        raise HG11EvaluatorAnalysisError(code)


def _validate_hash(path: Path, expected: str, code: str) -> None:
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise HG11EvaluatorAnalysisError(code) from exc
    if actual != expected:
        raise HG11EvaluatorAnalysisError(code)


def _load_json_model(path: Path, model: type[BaseModel], code: str) -> BaseModel:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError, ValueError) as exc:
        raise HG11EvaluatorAnalysisError(code) from exc


def _public_scenarios(parent_path: Path, parent: HG1PilotConfig) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    for material in parent.scenario_materials:
        scenario_path = _rooted(material.scenario_path)
        graph_path = _rooted(material.graph_path)
        _validate_hash(scenario_path, material.scenario_sha256, "scenario_hash_mismatch")
        _validate_hash(graph_path, material.graph_sha256, "graph_hash_mismatch")
        try:
            raw_graph = json.loads(graph_path.read_text(encoding="utf-8"))
            _assert_public_safe(raw_graph)
            scenario = CascadeScenarioLoader.load(scenario_path)
        except Exception as exc:
            raise HG11EvaluatorAnalysisError("public_scenario_invalid") from exc
        _require(scenario.scenario_id == material.scenario_id, "scenario_coordinate_mismatch")
        scenarios[scenario.scenario_id] = scenario
    _require(len(scenarios) == 4, "scenario_set_incomplete")
    return scenarios


def _specs(config: HG11PilotConfig, parent: HG1PilotConfig) -> tuple[HG11RunSpec, ...]:
    specs = tuple(
        HG11RunSpec(
            run_id=f"hg11-{material.scenario_id}-{seed}-{condition.value}",
            matched_group_id=f"hg11-{material.scenario_id}-{seed}",
            scenario_id=material.scenario_id,
            condition=condition,
            seed=seed,
        )
        for material in parent.scenario_materials
        for seed in config.seeds
        for condition in config.conditions
    )
    _require(len(specs) == 48 and len({item.run_id for item in specs}) == 48, "run_set_incomplete")
    return specs


def _validate_coordinates(specs: tuple[HG11RunSpec, ...], batch: HG11PilotBatchRecord) -> None:
    states = {item.run_id: item for item in batch.runs}
    _require(len(states) == 48 and set(states) == {item.run_id for item in specs}, "run_set_incomplete")
    groups: dict[str, set[str]] = defaultdict(set)
    for spec in specs:
        state = states[spec.run_id]
        groups[spec.matched_group_id].add(spec.condition.value)
        _require(
            state.scenario_id == spec.scenario_id and state.seed == spec.seed and state.condition == spec.condition.value,
            "run_coordinate_mismatch",
        )
        _require(state.status is HG11PilotRunStatus.COMPLETED and state.error_code is None, "run_not_completed")
        _require(state.logical_request_count == 18 and state.transport_attempt_count >= 18, "run_request_audit_mismatch")
        _require(
            state.cascade_replay_status == "passed" and state.application_replay_status == "passed" and state.outcome_replay_status == "passed",
            "replay_not_passed",
        )
    _require(len(groups) == 12 and all(items == {item.value for item in config_conditions()} for items in groups.values()), "matched_group_incomplete")


def config_conditions():
    from .provenance_cascade_pilot_analysis_plan import EXPECTED_CONDITIONS
    return EXPECTED_CONDITIONS


def _scenario_condition_summaries(metrics: list[HFRunMetric], scenario_ids: tuple[str, ...], conditions: tuple[str, ...]) -> tuple[HG11ScenarioConditionMetricSummary, ...]:
    output: list[HG11ScenarioConditionMetricSummary] = []
    for scenario_id in scenario_ids:
        for condition in conditions:
            for metric_name in EXPECTED_METRICS:
                rows = [item for item in metrics if item.scenario_id == scenario_id and item.condition.value == condition and item.metric_name == metric_name]
                values = [item.value for item in rows if item.status is HFMetricStatus.VALUE and item.value is not None]
                numerators = [item.numerator for item in rows if item.status is HFMetricStatus.VALUE and item.numerator is not None]
                denominators = [item.denominator for item in rows if item.status is HFMetricStatus.VALUE]
                numerator_sum = float(sum(numerators)) if numerators and len(numerators) == len(values) else None
                denominator_sum = sum(denominators)
                output.append(HG11ScenarioConditionMetricSummary(
                    scenario_id=scenario_id, condition=condition, metric_name=metric_name,
                    run_count=len(rows), applicable_count=len(values), not_applicable_count=len(rows) - len(values),
                    denominator_sum=denominator_sum, numerator_sum=numerator_sum,
                    pooled_value=(numerator_sum / denominator_sum) if numerator_sum is not None and denominator_sum else None,
                    mean=float(statistics.mean(values)) if values else None,
                    median=float(statistics.median(values)) if values else None,
                    standard_deviation=float(statistics.pstdev(values)) if values else None,
                ))
    return tuple(output)


def _separation_summaries(summaries: tuple[HFConditionMetricSummary, ...]) -> tuple[HG11MetricSeparationSummary, ...]:
    output: list[HG11MetricSeparationSummary] = []
    for metric_name in EXPECTED_METRICS:
        means = {item.condition.value: item.mean for item in summaries if item.metric_name == metric_name}
        values = [value for value in means.values() if value is not None]
        observed_range = float(max(values) - min(values)) if values else None
        output.append(HG11MetricSeparationSummary(
            metric_name=metric_name, applicable_condition_count=len(values), condition_means=means,
            range_across_condition_means=observed_range,
            condition_mean_difference_observed=bool(observed_range is not None and observed_range > 0.0),
        ))
    return tuple(output)


def analyze_hg11_pilot(plan: HG11AnalysisPlan) -> tuple[HG11AnalysisManifest, HG11AnalysisReport]:
    paths = {kind: _rooted(value) for kind, value in {
        "config": plan.config_path, "protocol": plan.protocol_path, "controller": plan.controller_path,
        "replay": plan.replay_path, "amendment": plan.amendment_path, "approval": plan.approval_path,
        "compatibility_receipt": plan.compatibility_receipt_path, "amendment_receipt": plan.amendment_receipt_path,
        "preregistration": plan.preregistration_path, "pilot_receipt": plan.pilot_receipt_path,
        "batch_record": plan.batch_record_path,
    }.items()}
    expected = {
        "config": plan.config_sha256, "protocol": plan.protocol_sha256, "controller": plan.controller_sha256,
        "replay": plan.replay_sha256, "amendment": plan.amendment_sha256, "approval": plan.approval_sha256,
        "compatibility_receipt": plan.compatibility_receipt_sha256, "amendment_receipt": plan.amendment_receipt_sha256,
        "preregistration": plan.preregistration_sha256, "pilot_receipt": plan.pilot_receipt_sha256,
        "batch_record": plan.batch_record_sha256,
    }
    for kind, path in paths.items():
        _validate_hash(path, expected[kind], f"{kind}_hash_mismatch")

    try:
        config = HG11PilotConfig.model_validate(tomllib.loads(paths["config"].read_text(encoding="utf-8")))
        parent_path = _rooted(config.parent_config_path)
        _validate_hash(parent_path, config.parent_config_sha256, "parent_config_hash_mismatch")
        parent = HG1PilotConfig.model_validate(tomllib.loads(parent_path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise HG11EvaluatorAnalysisError("hg11_config_invalid") from exc
    _require(config.study_id == "evicon-provenance-cascade-pilot-hg11", "study_id_mismatch")
    _require(config.request_cap == plan.logical_request_count and config.completion_reservation_cap == plan.completion_reservation_cap, "budget_binding_mismatch")
    _require(config.output_root == plan.pilot_root and config.seeds == plan.seeds, "pilot_plan_mismatch")
    _require(tuple(item.value for item in config.conditions) == plan.conditions and parent.agent_ids == config.agent_ids, "matched_plan_mismatch")
    _require(tuple(item.scenario_id for item in parent.scenario_materials) == plan.scenario_ids, "scenario_plan_mismatch")
    _require(sha256_file(_rooted(config.controller_path)) == plan.controller_sha256, "controller_hash_mismatch")
    _require(sha256_file(_rooted(config.replay_path)) == plan.replay_sha256, "replay_hash_mismatch")

    try:
        approval = HG11Approval.model_validate(tomllib.loads(paths["approval"].read_text(encoding="utf-8")))
    except Exception as exc:
        raise HG11EvaluatorAnalysisError("approval_invalid") from exc
    _require(approval.ready and approval.acceptance_status == "accepted", "approval_not_accepted")
    _require(
        approval.config_sha256 == plan.config_sha256 and approval.protocol_sha256 == plan.protocol_sha256
        and approval.controller_sha256 == plan.controller_sha256 and approval.replay_sha256 == plan.replay_sha256
        and approval.amendment_sha256 == plan.amendment_sha256 and approval.compatibility_receipt_sha256 == plan.compatibility_receipt_sha256,
        "approval_binding_mismatch",
    )
    try:
        compatibility = validate_compatibility_receipt(
            paths["compatibility_receipt"], expected_hash=plan.compatibility_receipt_sha256,
            config_path=paths["config"], protocol_path=paths["protocol"], amendment_sha256=plan.amendment_sha256,
        )
    except Exception as exc:
        raise HG11EvaluatorAnalysisError("compatibility_receipt_invalid") from exc
    _require(compatibility.status == "completed" and compatibility.parser_valid, "compatibility_receipt_incomplete")

    try:
        amendment_receipt = json.loads(paths["amendment_receipt"].read_text(encoding="utf-8"))
    except Exception as exc:
        raise HG11EvaluatorAnalysisError("amendment_receipt_invalid") from exc
    _require(
        amendment_receipt.get("config_sha256") == plan.config_sha256
        and amendment_receipt.get("protocol_sha256") == plan.protocol_sha256
        and amendment_receipt.get("amendment_sha256") == plan.amendment_sha256
        and amendment_receipt.get("fake_provider_replay_passed_count") == 48,
        "amendment_receipt_binding_mismatch",
    )

    receipt = _load_json_model(paths["pilot_receipt"], HG11PilotReceipt, "pilot_receipt_invalid")
    batch = _load_json_model(paths["batch_record"], HG11PilotBatchRecord, "batch_record_invalid")
    assert isinstance(receipt, HG11PilotReceipt) and isinstance(batch, HG11PilotBatchRecord)
    _require(
        receipt.status == "completed" and receipt.run_count == receipt.completed_run_count == 48
        and receipt.failed_run_count == 0 and receipt.matched_group_count == 12 and receipt.replay_passed_count == 48,
        "pilot_receipt_incomplete",
    )
    _require(receipt.logical_request_count == receipt.provider_call_count == 864 and receipt.transport_attempt_count >= 864, "pilot_request_audit_mismatch")
    _require(receipt.request_cap == 864 and receipt.completion_reservation_cap == plan.completion_reservation_cap, "pilot_budget_mismatch")
    _require(
        receipt.config_sha256 == plan.config_sha256 and receipt.protocol_sha256 == plan.protocol_sha256
        and receipt.template_sha256 == plan.protocol_sha256 and receipt.controller_sha256 == plan.controller_sha256
        and receipt.replay_sha256 == plan.replay_sha256 and receipt.amendment_sha256 == plan.amendment_sha256
        and receipt.approval_sha256 == plan.approval_sha256 and receipt.compatibility_receipt_sha256 == plan.compatibility_receipt_sha256,
        "pilot_receipt_binding_mismatch",
    )
    _require(receipt.outcome_replay_contract_version == HG1_OUTCOME_REPLAY_VERSION and batch.outcome_replay_contract_version == HG1_OUTCOME_REPLAY_VERSION, "outcome_replay_contract_mismatch")
    _require(not receipt.legacy_assets_touched and not receipt.private_truth_exposed and not receipt.parser_recovery_enabled, "pilot_safety_binding_mismatch")
    _require(batch.status is HG11PilotRunStatus.COMPLETED and batch.failure_code is None and len(batch.runs) == 48, "batch_record_incomplete")
    _require(batch.config_sha256 == plan.config_sha256 and batch.protocol_sha256 == plan.protocol_sha256 and batch.template_sha256 == plan.protocol_sha256, "batch_record_binding_mismatch")

    specs = _specs(config, parent)
    _validate_coordinates(specs, batch)
    preregistration = ProvenanceCascadePreregistrationLoader.load(paths["preregistration"])
    _require(tuple(preregistration.primary_metrics) == EXPECTED_METRICS and preregistration.success_criterion.no_unregistered_thresholds, "preregistered_metric_suite_mismatch")
    scenarios = _public_scenarios(parent_path, parent)
    pilot_root = _rooted(plan.pilot_root)
    states = {item.run_id: item for item in batch.runs}
    public_records: list[tuple[HG11RunSpec, CascadeRealAgentRunRecord, list[Any]]] = []
    run_hashes: dict[str, str] = {}
    for spec in specs:
        state = states[spec.run_id]
        run_path = pilot_root / spec.run_id / "run_record.json"
        _validate_hash(run_path, state.run_record_sha256 or "", "run_record_hash_mismatch")
        try:
            raw = json.loads(run_path.read_text(encoding="utf-8"))
            _assert_public_safe(raw)
            record = CascadeRealAgentRunRecord.model_validate(raw)
        except Exception as exc:
            raise HG11EvaluatorAnalysisError("run_record_invalid") from exc
        _require(record.run_id == spec.run_id and record.scenario_id == spec.scenario_id and record.seed == spec.seed and record.condition is spec.condition, "run_coordinate_mismatch")
        _require(record.status == "completed" and record.error_code is None and record.exposure_ledger is not None and record.application_ledger is not None and record.outcome_ledger is not None and record.replay is not None, "run_record_incomplete")
        _require(record.exposure_ledger_sha256 == _stable_sha(record.exposure_ledger.model_dump(mode="json")), "exposure_hash_mismatch")
        _require(record.application_ledger_sha256 == _stable_sha(record.application_ledger.model_dump(mode="json")), "application_hash_mismatch")
        _require(record.outcome_ledger_sha256 == record.outcome_ledger.sha256(), "outcome_hash_mismatch")
        replay = HG1OutcomeReplayValidator.validate(scenarios[spec.scenario_id].graph, record.exposure_ledger, record.outcome_ledger, record.application_ledger, record.round_contexts)
        _require(replay.status is CascadeOutcomeReplayStatus.PASSED and replay.cascade_replay.status.value == "passed" and replay.application_replay.status.value == "passed", "replay_not_passed")
        entries = RequestLedger(pilot_root / spec.run_id / "request_ledger.jsonl").entries()
        started = [item for item in entries if item.status is RequestLedgerStatus.STARTED]
        completed = [item for item in entries if item.status is RequestLedgerStatus.COMPLETED]
        _require(len({item.fingerprint for item in started}) == len({item.fingerprint for item in completed}) == 18, "request_ledger_incomplete")
        _require({item.fingerprint for item in started} == {item.fingerprint for item in completed}, "request_ledger_incomplete")
        summary = RequestLedger(pilot_root / spec.run_id / "request_ledger.jsonl").summary(request_cap=18, completion_reservation_cap=spec.completion_reservation)
        _require(summary.unique_logical_request_count == state.logical_request_count == 18 and summary.transport_attempt_count == state.transport_attempt_count, "request_ledger_state_mismatch")
        _require(summary.actual_prompt_token_count == state.actual_prompt_token_count and summary.actual_completion_token_count == state.actual_completion_token_count and summary.actual_total_token_count == state.actual_total_token_count, "request_ledger_state_mismatch")
        public_records.append((spec, record, entries))
        run_hashes[spec.run_id] = state.run_record_sha256 or ""
    _require(_records_hash([record for _, record, _ in public_records]) == receipt.ledger_hash, "pilot_record_aggregate_hash_mismatch")

    # Private truth is intentionally loaded only after all public data integrity gates pass.
    truth_by_scenario: dict[str, Any] = {}
    truth_bindings = {item.scenario_id: item for item in plan.truth_fixtures}
    materials = {item.scenario_id: item for item in parent.scenario_materials}
    for scenario_id in plan.scenario_ids:
        binding = truth_bindings[scenario_id]
        material = materials[scenario_id]
        truth_path = _rooted(binding.path)
        _require(binding.sha256 == material.truth_sha256 and truth_path == _rooted(material.truth_path), "evaluator_fixture_binding_mismatch")
        _validate_hash(truth_path, binding.sha256, "evaluator_fixture_hash_mismatch")
        try:
            truth = EvaluatorTruthLoader.load(truth_path)
            validate_fixture_pair(scenarios[scenario_id].graph, truth)
        except Exception as exc:
            raise HG11EvaluatorAnalysisError("evaluator_fixture_binding_mismatch") from exc
        truth_by_scenario[scenario_id] = truth

    run_metrics: list[HFRunMetric] = []
    run_operations: list[HFRunOperations] = []
    for spec, record, entries in public_records:
        metrics, operations = _evaluate_run(spec, record, scenarios[spec.scenario_id].graph, truth_by_scenario[spec.scenario_id], entries)
        run_metrics.extend(metrics)
        run_operations.append(operations)
    _require(len(run_metrics) == 48 * 9, "analysis_coordinate_incomplete")
    _require(sum(item.logical_request_count for item in run_operations) == 864, "logical_request_audit_mismatch")
    _require(sum(item.transport_attempt_count for item in run_operations) == receipt.transport_attempt_count, "transport_attempt_audit_mismatch")
    _require(sum(item.applied_count for item in run_operations) == receipt.directive_applied_count, "directive_count_mismatch")
    _require(sum(item.prompt_tokens or 0 for item in run_operations) == receipt.actual_prompt_token_count, "prompt_token_mismatch")
    _require(sum(item.completion_tokens or 0 for item in run_operations) == receipt.actual_completion_token_count, "completion_token_mismatch")
    _require(sum(item.total_tokens or 0 for item in run_operations) == receipt.actual_total_token_count, "total_token_mismatch")

    condition_summaries = _condition_summaries(run_metrics)
    scenario_condition_summaries = _scenario_condition_summaries(run_metrics, plan.scenario_ids, plan.conditions)
    matched_differences = _matched_differences(run_metrics, plan.scenario_ids, plan.seeds)
    operations = _condition_operations(run_operations)
    separation = _separation_summaries(condition_summaries)
    warnings = {
        "three_seed_pilot_descriptive_only", "no_causal_conclusion", "thresholds_not_pre_registered",
        "manual_go_no_go_decision_required", "h_g1_and_hd21_pilots_excluded", "wvs_and_calibration_excluded",
        "condition_mean_differences_are_not_significance_tests", "transport_attempts_include_limited_retries",
    }
    for metric in run_metrics:
        warnings.update(metric.warnings)
    input_hashes = tuple(HFSafeInputHash(kind=kind, sha256=expected[kind]) for kind in expected) + tuple(
        HFSafeInputHash(kind=f"evaluator_fixture:{item.scenario_id}", sha256=item.sha256) for item in plan.truth_fixtures
    )
    manifest = HG11AnalysisManifest(
        analysis_id=plan.analysis_id, analysis_version=HG11_ANALYSIS_VERSION, study_id=config.study_id,
        input_hashes=input_hashes, run_record_sha256=run_hashes, scenario_count=4, seed_count=3,
        condition_count=4, run_count=48, matched_group_count=12, logical_request_count=864,
        transport_attempt_count=receipt.transport_attempt_count, primary_metrics=EXPECTED_METRICS,
        h_g1_and_hd21_excluded=True, wvs_and_calibration_excluded=True, development_only=True,
        pilot_only=True, not_paper_result=True, no_causal_conclusion=True, private_truth_exposed=False,
        network_used_for_analysis=False,
    )
    report = HG11AnalysisReport(
        analysis_id=plan.analysis_id, analysis_version=HG11_ANALYSIS_VERSION, data_integrity_status="passed",
        effect_evidence_status="descriptive_pilot_only", decision_status="thresholds_not_pre_registered",
        decision_reason_codes=("thresholds_not_pre_registered", "manual_review_required"), run_metrics=tuple(run_metrics),
        scenario_condition_summaries=scenario_condition_summaries, condition_summaries=condition_summaries,
        matched_differences=matched_differences, separation_summaries=separation, run_operations=tuple(run_operations),
        condition_operations=operations, warnings=tuple(sorted(warnings)), development_only=True, pilot_only=True,
        not_paper_result=True, no_causal_conclusion=True, private_truth_exposed=False,
    )
    serialized = json.dumps({"manifest": manifest.model_dump(mode="json"), "report": report.model_dump(mode="json")}, ensure_ascii=True, sort_keys=True).lower()
    for forbidden in ("ground_truth_label", "source_independence_label", "system_prompt", "user_prompt", "provider_metadata", "api_key", "authorization", "model_response", "public_statement", "private_fixture"):
        _require(forbidden not in serialized, "analysis_output_sensitive_field_detected")
    return manifest, report


def _write_json(path: Path, value: BaseModel) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def run_hg11_analysis(config_path: str | Path, *, write_outputs: bool, output_dir: str | Path | None = None) -> HG11AnalysisResult:
    try:
        plan = load_hg11_analysis_plan(config_path)
        manifest, report = analyze_hg11_pilot(plan)
        destination = _rooted(output_dir or plan.output_dir)
        if write_outputs:
            if destination.exists():
                raise HG11EvaluatorAnalysisError("analysis_output_exists")
            destination.mkdir(parents=True)
            _write_json(destination / "analysis_manifest.json", manifest)
            _write_json(destination / "analysis_report.json", report)
        return HG11AnalysisResult(
            status="completed" if write_outputs else "validated", analysis_id=plan.analysis_id,
            run_count=48, matched_group_count=12, metric_count=len(report.run_metrics),
            decision_status=report.decision_status, output_dir=str(destination) if write_outputs else None,
        )
    except Exception as exc:
        code = exc.code if isinstance(exc, HG11EvaluatorAnalysisError) else "hg11_pilot_analysis_failed"
        return HG11AnalysisResult(status="blocked", analysis_id="provenance-cascade-hg11-pilot-analysis", run_count=0, matched_group_count=0, metric_count=0, blocking_reasons=(code,))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline evaluator-only H-G.1.1 Pilot analysis")
    parser.add_argument("--config", default="configs/provenance_cascade/pilot/provenance_cascade_pilot_hg11_analysis.v1.toml")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    result = run_hg11_analysis(args.config, write_outputs=not args.validate_only)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))
    return 0 if result.status != "blocked" else 1


__all__ = ["HG11AnalysisManifest", "HG11AnalysisReport", "HG11AnalysisResult", "HG11EvaluatorAnalysisError", "analyze_hg11_pilot", "run_hg11_analysis", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
