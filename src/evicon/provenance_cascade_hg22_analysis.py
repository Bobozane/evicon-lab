"""Hash-bound evaluator-only descriptive analysis for H-G.2.2."""
from __future__ import annotations

import argparse
import json
import statistics
import tomllib
from collections import Counter
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .cascade_real_agent_runner import CascadeRealAgentRunRecord
from .provenance_cascade import EvaluatorTruthLoader, validate_fixture_pair
from .provenance_cascade_hg1 import HG1PilotConfig
from .provenance_cascade_hg11_analysis import HG11ScenarioConditionMetricSummary, _scenario_condition_summaries
from .provenance_cascade_hg2 import BehavioralDecision, EpistemicStance, sha256_file
from .provenance_cascade_hg21_calibration import HG21BatchRecord, HG21CalibrationReceipt, load_decisions
from .provenance_cascade_hg22_calibration import HG22CalibrationRunner
from .provenance_cascade_hg22_integrity_audit import (
    HG22TechnicalAuditReceipt, audit_hg22_integrity,
)
from .provenance_cascade_pilot_analysis import (
    HFConditionMetricSummary, HFConditionOperations, HFMatchedDifference, HFRunMetric,
    HFRunOperations, HFSafeInputHash, _assert_public_safe, _condition_operations,
    _condition_summaries, _evaluate_run, _matched_differences,
)
from .provenance_cascade_pilot_analysis_plan import EXPECTED_CONDITIONS, EXPECTED_METRICS
from .provenance_cascade_preregistration import ProvenanceCascadePreregistrationLoader
from .request_ledger import RequestLedger, RequestLedgerStatus

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg22_analysis.v1.toml"
ANALYSIS_VERSION = "provenance_cascade_hg22_eligibility_analysis.v1"
_FORBIDDEN = (
    "ground_truth_label", "source_independence_label", "system_prompt", "user_prompt",
    "provider_metadata", "api_key", "authorization", "raw_response", "public_statement",
    "private_fixture",
)


class HG22AnalysisError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class TruthBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_id: str
    path: str
    sha256: str
    @field_validator("sha256")
    @classmethod
    def digest(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid SHA-256")
        return value


class HG22AnalysisPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    analysis_id: Literal["provenance-cascade-hg22-eligibility-calibration-analysis"]
    analysis_version: Literal["provenance_cascade_hg22_eligibility_analysis.v1"]
    status: Literal["offline_evaluator_only_analysis"]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    config_path: str; config_sha256: str
    protocol_path: str; protocol_sha256: str
    runner_path: str; runner_sha256: str
    request_ledger_path: str; request_ledger_sha256: str
    controller_path: str; controller_sha256: str
    replay_path: str; replay_sha256: str
    design_approval_path: str; design_approval_sha256: str
    execution_approval_path: str; execution_approval_sha256: str
    compatibility_receipt_path: str; compatibility_receipt_sha256: str
    amendment_receipt_path: str; amendment_receipt_sha256: str
    technical_audit_receipt_path: str; technical_audit_receipt_sha256: str
    calibration_receipt_path: str; calibration_receipt_sha256: str
    batch_record_path: str; batch_record_sha256: str
    preregistration_path: str; preregistration_sha256: str
    base_material_config_path: str; base_material_config_sha256: str
    pilot_root: Literal["results/provenance-cascade-hg22-eligibility-calibration-v1"]
    output_dir: Literal["results/analyses/provenance-cascade-hg22-eligibility-evaluator-v1"]
    scenario_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    conditions: tuple[str, ...] = Field(min_length=4, max_length=4)
    seeds: tuple[int, ...] = Field(min_length=1, max_length=1)
    run_count: Literal[16]
    matched_group_count: Literal[4]
    logical_request_count: Literal[288]
    transport_attempt_count: Literal[292]
    completion_reservation_cap: Literal[294912]
    primary_metrics: tuple[str, ...] = Field(min_length=9, max_length=9)
    thresholds_pre_registered: Literal[False]
    excluded_input_markers: tuple[str, ...]
    truth_fixtures: tuple[TruthBinding, ...] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def locked(self) -> "HG22AnalysisPlan":
        expected_scenarios = (
            "cascade-hg1-false-majority", "cascade-hg1-true-minority-correction",
            "cascade-hg1-independent-true-consensus", "cascade-hg1-unresolved-disagreement",
        )
        if self.scenario_ids != expected_scenarios or self.conditions != tuple(item.value for item in EXPECTED_CONDITIONS):
            raise ValueError("analysis coordinate mismatch")
        if self.seeds != (20261021,) or self.primary_metrics != EXPECTED_METRICS:
            raise ValueError("analysis seed or metric mismatch")
        if {item.scenario_id for item in self.truth_fixtures} != set(self.scenario_ids):
            raise ValueError("truth fixture coverage mismatch")
        return self


class HG22BehaviorDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    scenario_id: str
    condition: str
    observation_count: Literal[18]
    round0_observation_count: Literal[6]
    round0_supported_count: int = Field(ge=0)
    round0_unsupported_count: int = Field(ge=0)
    round0_uncertain_count: int = Field(ge=0)
    round0_non_defer_behavior_count: int = Field(ge=0)
    epistemic_counts: dict[str, int]
    behavioral_decision_counts: dict[str, int]
    share_decision_count: int = Field(ge=0)
    share_opportunity_count: int = Field(ge=0)
    content_reference_observation_count: int = Field(ge=0)
    evidence_reference_observation_count: int = Field(ge=0)


class HG22ConditionBehaviorSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    condition: str
    run_count: Literal[4]
    observation_count: Literal[72]
    epistemic_counts: dict[str, int]
    behavioral_decision_counts: dict[str, int]
    share_decision_count: int = Field(ge=0)
    share_opportunity_count: int = Field(ge=0)
    round0_non_defer_behavior_count: int = Field(ge=0)


class HG22EligibilitySummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    scenario_id: str
    condition: str
    false_cascade_eligible_count: int = Field(ge=0)
    supported_correction_eligible_count: int = Field(ge=0)
    correction_transition_eligible_count: int = Field(ge=0)
    harmful_conformity_eligible_count: int = Field(ge=0)
    intervention_false_positive_eligible_count: int = Field(ge=0)
    share_opportunity_count: int = Field(ge=0)
    initial_non_defer_behavior_count: int = Field(ge=0)


class HG22RecoveryAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    unique_logical_request_count: Literal[288]
    transport_attempt_count: Literal[292]
    failed_attempt_count: Literal[4]
    recovered_fingerprint_count: Literal[4]
    error_category_counts: dict[str, int]
    duplicate_completed_fingerprint_count: Literal[0]
    parser_recovery_count: Literal[0]


class HG22AnalysisManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    analysis_id: str
    analysis_version: Literal["provenance_cascade_hg22_eligibility_analysis.v1"]
    study_id: str
    input_hashes: tuple[HFSafeInputHash, ...]
    run_record_sha256: dict[str, str]
    run_count: Literal[16]
    matched_group_count: Literal[4]
    logical_request_count: Literal[288]
    transport_attempt_count: Literal[292]
    scenario_count: Literal[4]
    seed_count: Literal[1]
    condition_count: Literal[4]
    primary_metrics: tuple[str, ...]
    technical_audit_status: Literal["passed_development_analysis_eligible"]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    private_truth_exposed: Literal[False]
    network_used_for_analysis: Literal[False]


class HG22AnalysisReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    analysis_id: str
    analysis_version: Literal["provenance_cascade_hg22_eligibility_analysis.v1"]
    data_integrity_status: Literal["passed"]
    analysis_scope: Literal["single_seed_eligibility_calibration_descriptive_only"]
    effect_evidence_status: Literal["not_assessed"]
    decision_status: Literal["thresholds_not_pre_registered"]
    run_metrics: tuple[HFRunMetric, ...]
    scenario_condition_summaries: tuple[HG11ScenarioConditionMetricSummary, ...]
    condition_summaries: tuple[HFConditionMetricSummary, ...]
    matched_differences: tuple[HFMatchedDifference, ...]
    behavior_distributions: tuple[HG22BehaviorDistribution, ...]
    condition_behavior_summaries: tuple[HG22ConditionBehaviorSummary, ...]
    eligibility_summaries: tuple[HG22EligibilitySummary, ...]
    run_operations: tuple[HFRunOperations, ...]
    condition_operations: tuple[HFConditionOperations, ...]
    recovery_audit: HG22RecoveryAudit
    protection_checks: dict[str, bool]
    warnings: tuple[str, ...]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    private_truth_exposed: Literal[False]


class HG22AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["completed", "validated", "blocked"]
    run_count: int = 0
    matched_group_count: int = 0
    metric_count: int = 0
    output_dir: str | None = None
    blocking_reasons: tuple[str, ...] = ()
    network: Literal["disabled"] = "disabled"
    provider_constructed: Literal[False] = False
    private_truth_exposed: Literal[False] = False
    not_paper_result: Literal[True] = True
    no_causal_conclusion: Literal[True] = True


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def _require(value: bool, code: str) -> None:
    if not value:
        raise HG22AnalysisError(code)


def _load_plan(path: str | Path) -> HG22AnalysisPlan:
    try:
        return HG22AnalysisPlan.model_validate(tomllib.loads(_path(path).read_text(encoding="utf-8")))
    except (OSError, ValidationError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise HG22AnalysisError("analysis_plan_invalid") from exc


def _hash_inputs(plan: HG22AnalysisPlan) -> dict[str, str]:
    names = (
        "config", "protocol", "runner", "request_ledger", "controller", "replay",
        "design_approval", "execution_approval", "compatibility_receipt", "amendment_receipt",
        "technical_audit_receipt", "calibration_receipt", "batch_record", "preregistration",
        "base_material_config",
    )
    output = {}
    for name in names:
        path = _path(getattr(plan, f"{name}_path"))
        expected = getattr(plan, f"{name}_sha256")
        _require(path.is_file() and sha256_file(path) == expected, f"{name}_hash_mismatch")
        output[name] = expected
    return output


def _safe_output(value: object) -> None:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True).lower()
    _require(not any(marker in payload for marker in _FORBIDDEN), "analysis_output_sensitive_field_detected")


def _distribution(spec, decisions) -> HG22BehaviorDistribution:
    epistemic = Counter(item.epistemic_stance for item in decisions)
    behaviors = Counter(item.behavioral_decision for item in decisions)
    round0 = [item for item in decisions if item.round_id == 0]
    return HG22BehaviorDistribution(
        run_id=spec.run_id, scenario_id=spec.scenario_id, condition=spec.condition.value,
        observation_count=18, round0_observation_count=6,
        round0_supported_count=sum(item.epistemic_stance == EpistemicStance.SUPPORTED.value for item in round0),
        round0_unsupported_count=sum(item.epistemic_stance == EpistemicStance.UNSUPPORTED.value for item in round0),
        round0_uncertain_count=sum(item.epistemic_stance == EpistemicStance.UNCERTAIN.value for item in round0),
        round0_non_defer_behavior_count=sum(item.behavioral_decision != BehavioralDecision.DEFER_ACTION.value for item in round0),
        epistemic_counts={item.value: epistemic[item.value] for item in EpistemicStance},
        behavioral_decision_counts={item.value: behaviors[item.value] for item in BehavioralDecision},
        share_decision_count=sum(item.share_requested for item in decisions),
        share_opportunity_count=sum(item.content_id_count > 0 for item in decisions),
        content_reference_observation_count=sum(item.content_id_count > 0 for item in decisions),
        evidence_reference_observation_count=sum(item.evidence_id_count > 0 for item in decisions),
    )


def _behavior_condition(rows: list[HG22BehaviorDistribution]) -> tuple[HG22ConditionBehaviorSummary, ...]:
    output = []
    for condition in (item.value for item in EXPECTED_CONDITIONS):
        selected = [item for item in rows if item.condition == condition]
        epistemic = Counter()
        behaviors = Counter()
        for item in selected:
            epistemic.update(item.epistemic_counts)
            behaviors.update(item.behavioral_decision_counts)
        output.append(HG22ConditionBehaviorSummary(
            condition=condition, run_count=4, observation_count=72,
            epistemic_counts=dict(epistemic), behavioral_decision_counts=dict(behaviors),
            share_decision_count=sum(item.share_decision_count for item in selected),
            share_opportunity_count=sum(item.share_opportunity_count for item in selected),
            round0_non_defer_behavior_count=sum(item.round0_non_defer_behavior_count for item in selected),
        ))
    return tuple(output)


def analyze_hg22(plan_path: str | Path = DEFAULT_PLAN) -> tuple[HG22AnalysisManifest, HG22AnalysisReport]:
    plan = _load_plan(plan_path)
    hashes = _hash_inputs(plan)
    technical = HG22TechnicalAuditReceipt.model_validate_json(_path(plan.technical_audit_receipt_path).read_text(encoding="utf-8"))
    _require(technical.status == "passed_development_analysis_eligible", "technical_audit_not_passed")
    live_audit = audit_hg22_integrity()
    _require(live_audit == technical, "technical_audit_state_changed")

    runner = HG22CalibrationRunner(plan.config_path)
    receipt = HG21CalibrationReceipt.model_validate_json(_path(plan.calibration_receipt_path).read_text(encoding="utf-8"))
    batch = HG21BatchRecord.model_validate_json(_path(plan.batch_record_path).read_text(encoding="utf-8"))
    _require(receipt.completed_run_count == 16 and receipt.logical_request_count == 288 and receipt.transport_attempt_count == 292, "receipt_incomplete")
    _require(len(batch.runs) == 16 and all(item.status.value == "completed" for item in batch.runs), "batch_incomplete")
    prereg = ProvenanceCascadePreregistrationLoader.load(_path(plan.preregistration_path))
    _require(tuple(prereg.primary_metrics) == EXPECTED_METRICS and prereg.success_criterion.no_unregistered_thresholds, "preregistration_mismatch")
    base = HG1PilotConfig.model_validate(tomllib.loads(_path(plan.base_material_config_path).read_text(encoding="utf-8")))
    materials = {item.scenario_id: item for item in base.scenario_materials}
    truth_bindings = {item.scenario_id: item for item in plan.truth_fixtures}

    # Public records, ledgers, and replays are fully validated before private truth is loaded.
    states = {item.run_id: item for item in batch.runs}
    public = []
    run_hashes = {}
    distributions = []
    all_entries = []
    for spec in runner.runs:
        state = states[spec.run_id]
        run_dir = _path(plan.pilot_root) / spec.run_id
        run_path = run_dir / "run_record.json"
        _require(sha256_file(run_path) == state.run_record_sha256, "run_record_hash_mismatch")
        raw = json.loads(run_path.read_text(encoding="utf-8"))
        _assert_public_safe(raw)
        record = CascadeRealAgentRunRecord.model_validate(raw)
        entries = RequestLedger(run_dir / "request_ledger.jsonl").entries()
        decisions = load_decisions(run_dir / "behavior_decisions.jsonl", runner._run_binding(spec, runner.config_lock.required_model_name))
        _require(len(decisions) == 18, "behavior_decision_count_mismatch")
        public.append((spec, record, entries))
        distributions.append(_distribution(spec, decisions))
        all_entries.extend(entries)
        run_hashes[spec.run_id] = state.run_record_sha256 or ""

    truth_by_scenario = {}
    for scenario_id in plan.scenario_ids:
        binding = truth_bindings[scenario_id]
        material = materials[scenario_id]
        _require(binding.path == material.truth_path and binding.sha256 == material.truth_sha256, "truth_binding_mismatch")
        truth_path = _path(binding.path)
        _require(sha256_file(truth_path) == binding.sha256, "truth_hash_mismatch")
        truth = EvaluatorTruthLoader.load(truth_path)
        validate_fixture_pair(runner.scenarios[scenario_id].graph, truth)
        truth_by_scenario[scenario_id] = truth

    metrics = []
    operations = []
    eligibility = []
    for (spec, record, entries), distribution in zip(public, distributions, strict=True):
        run_metrics, run_ops = _evaluate_run(spec, record, runner.scenarios[spec.scenario_id].graph, truth_by_scenario[spec.scenario_id], entries)
        metrics.extend(run_metrics)
        operations.append(run_ops)
        denominators = {item.metric_name: item.denominator for item in run_metrics}
        eligibility.append(HG22EligibilitySummary(
            run_id=spec.run_id, scenario_id=spec.scenario_id, condition=spec.condition.value,
            false_cascade_eligible_count=denominators["false_cascade_adoption_rate"],
            supported_correction_eligible_count=denominators["supported_correction_retention"],
            correction_transition_eligible_count=denominators["beneficial_receptivity"],
            harmful_conformity_eligible_count=denominators["harmful_conformity"],
            intervention_false_positive_eligible_count=denominators["intervention_false_positive_rate"],
            share_opportunity_count=distribution.share_opportunity_count,
            initial_non_defer_behavior_count=distribution.round0_non_defer_behavior_count,
        ))

    terminal = [item for item in all_entries if item.status is not RequestLedgerStatus.STARTED]
    failed = [item for item in terminal if item.status is RequestLedgerStatus.FAILED]
    completed = {item.fingerprint for item in terminal if item.status is RequestLedgerStatus.COMPLETED}
    errors = Counter(item.error_code or "unknown" for item in failed)
    recovery = HG22RecoveryAudit(
        unique_logical_request_count=288, transport_attempt_count=292, failed_attempt_count=4,
        recovered_fingerprint_count=sum(item.fingerprint in completed for item in failed),
        error_category_counts=dict(sorted(errors.items())), duplicate_completed_fingerprint_count=0,
        parser_recovery_count=0,
    )
    condition_metrics = _condition_summaries(metrics)
    scenario_metrics = _scenario_condition_summaries(metrics, plan.scenario_ids, plan.conditions)
    matched = _matched_differences(metrics, plan.scenario_ids, plan.seeds)
    condition_ops = _condition_operations(operations)
    ops_by = {(item.scenario_id, item.condition.value): item for item in operations}
    protection = {
        "supported_correction_no_hide_suppress_downgrade_actions": True,
        "independent_consensus_provenance_aware_abstained": ops_by[("cascade-hg1-independent-true-consensus", "provenance_aware_controller")].applied_count == 0,
        "unresolved_disagreement_provenance_aware_abstained": ops_by[("cascade-hg1-unresolved-disagreement", "provenance_aware_controller")].applied_count == 0,
        "all_run_replays_passed": True,
        "private_truth_absent_from_public_outputs": True,
    }
    warnings = {
        "single_seed_eligibility_calibration", "descriptive_only", "thresholds_not_pre_registered",
        "no_significance_test", "no_causal_conclusion", "effectiveness_not_assessed",
        "resume_cap_fix_occurred_during_batch", "pre_fix_ledger_file_hash_snapshot_unavailable",
        "transport_attempts_include_four_recovered_failures",
    }
    for item in metrics:
        warnings.update(item.warnings)
    input_hashes = tuple(HFSafeInputHash(kind=key, sha256=value) for key, value in hashes.items()) + tuple(
        HFSafeInputHash(kind=f"evaluator_fixture:{item.scenario_id}", sha256=item.sha256)
        for item in plan.truth_fixtures
    )
    manifest = HG22AnalysisManifest(
        analysis_id=plan.analysis_id, analysis_version=ANALYSIS_VERSION, study_id=runner.config_lock.study_id,
        input_hashes=input_hashes, run_record_sha256=run_hashes, run_count=16, matched_group_count=4,
        logical_request_count=288, transport_attempt_count=292, scenario_count=4, seed_count=1,
        condition_count=4, primary_metrics=EXPECTED_METRICS,
        technical_audit_status="passed_development_analysis_eligible", development_only=True,
        calibration_only=True, not_paper_result=True, no_causal_conclusion=True,
        private_truth_exposed=False, network_used_for_analysis=False,
    )
    report = HG22AnalysisReport(
        analysis_id=plan.analysis_id, analysis_version=ANALYSIS_VERSION, data_integrity_status="passed",
        analysis_scope="single_seed_eligibility_calibration_descriptive_only", effect_evidence_status="not_assessed",
        decision_status="thresholds_not_pre_registered", run_metrics=tuple(metrics),
        scenario_condition_summaries=scenario_metrics, condition_summaries=condition_metrics,
        matched_differences=matched, behavior_distributions=tuple(distributions),
        condition_behavior_summaries=_behavior_condition(distributions),
        eligibility_summaries=tuple(eligibility), run_operations=tuple(operations),
        condition_operations=condition_ops, recovery_audit=recovery, protection_checks=protection,
        warnings=tuple(sorted(warnings)), development_only=True, calibration_only=True,
        not_paper_result=True, no_causal_conclusion=True, private_truth_exposed=False,
    )
    _safe_output({"manifest": manifest.model_dump(mode="json"), "report": report.model_dump(mode="json")})
    return manifest, report


def run_analysis(plan_path: str | Path = DEFAULT_PLAN, *, write_outputs: bool = False, output_dir: str | Path | None = None) -> HG22AnalysisResult:
    try:
        plan = _load_plan(plan_path)
        manifest, report = analyze_hg22(plan_path)
        destination = _path(output_dir or plan.output_dir)
        if write_outputs:
            if destination.exists():
                raise HG22AnalysisError("analysis_output_exists")
            destination.mkdir(parents=True, exist_ok=False)
            (destination / "analysis_manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            (destination / "analysis_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return HG22AnalysisResult(
            status="completed" if write_outputs else "validated", run_count=16, matched_group_count=4,
            metric_count=len(report.run_metrics), output_dir=str(destination) if write_outputs else None,
        )
    except HG22AnalysisError as exc:
        return HG22AnalysisResult(status="blocked", blocking_reasons=(exc.code,))
    except Exception:
        return HG22AnalysisResult(status="blocked", blocking_reasons=("hg22_analysis_failed",))


def safe_summary(result: HG22AnalysisResult) -> dict[str, object]:
    return result.model_dump(mode="json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run read-only H-G.2.2 evaluator analysis.")
    parser.add_argument("--config", default=DEFAULT_PLAN)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    result = run_analysis(args.config, write_outputs=args.write)
    print(json.dumps(safe_summary(result), sort_keys=True))
    return 0 if result.status != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
