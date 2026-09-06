"""Offline, seed-level analysis for the locked protocol-blind v2 study.

This module consumes only safe batch receipts, evaluation metrics, manifest
coordinates, and run-record metadata. It intentionally never loads prompts,
raw probe answers, provider receipts, or conversational content.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .batch_experiment import BatchExperimentManifest
from .models import ProtocolCondition
from .wvs7_confirmatory_analysis_plan import (
    ANALYSIS_VERSION,
    EXPECTED_CONDITIONS,
    EXPECTED_EXCLUDED_RUN_IDS,
    EXPECTED_PRIMARY_METRICS,
    ConfirmatoryAnalysisPlan,
)


_ROOT = Path(__file__).resolve().parents[2]
_REQUIRED_EVALUATION_WARNINGS = (
    "representation: skipped minority retention because coverage_threshold and explicit minority_dimensions are required",
    "holdout_social_influence_loss: skipped because no explicit matched evidence_only/evidence_social pair exists",
    "audit: token_count_total skipped because no numeric token count exists in events",
)
_CONDITION_BY_NAME = {condition.value: condition for condition in EXPECTED_CONDITIONS}


class WVS7ConfirmatoryAnalysisError(ValueError):
    """Stable safe failure. Source content is never included in messages."""


class AnalysisInputFile(BaseModel):
    """Hash-only audit entry for one permitted local analysis input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(ge=0)


class SeedConditionValue(BaseModel):
    """One aggregate metric coordinate. The unit is always one matched seed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    condition: ProtocolCondition
    run_id: str = Field(min_length=1)
    metric_name: str = Field(min_length=1)
    value: float

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("metric value must be finite")
        return value


class SeedPairedValue(BaseModel):
    """One explicitly paired seed-level counterfactual metric coordinate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    metric_name: Literal["social_influence_loss"]
    left_condition: Literal[ProtocolCondition.EVIDENCE_ONLY]
    right_condition: Literal[ProtocolCondition.EVIDENCE_SOCIAL]
    value: float

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("metric value must be finite")
        return value


class ConditionSummary(BaseModel):
    """Descriptive condition summary over exactly the matched seed units."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_name: str
    condition: ProtocolCondition
    n: int = Field(ge=0)
    mean: float
    standard_deviation: float
    median: float


class ExactSignFlipResult(BaseModel):
    """Two-sided exact sign-flip result, explicitly limited to n=3 pilot data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["exact_paired_sign_flip_two_sided"]
    n: Literal[3]
    permutation_count: Literal[8]
    observed_mean_difference: float
    two_sided_p_value: float = Field(ge=0.0, le=1.0)
    evidence_limit: Literal["n_equals_3_not_strong_significance_evidence"]


class PairedComparison(BaseModel):
    """A preregistered seed-level contrast, not an Agent or probe-level test."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    comparison_id: str
    metric_name: str
    left_condition: ProtocolCondition
    right_condition: ProtocolCondition
    n: Literal[3]
    seed_differences: list[float] = Field(min_length=3, max_length=3)
    mean_difference: float
    standard_deviation: float
    effect_direction: Literal["positive", "negative", "zero"]
    exact_sign_flip: ExactSignFlipResult


class LeaveOneSeedOutResult(BaseModel):
    """Deterministic sensitivity summary that omits one matched group at a time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    comparison_id: str
    excluded_seed: int
    remaining_seed_count: Literal[2]
    mean_difference: float
    standard_deviation: float
    effect_direction: Literal["positive", "negative", "zero"]


class ConfirmatoryAnalysisManifest(BaseModel):
    """Safe, written receipt of the analysis design and exact input hashes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis_id: str
    analysis_version: Literal["wvs7_protocol_blind_v2_confirmatory_analysis.v1"]
    batch_id: str
    seed_count: Literal[3]
    conditions: list[ProtocolCondition]
    primary_metrics: list[str]
    input_files: list[AnalysisInputFile]
    immutable_source_sha256: dict[str, str]
    confirmatory: Literal[True]
    statistical_analysis: Literal[True]
    not_causal_conclusion: Literal[True]
    v1_002_excluded: Literal[True]
    v2_003_excluded: Literal[True]


class ConfirmatoryAnalysisReport(BaseModel):
    """Safe statistical output without source text, raw answers, or prompts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis_id: str
    analysis_version: Literal["wvs7_protocol_blind_v2_confirmatory_analysis.v1"]
    batch_id: str
    statistical_unit: Literal["matched_group_seed"]
    n_matched_groups: Literal[3]
    raw_metric_values: list[SeedConditionValue]
    social_influence_loss_values: list[SeedPairedValue]
    condition_summaries: list[ConditionSummary]
    paired_comparisons: list[PairedComparison]
    profile_drift_descriptive: list[ConditionSummary]
    source_evaluation_warnings: list[str]
    input_files: list[AnalysisInputFile]
    immutable_source_sha256: dict[str, str]
    confirmatory: Literal[True]
    statistical_analysis: Literal[True]
    not_causal_conclusion: Literal[True]
    v1_002_excluded: Literal[True]
    v2_003_excluded: Literal[True]
    interpretation_limit: Literal["limited_sample_confirmatory_pilot_no_causal_conclusion"]


class RobustnessReport(BaseModel):
    """Safe leave-one-seed-out sensitivity report for preregistered contrasts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis_id: str
    analysis_version: Literal["wvs7_protocol_blind_v2_confirmatory_analysis.v1"]
    statistical_unit: Literal["matched_group_seed"]
    leave_one_seed_out: list[LeaveOneSeedOutResult]
    immutable_source_sha256: dict[str, str]
    confirmatory: Literal[True]
    statistical_analysis: Literal[True]
    not_causal_conclusion: Literal[True]
    v1_002_excluded: Literal[True]
    v2_003_excluded: Literal[True]
    interpretation_limit: Literal["limited_sample_confirmatory_pilot_no_causal_conclusion"]


class ConfirmatoryAnalysisResult(BaseModel):
    """CLI-safe completion summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["completed", "blocked"]
    analysis_id: str
    seed_count: int = Field(ge=0)
    condition_count: int = Field(ge=0)
    primary_metrics: list[str]
    output_dir: str | None = None
    warnings: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _LoadedInputs:
    receipt: dict[str, Any]
    completeness_audit: dict[str, Any]
    evaluation_report: dict[str, Any]
    batch_manifest: BatchExperimentManifest
    input_files: list[AnalysisInputFile]


def _rooted(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else _ROOT / value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WVS7ConfirmatoryAnalysisError(code) from exc
    if not isinstance(value, dict):
        raise WVS7ConfirmatoryAnalysisError(code)
    return value


def _audit(path: Path, kind: str) -> AnalysisInputFile:
    try:
        return AnalysisInputFile(
            kind=kind,
            path=str(path),
            sha256=_sha256_file(path),
            byte_count=path.stat().st_size,
        )
    except OSError as exc:
        raise WVS7ConfirmatoryAnalysisError("analysis_input_missing") from exc


def _require(value: bool, code: str) -> None:
    if not value:
        raise WVS7ConfirmatoryAnalysisError(code)


def _finite_metric(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise WVS7ConfirmatoryAnalysisError("analysis_metric_value_invalid")
    return float(value)


def _parse_run_id(value: object, *, phase: str | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise WVS7ConfirmatoryAnalysisError("analysis_metric_coordinate_invalid")
    if phase is None:
        return value
    suffix = f"::{phase}"
    if not value.endswith(suffix):
        raise WVS7ConfirmatoryAnalysisError("analysis_metric_coordinate_invalid")
    return value[: -len(suffix)]


def _safe_run_record_audit(path: Path, spec: object) -> AnalysisInputFile:
    """Read only status/config fields; turns and value content are never inspected or emitted."""
    record = _load_json(path, "analysis_run_record_invalid")
    config = record.get("config")
    if not isinstance(config, dict):
        raise WVS7ConfirmatoryAnalysisError("analysis_run_record_invalid")
    _require(record.get("status") == "completed", "analysis_run_not_completed")
    _require(
        config.get("run_id") == spec.run_id
        and config.get("scenario_id") == spec.scenario_id
        and config.get("protocol") == spec.condition.value
        and config.get("seed") == spec.seed
        and config.get("model_name") == spec.model_name
        and config.get("max_rounds") == spec.max_rounds,
        "analysis_run_record_coordinate_mismatch",
    )
    return _audit(path, "run_record")


def _load_and_validate_inputs(plan: ConfirmatoryAnalysisPlan) -> _LoadedInputs:
    paths = plan.inputs
    receipt_path = _rooted(paths.batch_receipt_path)
    audit_path = _rooted(paths.completeness_audit_path)
    evaluation_path = _rooted(paths.offline_evaluation_report_path)
    manifest_path = _rooted(paths.batch_manifest_path)
    receipt = _load_json(receipt_path, "analysis_receipt_invalid")
    completeness = _load_json(audit_path, "analysis_completeness_audit_invalid")
    evaluation = _load_json(evaluation_path, "analysis_evaluation_report_invalid")
    manifest_raw = _load_json(manifest_path, "analysis_batch_manifest_invalid")
    try:
        batch_manifest = BatchExperimentManifest.model_validate(manifest_raw)
    except ValidationError as exc:
        raise WVS7ConfirmatoryAnalysisError("analysis_batch_manifest_invalid") from exc

    _require(receipt.get("status") == "confirmatory", "analysis_receipt_not_confirmatory")
    _require(receipt.get("batch_id") == plan.batch_id, "analysis_batch_id_mismatch")
    _require(receipt.get("completed_run_count") == 12 and receipt.get("failed_run_count") == 0, "analysis_runs_incomplete")
    _require(receipt.get("matched_group_count") == 3, "analysis_matched_groups_incomplete")
    receipt_replays = receipt.get("replay_statuses")
    _require(isinstance(receipt_replays, dict) and len(receipt_replays) == 12 and set(receipt_replays.values()) == {"passed"}, "analysis_replay_not_passed")
    _require(
        receipt.get("plan_sha256") == plan.expected_plan_sha256
        and receipt.get("plan_lock_sha256") == plan.expected_plan_lock_sha256
        and receipt.get("frozen_probe_set_sha256") == plan.expected_frozen_probe_set_sha256
        and receipt.get("test_scenario_sha256") == plan.expected_test_scenario_sha256,
        "analysis_receipt_hash_mismatch",
    )
    _require(
        completeness.get("status") == "confirmatory"
        and completeness.get("completed_run_count") == 12
        and completeness.get("failed_run_count") == 0
        and completeness.get("matched_group_count") == 3
        and completeness.get("matched_groups_complete") is True,
        "analysis_completeness_audit_mismatch",
    )
    audit_replays = completeness.get("replay_statuses")
    _require(isinstance(audit_replays, dict) and audit_replays == receipt_replays, "analysis_replay_not_passed")
    _require(
        completeness.get("frozen_probe_set_sha256") == receipt.get("frozen_probe_set_sha256")
        and completeness.get("test_scenario_sha256") == receipt.get("test_scenario_sha256"),
        "analysis_completeness_hash_mismatch",
    )
    _require(batch_manifest.batch_id == plan.batch_id, "analysis_batch_id_mismatch")
    _require(tuple(batch_manifest.seeds) == tuple(plan.seeds), "analysis_seed_mismatch")
    _require(tuple(batch_manifest.conditions) == tuple(plan.conditions), "analysis_condition_mismatch")
    _require(len(batch_manifest.runs) == 12 and len(batch_manifest.run_order) == 12, "analysis_run_plan_incomplete")
    metadata = batch_manifest.metadata
    _require(
        metadata.get("confirmatory") is True
        and metadata.get("pilot_version") == "protocol_blind_v2"
        and metadata.get("frozen_probe_set_sha256") == plan.expected_frozen_probe_set_sha256
        and metadata.get("test_scenario_sha256") == plan.expected_test_scenario_sha256,
        "analysis_batch_manifest_binding_mismatch",
    )
    _require(
        not (set(plan.excluded_run_ids) & {run.run_id for run in batch_manifest.runs}),
        "analysis_excluded_run_mixed",
    )
    _require(
        not (set(EXPECTED_EXCLUDED_RUN_IDS) & {run.run_id for run in batch_manifest.runs}),
        "analysis_excluded_run_mixed",
    )

    pairs = {(run.seed, run.condition) for run in batch_manifest.runs}
    expected_pairs = {(seed, condition) for seed in plan.seeds for condition in plan.conditions}
    _require(pairs == expected_pairs, "analysis_run_plan_incomplete")
    groups_by_seed: dict[int, set[str]] = {seed: set() for seed in plan.seeds}
    for run in batch_manifest.runs:
        groups_by_seed[run.seed].add(run.matched_group_id)
        _require(run.probe_set_id == "wvs7-english-core-23" and run.is_holdout is False, "analysis_probe_contract_mismatch")
    _require(all(len(groups) == 1 for groups in groups_by_seed.values()), "analysis_matched_group_mismatch")

    input_files = [
        _audit(receipt_path, "confirmatory_batch_receipt"),
        _audit(audit_path, "confirmatory_completeness_audit"),
        _audit(evaluation_path, "offline_evaluation_report_initial_final_v2"),
        _audit(manifest_path, "batch_manifest"),
    ]
    for run in batch_manifest.runs:
        run_path = Path(run.artifacts.run_record_path)
        _require(run_path.name == "run_record.json", "analysis_run_record_path_invalid")
        input_files.append(_safe_run_record_audit(run_path, run))
    return _LoadedInputs(receipt, completeness, evaluation, batch_manifest, input_files)


def _collect_primary_values(
    evaluation: dict[str, Any],
    batch_manifest: BatchExperimentManifest,
) -> tuple[dict[tuple[int, ProtocolCondition], float], dict[tuple[int, ProtocolCondition], float], dict[int, float]]:
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, list):
        raise WVS7ConfirmatoryAnalysisError("analysis_evaluation_report_invalid")
    run_by_id = {run.run_id: run for run in batch_manifest.runs}
    diversity: dict[tuple[int, ProtocolCondition], float] = {}
    drift: dict[tuple[int, ProtocolCondition], float] = {}
    social_loss: dict[int, float] = {}
    for metric in metrics:
        if not isinstance(metric, dict) or metric.get("valid") is not True:
            continue
        name = metric.get("metric_name")
        inputs = metric.get("input_ids")
        if not isinstance(inputs, list):
            continue
        value = _finite_metric(metric.get("value"))
        if name == "pairwise_diversity":
            if len(inputs) != 1:
                raise WVS7ConfirmatoryAnalysisError("analysis_metric_condition_pairing_invalid")
            run_id = _parse_run_id(inputs[0])
            run = run_by_id.get(run_id)
            if run is None:
                raise WVS7ConfirmatoryAnalysisError("analysis_metric_coordinate_invalid")
            key = (run.seed, run.condition)
            if key in diversity:
                raise WVS7ConfirmatoryAnalysisError("analysis_metric_duplicate")
            diversity[key] = value
        elif name == "profile_drift":
            if len(inputs) != 2:
                raise WVS7ConfirmatoryAnalysisError("analysis_metric_condition_pairing_invalid")
            first = _parse_run_id(inputs[0], phase="initial")
            second = _parse_run_id(inputs[1], phase="final")
            if first != second or first not in run_by_id:
                raise WVS7ConfirmatoryAnalysisError("analysis_metric_condition_pairing_invalid")
            run = run_by_id[first]
            key = (run.seed, run.condition)
            if key in drift:
                raise WVS7ConfirmatoryAnalysisError("analysis_metric_duplicate")
            drift[key] = value
        elif name == "social_influence_loss":
            if len(inputs) != 2:
                raise WVS7ConfirmatoryAnalysisError("analysis_metric_condition_pairing_invalid")
            left = _parse_run_id(inputs[0], phase="final")
            right = _parse_run_id(inputs[1], phase="final")
            left_run, right_run = run_by_id.get(left), run_by_id.get(right)
            if (
                left_run is None or right_run is None
                or left_run.seed != right_run.seed
                or left_run.condition is not ProtocolCondition.EVIDENCE_ONLY
                or right_run.condition is not ProtocolCondition.EVIDENCE_SOCIAL
                or left_run.matched_group_id != right_run.matched_group_id
                or left_run.seed in social_loss
            ):
                raise WVS7ConfirmatoryAnalysisError("analysis_metric_condition_pairing_invalid")
            social_loss[left_run.seed] = value
    expected = {(seed, condition) for seed in batch_manifest.seeds for condition in batch_manifest.conditions}
    if set(diversity) != expected or set(drift) != expected or set(social_loss) != set(batch_manifest.seeds):
        raise WVS7ConfirmatoryAnalysisError("analysis_primary_metric_missing")
    for seed in batch_manifest.seeds:
        derived = diversity[(seed, ProtocolCondition.EVIDENCE_ONLY)] - diversity[(seed, ProtocolCondition.EVIDENCE_SOCIAL)]
        if not math.isclose(social_loss[seed], derived, rel_tol=0.0, abs_tol=1e-12):
            raise WVS7ConfirmatoryAnalysisError("analysis_social_influence_loss_inconsistent")
    return diversity, drift, social_loss


def _summary(metric_name: str, condition: ProtocolCondition, values: list[float]) -> ConditionSummary:
    return ConditionSummary(
        metric_name=metric_name,
        condition=condition,
        n=len(values),
        mean=statistics.fmean(values),
        standard_deviation=statistics.stdev(values) if len(values) > 1 else 0.0,
        median=statistics.median(values),
    )


def _direction(value: float) -> Literal["positive", "negative", "zero"]:
    if math.isclose(value, 0.0, abs_tol=1e-12):
        return "zero"
    return "positive" if value > 0 else "negative"


def exact_paired_sign_flip(differences: list[float]) -> ExactSignFlipResult:
    """Exact two-sided sign-flip test for the preregistered three-seed unit."""
    if len(differences) != 3 or any(not math.isfinite(value) for value in differences):
        raise WVS7ConfirmatoryAnalysisError("analysis_sign_flip_requires_three_finite_seed_differences")
    observed = statistics.fmean(differences)
    permutations = [statistics.fmean([sign * value for sign, value in zip(signs, differences)]) for signs in itertools.product((-1.0, 1.0), repeat=3)]
    extreme = sum(abs(value) >= abs(observed) - 1e-15 for value in permutations)
    return ExactSignFlipResult(
        method="exact_paired_sign_flip_two_sided",
        n=3,
        permutation_count=8,
        observed_mean_difference=observed,
        two_sided_p_value=extreme / 8.0,
        evidence_limit="n_equals_3_not_strong_significance_evidence",
    )


def _comparison(
    comparison_id: str,
    metric_name: str,
    left_condition: ProtocolCondition,
    right_condition: ProtocolCondition,
    differences: list[float],
) -> PairedComparison:
    mean_difference = statistics.fmean(differences)
    return PairedComparison(
        comparison_id=comparison_id,
        metric_name=metric_name,
        left_condition=left_condition,
        right_condition=right_condition,
        n=3,
        seed_differences=differences,
        mean_difference=mean_difference,
        standard_deviation=statistics.stdev(differences),
        effect_direction=_direction(mean_difference),
        exact_sign_flip=exact_paired_sign_flip(differences),
    )


def _leave_one_out(comparison: PairedComparison, seeds: list[int]) -> list[LeaveOneSeedOutResult]:
    values: list[LeaveOneSeedOutResult] = []
    for index, seed in enumerate(seeds):
        remaining = comparison.seed_differences[:index] + comparison.seed_differences[index + 1 :]
        mean_difference = statistics.fmean(remaining)
        values.append(LeaveOneSeedOutResult(
            comparison_id=comparison.comparison_id,
            excluded_seed=seed,
            remaining_seed_count=2,
            mean_difference=mean_difference,
            standard_deviation=statistics.stdev(remaining),
            effect_direction=_direction(mean_difference),
        ))
    return values


def _analysis_objects(plan: ConfirmatoryAnalysisPlan, loaded: _LoadedInputs) -> tuple[ConfirmatoryAnalysisManifest, ConfirmatoryAnalysisReport, RobustnessReport]:
    evaluation_warnings = loaded.evaluation_report.get("warnings")
    if not isinstance(evaluation_warnings, list) or not all(isinstance(item, str) for item in evaluation_warnings):
        raise WVS7ConfirmatoryAnalysisError("analysis_evaluation_warnings_invalid")
    if not set(_REQUIRED_EVALUATION_WARNINGS).issubset(evaluation_warnings):
        raise WVS7ConfirmatoryAnalysisError("analysis_required_evaluation_warning_missing")
    diversity, drift, social_loss = _collect_primary_values(loaded.evaluation_report, loaded.batch_manifest)
    run_by_key = {(run.seed, run.condition): run for run in loaded.batch_manifest.runs}
    source_hashes = {
        "plan_sha256": plan.expected_plan_sha256,
        "plan_lock_sha256": plan.expected_plan_lock_sha256,
        "frozen_probe_set_sha256": plan.expected_frozen_probe_set_sha256,
        "test_scenario_sha256": plan.expected_test_scenario_sha256,
    }
    raw: list[SeedConditionValue] = []
    for seed in plan.seeds:
        for condition in plan.conditions:
            raw.append(SeedConditionValue(
                seed=seed, condition=condition, run_id=run_by_key[(seed, condition)].run_id,
                metric_name="pairwise_diversity", value=diversity[(seed, condition)],
            ))
            raw.append(SeedConditionValue(
                seed=seed, condition=condition, run_id=run_by_key[(seed, condition)].run_id,
                metric_name="profile_drift", value=drift[(seed, condition)],
            ))
    social_values = [
        SeedPairedValue(
            seed=seed,
            metric_name="social_influence_loss",
            left_condition=ProtocolCondition.EVIDENCE_ONLY,
            right_condition=ProtocolCondition.EVIDENCE_SOCIAL,
            value=social_loss[seed],
        )
        for seed in plan.seeds
    ]
    condition_summaries = [
        _summary("pairwise_diversity", condition, [diversity[(seed, condition)] for seed in plan.seeds])
        for condition in plan.conditions
    ]
    drift_summaries = [
        _summary("profile_drift", condition, [drift[(seed, condition)] for seed in plan.seeds])
        for condition in plan.conditions
    ]
    comparisons = [
        _comparison(
            "social_only_minus_independent", "pairwise_diversity",
            ProtocolCondition.SOCIAL_ONLY, ProtocolCondition.INDEPENDENT,
            [diversity[(seed, ProtocolCondition.SOCIAL_ONLY)] - diversity[(seed, ProtocolCondition.INDEPENDENT)] for seed in plan.seeds],
        ),
        _comparison(
            "evidence_only_minus_independent", "pairwise_diversity",
            ProtocolCondition.EVIDENCE_ONLY, ProtocolCondition.INDEPENDENT,
            [diversity[(seed, ProtocolCondition.EVIDENCE_ONLY)] - diversity[(seed, ProtocolCondition.INDEPENDENT)] for seed in plan.seeds],
        ),
        _comparison(
            "evidence_social_minus_evidence_only", "pairwise_diversity",
            ProtocolCondition.EVIDENCE_SOCIAL, ProtocolCondition.EVIDENCE_ONLY,
            [diversity[(seed, ProtocolCondition.EVIDENCE_SOCIAL)] - diversity[(seed, ProtocolCondition.EVIDENCE_ONLY)] for seed in plan.seeds],
        ),
        _comparison(
            "social_influence_loss_evidence_only_minus_evidence_social", "social_influence_loss",
            ProtocolCondition.EVIDENCE_ONLY, ProtocolCondition.EVIDENCE_SOCIAL,
            [social_loss[seed] for seed in plan.seeds],
        ),
    ]
    analysis_manifest = ConfirmatoryAnalysisManifest(
        analysis_id=plan.analysis_id,
        analysis_version=ANALYSIS_VERSION,
        batch_id=plan.batch_id,
        seed_count=3,
        conditions=list(plan.conditions),
        primary_metrics=list(plan.primary_metrics),
        input_files=loaded.input_files,
        immutable_source_sha256=source_hashes,
        confirmatory=True,
        statistical_analysis=True,
        not_causal_conclusion=True,
        v1_002_excluded=True,
        v2_003_excluded=True,
    )
    report = ConfirmatoryAnalysisReport(
        analysis_id=plan.analysis_id,
        analysis_version=ANALYSIS_VERSION,
        batch_id=plan.batch_id,
        statistical_unit="matched_group_seed",
        n_matched_groups=3,
        raw_metric_values=raw,
        social_influence_loss_values=social_values,
        condition_summaries=condition_summaries,
        paired_comparisons=comparisons,
        profile_drift_descriptive=drift_summaries,
        source_evaluation_warnings=list(evaluation_warnings),
        input_files=loaded.input_files,
        immutable_source_sha256=source_hashes,
        confirmatory=True,
        statistical_analysis=True,
        not_causal_conclusion=True,
        v1_002_excluded=True,
        v2_003_excluded=True,
        interpretation_limit="limited_sample_confirmatory_pilot_no_causal_conclusion",
    )
    robustness = RobustnessReport(
        analysis_id=plan.analysis_id,
        analysis_version=ANALYSIS_VERSION,
        statistical_unit="matched_group_seed",
        leave_one_seed_out=[entry for comparison in comparisons for entry in _leave_one_out(comparison, list(plan.seeds))],
        immutable_source_sha256=source_hashes,
        confirmatory=True,
        statistical_analysis=True,
        not_causal_conclusion=True,
        v1_002_excluded=True,
        v2_003_excluded=True,
        interpretation_limit="limited_sample_confirmatory_pilot_no_causal_conclusion",
    )
    return analysis_manifest, report, robustness


def _write_idempotent(path: Path, payload: BaseModel) -> None:
    serialized = payload.model_dump_json(indent=2) + "\n"
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == serialized:
                return
        except OSError as exc:
            raise WVS7ConfirmatoryAnalysisError("analysis_output_collision") from exc
        raise WVS7ConfirmatoryAnalysisError("analysis_output_collision")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def run_confirmatory_analysis(plan: ConfirmatoryAnalysisPlan) -> ConfirmatoryAnalysisResult:
    """Validate safe inputs and write deterministic, content-free analysis reports."""
    loaded = _load_and_validate_inputs(plan)
    manifest, report, robustness = _analysis_objects(plan, loaded)
    output_dir = _rooted(plan.output_dir)
    _write_idempotent(output_dir / "analysis_manifest.json", manifest)
    _write_idempotent(output_dir / "analysis_report.json", report)
    _write_idempotent(output_dir / "robustness_report.json", robustness)
    return ConfirmatoryAnalysisResult(
        status="completed",
        analysis_id=plan.analysis_id,
        seed_count=3,
        condition_count=len(plan.conditions),
        primary_metrics=list(plan.primary_metrics),
        output_dir=str(output_dir),
        warnings=list(report.source_evaluation_warnings),
    )


def validate_confirmatory_analysis(plan: ConfirmatoryAnalysisPlan) -> ConfirmatoryAnalysisResult:
    """Validate all inputs and calculations without writing an output directory."""
    loaded = _load_and_validate_inputs(plan)
    _analysis_objects(plan, loaded)
    return ConfirmatoryAnalysisResult(
        status="completed",
        analysis_id=plan.analysis_id,
        seed_count=3,
        condition_count=len(plan.conditions),
        primary_metrics=list(plan.primary_metrics),
        warnings=list(loaded.evaluation_report.get("warnings", [])),
    )


__all__ = [
    "ANALYSIS_VERSION",
    "AnalysisInputFile",
    "ConditionSummary",
    "ConfirmatoryAnalysisManifest",
    "ConfirmatoryAnalysisReport",
    "ConfirmatoryAnalysisResult",
    "ExactSignFlipResult",
    "LeaveOneSeedOutResult",
    "PairedComparison",
    "RobustnessReport",
    "SeedConditionValue",
    "SeedPairedValue",
    "WVS7ConfirmatoryAnalysisError",
    "exact_paired_sign_flip",
    "run_confirmatory_analysis",
    "validate_confirmatory_analysis",
]
