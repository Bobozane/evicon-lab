"""No-network orchestration that delegates all formulas to ``evicon.metrics``."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, TypeVar

from ..metrics import (
    MetricError,
    MetricResult,
    holdout_profile_drift,
    holdout_social_influence_loss,
    minority_retention,
    pairwise_diversity,
    social_influence_loss,
    structural_diversity,
    value_dimension_coverage,
)
from ..models import RunStatus, ValueProfile
from .loaders import EvaluationManifestLoader, LoadedEvaluationRun, load_declared_run
from .matching import MatchingResult, match_runs
from .models import (
    EvaluationManifest,
    EvaluationReport,
    EvaluationStatus,
    InputFileAudit,
    MatchedRunPair,
    MetricSuite,
)


MetricValue = TypeVar("MetricValue")


class OfflineEvaluator:
    """Load declared artifacts, match them explicitly, and write no online state."""

    def __init__(self, manifest: EvaluationManifest, *, manifest_directory: str | Path) -> None:
        self.manifest = manifest
        self.manifest_directory = Path(manifest_directory)

    @classmethod
    def from_manifest_file(cls, path: str | Path) -> "OfflineEvaluator":
        manifest_path = Path(path)
        return cls(
            EvaluationManifestLoader.load(manifest_path),
            manifest_directory=manifest_path.parent,
        )

    def evaluate(self) -> EvaluationReport:
        """Run a local, deterministic assessment over completed declared artifacts."""
        loaded_runs = tuple(
            load_declared_run(
                entry,
                self.manifest,
                manifest_directory=self.manifest_directory,
            )
            for entry in self.manifest.runs
        )
        matching = match_runs(loaded_runs)
        warnings = [
            f"{entry.comparison}: {entry.run_id}: {entry.reason}"
            for entry in matching.unmatched_entries
        ]
        metrics: list[MetricResult] = []
        enabled = _enabled_suites(self.manifest.metric_suite)

        if MetricSuite.BASELINE_DIVERSITY in enabled:
            self._evaluate_baseline_diversity(loaded_runs, metrics, warnings)
        if MetricSuite.REPRESENTATION in enabled:
            self._evaluate_representation(loaded_runs, matching, metrics, warnings)
        if MetricSuite.HOLDOUT in enabled:
            self._evaluate_holdout(matching, loaded_runs, metrics, warnings)
        if MetricSuite.COUNTERFACTUAL in enabled:
            self._evaluate_counterfactual(matching, loaded_runs, metrics, warnings)
        if MetricSuite.AUDIT in enabled:
            self._evaluate_audit(loaded_runs, metrics, warnings)

        audit = _unique_audit(audit for loaded in loaded_runs for audit in loaded.input_audit)
        return EvaluationReport(
            evaluation_id=self.manifest.evaluation_id,
            scenario_id=self.manifest.scenario_id,
            metric_suite=self.manifest.metric_suite,
            status=(
                EvaluationStatus.COMPLETED_WITH_WARNINGS
                if warnings
                else EvaluationStatus.COMPLETED
            ),
            metrics=metrics,
            matched_pairs=list(matching.matched_pairs),
            unmatched_entries=list(matching.unmatched_entries),
            input_audit=audit,
            warnings=warnings,
        )

    def write_report(self, path: str | Path) -> Path:
        """Evaluate and write one new JSON report; existing reports are preserved."""
        return write_evaluation_report(self.evaluate(), path)

    def _evaluate_baseline_diversity(
        self,
        runs: tuple[LoadedEvaluationRun, ...],
        metrics: list[MetricResult],
        warnings: list[str],
    ) -> None:
        for run in _completed_final_runs(runs):
            profiles = _profiles_for_scope(run, "standard")
            if not profiles:
                warnings.append(
                    f"pairwise_diversity: {run.entry.run_id}: skipped because no standard ValueProfile is available"
                )
                continue
            result = _call_metric(
                "pairwise_diversity",
                run.entry.run_id,
                lambda: pairwise_diversity(profiles, strict=self.manifest.metric_options.strict),
                warnings,
            )
            if result is None:
                continue
            metrics.append(
                _metric_result(
                    "pairwise_diversity",
                    result.mean_distance,
                    "normalized_distance",
                    [run.entry.run_id],
                    result,
                )
            )
            structural = _call_metric(
                "structural_diversity",
                run.entry.run_id,
                lambda: structural_diversity(
                    result.pair_distances,
                    result.agent_ids,
                    strict=self.manifest.metric_options.strict,
                ),
                warnings,
            )
            if structural is not None:
                metrics.append(
                    _metric_result(
                        "structural_diversity",
                        structural.mst_span,
                        "normalized_distance",
                        [run.entry.run_id],
                        structural,
                        valid=structural.valid,
                    )
                )

    def _evaluate_representation(
        self,
        runs: tuple[LoadedEvaluationRun, ...],
        matching: MatchingResult,
        metrics: list[MetricResult],
        warnings: list[str],
    ) -> None:
        threshold = self.manifest.metric_options.coverage_threshold
        if threshold is None:
            warnings.append(
                "representation: skipped value dimension coverage because metric_options.coverage_threshold is required"
            )
        else:
            for run in _completed_final_runs(runs):
                profiles = _profiles_for_scope(run, "standard")
                if not profiles:
                    warnings.append(
                        f"value_dimension_coverage: {run.entry.run_id}: skipped because no standard ValueProfile is available"
                    )
                    continue
                result = _call_metric(
                    "value_dimension_coverage",
                    run.entry.run_id,
                    lambda: value_dimension_coverage(profiles, threshold=threshold),
                    warnings,
                )
                if result is not None:
                    metrics.append(
                        _metric_result(
                            "value_dimension_coverage",
                            result.coverage_rate,
                            "proportion",
                            [run.entry.run_id],
                            result,
                        )
                    )

        if threshold is None or not self.manifest.metric_options.minority_dimensions:
            warnings.append(
                "representation: skipped minority retention because coverage_threshold and explicit minority_dimensions are required"
            )
        pairs = _pairs_for(matching, comparison="initial_final", scope="standard")
        if not pairs:
            warnings.append("minority_retention: skipped because no explicit initial/final standard probe pair exists")
        for pair in pairs:
            initial, final = _pair_runs(pair, runs)
            drift = _call_metric(
                "profile_drift",
                f"{pair.left_run_id},{pair.right_run_id}",
                lambda: holdout_profile_drift(
                    _profiles_for_scope(initial, "standard"),
                    _profiles_for_scope(final, "standard"),
                ),
                warnings,
            )
            if drift is not None:
                metrics.append(
                    _metric_result(
                        "profile_drift",
                        drift.mean_drift,
                        "normalized_distance",
                        [pair.left_run_id, pair.right_run_id],
                        drift,
                    )
                )
            if threshold is None or not self.manifest.metric_options.minority_dimensions:
                continue
            result = _call_metric(
                "minority_retention",
                f"{pair.left_run_id},{pair.right_run_id}",
                lambda: minority_retention(
                    _profiles_for_scope(initial, "standard"),
                    _profiles_for_scope(final, "standard"),
                    minority_dimensions=self.manifest.metric_options.minority_dimensions,
                    threshold=threshold,
                ),
                warnings,
            )
            if result is not None:
                metrics.append(
                    _metric_result(
                        "minority_retention",
                        result.retention_rate,
                        "proportion",
                        [pair.left_run_id, pair.right_run_id],
                        result,
                    )
                )

    def _evaluate_holdout(
        self,
        matching: MatchingResult,
        runs: tuple[LoadedEvaluationRun, ...],
        metrics: list[MetricResult],
        warnings: list[str],
    ) -> None:
        pairs = _pairs_for(matching, comparison="initial_final", scope="holdout")
        if not pairs:
            warnings.append("holdout_profile_drift: skipped because no explicit initial/final holdout probe pair exists")
        for pair in pairs:
            initial, final = _pair_runs(pair, runs)
            result = _call_metric(
                "holdout_profile_drift",
                f"{pair.left_run_id},{pair.right_run_id}",
                lambda: holdout_profile_drift(
                    _profiles_for_scope(initial, "holdout"),
                    _profiles_for_scope(final, "holdout"),
                ),
                warnings,
            )
            if result is not None:
                metrics.append(
                    _metric_result(
                        "holdout_profile_drift",
                        result.mean_drift,
                        "normalized_distance",
                        [pair.left_run_id, pair.right_run_id],
                        result,
                    )
                )

    def _evaluate_counterfactual(
        self,
        matching: MatchingResult,
        runs: tuple[LoadedEvaluationRun, ...],
        metrics: list[MetricResult],
        warnings: list[str],
    ) -> None:
        _evaluate_social_loss_pair_set(
            pairs=_pairs_for(matching, comparison="evidence_social", scope="standard"),
            runs=runs,
            metric_name="social_influence_loss",
            calculation=lambda left, right: social_influence_loss(
                _profiles_for_scope(left, "standard"),
                _profiles_for_scope(right, "standard"),
                strict=self.manifest.metric_options.strict,
            ),
            metrics=metrics,
            warnings=warnings,
        )
        _evaluate_social_loss_pair_set(
            pairs=_pairs_for(matching, comparison="holdout_evidence_social", scope="holdout"),
            runs=runs,
            metric_name="holdout_social_influence_loss",
            calculation=lambda left, right: holdout_social_influence_loss(
                _profiles_for_scope(left, "holdout"),
                _profiles_for_scope(right, "holdout"),
                strict=self.manifest.metric_options.strict,
            ),
            metrics=metrics,
            warnings=warnings,
        )

    def _evaluate_audit(
        self,
        runs: tuple[LoadedEvaluationRun, ...],
        metrics: list[MetricResult],
        warnings: list[str],
    ) -> None:
        token_values: list[float] = []
        latency_values: list[float] = []
        for run in _audit_runs(runs):
            metrics.extend(_run_audit_metrics(run))
            tokens, latency = _optional_log_totals(run.events)
            token_values.extend(tokens)
            latency_values.extend(latency)
            if run.record.status is RunStatus.FAILED:
                warnings.append(f"audit: {run.entry.run_id}: failed run was retained only for partial audit")
        if token_values:
            metrics.append(
                MetricResult(
                    metric_name="token_count_total",
                    value=sum(token_values),
                    unit="tokens",
                    input_ids=[run.entry.evaluation_entry_id for run in runs],
                    configuration={"observed_value_count": len(token_values)},
                    valid=True,
                )
            )
        else:
            warnings.append("audit: token_count_total skipped because no numeric token count exists in events")
        if latency_values:
            metrics.append(
                MetricResult(
                    metric_name="latency_ms_total",
                    value=sum(latency_values),
                    unit="milliseconds",
                    input_ids=[run.entry.evaluation_entry_id for run in runs],
                    configuration={"observed_value_count": len(latency_values)},
                    valid=True,
                )
            )
        else:
            warnings.append("audit: latency_ms_total skipped because no numeric latency exists in events")


def write_evaluation_report(report: EvaluationReport, path: str | Path) -> Path:
    """Write one new JSON report without mutating or overwriting source artifacts."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as handle:
            json.dump(report.model_dump(mode="json"), handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\\n")
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite existing evaluation report: {output_path}") from None
    return output_path


def _enabled_suites(suites: list[MetricSuite]) -> set[MetricSuite]:
    if MetricSuite.ALL in suites:
        return {
            MetricSuite.BASELINE_DIVERSITY,
            MetricSuite.REPRESENTATION,
            MetricSuite.HOLDOUT,
            MetricSuite.COUNTERFACTUAL,
            MetricSuite.AUDIT,
        }
    return set(suites)


def _completed_final_runs(runs: Iterable[LoadedEvaluationRun]) -> tuple[LoadedEvaluationRun, ...]:
    return tuple(
        run
        for run in runs
        if run.record.status is RunStatus.COMPLETED and run.entry.role == "final"
    )


def _audit_runs(runs: Iterable[LoadedEvaluationRun]) -> tuple[LoadedEvaluationRun, ...]:
    """Audit one physical trajectory once even when it has initial/final probe entries."""
    selected: dict[str, LoadedEvaluationRun] = {}
    for run in runs:
        current = selected.get(run.entry.run_id)
        if current is None or run.entry.role == "final":
            selected[run.entry.run_id] = run
    return tuple(selected[run_id] for run_id in sorted(selected))


def _profiles_for_scope(run: LoadedEvaluationRun, scope: str) -> list[ValueProfile]:
    if run.probe_results:
        target_holdout = scope == "holdout"
        return [
            result.value_profile
            for result in run.probe_results
            if result.is_holdout is target_holdout
        ]
    if scope == "standard":
        return list(run.record.value_profiles)
    return []


def _pairs_for(
    matching: MatchingResult, *, comparison: str, scope: str
) -> tuple[MatchedRunPair, ...]:
    return tuple(
        pair
        for pair in matching.matched_pairs
        if pair.comparison == comparison and pair.scope == scope
    )


def _pair_runs(
    pair: MatchedRunPair, runs: Iterable[LoadedEvaluationRun]
) -> tuple[LoadedEvaluationRun, LoadedEvaluationRun]:
    by_id = {run.entry.evaluation_entry_id: run for run in runs}
    return by_id[pair.left_run_id], by_id[pair.right_run_id]


def _call_metric(
    metric_name: str,
    input_label: str,
    calculation: Callable[[], MetricValue],
    warnings: list[str],
) -> MetricValue | None:
    try:
        return calculation()
    except (MetricError, ValueError) as exc:
        warnings.append(f"{metric_name}: {input_label}: skipped: {exc}")
        return None


def _metric_result(
    metric_name: str,
    value: float | None,
    unit: str,
    input_ids: list[str],
    detail: Any,
    *,
    valid: bool = True,
) -> MetricResult:
    return MetricResult(
        metric_name=metric_name,
        value=value,
        unit=unit,
        input_ids=input_ids,
        configuration={"detail": detail.model_dump(mode="json")},
        warnings=list(getattr(detail, "warnings", [])),
        valid=valid,
    )


def _evaluate_social_loss_pair_set(
    *,
    pairs: tuple[MatchedRunPair, ...],
    runs: tuple[LoadedEvaluationRun, ...],
    metric_name: str,
    calculation: Callable[[LoadedEvaluationRun, LoadedEvaluationRun], Any],
    metrics: list[MetricResult],
    warnings: list[str],
) -> None:
    if not pairs:
        warnings.append(f"{metric_name}: skipped because no explicit matched evidence_only/evidence_social pair exists")
    for pair in pairs:
        left, right = _pair_runs(pair, runs)
        result = _call_metric(
            metric_name,
            f"{pair.left_run_id},{pair.right_run_id}",
            lambda: calculation(left, right),
            warnings,
        )
        if result is not None:
            metrics.append(
                _metric_result(
                    metric_name,
                    result.social_influence_loss,
                    "normalized_distance_difference",
                    [pair.left_run_id, pair.right_run_id],
                    result,
                )
            )


def _run_audit_metrics(run: LoadedEvaluationRun) -> list[MetricResult]:
    completed_rounds = {
        event["round_id"]
        for event in run.events
        if event.get("event_type") == "round_completed"
    }
    max_rounds = run.record.config.max_rounds
    completeness = len(completed_rounds) / max_rounds
    evidence_consistent, evidence_warnings = _evidence_exposure_consistency(run)
    return [
        MetricResult(
            metric_name="run_completeness",
            value=completeness,
            unit="proportion",
            input_ids=[run.entry.evaluation_entry_id],
            configuration={
                "completed_round_count": len(completed_rounds),
                "configured_round_count": max_rounds,
                "run_status": run.record.status.value,
            },
            valid=run.record.status is RunStatus.COMPLETED and completeness == 1.0,
        ),
        MetricResult(
            metric_name="evidence_exposure_consistency",
            value=1.0 if evidence_consistent else 0.0,
            unit="boolean",
            input_ids=[run.entry.evaluation_entry_id],
            configuration={"recorded_exposure_count": len(run.record.evidence_exposures)},
            warnings=evidence_warnings,
            valid=evidence_consistent,
        ),
        MetricResult(
            metric_name="intervention_count",
            value=float(len(run.record.intervention_decisions)),
            unit="count",
            input_ids=[run.entry.evaluation_entry_id],
            configuration={},
            valid=True,
        ),
    ]


def _evidence_exposure_consistency(run: LoadedEvaluationRun) -> tuple[bool, list[str]]:
    scenario = run.record.scenario
    assert scenario is not None
    known_agents = {agent.agent_id for agent in scenario.agents}
    cards = {card.evidence_id: card for card in scenario.evidence_cards}
    warnings: list[str] = []
    recorded: set[tuple[int, str, str]] = set()
    for exposure in run.record.evidence_exposures:
        card = cards.get(exposure.evidence_id)
        if card is None:
            warnings.append(f"unknown evidence_id {exposure.evidence_id!r} in RunRecord")
            continue
        agents = known_agents if "*" in exposure.exposed_to else set(exposure.exposed_to)
        for agent_id in agents:
            if agent_id not in known_agents:
                warnings.append(f"unknown exposed agent_id {agent_id!r}")
            elif exposure.round_id < card.introduced_round:
                warnings.append(f"evidence {card.evidence_id!r} exposed before introduced_round")
            elif not card.is_visible_to(agent_id):
                warnings.append(f"evidence {card.evidence_id!r} exposed to an invisible agent")
            else:
                recorded.add((exposure.round_id, agent_id, card.evidence_id))

    logged: set[tuple[int, str, str]] = set()
    for event in run.events:
        if event.get("event_type") != "exposure_created":
            continue
        payload = event["payload"]
        agent_id = payload.get("agent_id")
        evidence_ids = payload.get("visible_evidence_ids")
        if not isinstance(agent_id, str) or not isinstance(evidence_ids, list):
            warnings.append("exposure_created event has invalid agent_id or visible_evidence_ids")
            continue
        for evidence_id in evidence_ids:
            if isinstance(evidence_id, str):
                logged.add((event["round_id"], agent_id, evidence_id))
            else:
                warnings.append("exposure_created event contains a non-string evidence ID")
    if recorded != logged:
        warnings.append("RunRecord evidence_exposures do not exactly match exposure_created events")
    return not warnings, warnings


def _optional_log_totals(events: Iterable[dict[str, Any]]) -> tuple[list[float], list[float]]:
    tokens: list[float] = []
    latency: list[float] = []
    for event in events:
        payload = event.get("payload", {})
        for key in ("token_count", "tokens"):
            value = payload.get(key)
            if _is_nonnegative_number(value):
                tokens.append(float(value))
        value = payload.get("latency_ms")
        if _is_nonnegative_number(value):
            latency.append(float(value))
    return tokens, latency


def _is_nonnegative_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _unique_audit(entries: Iterable[InputFileAudit]) -> list[InputFileAudit]:
    by_path = {entry.path: entry for entry in entries}
    return [by_path[path] for path in sorted(by_path)]
