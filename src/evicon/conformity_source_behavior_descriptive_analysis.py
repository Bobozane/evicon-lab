"""Pure, outcome-safe aggregation for the accepted source-behavior v2 lock.

This module intentionally has no paths, CLI, file I/O, Provider integration,
or report writer. It is exercised only with caller-supplied in-memory audits.
Reading the real audit and producing a report requires a later, separately
authorized execution boundary.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_identification import IdentificationError
from .conformity_source_behavior_analysis_lock import (
    DescriptiveContrastId,
    SourceBehaviorAnalysisLock,
)
from .conformity_source_behavior_qualification import (
    AdoptionDecision,
    SafeBehaviorCaseAudit,
    SharingDecision,
)
from .conformity_source_manipulation import SourceProjection


BinaryOutcome = Literal["adoption_decision", "sharing_decision"]


class PairedDifferenceSummary(BaseModel):
    """One aggregate-only outcome/contrast summary; no case coordinates persist."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: BinaryOutcome
    contrast_id: DescriptiveContrastId
    eligible_scenario_count: int = Field(ge=0, le=4)
    positive_difference_count: int = Field(ge=0, le=4)
    zero_difference_count: int = Field(ge=0, le=4)
    negative_difference_count: int = Field(ge=0, le=4)
    mean_paired_difference: float = Field(ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def counts_cover_eligible_scenarios(self) -> "PairedDifferenceSummary":
        if (
            self.positive_difference_count
            + self.zero_difference_count
            + self.negative_difference_count
            != self.eligible_scenario_count
        ):
            raise ValueError("paired difference counts do not cover eligibility")
        return self


class SourceBehaviorDescriptiveSummary(BaseModel):
    """The only permissible in-memory result shape for the current lock."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis_id: Literal["evicon-conformity-source-behavior-analysis-v2"]
    analysis_version: Literal["conformity_source_behavior_analysis_lock.v2"]
    summary_version: Literal["conformity_source_behavior_descriptive_summary.v2"]
    scenario_count: Literal[4]
    primary_outcome: Literal["adoption_decision"]
    secondary_outcome: Literal["sharing_decision"]
    paired_summaries: tuple[PairedDifferenceSummary, ...]
    fixed_projection_order_confounding_acknowledged: Literal[True] = True
    inferential_tests_run: Literal[False] = False
    p_values_reported: Literal[False] = False
    confidence_intervals_reported: Literal[False] = False
    case_level_values_included: Literal[False] = False
    historical_results_joined: Literal[False] = False
    evaluator_private_truth_loaded: Literal[False] = False
    not_paper_result: Literal[True] = True
    no_causal_conclusion: Literal[True] = True
    no_population_generalization: Literal[True] = True

    @model_validator(mode="after")
    def exact_summary_set(self) -> "SourceBehaviorDescriptiveSummary":
        expected = tuple(
            (outcome, contrast)
            for outcome in ("adoption_decision", "sharing_decision")
            for contrast in DescriptiveContrastId
        )
        observed = tuple(
            (item.outcome, item.contrast_id) for item in self.paired_summaries
        )
        if observed != expected:
            raise ValueError("paired summary set mismatch")
        return self


_EXPECTED_ROOT_COUNTS = {
    SourceProjection.SOURCE_FREE: 0,
    SourceProjection.SAME_ROOT: 1,
    SourceProjection.INDEPENDENT_ROOTS: 2,
}


def _validate_audit_shape(
    audits: tuple[SafeBehaviorCaseAudit, ...],
    plan: SourceBehaviorAnalysisLock,
) -> tuple[str, ...]:
    if len(audits) != plan.case_count:
        raise IdentificationError("source_behavior_analysis_case_count_mismatch")
    if len({item.case_id for item in audits}) != plan.case_count:
        raise IdentificationError("source_behavior_analysis_duplicate_case_id")
    scenarios = tuple(sorted({item.scenario_id for item in audits}))
    if len(scenarios) != plan.scenario_count:
        raise IdentificationError("source_behavior_analysis_scenario_count_mismatch")
    coverage = Counter((item.scenario_id, item.projection) for item in audits)
    if any(count != 1 for count in coverage.values()):
        raise IdentificationError("source_behavior_analysis_duplicate_scenario_projection")
    expected_pairs = {
        (scenario_id, projection)
        for scenario_id in scenarios
        for projection in SourceProjection
    }
    if set(coverage) != expected_pairs:
        raise IdentificationError("source_behavior_analysis_projection_coverage_mismatch")
    if any(
        item.visible_root_count != _EXPECTED_ROOT_COUNTS[item.projection]
        for item in audits
    ):
        raise IdentificationError("source_behavior_analysis_visible_root_count_mismatch")
    return scenarios


def _binary_value(item: SafeBehaviorCaseAudit, outcome: BinaryOutcome) -> int:
    if outcome == "adoption_decision":
        return int(item.adoption_decision is AdoptionDecision.ADOPT_VISIBLE_CLAIM)
    return int(item.sharing_decision is SharingDecision.SHARE_VISIBLE_CONTENT)


def aggregate_safe_case_audits(
    audits: Iterable[SafeBehaviorCaseAudit],
    *,
    plan: SourceBehaviorAnalysisLock,
) -> SourceBehaviorDescriptiveSummary:
    """Aggregate caller-supplied audits without reading or writing any files."""

    observed = tuple(audits)
    scenarios = _validate_audit_shape(observed, plan)
    by_coordinate = {(item.scenario_id, item.projection): item for item in observed}
    summaries: list[PairedDifferenceSummary] = []
    for outcome in ("adoption_decision", "sharing_decision"):
        for contrast in plan.primary_contrasts:
            differences = tuple(
                _binary_value(by_coordinate[(scenario_id, contrast.minuend)], outcome)
                - _binary_value(by_coordinate[(scenario_id, contrast.subtrahend)], outcome)
                for scenario_id in scenarios
            )
            summaries.append(PairedDifferenceSummary(
                outcome=outcome,
                contrast_id=contrast.contrast_id,
                eligible_scenario_count=len(differences),
                positive_difference_count=sum(value > 0 for value in differences),
                zero_difference_count=sum(value == 0 for value in differences),
                negative_difference_count=sum(value < 0 for value in differences),
                mean_paired_difference=sum(differences) / len(differences),
            ))
    return SourceBehaviorDescriptiveSummary(
        analysis_id=plan.analysis_id,
        analysis_version=plan.analysis_version,
        summary_version="conformity_source_behavior_descriptive_summary.v2",
        scenario_count=plan.scenario_count,
        primary_outcome=plan.primary_outcome,
        secondary_outcome=plan.secondary_outcome,
        paired_summaries=tuple(summaries),
    )


def safe_summary_json(summary: SourceBehaviorDescriptiveSummary) -> str:
    """Serialize an already-aggregate-only summary without creating a file."""

    return json.dumps(summary.model_dump(mode="json"), ensure_ascii=True, sort_keys=True)


__all__ = [
    "PairedDifferenceSummary",
    "SourceBehaviorDescriptiveSummary",
    "aggregate_safe_case_audits",
    "safe_summary_json",
]
