"""Pure value-expression, minority-retention, and within-agent drift metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..models import ValueProfile
from .distances import profile_distance
from .errors import DimensionMismatchError, InsufficientDataError, MetricError
from .models import (
    DimensionSummary,
    HoldoutProfileDriftResult,
    MinorityRetentionResult,
    ValueDimensionCoverageResult,
)


def value_dimension_coverage(
    profiles: Sequence[ValueProfile] | ValueProfile,
    *,
    threshold: float,
) -> ValueDimensionCoverageResult:
    """Summarize dimensions with at least one score at an explicit expression threshold."""
    profile_list = _as_profile_list(profiles)
    threshold_value = _threshold(threshold)
    dimensions = _shared_dimensions(profile_list)
    summaries: dict[str, DimensionSummary] = {}
    expressed: list[str] = []
    for dimension in dimensions:
        values = [profile.dimension_scores[dimension] for profile in profile_list]
        summary = DimensionSummary(
            minimum=min(values),
            maximum=max(values),
            mean=sum(values) / len(values),
        )
        summaries[dimension] = summary
        if summary.maximum >= threshold_value:
            expressed.append(dimension)
    return ValueDimensionCoverageResult(
        expressed_dimensions=expressed,
        coverage_rate=len(expressed) / len(dimensions),
        dimension_summaries=summaries,
        threshold=threshold_value,
    )


def minority_retention(
    initial_profiles: Sequence[ValueProfile],
    final_profiles: Sequence[ValueProfile],
    *,
    minority_dimensions: Sequence[str],
    threshold: float,
) -> MinorityRetentionResult:
    """Retain only explicitly nominated, initially expressed dimensions; never infer a majority."""
    initial_list = _as_profile_list(initial_profiles)
    final_list = _as_profile_list(final_profiles)
    _require_same_agent_set(initial_list, final_list)
    dimensions = _shared_dimensions(initial_list)
    if set(dimensions) != set(_shared_dimensions(final_list)):
        raise DimensionMismatchError("initial and final profiles must name the same dimensions")
    threshold_value = _threshold(threshold)
    nominated = _minority_dimensions(minority_dimensions, dimensions)
    initial_coverage = value_dimension_coverage(initial_list, threshold=threshold_value)
    final_coverage = value_dimension_coverage(final_list, threshold=threshold_value)
    initial_minority = [
        dimension for dimension in dimensions if dimension in nominated and dimension in initial_coverage.expressed_dimensions
    ]
    if not initial_minority:
        raise InsufficientDataError("no nominated minority dimensions are expressed in initial profiles")
    retained = [
        dimension
        for dimension in initial_minority
        if dimension in final_coverage.expressed_dimensions
    ]
    lost = [dimension for dimension in initial_minority if dimension not in retained]
    return MinorityRetentionResult(
        initial_minority_dimensions=initial_minority,
        retained_minority_dimensions=retained,
        retention_rate=len(retained) / len(initial_minority),
        lost_dimensions=lost,
        threshold=threshold_value,
    )


def holdout_profile_drift(
    initial_profiles: Sequence[ValueProfile],
    final_profiles: Sequence[ValueProfile],
) -> HoldoutProfileDriftResult:
    """Measure matched within-agent profile changes without assigning a normative direction."""
    initial_list = _as_profile_list(initial_profiles)
    final_list = _as_profile_list(final_profiles)
    _require_same_agent_set(initial_list, final_list)
    initial_by_agent = {profile.agent_id: profile for profile in initial_list}
    final_by_agent = {profile.agent_id: profile for profile in final_list}
    drifts = {
        agent_id: profile_distance(initial_by_agent[agent_id], final_by_agent[agent_id])
        for agent_id in sorted(initial_by_agent)
    }
    values = list(drifts.values())
    return HoldoutProfileDriftResult(
        agent_drifts=drifts,
        mean_drift=sum(values) / len(values),
        max_drift=max(values),
    )


def _as_profile_list(profiles: Sequence[ValueProfile] | ValueProfile) -> list[ValueProfile]:
    profile_list = [profiles] if isinstance(profiles, ValueProfile) else list(profiles)
    if not profile_list:
        raise InsufficientDataError("metric requires at least one ValueProfile")
    profile_by_agent = {profile.agent_id: profile for profile in profile_list}
    if len(profile_by_agent) != len(profile_list):
        raise MetricError("profiles must contain at most one ValueProfile per agent_id")
    return profile_list


def _shared_dimensions(profiles: Sequence[ValueProfile]) -> list[str]:
    dimensions = list(profiles[0].dimensions)
    expected = set(dimensions)
    for profile in profiles[1:]:
        if set(profile.dimensions) != expected:
            raise DimensionMismatchError("profiles must name the same dimensions")
    return dimensions


def _require_same_agent_set(
    initial_profiles: Sequence[ValueProfile], final_profiles: Sequence[ValueProfile]
) -> None:
    initial_agents = {profile.agent_id for profile in initial_profiles}
    final_agents = {profile.agent_id for profile in final_profiles}
    if initial_agents != final_agents:
        raise DimensionMismatchError(
            f"initial and final profiles must name the same agents; "
            f"initial_only={sorted(initial_agents - final_agents)}, "
            f"final_only={sorted(final_agents - initial_agents)}"
        )


def _threshold(value: float) -> float:
    if isinstance(value, bool):
        raise MetricError("threshold must be a finite float in [0, 1]")
    threshold = float(value)
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise MetricError("threshold must be a finite float in [0, 1]")
    return threshold


def _minority_dimensions(values: Sequence[str], dimensions: Sequence[str]) -> set[str]:
    if not values:
        raise InsufficientDataError("minority_dimensions must explicitly name at least one dimension")
    nominated = set(values)
    if any(not isinstance(value, str) or not value.strip() for value in nominated):
        raise MetricError("minority_dimensions must be non-blank strings")
    unknown = nominated - set(dimensions)
    if unknown:
        raise DimensionMismatchError(f"minority_dimensions names unknown dimensions: {sorted(unknown)}")
    return nominated
