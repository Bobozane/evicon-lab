"""Pure coverage, explicitly defined minority retention, and drift tests."""

from __future__ import annotations

import math

import pytest

from evicon.metrics import (
    DimensionMismatchError,
    InsufficientDataError,
    holdout_profile_drift,
    minority_retention,
    value_dimension_coverage,
)
from evicon.models import ValueProfile


def profile(agent_id: str, scores: list[float], *, dimensions: list[str] | None = None) -> ValueProfile:
    return ValueProfile(
        agent_id=agent_id,
        round_id=0,
        dimensions=dimensions or ["fairness", "autonomy"],
        scores=scores,
        source="fixture",
        probe_id=None,
    )


def test_value_dimension_coverage_uses_explicit_threshold_and_descriptive_summaries() -> None:
    result = value_dimension_coverage(
        [profile("a", [0.8, 0.2]), profile("b", [0.2, 0.6])],
        threshold=0.7,
    )

    assert result.expressed_dimensions == ["fairness"]
    assert result.coverage_rate == 0.5
    assert result.dimension_summaries["fairness"].minimum == 0.2
    assert result.dimension_summaries["fairness"].maximum == 0.8
    assert result.dimension_summaries["fairness"].mean == 0.5
    assert result.dimension_summaries["autonomy"].mean == pytest.approx(0.4)
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_minority_retention_uses_explicit_nominated_dimensions_not_majority_inference() -> None:
    initial = [profile("a", [0.2, 0.9]), profile("b", [0.3, 0.7])]
    final = [profile("a", [0.2, 0.1]), profile("b", [0.3, 0.2])]
    before = [entry.model_dump() for entry in initial + final]

    result = minority_retention(
        initial,
        final,
        minority_dimensions=["autonomy"],
        threshold=0.6,
    )

    assert result.initial_minority_dimensions == ["autonomy"]
    assert result.retained_minority_dimensions == []
    assert result.retention_rate == 0.0
    assert result.lost_dimensions == ["autonomy"]
    assert [entry.model_dump() for entry in initial + final] == before


def test_minority_retention_rejects_implicit_or_unexpressed_minority_definitions() -> None:
    profiles = [profile("a", [0.2, 0.2]), profile("b", [0.3, 0.3])]
    with pytest.raises(InsufficientDataError):
        minority_retention(profiles, profiles, minority_dimensions=[], threshold=0.6)
    with pytest.raises(InsufficientDataError, match="no nominated"):
        minority_retention(profiles, profiles, minority_dimensions=["fairness"], threshold=0.6)


def test_holdout_profile_drift_is_matched_per_agent_and_not_normative() -> None:
    initial = [profile("a", [0.0, 0.0]), profile("b", [0.5, 0.5])]
    final = [profile("a", [1.0, 0.0]), profile("b", [0.5, 0.5])]

    result = holdout_profile_drift(initial, final)

    expected = 1.0 / math.sqrt(2)
    assert result.agent_drifts == pytest.approx({"a": expected, "b": 0.0})
    assert result.mean_drift == pytest.approx(expected / 2)
    assert result.max_drift == pytest.approx(expected)


def test_representation_metrics_reject_mismatched_agents_and_dimensions() -> None:
    with pytest.raises(DimensionMismatchError):
        holdout_profile_drift([profile("a", [0.0, 0.0])], [profile("b", [0.0, 0.0])])
    with pytest.raises(DimensionMismatchError):
        value_dimension_coverage(
            [profile("a", [0.0, 0.0]), profile("b", [0.0], dimensions=["fairness"])],
            threshold=0.5,
        )
