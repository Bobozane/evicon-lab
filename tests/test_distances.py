"""Hand-calculated tests for pure normalized Euclidean metric primitives."""

from __future__ import annotations

import math

import pytest

from evicon.metrics import (
    DimensionMismatchError,
    InsufficientDataError,
    MetricError,
    answer_distance,
    normalized_euclidean_distance,
    profile_distance,
)
from evicon.models import ValueProfile


def profile(agent_id: str, dimensions: list[str], scores: list[float]) -> ValueProfile:
    return ValueProfile(
        agent_id=agent_id,
        round_id=0,
        dimensions=dimensions,
        scores=scores,
        source="fixture",
        probe_id=None,
    )


def test_normalized_euclidean_distance_known_values() -> None:
    assert normalized_euclidean_distance([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert normalized_euclidean_distance([0.0, 0.0], [1.0, 1.0]) == 1.0
    assert normalized_euclidean_distance([0.0, 0.0], [0.5, 0.5]) == pytest.approx(0.5)
    assert normalized_euclidean_distance([0.0, 0.0], [2.0, 4.0], max_difference=[2.0, 4.0]) == 1.0


@pytest.mark.parametrize(
    ("left", "right", "error"),
    [([], [], InsufficientDataError), ([0.0], [], InsufficientDataError), ([0.0], [0.0, 1.0], DimensionMismatchError)],
)
def test_normalized_euclidean_distance_rejects_missing_or_misaligned_data(
    left: list[float], right: list[float], error: type[Exception]
) -> None:
    with pytest.raises(error):
        normalized_euclidean_distance(left, right)


def test_answer_distance_uses_overlap_and_per_question_option_ranges() -> None:
    result = answer_distance(
        {"q1": 0, "q2": 3, "outside": 0},
        {"q1": 2, "q2": 1, "other": 2},
        {"q1": 3, "q2": 5},
    )

    assert result.valid_question_count == 2
    assert result.distance == pytest.approx(math.sqrt(8) / math.sqrt(20))
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_answer_distance_rejects_no_shared_questions_and_invalid_answers() -> None:
    with pytest.raises(InsufficientDataError):
        answer_distance({"a": 0}, {"b": 0}, {"q": 2})
    with pytest.raises(MetricError, match="zero-based"):
        answer_distance({"q": 2}, {"q": 0}, {"q": 2})


def test_profile_distance_aligns_by_dimension_name_without_cosine_similarity() -> None:
    first = profile("a", ["fairness", "autonomy"], [0.0, 1.0])
    reordered_equal = profile("b", ["autonomy", "fairness"], [1.0, 0.0])
    different = profile("c", ["fairness", "care"], [0.0, 1.0])

    assert profile_distance(first, reordered_equal) == 0.0
    with pytest.raises(DimensionMismatchError):
        profile_distance(first, different)
