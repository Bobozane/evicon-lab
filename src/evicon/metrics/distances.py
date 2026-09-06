"""Pure normalized Euclidean distances for answers and value profiles."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from ..models import ValueProfile
from .errors import DimensionMismatchError, InsufficientDataError, MetricError
from .models import AnswerDistanceResult


def normalized_euclidean_distance(
    vector_a: Sequence[float],
    vector_b: Sequence[float],
    *,
    max_difference: float | Sequence[float] = 1.0,
) -> float:
    """Return Euclidean distance normalized by declared per-dimension maxima."""
    if not vector_a or not vector_b:
        raise InsufficientDataError("normalized_euclidean_distance requires non-empty vectors")
    if len(vector_a) != len(vector_b):
        raise DimensionMismatchError("vectors must contain the same number of dimensions")
    maxima = _resolve_maxima(max_difference, len(vector_a))
    squared_difference = 0.0
    squared_maximum = 0.0
    for index, (left, right, maximum) in enumerate(zip(vector_a, vector_b, maxima, strict=True)):
        left_value = _finite_float(left, f"vector_a[{index}]")
        right_value = _finite_float(right, f"vector_b[{index}]")
        difference = abs(left_value - right_value)
        if difference > maximum:
            raise MetricError(
                f"absolute difference at dimension {index} exceeds max_difference {maximum}"
            )
        squared_difference += difference * difference
        squared_maximum += maximum * maximum
    return math.sqrt(squared_difference) / math.sqrt(squared_maximum)


def answer_distance(
    agent_a: Mapping[str, int],
    agent_b: Mapping[str, int],
    reference: Mapping[str, int],
) -> AnswerDistanceResult:
    """Reproduce the baseline overlap-only answer distance without defaulting missing data to zero."""
    common_questions = sorted(set(agent_a) & set(agent_b) & set(reference))
    if not common_questions:
        raise InsufficientDataError("answer_distance requires at least one shared referenced question")

    squared_difference = 0.0
    squared_maximum = 0.0
    for question_id in common_questions:
        option_count = reference[question_id]
        if type(option_count) is not int or option_count < 2:
            raise MetricError(f"reference option_count for {question_id!r} must be an integer >= 2")
        answer_a = _answer_index(agent_a[question_id], question_id, option_count)
        answer_b = _answer_index(agent_b[question_id], question_id, option_count)
        maximum = option_count - 1
        squared_difference += (answer_a - answer_b) ** 2
        squared_maximum += maximum**2
    return AnswerDistanceResult(
        distance=math.sqrt(squared_difference) / math.sqrt(squared_maximum),
        valid_question_count=len(common_questions),
    )


def profile_distance(profile_a: ValueProfile, profile_b: ValueProfile) -> float:
    """Align profiles by dimension name in profile A's declared stable order."""
    dimensions_a = list(profile_a.dimensions)
    dimensions_b = list(profile_b.dimensions)
    if set(dimensions_a) != set(dimensions_b):
        missing_from_b = sorted(set(dimensions_a) - set(dimensions_b))
        missing_from_a = sorted(set(dimensions_b) - set(dimensions_a))
        raise DimensionMismatchError(
            "profiles must name the same dimensions; "
            f"missing_from_b={missing_from_b}, missing_from_a={missing_from_a}"
        )
    scores_b = profile_b.dimension_scores
    aligned_b = [scores_b[dimension] for dimension in dimensions_a]
    return normalized_euclidean_distance(profile_a.scores, aligned_b)


def _resolve_maxima(max_difference: float | Sequence[float], length: int) -> list[float]:
    if isinstance(max_difference, (int, float)) and not isinstance(max_difference, bool):
        maximum = _finite_float(max_difference, "max_difference")
        if maximum <= 0.0:
            raise MetricError("max_difference must be positive")
        return [maximum] * length
    if not isinstance(max_difference, Sequence) or isinstance(max_difference, (str, bytes)):
        raise MetricError("max_difference must be a positive float or a sequence of positive floats")
    if len(max_difference) != length:
        raise DimensionMismatchError("per-dimension max_difference must align with vectors")
    maxima = [_finite_float(value, "max_difference") for value in max_difference]
    if any(value <= 0.0 for value in maxima):
        raise MetricError("all max_difference values must be positive")
    return maxima


def _finite_float(value: float, field_name: str) -> float:
    if isinstance(value, bool):
        raise MetricError(f"{field_name} must be a finite float")
    converted = float(value)
    if not math.isfinite(converted):
        raise MetricError(f"{field_name} must be a finite float")
    return converted


def _answer_index(value: int, question_id: str, option_count: int) -> int:
    if type(value) is not int or value < 0 or value >= option_count:
        raise MetricError(
            f"answer for {question_id!r} must be a zero-based integer in [0, {option_count - 1}]"
        )
    return value
