"""Pairwise and MST structural diversity tests using hand-built profiles."""

from __future__ import annotations

import pytest

from evicon.metrics import (
    DimensionMismatchError,
    InsufficientDataError,
    MetricResult,
    MissingPairError,
    PairDistance,
    pairwise_diversity,
    structural_diversity,
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


def test_pairwise_diversity_identical_extreme_and_three_agent_mean() -> None:
    same = pairwise_diversity([profile("a", ["v"], [0.4]), profile("b", ["v"], [0.4])])
    extreme = pairwise_diversity([profile("a", ["v"], [0.0]), profile("b", ["v"], [1.0])])
    three = pairwise_diversity(
        [profile("a", ["v"], [0.0]), profile("b", ["v"], [0.5]), profile("c", ["v"], [1.0])]
    )

    assert same.mean_distance == 0.0
    assert extreme.mean_distance == 1.0
    assert three.mean_distance == pytest.approx(2 / 3)
    assert three.valid_pair_count == 3


def test_pairwise_diversity_handles_dimension_mismatch_only_when_non_strict() -> None:
    profiles = [
        profile("a", ["fairness"], [0.0]),
        profile("b", ["fairness"], [1.0]),
        profile("c", ["autonomy"], [0.5]),
    ]
    before = [item.model_dump() for item in profiles]

    with pytest.raises(DimensionMismatchError):
        pairwise_diversity(profiles, strict=True)
    result = pairwise_diversity(profiles, strict=False)

    assert result.mean_distance == 1.0
    assert result.valid_pair_count == 1
    assert result.skipped_pair_count == 2
    assert len(result.skipped_pairs) == 2
    assert [item.model_dump() for item in profiles] == before


def test_pairwise_diversity_requires_at_least_two_agents() -> None:
    with pytest.raises(InsufficientDataError):
        pairwise_diversity([profile("a", ["v"], [0.0])])


def test_structural_diversity_two_agents_line_triangle_zero_and_order_independence() -> None:
    two = structural_diversity([PairDistance(agent_a="a", agent_b="b", distance=0.6)], ["b", "a"])
    line = structural_diversity(
        [
            PairDistance(agent_a="a", agent_b="b", distance=0.2),
            PairDistance(agent_a="b", agent_b="c", distance=0.3),
            PairDistance(agent_a="a", agent_b="c", distance=0.5),
        ],
        ["a", "b", "c"],
    )
    triangle = structural_diversity(
        {
            ("a", "b"): 0.3,
            ("a", "c"): 0.4,
            ("b", "c"): 0.5,
        },
        ["c", "a", "b"],
    )
    zero = structural_diversity(
        [
            PairDistance(agent_a="a", agent_b="b", distance=0.0),
            PairDistance(agent_a="a", agent_b="c", distance=0.0),
            PairDistance(agent_a="b", agent_b="c", distance=0.0),
        ],
        ["a", "b", "c"],
    )

    assert two.mst_span == 0.6
    assert line.mst_total_length == pytest.approx(0.5)
    assert line.mst_span == pytest.approx(0.25)
    assert triangle.mst_total_length == pytest.approx(0.7)
    assert triangle.agent_ids == ["a", "b", "c"]
    assert zero.mst_span == 0.0
    assert structural_diversity(
        list(reversed(line.mst_edges)) + [PairDistance(agent_a="a", agent_b="c", distance=0.5)],
        ["c", "b", "a"],
    ).mst_span == line.mst_span


def test_structural_diversity_missing_pairs_have_explicit_strict_and_non_strict_behavior() -> None:
    edges = [
        PairDistance(agent_a="a", agent_b="b", distance=0.2),
        PairDistance(agent_a="b", agent_b="c", distance=0.3),
    ]
    with pytest.raises(MissingPairError):
        structural_diversity(edges, ["a", "b", "c"], strict=True)
    connected_partial = structural_diversity(edges, ["a", "b", "c"], strict=False)
    disconnected = structural_diversity(edges[:1], ["a", "b", "c"], strict=False)

    assert connected_partial.valid
    assert connected_partial.mst_span == pytest.approx(0.25)
    assert connected_partial.warnings
    assert not disconnected.valid
    assert disconnected.mst_span is None


def test_metric_result_is_json_serializable_without_prompts() -> None:
    result = MetricResult(
        metric_name="fixture_metric",
        value=0.5,
        unit="normalized_score",
        input_ids=["agent-a", "agent-b"],
        configuration={"strict": True},
        warnings=[],
        valid=True,
    )

    assert type(result).model_validate_json(result.model_dump_json()) == result
