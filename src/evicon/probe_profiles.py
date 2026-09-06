"""Pure aggregation from private item responses to one normalized value profile."""

from __future__ import annotations

import math
from collections import defaultdict

from .models import ProbeSet, ValueProbeResponse, ValueProfile


class ProbeProfileError(ValueError):
    """Raised when responses cannot form a complete, valid profile."""


class ProbeProfileBuilder:
    """Namespace for deterministic, offline-only profile aggregation."""

    @staticmethod
    def build(
        probe_set: ProbeSet,
        responses: list[ValueProbeResponse],
        *,
        agent_id: str,
        round_id: int,
        is_holdout: bool,
    ) -> ValueProfile:
        return build_value_profile(
            probe_set,
            responses,
            agent_id=agent_id,
            round_id=round_id,
            is_holdout=is_holdout,
        )


def build_value_profile(
    probe_set: ProbeSet,
    responses: list[ValueProbeResponse],
    *,
    agent_id: str,
    round_id: int,
    is_holdout: bool,
) -> ValueProfile:
    """Aggregate a complete ordered probe subset without comparing agents."""
    expected_items = probe_set.items_for_holdout(is_holdout)
    if not expected_items:
        raise ProbeProfileError("probe set has no items for the requested is_holdout selection")
    expected_ids = {item.probe_id for item in expected_items}
    response_by_id: dict[str, ValueProbeResponse] = {}
    for response in responses:
        if response.agent_id != agent_id or response.round_id != round_id:
            raise ProbeProfileError("responses must match the requested agent_id and round_id")
        if response.probe_id in response_by_id:
            raise ProbeProfileError(f"duplicate response for probe_id {response.probe_id}")
        response_by_id[response.probe_id] = response

    actual_ids = set(response_by_id)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise ProbeProfileError(
            f"responses must exactly cover selected probe items; missing={missing}, unexpected={unexpected}"
        )

    scores_by_dimension: dict[str, list[float]] = defaultdict(list)
    for item in expected_items:
        response = response_by_id[item.probe_id]
        try:
            expected_score = item.normalized_score_for(response.raw_response)
        except ValueError as exc:
            raise ProbeProfileError(str(exc)) from exc
        if not math.isclose(response.normalized_score, expected_score, abs_tol=1e-12):
            raise ProbeProfileError(
                f"normalized_score does not match response_scale for probe_id {item.probe_id}"
            )
        score = 1.0 - expected_score if item.reverse_scored else expected_score
        scores_by_dimension[item.dimension].append(score)

    dimensions = list(probe_set.dimensions)
    missing_dimensions = [dimension for dimension in dimensions if not scores_by_dimension[dimension]]
    if missing_dimensions:
        raise ProbeProfileError(f"selected probe items do not cover dimensions: {missing_dimensions}")
    scores = [sum(scores_by_dimension[dimension]) / len(scores_by_dimension[dimension]) for dimension in dimensions]
    source_kind = "holdout" if is_holdout else "standard"
    return ValueProfile(
        agent_id=agent_id,
        round_id=round_id,
        dimensions=dimensions,
        scores=scores,
        source=f"probe:{probe_set.probe_set_id}:{source_kind}",
        probe_id=probe_set.probe_set_id,
    )
