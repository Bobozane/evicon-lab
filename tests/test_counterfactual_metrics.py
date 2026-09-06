"""Matched-condition social-influence contrast tests without experimental claims."""

from __future__ import annotations

import pytest

from evicon.metrics import (
    DimensionMismatchError,
    holdout_social_influence_loss,
    social_influence_loss,
)
from evicon.models import ValueProfile


def profile(agent_id: str, score: float, *, dimension: str = "fairness") -> ValueProfile:
    return ValueProfile(
        agent_id=agent_id,
        round_id=0,
        dimensions=[dimension],
        scores=[score],
        source="fixture",
        probe_id=None,
    )


def test_social_influence_loss_preserves_positive_and_negative_directions() -> None:
    lower_social = social_influence_loss(
        [profile("a", 0.0), profile("b", 1.0)],
        [profile("a", 0.5), profile("b", 0.5)],
    )
    higher_social = social_influence_loss(
        [profile("a", 0.4), profile("b", 0.6)],
        [profile("a", 0.0), profile("b", 1.0)],
    )

    assert lower_social.evidence_only_diversity == 1.0
    assert lower_social.evidence_social_diversity == 0.0
    assert lower_social.social_influence_loss == 1.0
    assert lower_social.interpretation == "social_diversity_lower"
    assert higher_social.social_influence_loss == pytest.approx(-0.8)
    assert higher_social.interpretation == "social_diversity_higher"
    assert type(higher_social).model_validate_json(higher_social.model_dump_json()) == higher_social


def test_holdout_social_influence_loss_marks_holdout_scope() -> None:
    result = holdout_social_influence_loss(
        [profile("a", 0.0), profile("b", 1.0)],
        [profile("a", 0.0), profile("b", 1.0)],
    )

    assert result.scope == "holdout"
    assert result.social_influence_loss == 0.0
    assert result.interpretation == "no_diversity_difference"


def test_social_influence_loss_requires_matched_agents_and_dimensions() -> None:
    with pytest.raises(DimensionMismatchError, match="same agents"):
        social_influence_loss(
            [profile("a", 0.0), profile("b", 1.0)],
            [profile("a", 0.0), profile("c", 1.0)],
        )
    with pytest.raises(DimensionMismatchError, match="value dimensions"):
        social_influence_loss(
            [profile("a", 0.0), profile("b", 1.0)],
            [profile("a", 0.0, dimension="autonomy"), profile("b", 1.0, dimension="autonomy")],
        )
