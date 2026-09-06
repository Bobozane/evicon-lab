"""Matched-condition diversity contrasts for controlled social-influence estimates."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..models import ValueProfile
from .diversity import pairwise_diversity
from .errors import DimensionMismatchError, MetricError
from .models import SocialInfluenceLossResult


def social_influence_loss(
    evidence_only_profiles: Sequence[ValueProfile],
    evidence_social_profiles: Sequence[ValueProfile],
    *,
    strict: bool = True,
) -> SocialInfluenceLossResult:
    """Return a matched diversity contrast; it is not a proof of causal social influence."""
    return _social_influence_loss(
        evidence_only_profiles,
        evidence_social_profiles,
        strict=strict,
        scope="exposed",
    )


def holdout_social_influence_loss(
    evidence_only_holdout_profiles: Sequence[ValueProfile],
    evidence_social_holdout_profiles: Sequence[ValueProfile],
    *,
    strict: bool = True,
) -> SocialInfluenceLossResult:
    """Apply the same matched contrast to profiles from hidden, unexposed probes."""
    return _social_influence_loss(
        evidence_only_holdout_profiles,
        evidence_social_holdout_profiles,
        strict=strict,
        scope="holdout",
    )


def _social_influence_loss(
    evidence_only_profiles: Sequence[ValueProfile],
    evidence_social_profiles: Sequence[ValueProfile],
    *,
    strict: bool,
    scope: str,
) -> SocialInfluenceLossResult:
    evidence_only_by_agent = _profile_by_agent(evidence_only_profiles, "evidence_only_profiles")
    evidence_social_by_agent = _profile_by_agent(
        evidence_social_profiles, "evidence_social_profiles"
    )
    if set(evidence_only_by_agent) != set(evidence_social_by_agent):
        raise DimensionMismatchError(
            "evidence-only and evidence-social inputs must name the same agents"
        )
    reference_dimensions = set(next(iter(evidence_only_by_agent.values())).dimensions)
    for condition_name, profiles_by_agent in (
        ("evidence-only", evidence_only_by_agent),
        ("evidence-social", evidence_social_by_agent),
    ):
        for agent_id, profile in profiles_by_agent.items():
            if set(profile.dimensions) != reference_dimensions:
                raise DimensionMismatchError(
                    f"{condition_name} profile for {agent_id} does not match shared value dimensions"
                )
    evidence_only = pairwise_diversity(list(evidence_only_by_agent.values()), strict=strict)
    evidence_social = pairwise_diversity(list(evidence_social_by_agent.values()), strict=strict)
    loss = evidence_only.mean_distance - evidence_social.mean_distance
    if math.isclose(loss, 0.0, abs_tol=1e-12):
        interpretation = "no_diversity_difference"
    elif loss > 0.0:
        interpretation = "social_diversity_lower"
    else:
        interpretation = "social_diversity_higher"
    warnings = [
        *(f"evidence_only: {pair.reason}" for pair in evidence_only.skipped_pairs),
        *(f"evidence_social: {pair.reason}" for pair in evidence_social.skipped_pairs),
    ]
    return SocialInfluenceLossResult(
        evidence_only_diversity=evidence_only.mean_distance,
        evidence_social_diversity=evidence_social.mean_distance,
        social_influence_loss=loss,
        interpretation=interpretation,
        scope=scope,
        warnings=warnings,
    )


def _profile_by_agent(
    profiles: Sequence[ValueProfile], field_name: str
) -> dict[str, ValueProfile]:
    if len(profiles) < 2:
        raise MetricError(f"{field_name} must contain at least two agent profiles")
    profile_by_agent = {profile.agent_id: profile for profile in profiles}
    if len(profile_by_agent) != len(profiles):
        raise MetricError(f"{field_name} must not contain duplicate agent_id values")
    return profile_by_agent
