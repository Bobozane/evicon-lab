"""Boundary reason coverage for deterministic target selection."""

from __future__ import annotations

from evicon.policy import PolicyConfig, TargetCandidate, TargetSelectionReason, select_targets


def _config() -> PolicyConfig:
    return PolicyConfig(
        max_targets=2,
        min_marginal_gain=0.2,
        max_target_safety_risk=0.5,
        request_evidence_cost=0.1,
        blind_reflection_cost=0.2,
        solicit_dissent_cost=0.3,
        adaptive_exposure_cost=0.4,
        minority_report_cost=0.4,
        restructure_cost=0.8,
        high_risk_restructure_threshold=0.7,
        cooldown_rounds=2,
        restructure_failure_threshold=2,
        version="boundary-v1",
    )


def _candidate(agent_id: str, *, gain: float = 0.8, safety: float = 0.1) -> TargetCandidate:
    return TargetCandidate(
        agent_id=agent_id,
        marginal_coverage_gain=gain,
        minority_representation_score=0.5,
        safety_risk=safety,
        estimated_cost=0.2,
        eligible=True,
        reason_codes=["boundary"],
    )


def test_all_low_gain_candidates_report_a_stable_empty_reason() -> None:
    result = select_targets([_candidate("a", gain=0.19)], _config())

    assert result.target_agent_ids == []
    assert result.reason_codes == [TargetSelectionReason.ALL_BELOW_MINIMUM_GAIN]


def test_all_unsafe_candidates_report_a_stable_empty_reason() -> None:
    result = select_targets([_candidate("a", safety=0.51)], _config())

    assert result.target_agent_ids == []
    assert result.reason_codes == [TargetSelectionReason.ALL_ABOVE_SAFETY_LIMIT]
