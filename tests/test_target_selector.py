"""Deterministic filtering and ordering coverage for TargetSelector."""

from __future__ import annotations

import copy

import pytest

from evicon.policy import PolicyConfig, TargetCandidate, TargetSelectionReason, select_targets


def config() -> PolicyConfig:
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
        version="test-policy-v1",
    )


def candidate(agent_id: str, **overrides: object) -> TargetCandidate:
    values: dict[str, object] = {
        "agent_id": agent_id,
        "marginal_coverage_gain": 0.5,
        "minority_representation_score": 0.5,
        "safety_risk": 0.1,
        "estimated_cost": 0.2,
        "eligible": True,
        "reason_codes": ["fixture"],
    }
    values.update(overrides)
    return TargetCandidate.model_validate(values)


def test_selector_orders_candidates_stably_by_declared_priority() -> None:
    result = select_targets(
        [
            candidate("gamma", marginal_coverage_gain=0.7, minority_representation_score=0.2),
            candidate("beta", marginal_coverage_gain=0.7, minority_representation_score=0.8),
            candidate("alpha", marginal_coverage_gain=0.7, minority_representation_score=0.8),
        ],
        config(),
        max_targets=3,
    )

    assert result.target_agent_ids == ["alpha", "beta", "gamma"]
    assert result.reason_codes == [TargetSelectionReason.TARGETS_SELECTED]


def test_selector_excludes_ineligible_low_gain_and_high_safety_risk_candidates() -> None:
    result = select_targets(
        [
            candidate("ineligible", eligible=False),
            candidate("low-gain", marginal_coverage_gain=0.19),
            candidate("unsafe", safety_risk=0.51),
            candidate("safe"),
        ],
        config(),
    )

    assert result.target_agent_ids == ["safe"]


def test_selector_truncates_after_the_requested_maximum() -> None:
    result = select_targets(
        [candidate("a", marginal_coverage_gain=0.9), candidate("b", marginal_coverage_gain=0.8)],
        config(),
        max_targets=1,
    )

    assert result.target_agent_ids == ["a"]


def test_empty_candidates_return_a_structured_reason() -> None:
    result = select_targets([], config(), max_targets=2)

    assert result.target_agent_ids == []
    assert result.reason_codes == [TargetSelectionReason.EMPTY_CANDIDATES]


def test_selector_reports_when_no_candidate_is_eligible() -> None:
    result = select_targets([candidate("a", eligible=False)], config(), max_targets=2)

    assert result.reason_codes == [TargetSelectionReason.NO_ELIGIBLE_CANDIDATES]


def test_selector_does_not_modify_candidates() -> None:
    candidates = [candidate("a"), candidate("b")]
    before = copy.deepcopy([item.model_dump() for item in candidates])

    select_targets(candidates, config(), max_targets=2)

    assert [item.model_dump() for item in candidates] == before


def test_selector_rejects_a_negative_maximum() -> None:
    with pytest.raises(ValueError, match="max_targets"):
        select_targets([candidate("a")], config(), max_targets=-1)
