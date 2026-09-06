"""Public-agent deterministic candidate-provider coverage."""

from __future__ import annotations

import copy

from adaptive_cases import scenario

from evicon.adaptive_control import FakeTargetCandidateProvider
from evicon.conformity_monitor import MonitorResult, MonitorSignals, RiskLevel
from evicon.models import DialogueState


def _state() -> DialogueState:
    value = scenario()
    return DialogueState(
        run_id="candidate-run",
        scenario_id=value.scenario_id,
        current_round=0,
        agents=value.agents,
        turns=[],
        evidence_cards=value.evidence_cards,
        value_profiles=[],
        intervention_budget=0.5,
        metadata={},
    )


def _monitor() -> MonitorResult:
    return MonitorResult(
        run_id="candidate-run",
        scenario_id="adaptive-scenario",
        round_id=0,
        risk_score=0.8,
        risk_level=RiskLevel.HIGH,
        should_intervene=True,
        evidence_supported_update=False,
        signals=MonitorSignals(
            collapse_amount=0.8,
            coverage_loss=0.5,
            minority_loss=0.7,
            evidence_gain=0.0,
            evidence_quality=0.0,
            evidence_support=0.0,
            harm_risk=0.0,
            task_quality=0.8,
        ),
        reasons=[],
        config_version="fixture",
        valid=True,
    )


def test_default_candidates_are_public_deterministic_and_state_preserving() -> None:
    state = _state()
    provider = FakeTargetCandidateProvider()
    before = copy.deepcopy(state.model_dump())

    candidates = provider.build_candidates(state, _monitor(), scenario(), 0)

    assert [candidate.agent_id for candidate in candidates] == ["agent-a", "agent-b"]
    assert all(candidate.safety_risk == 0.1 for candidate in candidates)
    assert state.model_dump() == before


def test_scripted_minority_candidate_is_returned_without_model_calls() -> None:
    from evicon.intervention_executor import TargetCandidate

    candidate = TargetCandidate(
        agent_id="agent-b",
        marginal_coverage_gain=0.9,
        minority_representation_score=0.9,
        safety_risk=0.1,
        estimated_cost=0.1,
        eligible=True,
        reason_codes=["minority"],
    )
    provider = FakeTargetCandidateProvider({0: [candidate]})

    assert provider.build_candidates(_state(), _monitor(), scenario(), 0) == [candidate]
