"""Structured, no-side-effect policy tests for InterventionPolicy."""

from __future__ import annotations

import copy
import socket

import pytest
from pydantic import ValidationError

from evicon.conformity_monitor import MonitorReason, MonitorResult, MonitorSignals, RiskLevel
from evicon.models import AgentSpec, DialogueState, InterventionAction
from evicon.policy import (
    InterventionDecision,
    InterventionHistoryEntry,
    InterventionPolicy,
    PolicyConfig,
    PolicyInput,
    PolicyReason,
    TargetCandidate,
)


def config(**overrides: object) -> PolicyConfig:
    values: dict[str, object] = {
        "max_targets": 2,
        "min_marginal_gain": 0.2,
        "max_target_safety_risk": 0.5,
        "request_evidence_cost": 0.1,
        "blind_reflection_cost": 0.2,
        "solicit_dissent_cost": 0.3,
        "adaptive_exposure_cost": 0.4,
        "minority_report_cost": 0.4,
        "restructure_cost": 0.8,
        "high_risk_restructure_threshold": 0.7,
        "cooldown_rounds": 2,
        "restructure_failure_threshold": 2,
        "version": "test-policy-v1",
    }
    values.update(overrides)
    return PolicyConfig.model_validate(values)


def monitor_result(
    *,
    should_intervene: bool = True,
    evidence_supported_update: bool = False,
    risk_score: float = 0.9,
    risk_level: RiskLevel = RiskLevel.HIGH,
    minority_loss: float = 0.0,
    harm_risk: float = 0.0,
    reasons: list[MonitorReason] | None = None,
) -> MonitorResult:
    return MonitorResult(
        run_id="policy-run",
        scenario_id="policy-scenario",
        round_id=5,
        risk_score=risk_score,
        risk_level=risk_level,
        should_intervene=should_intervene,
        evidence_supported_update=evidence_supported_update,
        signals=MonitorSignals(
            collapse_amount=0.8 if should_intervene else 0.0,
            coverage_loss=0.2 if should_intervene else 0.0,
            minority_loss=minority_loss,
            evidence_gain=0.0,
            evidence_quality=0.0,
            evidence_support=0.0,
            harm_risk=harm_risk,
            task_quality=0.8,
        ),
        reasons=reasons or [],
        config_version="monitor-v1",
        valid=True,
    )


def candidate(agent_id: str = "agent-a", **overrides: object) -> TargetCandidate:
    values: dict[str, object] = {
        "agent_id": agent_id,
        "marginal_coverage_gain": 0.8,
        "minority_representation_score": 0.7,
        "safety_risk": 0.1,
        "estimated_cost": 0.2,
        "eligible": True,
        "reason_codes": ["fixture"],
    }
    values.update(overrides)
    return TargetCandidate.model_validate(values)


def policy_input(**overrides: object) -> PolicyInput:
    values: dict[str, object] = {
        "run_id": "policy-run",
        "scenario_id": "policy-scenario",
        "round_id": 5,
        "monitor_result": monitor_result(),
        "target_candidates": [candidate()],
        "remaining_budget": 1.0,
        "last_intervention_round": None,
        "cooldown_rounds": 2,
        "intervention_history": [],
        "metadata": {"fixture": True},
    }
    values.update(overrides)
    return PolicyInput.model_validate(values)


def test_low_risk_returns_zero_cost_no_op() -> None:
    decision = InterventionPolicy.decide(
        policy_input(
            monitor_result=monitor_result(
                should_intervene=False,
                risk_score=0.1,
                risk_level=RiskLevel.LOW,
            )
        ),
        config(),
    )

    assert decision.action is InterventionAction.NO_OP
    assert decision.estimated_cost == 0.0
    assert decision.reason_codes == [PolicyReason.RISK_BELOW_THRESHOLD]


def test_evidence_supported_update_returns_no_op() -> None:
    decision = InterventionPolicy.decide(
        policy_input(
            monitor_result=monitor_result(
                should_intervene=False,
                evidence_supported_update=True,
                risk_score=0.8,
            )
        ),
        config(),
    )

    assert decision.action is InterventionAction.NO_OP
    assert decision.reason_codes == [PolicyReason.EVIDENCE_SUPPORTED_UPDATE]


def test_exhausted_budget_returns_no_op() -> None:
    decision = InterventionPolicy.decide(policy_input(remaining_budget=0.0), config())

    assert decision.action is InterventionAction.NO_OP
    assert decision.reason_codes == [PolicyReason.BUDGET_EXHAUSTED]


def test_cooldown_prevents_repeated_non_no_op_action() -> None:
    decision = InterventionPolicy.decide(
        policy_input(last_intervention_round=4, cooldown_rounds=2),
        config(),
    )

    assert decision.action is InterventionAction.NO_OP
    assert decision.reason_codes == [PolicyReason.COOLDOWN_ACTIVE]
    assert decision.cooldown_until == 6


def test_high_risk_without_evidence_selects_request_evidence() -> None:
    decision = InterventionPolicy.decide(
        policy_input(
            monitor_result=monitor_result(reasons=[MonitorReason.INSUFFICIENT_EVIDENCE])
        ),
        config(),
    )

    assert decision.action is InterventionAction.REQUEST_EVIDENCE
    assert decision.target_agent_ids == []
    assert decision.estimated_cost == pytest.approx(0.1)


def test_minority_loss_with_safe_target_selects_solicit_dissent() -> None:
    decision = InterventionPolicy.decide(
        policy_input(
            monitor_result=monitor_result(
                minority_loss=0.8,
                reasons=[MonitorReason.MINORITY_LOSS, MonitorReason.INSUFFICIENT_EVIDENCE],
            )
        ),
        config(),
    )

    assert decision.action is InterventionAction.SOLICIT_DISSENT
    assert decision.target_agent_ids == ["agent-a"]
    assert PolicyReason.MINORITY_LOSS in decision.reason_codes


def test_high_safety_risk_target_is_not_selected() -> None:
    decision = InterventionPolicy.decide(
        policy_input(
            target_candidates=[candidate("unsafe", safety_risk=0.9)],
            monitor_result=monitor_result(reasons=[MonitorReason.INSUFFICIENT_EVIDENCE]),
        ),
        config(),
    )

    assert decision.action is InterventionAction.REQUEST_EVIDENCE
    assert decision.target_agent_ids == []


def test_empty_candidates_produce_a_structured_decision_reason() -> None:
    decision = InterventionPolicy.decide(
        policy_input(
            target_candidates=[],
            monitor_result=monitor_result(reasons=[MonitorReason.INSUFFICIENT_EVIDENCE]),
        ),
        config(),
    )

    assert decision.action is InterventionAction.REQUEST_EVIDENCE
    assert PolicyReason.NO_ELIGIBLE_TARGETS in decision.reason_codes


def test_cost_exceeding_budget_returns_no_op() -> None:
    decision = InterventionPolicy.decide(
        policy_input(
            remaining_budget=0.05,
            monitor_result=monitor_result(reasons=[MonitorReason.INSUFFICIENT_EVIDENCE]),
        ),
        config(),
    )

    assert decision.action is InterventionAction.NO_OP
    assert decision.reason_codes == [PolicyReason.COST_EXCEEDS_BUDGET]


def test_repeated_failed_interventions_can_escalate_once_to_restructure() -> None:
    history = [
        InterventionHistoryEntry(action=InterventionAction.REQUEST_EVIDENCE, round_id=1, succeeded=False),
        InterventionHistoryEntry(action=InterventionAction.SOLICIT_DISSENT, round_id=3, succeeded=False),
    ]
    decision = InterventionPolicy.decide(
        policy_input(intervention_history=history),
        config(),
    )

    assert decision.action is InterventionAction.RESTRUCTURE
    assert PolicyReason.REPEATED_INTERVENTION_FAILURES in decision.reason_codes
    assert decision.cooldown_until == 7


def test_restructure_is_not_repeated_indefinitely() -> None:
    history = [
        InterventionHistoryEntry(action=InterventionAction.RESTRUCTURE, round_id=3, succeeded=False),
        InterventionHistoryEntry(action=InterventionAction.SOLICIT_DISSENT, round_id=4, succeeded=False),
    ]
    decision = InterventionPolicy.decide(policy_input(intervention_history=history), config())

    assert decision.action is InterventionAction.SOLICIT_DISSENT
    assert PolicyReason.RESTRUCTURE_LIMIT_REACHED not in decision.reason_codes


def test_same_input_and_config_produce_an_identical_json_serializable_decision() -> None:
    policy_case = policy_input(monitor_result=monitor_result(reasons=[MonitorReason.INSUFFICIENT_EVIDENCE]))
    first = InterventionPolicy.decide(policy_case, config())
    second = InterventionPolicy.decide(policy_case, config())

    assert first == second
    assert InterventionDecision.model_validate_json(first.model_dump_json()) == first


def test_policy_rejects_misaligned_monitor_and_hidden_input_fields() -> None:
    values = policy_input().model_dump()
    values["run_id"] = "different-run"
    with pytest.raises(ValidationError, match="monitor_result"):
        PolicyInput.model_validate(values)

    values = policy_input().model_dump()
    values["hidden_probe_response"] = "private"
    with pytest.raises(ValidationError, match="hidden_probe_response"):
        PolicyInput.model_validate(values)


def test_policy_does_not_modify_input_or_dialogue_state() -> None:
    policy_case = policy_input(monitor_result=monitor_result(reasons=[MonitorReason.INSUFFICIENT_EVIDENCE]))
    state = DialogueState(
        run_id="state-run",
        scenario_id="state-scenario",
        current_round=0,
        agents=[AgentSpec(agent_id="state-agent", role="participant")],
        turns=[],
        evidence_cards=[],
        value_profiles=[],
        intervention_budget=1.0,
        metadata={},
    )
    before_input = copy.deepcopy(policy_case.model_dump())
    before_state = copy.deepcopy(state.model_dump())

    InterventionPolicy.decide(policy_case, config())

    assert policy_case.model_dump() == before_input
    assert state.model_dump() == before_state


def test_policy_generates_no_prompt_and_opens_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("InterventionPolicy attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", reject_connection)
    decision = InterventionPolicy.decide(policy_input(), config())

    assert "prompt" not in InterventionDecision.model_fields
    assert "prompt" not in decision.model_dump_json()
