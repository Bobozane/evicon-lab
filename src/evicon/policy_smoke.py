"""Local-only smoke cases for the deterministic InterventionPolicy."""

from __future__ import annotations

import json

from .conformity_monitor import MonitorReason, MonitorResult, MonitorSignals, RiskLevel
from .models.protocols import InterventionAction
from .policy import (
    InterventionHistoryEntry,
    InterventionPolicy,
    PolicyConfig,
    PolicyInput,
    TargetCandidate,
)


def smoke_config() -> PolicyConfig:
    """Return demonstration-only policy parameters, not calibrated thresholds."""
    return PolicyConfig(
        max_targets=1,
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
        version="policy-smoke-v1",
    )


def _monitor_result(
    *,
    should_intervene: bool,
    risk_score: float,
    risk_level: RiskLevel,
    minority_loss: float = 0.0,
    reasons: list[MonitorReason] | None = None,
) -> MonitorResult:
    return MonitorResult(
        run_id="policy-smoke",
        scenario_id="policy-smoke-scenario",
        round_id=5,
        risk_score=risk_score,
        risk_level=risk_level,
        should_intervene=should_intervene,
        evidence_supported_update=False,
        signals=MonitorSignals(
            collapse_amount=0.8 if should_intervene else 0.0,
            coverage_loss=0.4 if should_intervene else 0.0,
            minority_loss=minority_loss,
            evidence_gain=0.0,
            evidence_quality=0.0,
            evidence_support=0.0,
            harm_risk=0.0,
            task_quality=0.8,
        ),
        reasons=reasons or [],
        config_version="monitor-smoke-v1",
        valid=True,
    )


def _candidate() -> TargetCandidate:
    return TargetCandidate(
        agent_id="agent-minority",
        marginal_coverage_gain=0.8,
        minority_representation_score=0.9,
        safety_risk=0.1,
        estimated_cost=0.2,
        eligible=True,
        reason_codes=["smoke_candidate"],
    )


def smoke_cases() -> list[tuple[str, PolicyInput]]:
    """Return three independent inputs without reading run artifacts or prompts."""
    common = {
        "run_id": "policy-smoke",
        "scenario_id": "policy-smoke-scenario",
        "round_id": 5,
        "remaining_budget": 1.0,
        "last_intervention_round": None,
        "cooldown_rounds": 2,
        "metadata": {"purpose": "local smoke"},
    }
    return [
        (
            "stable",
            PolicyInput(
                **common,
                monitor_result=_monitor_result(
                    should_intervene=False,
                    risk_score=0.1,
                    risk_level=RiskLevel.LOW,
                ),
                target_candidates=[],
                intervention_history=[],
            ),
        ),
        (
            "social-collapse-without-evidence",
            PolicyInput(
                **common,
                monitor_result=_monitor_result(
                    should_intervene=True,
                    risk_score=0.9,
                    risk_level=RiskLevel.HIGH,
                    minority_loss=0.8,
                    reasons=[MonitorReason.DIVERSITY_COLLAPSE, MonitorReason.INSUFFICIENT_EVIDENCE],
                ),
                target_candidates=[_candidate()],
                intervention_history=[],
            ),
        ),
        (
            "repeated-failure-restructure",
            PolicyInput(
                **common,
                monitor_result=_monitor_result(
                    should_intervene=True,
                    risk_score=0.9,
                    risk_level=RiskLevel.HIGH,
                    reasons=[MonitorReason.DIVERSITY_COLLAPSE],
                ),
                target_candidates=[_candidate()],
                intervention_history=[
                    InterventionHistoryEntry(
                        action=InterventionAction.REQUEST_EVIDENCE,
                        round_id=1,
                        succeeded=False,
                    ),
                    InterventionHistoryEntry(
                        action=InterventionAction.SOLICIT_DISSENT,
                        round_id=3,
                        succeeded=False,
                    ),
                ],
            ),
        ),
    ]


def main() -> int:
    config = smoke_config()
    for label, policy_input in smoke_cases():
        decision = InterventionPolicy.decide(policy_input, config)
        print(
            json.dumps(
                {
                    "case": label,
                    "action": decision.action.value,
                    "target_agent_ids": decision.target_agent_ids,
                    "risk_score": decision.risk_score,
                    "estimated_cost": decision.estimated_cost,
                    "reason_codes": [reason.value for reason in decision.reason_codes],
                    "cooldown_until": decision.cooldown_until,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
