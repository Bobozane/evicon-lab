"""Shared deterministic fixtures for adaptive-control tests."""

from __future__ import annotations

from pathlib import Path

from evicon.adaptive_control import FakeOnlineStateProvider, FakeTargetCandidateProvider
from evicon.conformity_monitor import MonitorConfig
from evicon.intervention_executor import TargetCandidate
from evicon.models import AgentSpec, EvidenceCard, ProtocolCondition, RunConfig, ScenarioSpec
from evicon.policy import PolicyConfig


def scenario(max_rounds: int = 4) -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="adaptive-scenario",
        title="Adaptive local scenario",
        description="Synthetic scenario for deterministic adaptive tests.",
        initial_context="adaptive test context",
        agents=[
            AgentSpec(agent_id="agent-a", role="participant"),
            AgentSpec(agent_id="agent-b", role="participant"),
        ],
        evidence_cards=[
            EvidenceCard(
                evidence_id="evidence-0",
                claim="Synthetic adaptive evidence.",
                source="adaptive-test",
                supports=[],
                contradicts=[],
                introduced_round=0,
                visible_to=["*"],
                reliability=1.0,
            )
        ],
        max_rounds=max_rounds,
        metadata={},
    )


def config(tmp_path: Path, run_id: str, *, budget: float = 0.5, max_rounds: int = 4) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        scenario_id="adaptive-scenario",
        model_name="fake-llm",
        protocol=ProtocolCondition.EVIDENCE_SOCIAL,
        agent_count=2,
        max_rounds=max_rounds,
        seed=37,
        intervention_budget=budget,
        output_dir=str(tmp_path),
    )


def monitor_config() -> MonitorConfig:
    return MonitorConfig(
        collapse_weight=0.7,
        minority_weight=0.6,
        evidence_weight=0.3,
        evidence_quality_weight=0.1,
        harm_weight=0.5,
        collapse_trigger=0.5,
        minority_trigger=0.5,
        evidence_sufficient_threshold=0.7,
        high_risk_threshold=0.5,
        medium_risk_threshold=0.25,
        task_quality_sufficient_threshold=0.6,
        version="adaptive-test-monitor-v1",
    )


def policy_config() -> PolicyConfig:
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
        version="adaptive-test-policy-v1",
    )


def stable_signal() -> dict[str, float]:
    return {
        "current_diversity": 0.6,
        "previous_diversity": 0.6,
        "current_coverage": 0.8,
        "previous_coverage": 0.8,
        "minority_loss": 0.0,
        "evidence_gain": 0.0,
        "evidence_quality": 0.0,
        "harm_risk": 0.0,
        "task_quality": 0.8,
    }


def high_risk_signal() -> dict[str, float]:
    return {
        "current_diversity": 0.1,
        "previous_diversity": 0.9,
        "current_coverage": 0.2,
        "previous_coverage": 0.9,
        "minority_loss": 0.8,
        "evidence_gain": 0.0,
        "evidence_quality": 0.0,
        "harm_risk": 0.0,
        "task_quality": 0.8,
    }


def providers(signals: dict[int, dict[str, float]]):
    candidates = {
        round_id: [
            TargetCandidate(
                agent_id="agent-b",
                marginal_coverage_gain=0.9,
                minority_representation_score=0.9,
                safety_risk=0.1,
                estimated_cost=0.1,
                eligible=True,
                reason_codes=["adaptive_fixture"],
            )
        ]
        for round_id in signals
    }
    return FakeOnlineStateProvider(signals), FakeTargetCandidateProvider(candidates)
