"""Four-round local smoke run for the FakeLLM adaptive control loop."""

from __future__ import annotations

import json
from pathlib import Path

from .adaptive_control import FakeOnlineStateProvider, FakeTargetCandidateProvider
from .adaptive_runner import AdaptiveProtocolRunner
from .conformity_monitor import MonitorConfig
from .events import EventType
from .intervention_executor import TargetCandidate
from .models import AgentSpec, EvidenceCard, ProtocolCondition, RunConfig, ScenarioSpec
from .policy import PolicyConfig
from .replay import validate_replay


def _scenario() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="adaptive-smoke-scenario",
        title="Adaptive smoke scenario",
        description="A local scripted adaptive-control smoke scenario.",
        initial_context="adaptive smoke context",
        agents=[
            AgentSpec(agent_id="agent-a", role="participant"),
            AgentSpec(agent_id="agent-b", role="participant"),
        ],
        evidence_cards=[
            EvidenceCard(
                evidence_id="evidence-0",
                claim="Synthetic adaptive smoke evidence.",
                source="adaptive-smoke",
                supports=[],
                contradicts=[],
                introduced_round=0,
                visible_to=["*"],
                reliability=1.0,
            )
        ],
        max_rounds=4,
        metadata={},
    )


def _monitor_config() -> MonitorConfig:
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
        version="adaptive-smoke-monitor-v1",
    )


def _policy_config() -> PolicyConfig:
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
        version="adaptive-smoke-policy-v1",
    )


def main() -> int:
    scenario = _scenario()
    config = RunConfig(
        run_id="adaptive-smoke-run",
        scenario_id=scenario.scenario_id,
        model_name="fake-llm",
        protocol=ProtocolCondition.EVIDENCE_SOCIAL,
        agent_count=2,
        max_rounds=4,
        seed=31,
        intervention_budget=0.5,
        output_dir="results",
    )
    scripted = {
        0: {
            "current_diversity": 0.6,
            "previous_diversity": 0.6,
            "current_coverage": 0.8,
            "previous_coverage": 0.8,
            "minority_loss": 0.0,
            "evidence_gain": 0.0,
            "evidence_quality": 0.0,
            "harm_risk": 0.0,
            "task_quality": 0.8,
        },
        1: {
            "current_diversity": 0.1,
            "previous_diversity": 0.9,
            "current_coverage": 0.2,
            "previous_coverage": 0.9,
            "minority_loss": 0.8,
            "evidence_gain": 0.0,
            "evidence_quality": 0.0,
            "harm_risk": 0.0,
            "task_quality": 0.8,
        },
        2: {
            "current_diversity": 0.1,
            "previous_diversity": 0.9,
            "current_coverage": 0.2,
            "previous_coverage": 0.9,
            "minority_loss": 0.8,
            "evidence_gain": 0.0,
            "evidence_quality": 0.0,
            "harm_risk": 0.0,
            "task_quality": 0.8,
        },
        3: {
            "current_diversity": 0.1,
            "previous_diversity": 0.9,
            "current_coverage": 0.2,
            "previous_coverage": 0.9,
            "minority_loss": 0.8,
            "evidence_gain": 0.0,
            "evidence_quality": 0.0,
            "harm_risk": 0.0,
            "task_quality": 0.8,
        },
    }
    candidates = {
        round_id: [
            TargetCandidate(
                agent_id="agent-b",
                marginal_coverage_gain=0.9,
                minority_representation_score=0.9,
                safety_risk=0.1,
                estimated_cost=0.1,
                eligible=True,
                reason_codes=["scripted_minority_candidate"],
            )
        ]
        for round_id in range(4)
    }
    runner = AdaptiveProtocolRunner(
        config,
        scenario=scenario,
        online_state_provider=FakeOnlineStateProvider(scripted),
        target_candidate_provider=FakeTargetCandidateProvider(candidates),
        monitor_config=_monitor_config(),
        policy_config=_policy_config(),
    )
    runner.run()
    output_directory = Path(config.output_dir) / config.run_id
    events = [
        json.loads(line)
        for line in (output_directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    monitor_by_round = {
        event["round_id"]: event["payload"]
        for event in events
        if event["event_type"] == EventType.MONITOR_EVALUATED.value
    }
    policy_by_round = {
        event["round_id"]: event["payload"]
        for event in events
        if event["event_type"] == EventType.POLICY_DECIDED.value
    }
    replay = validate_replay(output_directory / "events.jsonl", output_directory / "run_record.json")
    print(
        json.dumps(
            {
                "rounds": {
                    str(round_id): {
                        "risk_score": monitor_by_round[round_id]["risk_score"],
                        "policy_action": policy_by_round[round_id]["action"],
                        "scheduled": any(
                            event["event_type"] == EventType.INTERVENTION_PLAN_SCHEDULED.value
                            and event["round_id"] == round_id
                            for event in events
                        ),
                        "applied_next_round": any(
                            event["event_type"] == EventType.INTERVENTION_PLAN_APPLIED.value
                            and event["round_id"] == round_id
                            for event in events
                        ),
                    }
                    for round_id in range(config.max_rounds)
                },
                "remaining_budget": runner.controller_state.remaining_budget,
                "spent_budget": runner.controller_state.spent_budget,
                "reserved_budget": runner.controller_state.reserved_budget,
                "cooldown_until": runner.controller_state.cooldown_until,
                "replay_validation": replay.validation,
                "output_directory": str(output_directory),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
