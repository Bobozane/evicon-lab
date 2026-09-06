"""Four-round local smoke run for the FakeLLM adaptive control loop."""

from __future__ import annotations

import json
from pathlib import Path

from .adaptive_control import FakeOnlineStateProvider, FakeTargetCandidateProvider
from .adaptive_runner import AdaptiveProtocolRunner
from .adaptive_smoke_previous import _monitor_config, _policy_config, _scenario
from .events import EventType
from .intervention_executor import TargetCandidate
from .models import ProtocolCondition, RunConfig
from .replay import validate_replay


_INTERVENTION_EVENTS = {
    EventType.INTERVENTION_PLAN_CREATED.value,
    EventType.INTERVENTION_PLAN_SCHEDULED.value,
    EventType.INTERVENTION_PLAN_APPLIED.value,
    EventType.INTERVENTION_PLAN_REJECTED.value,
    EventType.BUDGET_RESERVED.value,
    EventType.BUDGET_SPENT.value,
    EventType.BUDGET_RELEASED.value,
    EventType.COOLDOWN_UPDATED.value,
}


def _signals() -> dict[int, dict[str, float]]:
    stable = {
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
    high_risk = {
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
    return {0: stable, 1: high_risk, 2: high_risk, 3: high_risk}


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
        for round_id in range(config.max_rounds)
    }
    runner = AdaptiveProtocolRunner(
        config,
        scenario=scenario,
        online_state_provider=FakeOnlineStateProvider(_signals()),
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
                "intervention_event_count": sum(
                    event["event_type"] in _INTERVENTION_EVENTS for event in events
                ),
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
