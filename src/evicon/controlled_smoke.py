"""Run a local, explicit five-plan controlled schedule with FakeLLM only."""

from __future__ import annotations

import json
from pathlib import Path

from .controlled_runner import ControlledProtocolRunner, InterventionSchedule
from .executor import ExecutionContext, InterventionExecutor
from .models import AgentSpec, EvidenceCard, InterventionAction, ProtocolCondition, RunConfig, ScenarioSpec
from .policy import InterventionDecision, PolicyReason
from .replay import validate_replay


def _scenario() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="controlled-smoke-scenario",
        title="Controlled smoke scenario",
        description="A local-only deterministic schedule smoke scenario.",
        initial_context="controlled smoke context",
        agents=[
            AgentSpec(agent_id="agent-a", role="participant"),
            AgentSpec(agent_id="agent-b", role="participant"),
            AgentSpec(agent_id="agent-c", role="participant"),
        ],
        evidence_cards=[
            EvidenceCard(
                evidence_id="evidence-0",
                claim="Synthetic smoke evidence.",
                source="controlled-smoke",
                supports=[],
                contradicts=[],
                introduced_round=0,
                visible_to=["*"],
                reliability=1.0,
            )
        ],
        max_rounds=6,
        metadata={},
    )


def _plan(
    action: InterventionAction,
    *,
    round_id: int,
    targets: list[str] | None = None,
    peer_turn_ids: list[str] | None = None,
):
    peers = peer_turn_ids or []
    context = ExecutionContext(
        run_id="controlled-smoke-run",
        scenario_id="controlled-smoke-scenario",
        round_id=round_id,
        known_agent_ids=["agent-a", "agent-b", "agent-c"],
        visible_evidence_ids=["evidence-0"],
        visible_evidence_introduced_rounds={"evidence-0": 0},
        candidate_peer_turn_ids=peers,
        candidate_peer_turn_rounds={peer_id: round_id - 1 for peer_id in peers},
        current_budget=1.0,
        max_target_count=2,
        intervention_history=[],
        last_action=None,
        metadata={},
    )
    decision = InterventionDecision(
        action=action,
        target_agent_ids=targets or [],
        reason_codes=[PolicyReason.HIGH_RISK],
        risk_score=0.8,
        estimated_cost=0.0 if action is InterventionAction.NO_OP else 0.2,
        round_id=round_id,
        policy_version="controlled-smoke-v1",
        cooldown_until=round_id + 1,
        valid=True,
    )
    result = InterventionExecutor.plan(decision, context)
    if not result.valid:
        raise ValueError(f"invalid smoke plan: {[error.value for error in result.validation_errors]}")
    return result


def smoke_schedule() -> InterventionSchedule:
    """Return one action per round so each effect is observed in a later round."""
    return InterventionSchedule(
        schedule_id="controlled-smoke-schedule",
        run_id="controlled-smoke-run",
        scenario_id="controlled-smoke-scenario",
        plans=[
            _plan(InterventionAction.NO_OP, round_id=0),
            _plan(InterventionAction.BLIND_EVIDENCE_REFLECTION, round_id=1, targets=["agent-a"]),
            _plan(InterventionAction.SOLICIT_DISSENT, round_id=2, targets=["agent-c"]),
            _plan(
                InterventionAction.ADAPTIVE_EXPOSURE,
                round_id=3,
                targets=["agent-a"],
                peer_turn_ids=["turn-r2-agent-b"],
            ),
            _plan(InterventionAction.RESTRUCTURE, round_id=4, targets=["agent-b"]),
        ],
        metadata={"purpose": "local smoke"},
    )


def main() -> int:
    scenario = _scenario()
    config = RunConfig(
        run_id="controlled-smoke-run",
        scenario_id=scenario.scenario_id,
        model_name="fake-llm",
        protocol=ProtocolCondition.EVIDENCE_SOCIAL,
        agent_count=len(scenario.agents),
        max_rounds=scenario.max_rounds,
        seed=29,
        intervention_budget=1.0,
        output_dir="results",
    )
    runner = ControlledProtocolRunner(
        config,
        scenario=scenario,
        intervention_schedule=smoke_schedule(),
    )
    record = runner.run()
    output_directory = Path(config.output_dir) / config.run_id
    events = [
        json.loads(line)
        for line in (output_directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    intervention_events = [
        event
        for event in events
        if "intervention" in event["event_type"] or event["event_type"].endswith("_applied")
    ]
    round_visibility = {
        str(round_id): {
            turn.speaker_id: turn.visible_peer_turn_ids
            for turn in record.turns
            if turn.round_id == round_id
        }
        for round_id in range(config.max_rounds)
    }
    round_order = {
        str(round_id): [turn.speaker_id for turn in record.turns if turn.round_id == round_id]
        for round_id in range(config.max_rounds)
    }
    replay = validate_replay(output_directory / "events.jsonl", output_directory / "run_record.json")
    print(
        json.dumps(
            {
                "run_id": config.run_id,
                "applied_plan_count": len(runner.applied_interventions),
                "intervention_event_count": len(intervention_events),
                "round_visible_peer_turns": round_visibility,
                "round_turn_order": round_order,
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
