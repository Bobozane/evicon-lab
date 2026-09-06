"""Shared local fixtures for controlled-runner tests; not a collected test module."""

from __future__ import annotations

from pathlib import Path

from evicon.executor import ExecutionContext, InterventionExecutor
from evicon.models import AgentSpec, EvidenceCard, InterventionAction, ProtocolCondition, RunConfig, ScenarioSpec
from evicon.policy import InterventionDecision, PolicyReason


def scenario(max_rounds: int = 3) -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="controlled-scenario",
        title="Controlled local scenario",
        description="Synthetic scenario for deterministic controlled-runner tests.",
        initial_context="controlled test context",
        agents=[
            AgentSpec(agent_id="agent-a", role="participant"),
            AgentSpec(agent_id="agent-b", role="participant"),
            AgentSpec(agent_id="agent-c", role="participant"),
        ],
        evidence_cards=[
            EvidenceCard(
                evidence_id="evidence-0",
                claim="Synthetic visible evidence.",
                source="controlled-test",
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


def config(tmp_path: Path, run_id: str, *, max_rounds: int = 3) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        scenario_id="controlled-scenario",
        model_name="fake-llm",
        protocol=ProtocolCondition.EVIDENCE_SOCIAL,
        agent_count=3,
        max_rounds=max_rounds,
        seed=17,
        intervention_budget=1.0,
        output_dir=str(tmp_path),
    )


def execution_context(
    *,
    round_id: int,
    peer_turn_ids: list[str] | None = None,
) -> ExecutionContext:
    peers = peer_turn_ids or []
    return ExecutionContext(
        run_id="controlled-run",
        scenario_id="controlled-scenario",
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


def decision(
    action: InterventionAction,
    *,
    round_id: int,
    targets: list[str] | None = None,
) -> InterventionDecision:
    return InterventionDecision(
        action=action,
        target_agent_ids=targets or [],
        reason_codes=[PolicyReason.HIGH_RISK],
        risk_score=0.8,
        estimated_cost=0.0 if action is InterventionAction.NO_OP else 0.2,
        round_id=round_id,
        policy_version="controlled-test-v1",
        cooldown_until=round_id + 1,
        valid=True,
    )


def plan(
    action: InterventionAction,
    *,
    round_id: int,
    targets: list[str] | None = None,
    peer_turn_ids: list[str] | None = None,
):
    generated = InterventionExecutor.plan(
        decision(action, round_id=round_id, targets=targets),
        execution_context(round_id=round_id, peer_turn_ids=peer_turn_ids),
    )
    assert generated.valid
    return generated.model_copy(update={"run_id": "controlled-run"})
