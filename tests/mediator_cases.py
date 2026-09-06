"""Shared public-only fixtures for mediator prompt and parser tests."""

from __future__ import annotations

from controlled_cases import plan as controlled_plan

from evicon.executor import InterventionPlan
from evicon.mediator_prompts import PromptContext
from evicon.models import InterventionAction


def plan_for(action: InterventionAction) -> InterventionPlan:
    targets = [] if action is InterventionAction.NO_OP else ["agent-a"]
    peers = ["turn-r0-agent-b"] if action is InterventionAction.ADAPTIVE_EXPOSURE else None
    return controlled_plan(
        action,
        round_id=1,
        targets=targets,
        peer_turn_ids=peers,
    )


def prompt_context_for(plan: InterventionPlan) -> PromptContext:
    return PromptContext(
        run_id=plan.run_id,
        scenario_id=plan.scenario_id,
        round_id=plan.round_id,
        action=plan.action,
        target_agent_ids=list(plan.target_agent_ids),
        public_dialogue_summary="Agents discussed a public scenario and one visible evidence item.",
        visible_evidence_ids=["evidence-0"],
        requested_evidence_ids=["evidence-0"],
        preservation_constraints=[
            "Use only public evidence.",
            "Preserve a genuine remaining position.",
            "Separate factual claims from value preferences.",
        ],
        metadata={"fixture": "public_only"},
    )
