"""Pure prospective application coverage for schedule plans."""

from __future__ import annotations

import copy

from controlled_cases import plan, scenario

from evicon.intervention_application import apply_intervention_plan
from evicon.models import DialogueState, DialogueTurn, InterventionAction, ProtocolCondition


def _state(round_id: int = 1) -> DialogueState:
    current_scenario = scenario(max_rounds=3)
    return DialogueState(
        run_id="controlled-run",
        scenario_id=current_scenario.scenario_id,
        current_round=round_id,
        agents=current_scenario.agents,
        turns=[
            DialogueTurn(
                turn_id="turn-r0-agent-b",
                round_id=0,
                speaker_id="agent-b",
                message="synthetic prior turn",
                visible_to=["*"],
                visible_peer_turn_ids=[],
                visible_evidence_ids=["evidence-0"],
                protocol=ProtocolCondition.EVIDENCE_SOCIAL,
            )
        ],
        evidence_cards=current_scenario.evidence_cards,
        value_profiles=[],
        intervention_budget=1.0,
        metadata={},
    )


def test_no_op_and_minority_report_leave_current_dialogue_unchanged() -> None:
    state = _state()
    before = copy.deepcopy(state.model_dump())
    no_op = apply_intervention_plan(
        state,
        plan(InterventionAction.NO_OP, round_id=1),
        scenario(max_rounds=3),
    )
    minority = apply_intervention_plan(
        state,
        plan(InterventionAction.MINORITY_REPORT, round_id=1, targets=["agent-b"]),
        scenario(max_rounds=3),
    )

    assert no_op.applied is True
    assert no_op.next_visibility_overrides == []
    assert minority.applied is True
    assert minority.final_report_requirements == ["agent-b"]
    assert state.model_dump() == before


def test_request_blind_adaptive_and_restructure_describe_only_future_effects() -> None:
    state = _state()
    request = apply_intervention_plan(
        state,
        plan(InterventionAction.REQUEST_EVIDENCE, round_id=1, targets=["agent-a"]),
        scenario(max_rounds=3),
    )
    blind = apply_intervention_plan(
        state,
        plan(InterventionAction.BLIND_EVIDENCE_REFLECTION, round_id=1, targets=["agent-a"]),
        scenario(max_rounds=3),
    )
    adaptive = apply_intervention_plan(
        state,
        plan(
            InterventionAction.ADAPTIVE_EXPOSURE,
            round_id=1,
            targets=["agent-a"],
            peer_turn_ids=["turn-r0-agent-b"],
        ),
        scenario(max_rounds=3),
    )
    restructure = apply_intervention_plan(
        state,
        plan(InterventionAction.RESTRUCTURE, round_id=1, targets=["agent-c"]),
        scenario(max_rounds=3),
    )

    assert request.evidence_request_agent_ids == ["agent-a"]
    assert blind.next_visibility_overrides[0].visible_peer_turn_ids == []
    assert adaptive.next_visibility_overrides[0].visible_peer_turn_ids == ["turn-r0-agent-b"]
    assert restructure.next_turn_order == ["agent-c", "agent-a", "agent-b"]


def test_unknown_target_is_rejected_without_a_partial_effect() -> None:
    state = _state()
    result = apply_intervention_plan(
        state,
        plan(InterventionAction.SOLICIT_DISSENT, round_id=1, targets=["agent-a"]).model_copy(
            update={"target_agent_ids": ["unknown"]}
        ),
        scenario(max_rounds=3),
    )

    assert result.applied is False
    assert "unknown_target_agent" in result.errors
    assert result.next_turn_order == []
