"""Validation, serialization, and visibility tests for EviCon data contracts."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from evicon.models import (
    AgentSpec,
    DialogueState,
    DialogueTurn,
    EvidenceCard,
    EvidenceExposure,
    InterventionAction,
    InterventionDecision,
    ProtocolCondition,
    RunConfig,
    RunRecord,
    RunStatus,
    ValueProbeItem,
    ValueProbeResponse,
    ValueProfile,
)


def test_value_profile_alignment_and_serialization() -> None:
    profile = ValueProfile(
        agent_id="agent-a",
        round_id=2,
        dimensions=["fairness", "autonomy"],
        scores=[0.2, 0.8],
        source="held-out-probe",
        probe_id="probe-2",
    )

    restored = ValueProfile.model_validate_json(profile.model_dump_json())

    assert restored == profile
    assert profile.dimension_scores == {"fairness": 0.2, "autonomy": 0.8}


def test_value_profile_rejects_misaligned_dimensions_and_scores() -> None:
    with pytest.raises(ValidationError, match="dimensions and scores"):
        ValueProfile(
            agent_id="agent-a",
            round_id=0,
            dimensions=["fairness", "autonomy"],
            scores=[0.5],
            source="fixture",
            probe_id="probe-1",
        )


def test_probe_models_round_trip() -> None:
    item = ValueProbeItem(
        probe_id="probe-1",
        text="How important is fairness?",
        dimension="fairness",
        response_scale=["low", "medium", "high"],
        is_holdout=True,
    )
    response = ValueProbeResponse(
        agent_id="agent-a",
        probe_id="probe-1",
        round_id=1,
        raw_response="high",
        normalized_score=1.0,
    )

    assert ValueProbeItem.model_validate_json(item.model_dump_json()) == item
    assert ValueProbeResponse.model_validate_json(response.model_dump_json()) == response


def test_evidence_visibility_availability_and_round_trip(minimal_evidence_card: EvidenceCard) -> None:
    exposure = EvidenceExposure(
        evidence_id=minimal_evidence_card.evidence_id,
        round_id=0,
        exposed_to=["agent-a"],
        exposure_reason="evidence-only condition",
    )

    assert minimal_evidence_card.is_available_at(0)
    assert not minimal_evidence_card.is_visible_to("agent-b")
    assert exposure.is_exposed_to("agent-a")
    assert EvidenceCard.model_validate_json(minimal_evidence_card.model_dump_json()) == minimal_evidence_card
    assert EvidenceExposure.model_validate_json(exposure.model_dump_json()) == exposure


def test_evidence_card_rejects_negative_introduced_round() -> None:
    with pytest.raises(ValidationError):
        EvidenceCard(
            evidence_id="evidence-1",
            claim="A claim",
            source="fixture",
            supports=[],
            contradicts=[],
            introduced_round=-1,
            visible_to=["agent-a"],
            reliability=0.5,
        )


def test_dialogue_state_rejects_unknown_evidence_reference(minimal_agents: list[AgentSpec]) -> None:
    unknown_evidence_turn = DialogueTurn(
        turn_id="turn-unknown-evidence",
        round_id=0,
        speaker_id="agent-a",
        message="This turn cites an absent record.",
        visible_to=["agent-a"],
        visible_peer_turn_ids=[],
        visible_evidence_ids=["evidence-does-not-exist"],
        protocol=ProtocolCondition.EVIDENCE_ONLY,
    )

    with pytest.raises(ValidationError, match="unknown evidence IDs"):
        DialogueState(
            run_id="run-1",
            scenario_id="scenario-1",
            current_round=0,
            agents=minimal_agents,
            turns=[unknown_evidence_turn],
            evidence_cards=[],
            value_profiles=[],
            intervention_budget=0.0,
            metadata={},
        )


def test_dialogue_state_visibility_and_json_restore(minimal_dialogue_state: DialogueState) -> None:
    restored = DialogueState.model_validate_json(minimal_dialogue_state.model_dump_json())

    assert restored == minimal_dialogue_state
    assert [turn.turn_id for turn in restored.visible_turns_for("agent-a")] == ["turn-1"]
    assert restored.visible_turns_for("agent-b") == []
    assert [card.evidence_id for card in restored.visible_evidence_for("agent-a")] == ["evidence-1"]
    assert restored.visible_evidence_for("agent-b") == []


def test_dialogue_state_does_not_serialize_private_probe_response(
    minimal_dialogue_state: DialogueState,
) -> None:
    private_response = ValueProbeResponse(
        agent_id="agent-a",
        probe_id="probe-holdout",
        round_id=0,
        raw_response="private answer",
        normalized_score=0.4,
    )

    state_json = minimal_dialogue_state.model_dump_json()

    assert private_response.raw_response not in state_json
    assert "probe_responses" not in json.loads(state_json)


def test_intervention_decision_rejects_unknown_action() -> None:
    with pytest.raises(ValidationError):
        InterventionDecision(
            action="invented_action",
            target_agent_ids=["agent-a"],
            reason="invalid action test",
            risk_score=0.2,
            estimated_cost=0.1,
            round_id=0,
        )


@pytest.mark.parametrize("field_name", ["max_rounds", "agent_count", "intervention_budget"])
def test_run_config_rejects_negative_resource_values(field_name: str) -> None:
    payload: dict[str, object] = {
        "run_id": "run-1",
        "scenario_id": "scenario-1",
        "model_name": "fake-model",
        "protocol": ProtocolCondition.INDEPENDENT,
        "agent_count": 2,
        "max_rounds": 2,
        "seed": 11,
        "intervention_budget": 0.0,
        "output_dir": "results/test",
    }
    payload[field_name] = -1

    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_protocol_decision_and_run_record_round_trip(
    minimal_protocol_fixture: dict[str, object],
) -> None:
    config = minimal_protocol_fixture["run_config"]
    state = minimal_protocol_fixture["state"]
    assert isinstance(config, RunConfig)
    assert isinstance(state, DialogueState)

    decision = InterventionDecision(
        action=InterventionAction.REQUEST_EVIDENCE,
        target_agent_ids=["agent-a"],
        reason="The action was supplied by a test fixture.",
        risk_score=0.3,
        estimated_cost=0.1,
        round_id=0,
    )
    record = RunRecord(
        config=config,
        turns=state.turns,
        value_profiles=state.value_profiles,
        evidence_exposures=[
            EvidenceExposure(
                evidence_id="evidence-1",
                round_id=0,
                exposed_to=["agent-a"],
                exposure_reason="fixture exposure",
            )
        ],
        intervention_decisions=[decision],
        status=RunStatus.COMPLETED,
        error_message=None,
    )

    assert RunRecord.model_validate_json(record.model_dump_json()) == record
    assert ProtocolCondition("evidence_only") is ProtocolCondition.EVIDENCE_ONLY


def test_all_contracts_round_trip(minimal_dialogue_state: DialogueState) -> None:
    """Every public contract can serialize without model providers or API keys."""
    agent = AgentSpec(agent_id="agent-c", role="observer", initial_value_labels=[])
    profile = ValueProfile(
        agent_id="agent-c",
        round_id=0,
        dimensions=["care"],
        scores=[0.5],
        source="fixture",
        probe_id=None,
    )
    probe = ValueProbeItem(
        probe_id="probe-c",
        text="How important is care?",
        dimension="care",
        response_scale=["low", "high"],
        is_holdout=False,
    )
    response = ValueProbeResponse(
        agent_id="agent-c",
        probe_id="probe-c",
        round_id=0,
        raw_response="high",
        normalized_score=1.0,
    )
    card = EvidenceCard(
        evidence_id="evidence-c",
        claim="A fixture claim.",
        source="fixture",
        supports=[],
        contradicts=[],
        introduced_round=0,
        visible_to=["*"],
        reliability=0.5,
    )
    exposure = EvidenceExposure(
        evidence_id="evidence-c",
        round_id=0,
        exposed_to=["*"],
        exposure_reason="fixture",
    )
    turn = DialogueTurn(
        turn_id="turn-c",
        round_id=0,
        speaker_id="agent-c",
        message="A fixture message.",
        visible_to=["*"],
        visible_peer_turn_ids=[],
        visible_evidence_ids=[],
        protocol=ProtocolCondition.INDEPENDENT,
    )
    decision = InterventionDecision(
        action=InterventionAction.NO_OP,
        target_agent_ids=[],
        reason="No intervention is needed in this fixture.",
        risk_score=0.0,
        estimated_cost=0.0,
        round_id=0,
    )
    config = RunConfig(
        run_id="run-c",
        scenario_id="scenario-c",
        model_name="fake-model",
        protocol=ProtocolCondition.INDEPENDENT,
        agent_count=1,
        max_rounds=1,
        seed=0,
        intervention_budget=0.0,
        output_dir="results/c",
    )
    record = RunRecord(config=config, status=RunStatus.PENDING)

    contracts = [
        agent,
        profile,
        probe,
        response,
        card,
        exposure,
        turn,
        minimal_dialogue_state,
        decision,
        config,
        record,
    ]
    for contract in contracts:
        assert type(contract).model_validate_json(contract.model_dump_json()) == contract
