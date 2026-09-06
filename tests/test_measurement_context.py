"""Tests for read-only reconstruction of contextual probe inputs."""

from __future__ import annotations

import pytest

from evicon.measurement_context import OfflineMeasurementContext, OfflineMeasurementContextBuilder
from evicon.models import ProtocolCondition, RunStatus

from measurement_cases import completed_record


@pytest.mark.parametrize(
    ("protocol", "peer_count", "evidence_count"),
    [
        (ProtocolCondition.INDEPENDENT, 0, 0),
        (ProtocolCondition.SOCIAL_ONLY, 1, 0),
        (ProtocolCondition.EVIDENCE_ONLY, 0, 1),
        (ProtocolCondition.EVIDENCE_SOCIAL, 1, 1),
    ],
)
def test_builder_reconstructs_only_protocol_permitted_prior_public_context(
    protocol: ProtocolCondition,
    peer_count: int,
    evidence_count: int,
) -> None:
    context = OfflineMeasurementContextBuilder.build(completed_record(protocol), agent_id="agent-a", round_id=1)

    assert [turn.turn_id for turn in context.own_history] == ["turn-a-r0"]
    assert len(context.visible_peer_turns) == peer_count
    assert len(context.visible_evidence_cards) == evidence_count
    assert all(turn.round_id < context.round_id for turn in context.visible_peer_turns)
    assert context.model_dump(mode="json")["agent_id"] == "agent-a"


def test_builder_rejects_failed_and_protocol_violating_trajectories() -> None:
    failed = completed_record().model_copy(update={"status": RunStatus.FAILED, "error_message": "provider failure"})
    with pytest.raises(ValueError, match="completed"):
        OfflineMeasurementContextBuilder.build(failed, agent_id="agent-a", round_id=1)

    record = completed_record(ProtocolCondition.INDEPENDENT)
    invalid_turn = record.turns[-1].model_copy(update={"visible_peer_turn_ids": ["turn-a-r0"]})
    invalid = record.model_copy(update={"turns": [*record.turns[:-1], invalid_turn]})
    with pytest.raises(ValueError, match="does not permit peer"):
        OfflineMeasurementContextBuilder.build(invalid, agent_id="agent-a", round_id=1)


def test_context_rejects_hidden_or_future_material_and_unknown_fields() -> None:
    context = OfflineMeasurementContextBuilder.build(completed_record(), agent_id="agent-a", round_id=1)
    with pytest.raises(ValueError, match="hidden"):
        OfflineMeasurementContext.model_validate({**context.model_dump(mode="python"), "scenario_context": "hidden probe content"})
    with pytest.raises(ValueError, match="extra"):
        OfflineMeasurementContext.model_validate({**context.model_dump(mode="python"), "offline_evaluation": {}})
