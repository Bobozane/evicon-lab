"""Read-only reconstruction of public trajectory context for offline measurement."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import DialogueTurn, EvidenceCard, ProtocolCondition, RunRecord, RunStatus
from .models._validation import identifier_list, normalized_text


_PRIVATE = ("hidden_probe", "hidden probe", "hidden_profile", "hidden profile", "valueprofile", "evaluationreport", "evaluation_report", "proberesult", "probe result")


class OfflineMeasurementContext(BaseModel):
    """Frozen public-only input for one contextual probe; it has no online controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    initial_value_labels: list[str] = Field(default_factory=list)
    round_id: int = Field(ge=0)
    protocol: ProtocolCondition
    scenario_context: str = Field(max_length=4000)
    own_history: list[DialogueTurn] = Field(default_factory=list)
    visible_peer_turns: list[DialogueTurn] = Field(default_factory=list)
    visible_evidence_cards: list[EvidenceCard] = Field(default_factory=list)

    @field_validator("agent_id", "role", "scenario_context")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        field = getattr(info, "field_name", "text")
        result = value.strip() if field == "scenario_context" else normalized_text(value, field)
        _reject_private(result, field)
        return result

    @field_validator("initial_value_labels")
    @classmethod
    def validate_labels(cls, value: list[str]) -> list[str]:
        result = identifier_list(value, "initial_value_labels")
        for label in result:
            _reject_private(label, "initial_value_labels")
        return result

    @field_validator("own_history", "visible_peer_turns")
    @classmethod
    def reject_private_turns(cls, value: list[DialogueTurn], info: object) -> list[DialogueTurn]:
        for turn in value:
            _reject_private(turn.message, f"{getattr(info, 'field_name', 'turns')}.message")
        return value

    @field_validator("visible_evidence_cards")
    @classmethod
    def reject_private_evidence(cls, value: list[EvidenceCard]) -> list[EvidenceCard]:
        for card in value:
            for text in [card.evidence_id, card.claim, card.source, *card.supports, *card.contradicts]:
                _reject_private(text, "visible_evidence_cards")
        return value

    @model_validator(mode="after")
    def validate_visibility(self) -> "OfflineMeasurementContext":
        own_ids: set[str] = set()
        for turn in self.own_history:
            if turn.turn_id in own_ids or turn.speaker_id != self.agent_id or turn.round_id >= self.round_id:
                raise ValueError("own_history must contain unique earlier turns by agent_id")
            own_ids.add(turn.turn_id)
        peer_ids: set[str] = set()
        for turn in self.visible_peer_turns:
            if (
                turn.turn_id in own_ids | peer_ids
                or turn.speaker_id == self.agent_id
                or turn.round_id >= self.round_id
                or not turn.is_visible_to(self.agent_id)
            ):
                raise ValueError("visible_peer_turns must contain unique visible earlier peer turns")
            peer_ids.add(turn.turn_id)
        card_ids: set[str] = set()
        for card in self.visible_evidence_cards:
            if card.evidence_id in card_ids or not card.is_available_at(self.round_id) or not card.is_visible_to(self.agent_id):
                raise ValueError("visible_evidence_cards must contain unique available visible cards")
            card_ids.add(card.evidence_id)
        if self.protocol in {ProtocolCondition.INDEPENDENT, ProtocolCondition.EVIDENCE_ONLY} and peer_ids:
            raise ValueError("protocol does not permit visible peer turns")
        if self.protocol in {ProtocolCondition.INDEPENDENT, ProtocolCondition.SOCIAL_ONLY} and card_ids:
            raise ValueError("protocol does not permit visible evidence")
        return self


class OfflineMeasurementContextBuilder:
    """Pure builder accepting only a completed RunRecord and target public coordinates."""

    @staticmethod
    def build(record: RunRecord, *, agent_id: str, round_id: int) -> OfflineMeasurementContext:
        _validate_record(record)
        if type(round_id) is not int or round_id < 0 or round_id >= record.config.max_rounds:
            raise ValueError("round_id must be within the completed run")
        scenario = record.scenario
        assert scenario is not None
        agents = {agent.agent_id: agent for agent in scenario.agents}
        if agent_id not in agents:
            raise ValueError("agent_id must belong to the run scenario")
        _validate_trajectory(record)
        agent = agents[agent_id]
        own = [turn for turn in record.turns if turn.speaker_id == agent_id and turn.round_id < round_id]
        peers = [
            turn for turn in record.turns
            if turn.speaker_id != agent_id and turn.round_id < round_id and turn.is_visible_to(agent_id)
        ]
        exposure_ids = {
            exposure.evidence_id for exposure in record.evidence_exposures
            if exposure.round_id <= round_id and exposure.is_exposed_to(agent_id)
        }
        cards = [card for card in scenario.evidence_cards if card.evidence_id in exposure_ids]
        if len(cards) != len(exposure_ids):
            raise ValueError("EvidenceExposure references an evidence ID absent from the scenario")
        for card in cards:
            if not card.is_available_at(round_id) or not card.is_visible_to(agent_id):
                raise ValueError("EvidenceExposure includes unavailable or invisible evidence")
        return OfflineMeasurementContext(
            agent_id=agent.agent_id,
            role=agent.role,
            initial_value_labels=list(agent.initial_value_labels),
            round_id=round_id,
            protocol=record.config.protocol,
            scenario_context=scenario.initial_context,
            own_history=own,
            visible_peer_turns=peers,
            visible_evidence_cards=cards,
        )


def _validate_record(record: RunRecord) -> None:
    if record.status is not RunStatus.COMPLETED:
        raise ValueError("OfflineMeasurementContext requires a completed RunRecord")
    if record.scenario is None:
        raise ValueError("OfflineMeasurementContext requires a RunRecord scenario snapshot")
    if record.config.scenario_id != record.scenario.scenario_id or record.config.agent_count != len(record.scenario.agents) or record.config.max_rounds != record.scenario.max_rounds:
        raise ValueError("RunRecord config must match its scenario snapshot")


def _validate_trajectory(record: RunRecord) -> None:
    scenario = record.scenario
    assert scenario is not None
    agents = {agent.agent_id for agent in scenario.agents}
    cards = {card.evidence_id: card for card in scenario.evidence_cards}
    turns = {turn.turn_id: turn for turn in record.turns}
    if len(turns) != len(record.turns):
        raise ValueError("RunRecord turns must not contain duplicate turn IDs")
    for turn in record.turns:
        if turn.speaker_id not in agents or turn.round_id >= record.config.max_rounds or turn.protocol is not record.config.protocol:
            raise ValueError("RunRecord contains an invalid turn")
        if record.config.protocol in {ProtocolCondition.INDEPENDENT, ProtocolCondition.EVIDENCE_ONLY} and turn.visible_peer_turn_ids:
            raise ValueError("RunRecord protocol does not permit peer turns")
        if record.config.protocol in {ProtocolCondition.INDEPENDENT, ProtocolCondition.SOCIAL_ONLY} and turn.visible_evidence_ids:
            raise ValueError("RunRecord protocol does not permit evidence")
        for peer_id in turn.visible_peer_turn_ids:
            peer = turns.get(peer_id)
            if peer is None or peer.round_id >= turn.round_id or peer.speaker_id == turn.speaker_id or not peer.is_visible_to(turn.speaker_id):
                raise ValueError("RunRecord contains an invalid peer-turn reference")
        for evidence_id in turn.visible_evidence_ids:
            card = cards.get(evidence_id)
            if card is None or not card.is_available_at(turn.round_id) or not card.is_visible_to(turn.speaker_id):
                raise ValueError("RunRecord contains an invalid evidence reference")
    for exposure in record.evidence_exposures:
        card = cards.get(exposure.evidence_id)
        if card is None or exposure.round_id >= record.config.max_rounds or not card.is_available_at(exposure.round_id):
            raise ValueError("RunRecord contains an invalid EvidenceExposure")
        if any(value != "*" and value not in agents for value in exposure.exposed_to):
            raise ValueError("EvidenceExposure names an unknown agent")


def _reject_private(value: str, field: str) -> None:
    if any(token in value.lower() for token in _PRIVATE):
        raise ValueError(f"{field} must not contain hidden probe, profile, or offline evaluation content")


__all__ = ["OfflineMeasurementContext", "OfflineMeasurementContextBuilder"]
