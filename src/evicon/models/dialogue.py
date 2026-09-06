"""Dialogue records and restorable, visibility-aware dialogue state."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._validation import Metadata, can_view, identifier_list, normalized_text
from .agents import AgentSpec, ValueProfile
from .evidence import EvidenceCard
from .protocols import ProtocolCondition


class DialogueTurn(BaseModel):
    """One protocol-rendered message and the inputs visible to its speaker."""

    model_config = ConfigDict(extra="forbid")

    turn_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    speaker_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    visible_to: list[str] = Field(min_length=1)
    visible_peer_turn_ids: list[str] = Field(default_factory=list)
    visible_evidence_ids: list[str] = Field(default_factory=list)
    protocol: ProtocolCondition

    @field_validator("turn_id", "speaker_id", "message")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("visible_to")
    @classmethod
    def validate_visibility(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "visible_to", allow_public_audience=True)

    @field_validator("visible_peer_turn_ids", "visible_evidence_ids")
    @classmethod
    def validate_visible_ids(cls, value: list[str], info: object) -> list[str]:
        return identifier_list(value, getattr(info, "field_name", "visible_ids"))

    def is_visible_to(self, agent_id: str) -> bool:
        """Return whether this message is visible to ``agent_id``."""
        return can_view(self.visible_to, agent_id)


class DialogueState(BaseModel):
    """A complete serializable state with cross-record visibility validation.

    Value-probe responses are intentionally excluded. Held-out responses remain
    private records until a future reporting layer explicitly exports them.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    current_round: int = Field(ge=0)
    agents: list[AgentSpec] = Field(min_length=1)
    turns: list[DialogueTurn] = Field(default_factory=list)
    evidence_cards: list[EvidenceCard] = Field(default_factory=list)
    value_profiles: list[ValueProfile] = Field(default_factory=list)
    intervention_budget: float = Field(ge=0.0)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("run_id", "scenario_id")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @model_validator(mode="after")
    def validate_cross_references(self) -> "DialogueState":
        agent_ids = [agent.agent_id for agent in self.agents]
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("agents must not contain duplicate agent_id values")
        known_agents = set(agent_ids)

        evidence_by_id = {card.evidence_id: card for card in self.evidence_cards}
        if len(evidence_by_id) != len(self.evidence_cards):
            raise ValueError("evidence_cards must not contain duplicate evidence_id values")
        for card in self.evidence_cards:
            self._validate_audience(card.visible_to, known_agents, "EvidenceCard.visible_to")
            if not card.is_available_at(self.current_round):
                raise ValueError("evidence_cards must not include cards from future rounds")

        turn_by_id = {turn.turn_id: turn for turn in self.turns}
        if len(turn_by_id) != len(self.turns):
            raise ValueError("turns must not contain duplicate turn_id values")
        for turn in self.turns:
            if turn.speaker_id not in known_agents:
                raise ValueError(f"DialogueTurn speaker_id is not a known agent: {turn.speaker_id}")
            if turn.round_id > self.current_round:
                raise ValueError("turns must not include turns from future rounds")
            self._validate_audience(turn.visible_to, known_agents, "DialogueTurn.visible_to")
            missing_turn_ids = set(turn.visible_peer_turn_ids) - set(turn_by_id)
            if missing_turn_ids:
                raise ValueError(
                    f"DialogueTurn references unknown peer turn IDs: {sorted(missing_turn_ids)}"
                )
            if turn.turn_id in turn.visible_peer_turn_ids:
                raise ValueError("DialogueTurn must not list itself as a visible peer turn")
            missing_evidence_ids = set(turn.visible_evidence_ids) - set(evidence_by_id)
            if missing_evidence_ids:
                raise ValueError(
                    f"DialogueTurn references unknown evidence IDs: {sorted(missing_evidence_ids)}"
                )
            for evidence_id in turn.visible_evidence_ids:
                card = evidence_by_id[evidence_id]
                if not card.is_available_at(turn.round_id):
                    raise ValueError("DialogueTurn references evidence before its introduced_round")
                if not card.is_visible_to(turn.speaker_id):
                    raise ValueError("DialogueTurn speaker cannot see referenced evidence")

        for profile in self.value_profiles:
            if profile.agent_id not in known_agents:
                raise ValueError(f"ValueProfile agent_id is not a known agent: {profile.agent_id}")
            if profile.round_id > self.current_round:
                raise ValueError("value_profiles must not include profiles from future rounds")
        return self

    @staticmethod
    def _validate_audience(audience: list[str], known_agents: set[str], field_name: str) -> None:
        unknown = set(audience) - known_agents - {"*"}
        if unknown:
            raise ValueError(f"{field_name} names unknown agents: {sorted(unknown)}")

    def visible_turns_for(self, agent_id: str, *, through_round: int | None = None) -> list[DialogueTurn]:
        """Return messages visible to one known agent through a valid round."""
        limit = self._resolve_round(agent_id, through_round)
        return [turn for turn in self.turns if turn.round_id <= limit and turn.is_visible_to(agent_id)]

    def visible_evidence_for(self, agent_id: str, *, through_round: int | None = None) -> list[EvidenceCard]:
        """Return introduced evidence cards visible to one known agent."""
        limit = self._resolve_round(agent_id, through_round)
        return [
            card
            for card in self.evidence_cards
            if card.is_available_at(limit) and card.is_visible_to(agent_id)
        ]

    def _resolve_round(self, agent_id: str, through_round: int | None) -> int:
        if agent_id not in {agent.agent_id for agent in self.agents}:
            raise ValueError(f"unknown agent_id: {agent_id}")
        limit = self.current_round if through_round is None else through_round
        if limit < 0 or limit > self.current_round:
            raise ValueError("through_round must be between 0 and current_round")
        return limit
