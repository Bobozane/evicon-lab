"""Serializable scenario contracts for deterministic protocol runs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._validation import Metadata, normalized_text
from .agents import AgentSpec
from .evidence import EvidenceCard


class ScenarioSpec(BaseModel):
    """A strict, self-contained scenario with agents and evidence chronology."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    initial_context: str = ""
    agents: list[AgentSpec] = Field(min_length=1)
    evidence_cards: list[EvidenceCard] = Field(default_factory=list)
    max_rounds: int = Field(gt=0)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("scenario_id", "title", "description")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("initial_context")
    @classmethod
    def normalize_initial_context(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_scenario_references(self) -> "ScenarioSpec":
        agent_ids = [agent.agent_id for agent in self.agents]
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("agents must not contain duplicate agent_id values")
        evidence_ids = [card.evidence_id for card in self.evidence_cards]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("evidence_cards must not contain duplicate evidence_id values")

        known_agents = set(agent_ids)
        for card in self.evidence_cards:
            unknown = set(card.visible_to) - known_agents - {"*"}
            if unknown:
                raise ValueError(
                    "EvidenceCard.visible_to names unknown agents: "
                    f"{sorted(unknown)}"
                )
        return self
