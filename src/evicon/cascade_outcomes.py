"""Public, evaluator-neutral outcome observations for provenance cascades."""
from __future__ import annotations
import hashlib, json
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from .provenance_cascade import ProvenanceGraph
from .provenance_cascade_exposure import ExposureLedger
from .models._validation import identifier_list, normalized_text

class ClaimStance(str, Enum):
    ENDORSES = "endorses"
    REJECTS = "rejects"
    UNCERTAIN = "uncertain"
    NO_POSITION = "no_position"
class OutcomeSource(str, Enum):
    ACTOR_OBSERVATION = "actor_observation"
    DETERMINISTIC_FIXTURE = "deterministic_fixture"
class PublicClaimOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    stance: ClaimStance
    content_ids: tuple[str, ...] = ()
    evidence_card_ids: tuple[str, ...] = ()
    source: OutcomeSource
    @field_validator("scenario_id", "agent_id", "claim_id")
    @classmethod
    def ids(cls, value: str) -> str: return normalized_text(value, "id")
    @field_validator("content_ids", "evidence_card_ids")
    @classmethod
    def unique_ids(cls, value: tuple[str, ...], info: object) -> tuple[str, ...]: return tuple(identifier_list(list(value), getattr(info, "field_name", "ids")))
class CascadeOutcomeLedger(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_id: str = Field(min_length=1)
    agent_ids: tuple[str, ...] = Field(min_length=1)
    outcomes: tuple[PublicClaimOutcome, ...] = ()
    @field_validator("scenario_id")
    @classmethod
    def scenario(cls, value: str) -> str: return normalized_text(value, "scenario_id")
    @field_validator("agent_ids")
    @classmethod
    def agents(cls, value: tuple[str, ...]) -> tuple[str, ...]: return tuple(identifier_list(list(value), "agent_ids"))
    @model_validator(mode="after")
    def shape(self) -> "CascadeOutcomeLedger":
        seen=set(); last={}
        for o in self.outcomes:
            if o.scenario_id != self.scenario_id: raise ValueError("outcome scenario mismatch")
            if o.agent_id not in self.agent_ids: raise ValueError("outcome references unknown agent")
            key=(o.agent_id,o.claim_id,o.round_id)
            if key in seen: raise ValueError("duplicate outcome coordinate")
            seen.add(key); pair=(o.agent_id,o.claim_id)
            if pair in last and o.round_id < last[pair]: raise ValueError("outcomes for an agent and claim must be chronological")
            last[pair]=o.round_id
        return self
    def append(self, outcome: PublicClaimOutcome) -> "CascadeOutcomeLedger": return CascadeOutcomeLedger(scenario_id=self.scenario_id, agent_ids=self.agent_ids, outcomes=self.outcomes+(outcome,))
    def sha256(self) -> str:
        payload=json.dumps(self.model_dump(mode="json"),sort_keys=True,separators=(",",":"),ensure_ascii=True)
        return hashlib.sha256(payload.encode()).hexdigest()
    def outcome_sha256(self) -> str:
        """Stable alias used by run records and offline audit tooling."""
        return self.sha256()
    def validate_against_public(
        self,
        graph: ProvenanceGraph,
        exposure: ExposureLedger,
        *,
        exact_snapshot: bool = False,
    ) -> None:
        """Validate public references without exposing evaluator-only data.

        The historical offline evaluator uses the original event-ledger
        semantics (an outcome may cite content exposed by the run by its
        round).  The 24G runner opts into ``exact_snapshot`` so an actor can
        cite only the immutable round-start view it actually received.
        Keeping the mode explicit preserves the pre-24G evaluator contract
        while making the new online-safe boundary strict.
        """
        if graph.scenario_id != self.scenario_id or exposure.scenario_id != self.scenario_id:
            raise ValueError("outcome public scenario mismatch")
        if tuple(exposure.agent_ids) != self.agent_ids:
            raise ValueError("outcome agent set mismatch")
        exposure.validate_against_graph(graph)
        claims = {c.claim_id: c for c in graph.claims}
        nodes = {n.content_id: n for n in graph.nodes}
        cards = {c.evidence_id: c for c in graph.evidence_cards}
        snapshots = {(snapshot.agent_id, snapshot.round_id): snapshot for snapshot in exposure.snapshots}
        event_visibility = {}
        for event in exposure.events:
            event_visibility.setdefault((event.target_agent_id, event.round_id), {"content": set(), "evidence": set()})
            event_visibility[(event.target_agent_id, event.round_id)]["content"].add(event.content_id)
            event_visibility[(event.target_agent_id, event.round_id)]["evidence"].update(event.evidence_card_ids)
        max_round = max((snapshot.round_id for snapshot in exposure.snapshots), default=0)
        for outcome in self.outcomes:
            if outcome.round_id > max_round:
                raise ValueError("outcome occurs after the public run")
            claim = claims.get(outcome.claim_id)
            if claim is None:
                raise ValueError("outcome references unknown claim")
            if exact_snapshot:
                snapshot = snapshots.get((outcome.agent_id, outcome.round_id))
                if snapshot is None:
                    raise ValueError("outcome has no matching round-start snapshot")
                visible_content = set(snapshot.visible_content_ids)
                visible_evidence = set(snapshot.visible_evidence_ids)
            else:
                visible_content = set()
                visible_evidence = set()
                for (agent_id, round_id), values in event_visibility.items():
                    if agent_id == outcome.agent_id and round_id <= outcome.round_id:
                        visible_content.update(values["content"])
                        visible_evidence.update(values["evidence"])
            if not set(outcome.content_ids).issubset(visible_content):
                raise ValueError("outcome references content not exposed to agent")
            if not set(outcome.evidence_card_ids).issubset(visible_evidence):
                raise ValueError("outcome references evidence not exposed to agent")
            for content_id in outcome.content_ids:
                node = nodes.get(content_id)
                if node is None or node.claim_id != outcome.claim_id or node.round_id > outcome.round_id:
                    raise ValueError("outcome content is unknown or from the future")
            for evidence_id in outcome.evidence_card_ids:
                card = cards.get(evidence_id)
                if (
                    card is None
                    or evidence_id not in claim.evidence_card_ids
                    or card.introduced_round > outcome.round_id
                    or not (card.is_visible_to(outcome.agent_id) or "public" in card.visible_to or "*" in card.visible_to)
                ):
                    raise ValueError("outcome evidence is unknown, unauthorized, or from the future")

class CascadeOutcomeValidationError(ValueError): pass
__all__=["CascadeOutcomeLedger","CascadeOutcomeValidationError","ClaimStance","OutcomeSource","PublicClaimOutcome"]
