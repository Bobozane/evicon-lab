"""Immutable public exposure ledgers for provenance-cascade fixtures.

Events are recorded at the end of a round.  A snapshot for round ``r`` uses
only events from rounds strictly before ``r``.  This prevents same-round
ordering from changing the already-created controller input.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import EvidenceCard
from .models._validation import identifier_list, normalized_text
from .provenance_cascade import Claim, ProvenanceGraph, ProvenanceNode, SourceCategory, VerificationStatus


class ExposureChannel(str, Enum):
    TIMELINE = "timeline"
    REPLY = "reply"
    REPOST = "repost"
    QUOTED_EVIDENCE = "quoted_evidence"
    RECOMMENDATION = "recommendation"


class VisibilityReason(str, Enum):
    TIMELINE_DELIVERY = "timeline_delivery"
    REPLY_VISIBILITY = "reply_visibility"
    REPOST_VISIBILITY = "repost_visibility"
    EVIDENCE_DELIVERY = "evidence_delivery"
    RECOMMENDATION_DELIVERY = "recommendation_delivery"


class ExposureEvent(BaseModel):
    """One immutable public delivery of graph content to an agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1)
    target_agent_id: str = Field(min_length=1)
    source_actor_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    content_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    provenance_node_id: str = Field(min_length=1)
    channel: ExposureChannel
    visibility_reason: VisibilityReason
    evidence_card_ids: tuple[str, ...] = ()
    parent_content_id: str | None = None

    @field_validator(
        "event_id", "target_agent_id", "source_actor_id", "content_id", "claim_id", "provenance_node_id"
    )
    @classmethod
    def clean_ids(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "id"))

    @field_validator("parent_content_id")
    @classmethod
    def parent_content(cls, value: str | None) -> str | None:
        return None if value is None else normalized_text(value, "parent_content_id")

    @field_validator("evidence_card_ids")
    @classmethod
    def evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(identifier_list(list(value), "evidence_card_ids"))

    @model_validator(mode="after")
    def reason_matches_channel(self) -> "ExposureEvent":
        expected = {
            ExposureChannel.TIMELINE: VisibilityReason.TIMELINE_DELIVERY,
            ExposureChannel.REPLY: VisibilityReason.REPLY_VISIBILITY,
            ExposureChannel.REPOST: VisibilityReason.REPOST_VISIBILITY,
            ExposureChannel.QUOTED_EVIDENCE: VisibilityReason.EVIDENCE_DELIVERY,
            ExposureChannel.RECOMMENDATION: VisibilityReason.RECOMMENDATION_DELIVERY,
        }[self.channel]
        if self.visibility_reason is not expected:
            raise ValueError("visibility reason does not match exposure channel")
        return self


class VisibleRootRelation(BaseModel):
    """Public root relation for a node already visible in one snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provenance_node_id: str = Field(min_length=1)
    source_root_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("provenance_node_id")
    @classmethod
    def node_id(cls, value: str) -> str:
        return normalized_text(value, "provenance_node_id")

    @field_validator("source_root_ids")
    @classmethod
    def root_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(identifier_list(list(value), "source_root_ids"))


class CascadeExposureSnapshot(BaseModel):
    """Immutable, round-start public context for exactly one agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    captured_event_ids: tuple[str, ...] = ()
    visible_claim_ids: tuple[str, ...] = ()
    visible_content_ids: tuple[str, ...] = ()
    visible_evidence_ids: tuple[str, ...] = ()
    visible_provenance_node_ids: tuple[str, ...] = ()
    visible_root_relations: tuple[VisibleRootRelation, ...] = ()

    @field_validator("snapshot_id", "scenario_id", "agent_id")
    @classmethod
    def clean_ids(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "id"))

    @field_validator(
        "captured_event_ids", "visible_claim_ids", "visible_content_ids", "visible_evidence_ids", "visible_provenance_node_ids"
    )
    @classmethod
    def id_sets(cls, value: tuple[str, ...], info: object) -> tuple[str, ...]:
        return tuple(identifier_list(list(value), getattr(info, "field_name", "ids")))

    @model_validator(mode="after")
    def root_relation_shape(self) -> "CascadeExposureSnapshot":
        relation_nodes = [relation.provenance_node_id for relation in self.visible_root_relations]
        if len(relation_nodes) != len(set(relation_nodes)):
            raise ValueError("visible root relations must not repeat a provenance node")
        if set(relation_nodes) != set(self.visible_provenance_node_ids):
            raise ValueError("every visible provenance node requires one root relation")
        return self


class AgentTimeline(BaseModel):
    """A deterministic chronological collection of one agent's snapshots."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str = Field(min_length=1)
    snapshots: tuple[CascadeExposureSnapshot, ...] = ()

    @field_validator("agent_id")
    @classmethod
    def clean_agent(cls, value: str) -> str:
        return normalized_text(value, "agent_id")

    @model_validator(mode="after")
    def chronological(self) -> "AgentTimeline":
        rounds = [snapshot.round_id for snapshot in self.snapshots]
        if any(snapshot.agent_id != self.agent_id for snapshot in self.snapshots):
            raise ValueError("timeline snapshot has a different agent")
        if rounds != sorted(rounds) or len(rounds) != len(set(rounds)):
            raise ValueError("timeline snapshots must be unique and chronological")
        return self


class ControllerClaimView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    verification_status: VerificationStatus
    evidence_card_ids: tuple[str, ...]
    public_summary: str | None = None


class ControllerSourceRootView(BaseModel):
    """Root summary for visible nodes; it intentionally omits evidence IDs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_root_id: str
    public_source_category: SourceCategory


class ControllerPublicView(BaseModel):
    """Safe controller input derived only from a single verified snapshot.

    Root relationships are audit metadata for nodes in the snapshot. They do
    not include provenance parents, ancestor content IDs, or root evidence IDs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    agent_id: str
    round_id: int = Field(ge=0)
    claims: tuple[ControllerClaimView, ...]
    evidence_cards: tuple[EvidenceCard, ...]
    provenance_nodes: tuple[ProvenanceNode, ...]
    source_roots: tuple[ControllerSourceRootView, ...]
    root_relations: tuple[VisibleRootRelation, ...]

    @classmethod
    def from_snapshot(cls, snapshot: CascadeExposureSnapshot, graph: ProvenanceGraph) -> "ControllerPublicView":
        if snapshot.scenario_id != graph.scenario_id:
            raise ValueError("snapshot scenario does not match graph")
        claims = {claim.claim_id: claim for claim in graph.claims}
        cards = {card.evidence_id: card for card in graph.evidence_cards}
        nodes = {node.node_id: node for node in graph.nodes}
        roots = {root.source_root_id: root for root in graph.source_roots}
        if not set(snapshot.visible_claim_ids).issubset(claims):
            raise ValueError("snapshot has unknown claim")
        if not set(snapshot.visible_evidence_ids).issubset(cards):
            raise ValueError("snapshot has unknown evidence")
        if not set(snapshot.visible_provenance_node_ids).issubset(nodes):
            raise ValueError("snapshot has unknown provenance node")
        visible_nodes = tuple(nodes[node_id] for node_id in snapshot.visible_provenance_node_ids)
        visible_root_ids = {root_id for relation in snapshot.visible_root_relations for root_id in relation.source_root_ids}
        if not visible_root_ids.issubset(roots):
            raise ValueError("snapshot has unknown root relation")
        return cls(
            scenario_id=snapshot.scenario_id,
            agent_id=snapshot.agent_id,
            round_id=snapshot.round_id,
            claims=tuple(
                ControllerClaimView(
                    claim_id=claim_id,
                    verification_status=claims[claim_id].verification_status,
                    evidence_card_ids=tuple(
                        evidence_id for evidence_id in claims[claim_id].evidence_card_ids
                        if evidence_id in snapshot.visible_evidence_ids
                    ),
                    public_summary=claims[claim_id].public_summary,
                )
                for claim_id in snapshot.visible_claim_ids
            ),
            evidence_cards=tuple(cards[evidence_id] for evidence_id in snapshot.visible_evidence_ids),
            provenance_nodes=visible_nodes,
            source_roots=tuple(
                ControllerSourceRootView(
                    source_root_id=root_id,
                    public_source_category=roots[root_id].public_source_category,
                )
                for root_id in sorted(visible_root_ids)
            ),
            root_relations=snapshot.visible_root_relations,
        )

    def root_count_for_claim(self, claim_id: str) -> int:
        visible_nodes = {node.node_id for node in self.provenance_nodes if node.claim_id == claim_id}
        return len({root_id for relation in self.root_relations if relation.provenance_node_id in visible_nodes for root_id in relation.source_root_ids})


class ExposureLedger(BaseModel):
    """Public append-only ledger plus immutable round-start snapshots."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    agent_ids: tuple[str, ...] = Field(min_length=1)
    events: tuple[ExposureEvent, ...] = ()
    snapshots: tuple[CascadeExposureSnapshot, ...] = ()

    @field_validator("scenario_id")
    @classmethod
    def clean_scenario(cls, value: str) -> str:
        return normalized_text(value, "scenario_id")

    @field_validator("agent_ids")
    @classmethod
    def agents(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(identifier_list(list(value), "agent_ids"))

    def timeline_for(self, agent_id: str) -> AgentTimeline:
        if agent_id not in self.agent_ids:
            raise ValueError("unknown agent")
        snapshots = tuple(sorted((snapshot for snapshot in self.snapshots if snapshot.agent_id == agent_id), key=lambda snapshot: snapshot.round_id))
        return AgentTimeline(agent_id=agent_id, snapshots=snapshots)

    @model_validator(mode="after")
    def shape_only(self) -> "ExposureLedger":
        event_ids = [event.event_id for event in self.events]
        snapshot_ids = [snapshot.snapshot_id for snapshot in self.snapshots]
        coordinates = [(snapshot.agent_id, snapshot.round_id) for snapshot in self.snapshots]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("duplicate exposure event ID")
        if len(snapshot_ids) != len(set(snapshot_ids)) or len(coordinates) != len(set(coordinates)):
            raise ValueError("duplicate exposure snapshot")
        if any(event.target_agent_id not in self.agent_ids or (event.source_actor_id != "system" and event.source_actor_id not in self.agent_ids) for event in self.events):
            raise ValueError("exposure event names an unknown agent or source actor")
        if any(snapshot.scenario_id != self.scenario_id or snapshot.agent_id not in self.agent_ids for snapshot in self.snapshots):
            raise ValueError("snapshot belongs to a different scenario or unknown agent")
        return self

    def validate_against_graph(self, graph: ProvenanceGraph) -> None:
        if graph.scenario_id != self.scenario_id:
            raise ValueError("ledger scenario does not match public graph")
        nodes = {node.node_id: node for node in graph.nodes}
        claims = {claim.claim_id: claim for claim in graph.claims}
        cards = {card.evidence_id: card for card in graph.evidence_cards}
        event_by_id = {event.event_id: event for event in self.events}
        snapshots = {(snapshot.agent_id, snapshot.round_id): snapshot for snapshot in self.snapshots}
        for event in self.events:
            node = nodes.get(event.provenance_node_id)
            if node is None or node.content_id != event.content_id or node.claim_id != event.claim_id:
                raise ValueError("exposure event does not match provenance content")
            if node.round_id > event.round_id:
                raise ValueError("exposure occurs before public content exists")
            claim = claims[event.claim_id]
            for evidence_id in event.evidence_card_ids:
                card = cards.get(evidence_id)
                if card is None or evidence_id not in claim.evidence_card_ids:
                    raise ValueError("exposure event names unavailable evidence")
                if card.introduced_round > event.round_id:
                    raise ValueError("exposure event delivers evidence from the future")
            if event.source_actor_id != "system":
                source_snapshot = snapshots.get((event.source_actor_id, event.round_id))
                if source_snapshot is None:
                    raise ValueError("agent source has no current round snapshot")
                visible_nodes = set(source_snapshot.visible_provenance_node_ids)
                visible_content = set(source_snapshot.visible_content_ids)
                direct_parent_ids = self._direct_parent_nodes(graph, event.provenance_node_id)
                if event.provenance_node_id in visible_nodes:
                    if event.content_id not in visible_content or event.parent_content_id is not None:
                        raise ValueError("agent event has an invalid visible-content reference")
                else:
                    if not direct_parent_ids:
                        raise ValueError("agent source references unexposed root content")
                    direct_parent_content_ids = {nodes[parent_id].content_id for parent_id in direct_parent_ids}
                    if not direct_parent_ids.issubset(visible_nodes) or not direct_parent_content_ids.issubset(visible_content):
                        raise ValueError("agent source references an unexposed provenance parent")
                    if event.parent_content_id is not None and event.parent_content_id not in direct_parent_content_ids:
                        raise ValueError("event parent content does not match a direct provenance parent")
                    if event.parent_content_id is not None and event.parent_content_id not in visible_content:
                        raise ValueError("event parent content was not exposed to the source agent")
                if not set(event.evidence_card_ids).issubset(source_snapshot.visible_evidence_ids):
                    raise ValueError("agent source cites evidence it was not exposed to")
        for snapshot in self.snapshots:
            expected_events = tuple(
                event for event in self.events
                if event.target_agent_id == snapshot.agent_id and event.round_id < snapshot.round_id
            )
            if snapshot.captured_event_ids != tuple(event.event_id for event in expected_events):
                raise ValueError("snapshot is not immutable with respect to prior exposure events")
            expected_claim_ids = tuple(dict.fromkeys(event.claim_id for event in expected_events))
            expected_content_ids = tuple(dict.fromkeys(event.content_id for event in expected_events))
            expected_node_ids = tuple(dict.fromkeys(event.provenance_node_id for event in expected_events))
            expected_evidence_ids = tuple(dict.fromkeys(evidence_id for event in expected_events for evidence_id in event.evidence_card_ids))
            if (
                snapshot.visible_claim_ids != expected_claim_ids
                or snapshot.visible_content_ids != expected_content_ids
                or snapshot.visible_provenance_node_ids != expected_node_ids
                or snapshot.visible_evidence_ids != expected_evidence_ids
            ):
                raise ValueError("snapshot visible fields do not equal the captured exposure events")
            expected_relations = tuple(
                VisibleRootRelation(
                    provenance_node_id=node_id,
                    source_root_ids=tuple(sorted(graph.root_sources_for_node(node_id))),
                )
                for node_id in expected_node_ids
            )
            if snapshot.visible_root_relations != expected_relations:
                raise ValueError("snapshot root relations do not match public provenance")
            if any(event_id not in event_by_id for event_id in snapshot.captured_event_ids):
                raise ValueError("snapshot captures an unknown event")

    @staticmethod
    def _direct_parent_nodes(graph: ProvenanceGraph, node_id: str) -> set[str]:
        return {edge.source_node_id for edge in graph.edges if edge.target_node_id == node_id}


__all__ = [
    "AgentTimeline", "CascadeExposureSnapshot", "ControllerClaimView", "ControllerPublicView", "ControllerSourceRootView", "ExposureChannel",
    "ExposureEvent", "ExposureLedger", "VisibilityReason", "VisibleRootRelation",
]
