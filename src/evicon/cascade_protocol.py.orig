"""Deterministic, offline social-network runner for provenance-cascade pilots.

The runner operates only on public claims, evidence, provenance nodes, and the
24C exposure ledger.  It has no model/provider or evaluator-truth dependency;
``no_intervention`` is the only policy accepted in this stage.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .cascade_replay import CascadeReplayReport, CascadeReplayValidator
from .models._validation import identifier_list, normalized_text
from .provenance_cascade import (
    CascadeScenario as ScenarioType,
    ProvenanceGraph,
    ProvenanceRelation,
    PublicScenarioLoader,
    file_sha256,
)
from .provenance_cascade_exposure import (
    CascadeExposureSnapshot,
    ExposureChannel,
    ExposureEvent,
    ExposureLedger,
    VisibilityReason,
    VisibleRootRelation,
)
from .provenance_cascade_preregistration import NetworkTopology


_ROOT = Path(__file__).resolve().parents[2]
_AGENT_IDS = tuple(f"network-agent-{index:02d}" for index in range(1, 7))


class ActorAction(str, Enum):
    POST = "post"
    REPLY = "reply"
    REPOST = "repost"
    QUOTED_EVIDENCE = "quoted_evidence"
    ABSTAIN = "abstain"


class InitialPublicContent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    round_id: int = Field(ge=0, le=2)
    provenance_node_id: str = Field(min_length=1)
    target_agent_ids: tuple[str, ...] = Field(min_length=1)
    evidence_card_ids: tuple[str, ...] = ()

    @field_validator("provenance_node_id")
    @classmethod
    def node_id(cls, value: str) -> str:
        return normalized_text(value, "provenance_node_id")

    @field_validator("target_agent_ids", "evidence_card_ids")
    @classmethod
    def ids(cls, value: tuple[str, ...], info: object) -> tuple[str, ...]:
        return tuple(identifier_list(list(value), getattr(info, "field_name", "ids")))


class ActorScheduleEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    round_id: int = Field(ge=1, le=2)
    actor_id: str = Field(min_length=1)
    target_agent_ids: tuple[str, ...] = Field(min_length=1)
    action: ActorAction
    provenance_node_id: str | None = None
    parent_content_id: str | None = None
    evidence_card_ids: tuple[str, ...] = ()

    @field_validator("actor_id", "provenance_node_id", "parent_content_id")
    @classmethod
    def actor_text(cls, value: str | None, info: object) -> str | None:
        return None if value is None else normalized_text(value, getattr(info, "field_name", "id"))

    @field_validator("target_agent_ids", "evidence_card_ids")
    @classmethod
    def ids(cls, value: tuple[str, ...], info: object) -> tuple[str, ...]:
        return tuple(identifier_list(list(value), getattr(info, "field_name", "ids")))

    @model_validator(mode="after")
    def action_shape(self) -> "ActorScheduleEntry":
        if self.action is ActorAction.ABSTAIN and (self.provenance_node_id or self.parent_content_id or self.evidence_card_ids):
            raise ValueError("abstain schedule entries cannot carry content, parent, or evidence")
        if self.action is not ActorAction.ABSTAIN and not self.provenance_node_id:
            raise ValueError("non-abstaining schedule entries require a provenance node")
        return self


class CascadeScenarioFile(BaseModel):
    """Strict on-disk scenario declaration; graph content stays in public JSON."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    scenario_type: ScenarioType
    version: str = Field(min_length=1)
    graph_path: str = Field(min_length=1)
    graph_sha256: str = Field(min_length=64, max_length=64)
    claim_ids: tuple[str, ...] = Field(min_length=1)
    agent_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    network_topology: NetworkTopology
    max_rounds: Literal[3]
    initial_public_content: tuple[InitialPublicContent, ...] = Field(min_length=1)
    actor_schedule: tuple[ActorScheduleEntry, ...] = ()
    allowed_relations: tuple[ProvenanceRelation, ...] = Field(min_length=1)
    policy: Literal["no_intervention"] = "no_intervention"

    @field_validator("scenario_id", "version", "graph_path")
    @classmethod
    def text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("graph_sha256")
    @classmethod
    def sha(cls, value: str) -> str:
        value = value.lower().strip()
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("graph_sha256 must be a SHA-256 digest")
        return value

    @field_validator("claim_ids", "agent_ids")
    @classmethod
    def id_lists(cls, value: tuple[str, ...], info: object) -> tuple[str, ...]:
        return tuple(identifier_list(list(value), getattr(info, "field_name", "ids")))

    @model_validator(mode="after")
    def fixed_topology(self) -> "CascadeScenarioFile":
        if self.agent_ids != _AGENT_IDS:
            raise ValueError("cascade scenarios require the fixed six-agent order")
        if self.network_topology.agent_ids != list(self.agent_ids):
            raise ValueError("network topology agents must match scenario agents")
        if self.network_topology.topology_id != "ring_6_bidirectional" or self.network_topology.directed:
            raise ValueError("cascade scenarios require the undirected six-agent ring")
        if len(set(self.claim_ids)) != len(self.claim_ids):
            raise ValueError("claim_ids must be unique")
        return self


class CascadeScenarioSpec(BaseModel):
    """Loaded scenario with its public graph materialized."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    scenario_type: ScenarioType
    version: str
    graph_path: str
    graph_sha256: str
    config_sha256: str
    claim_ids: tuple[str, ...]
    agent_ids: tuple[str, ...]
    network_topology: NetworkTopology
    max_rounds: int
    initial_public_content: tuple[InitialPublicContent, ...]
    actor_schedule: tuple[ActorScheduleEntry, ...]
    allowed_relations: tuple[ProvenanceRelation, ...]
    graph: ProvenanceGraph
    policy: Literal["no_intervention"]

    @model_validator(mode="after")
    def schedule_shape(self) -> "CascadeScenarioSpec":
        coordinates = [(entry.actor_id, entry.round_id) for entry in self.actor_schedule]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("actor schedule may contain at most one entry per agent and round")
        known_agents = set(self.agent_ids)
        known_nodes = {node.node_id for node in self.graph.nodes}
        known_cards = {card.evidence_id for card in self.graph.evidence_cards}
        if any(entry.actor_id not in known_agents or any(target not in known_agents for target in entry.target_agent_ids) or (entry.provenance_node_id is not None and entry.provenance_node_id not in known_nodes) or any(card not in known_cards for card in entry.evidence_card_ids) for entry in self.actor_schedule):
            raise ValueError("actor schedule references unknown public data")
        return self


class CascadeScenarioLoader:
    @staticmethod
    def load(path: str | Path) -> CascadeScenarioSpec:
        config_path = Path(path)
        try:
            payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
            declaration = CascadeScenarioFile.model_validate(payload)
            graph_path = (config_path.parent / declaration.graph_path).resolve()
            graph = PublicScenarioLoader.load(graph_path)
            if file_sha256(graph_path) != declaration.graph_sha256:
                raise ValueError("scenario graph hash mismatch")
            if graph.scenario_id != declaration.scenario_id or graph.scenario_type != declaration.scenario_type:
                raise ValueError("scenario graph identity mismatch")
            graph_claim_ids = tuple(claim.claim_id for claim in graph.claims)
            if set(graph_claim_ids) != set(declaration.claim_ids):
                raise ValueError("scenario claim IDs do not match graph")
            known_agents = set(declaration.agent_ids)
            known_nodes = {node.node_id for node in graph.nodes}
            known_cards = {card.evidence_id for card in graph.evidence_cards}
            for item in declaration.initial_public_content:
                if item.provenance_node_id not in known_nodes or not set(item.target_agent_ids).issubset(known_agents) or not set(item.evidence_card_ids).issubset(known_cards):
                    raise ValueError("initial public content references unknown data")
            for item in declaration.actor_schedule:
                if item.actor_id not in known_agents or not set(item.target_agent_ids).issubset(known_agents):
                    raise ValueError("actor schedule references unknown agent")
                if item.provenance_node_id and item.provenance_node_id not in known_nodes:
                    raise ValueError("actor schedule references unknown provenance node")
                if not set(item.evidence_card_ids).issubset(known_cards):
                    raise ValueError("actor schedule references unknown evidence")
                action_relation = {
                    ActorAction.REPOST: ProvenanceRelation.REPOST.value,
                    ActorAction.REPLY: ProvenanceRelation.REPLY.value,
                    ActorAction.QUOTED_EVIDENCE: ProvenanceRelation.QUOTES_EVIDENCE.value,
                }.get(item.action)
                if action_relation is not None and action_relation not in {relation.value for relation in declaration.allowed_relations}:
                    raise ValueError("actor schedule action is not an allowed relation")
            loaded = declaration.model_dump(mode="python")
            loaded["graph_path"] = str(graph_path)
            loaded["config_sha256"] = file_sha256(config_path)
            loaded["graph"] = graph
            return CascadeScenarioSpec.model_validate(loaded)
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
            raise CascadeProtocolError("scenario_config_invalid") from exc


class CascadeRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    seed: int
    policy: Literal["no_intervention"]
    agent_ids: tuple[str, ...]
    round_count: int
    content_event_count: int
    exposure_event_count: int
    snapshot_count: int
    independent_root_count_by_claim: dict[str, int]
    trajectory_sha256: str
    replay: CascadeReplayReport
    ledger: ExposureLedger


class CascadeProtocolError(ValueError):
    """Stable, non-sensitive deterministic runner error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class CascadeProtocolRunner:
    """Run a fixed public actor schedule without models or intervention logic."""

    def run_scenario(
        self,
        scenario: CascadeScenarioSpec,
        seed: int,
        agent_schedule: tuple[ActorScheduleEntry, ...] | None = None,
        network_topology: NetworkTopology | None = None,
        policy: str = "no_intervention",
    ) -> CascadeRunRecord:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
            raise CascadeProtocolError("seed_invalid")
        if policy != "no_intervention":
            raise CascadeProtocolError("policy_not_implemented")
        topology = network_topology or scenario.network_topology
        if topology != scenario.network_topology:
            raise CascadeProtocolError("network_topology_mismatch")
        schedule = scenario.actor_schedule if agent_schedule is None else agent_schedule
        if schedule != scenario.actor_schedule:
            raise CascadeProtocolError("actor_schedule_mismatch")
        events: list[ExposureEvent] = []
        snapshots: list[CascadeExposureSnapshot] = []
        for round_id in range(scenario.max_rounds):
            round_snapshot = self.build_round_snapshot(scenario, tuple(events), round_id)
            snapshots.extend(round_snapshot)
            self._append_initial_content(scenario, round_id, events)
            for schedule_index, entry in enumerate(schedule):
                if entry.round_id != round_id or entry.action is ActorAction.ABSTAIN:
                    continue
                for target_index, target_agent_id in enumerate(entry.target_agent_ids):
                    event = self.append_public_event(
                        scenario=scenario,
                        events=tuple(events),
                        snapshots=tuple(snapshots),
                        entry=entry,
                        target_agent_id=target_agent_id,
                        event_id=f"{scenario.scenario_id}-s{seed}-r{round_id}-e{schedule_index}-t{target_index}",
                        round_snapshot=round_snapshot,
                    )
                    events.append(event)
        ledger = ExposureLedger(
            scenario_id=scenario.scenario_id,
            agent_ids=scenario.agent_ids,
            events=tuple(events),
            snapshots=tuple(snapshots),
        )
        try:
            replay = CascadeReplayValidator.validate(scenario.graph, ledger)
        except Exception as exc:
            raise CascadeProtocolError("cascade_replay_failed") from exc
        trajectory = json.dumps(ledger.model_dump(mode="json"), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        exposed_roots: dict[str, set[str]] = {}
        nodes = {node.node_id: node for node in scenario.graph.nodes}
        for snapshot in ledger.snapshots:
            for relation in snapshot.visible_root_relations:
                exposed_roots.setdefault(nodes[relation.provenance_node_id].claim_id, set()).update(relation.source_root_ids)
        roots = {claim.claim_id: len(exposed_roots.get(claim.claim_id, set())) for claim in scenario.graph.claims}
        return CascadeRunRecord(
            scenario_id=scenario.scenario_id,
            seed=seed,
            policy="no_intervention",
            agent_ids=scenario.agent_ids,
            round_count=scenario.max_rounds,
            content_event_count=len(events),
            exposure_event_count=len(events),
            snapshot_count=len(snapshots),
            independent_root_count_by_claim=roots,
            trajectory_sha256=hashlib.sha256(trajectory.encode("utf-8")).hexdigest(),
            replay=replay,
            ledger=ledger,
        )

    def run_round(self, scenario: CascadeScenarioSpec, seed: int, round_id: int) -> CascadeExposureSnapshot:
        """Build an empty round-start snapshot for API discoverability."""
        if round_id < 0 or round_id >= scenario.max_rounds:
            raise CascadeProtocolError("round_invalid")
        return self.build_round_snapshot(scenario, (), round_id)[0]

    def build_round_snapshot(
        self,
        scenario: CascadeScenarioSpec,
        events: tuple[ExposureEvent, ...],
        round_id: int,
    ) -> tuple[CascadeExposureSnapshot, ...]:
        if round_id < 0 or round_id >= scenario.max_rounds:
            raise CascadeProtocolError("round_invalid")
        graph = scenario.graph
        snapshots: list[CascadeExposureSnapshot] = []
        for agent_id in scenario.agent_ids:
            prior = tuple(event for event in events if event.target_agent_id == agent_id and event.round_id < round_id)
            node_ids = tuple(dict.fromkeys(event.provenance_node_id for event in prior))
            root_relations = tuple(
                VisibleRootRelation(
                    provenance_node_id=node_id,
                    source_root_ids=tuple(sorted(graph.root_sources_for_node(node_id))),
                )
                for node_id in node_ids
            )
            snapshots.append(
                CascadeExposureSnapshot(
                    snapshot_id=f"{scenario.scenario_id}-snapshot-{agent_id}-r{round_id}",
                    scenario_id=scenario.scenario_id,
                    agent_id=agent_id,
                    round_id=round_id,
                    captured_event_ids=tuple(event.event_id for event in prior),
                    visible_claim_ids=tuple(dict.fromkeys(event.claim_id for event in prior)),
                    visible_content_ids=tuple(dict.fromkeys(event.content_id for event in prior)),
                    visible_evidence_ids=tuple(dict.fromkeys(evidence_id for event in prior for evidence_id in event.evidence_card_ids)),
                    visible_provenance_node_ids=node_ids,
                    visible_root_relations=root_relations,
                )
            )
        return tuple(snapshots)

    def append_public_event(
        self,
        scenario: CascadeScenarioSpec,
        events: tuple[ExposureEvent, ...],
        snapshots: tuple[CascadeExposureSnapshot, ...],
        entry: ActorScheduleEntry,
        target_agent_id: str,
        event_id: str,
        round_snapshot: tuple[CascadeExposureSnapshot, ...],
    ) -> ExposureEvent:
        if target_agent_id not in scenario.agent_ids or entry.actor_id not in scenario.agent_ids:
            raise CascadeProtocolError("agent_unknown")
        if entry.provenance_node_id is None:
            raise CascadeProtocolError("provenance_node_required")
        if event_id in {event.event_id for event in events}:
            raise CascadeProtocolError("event_id_duplicate")
        node = next((node for node in scenario.graph.nodes if node.node_id == entry.provenance_node_id), None)
        if node is None:
            raise CascadeProtocolError("provenance_node_unknown")
        if node.round_id > entry.round_id:
            raise CascadeProtocolError("future_provenance_node")
        if node.claim_id not in {claim.claim_id for claim in scenario.graph.claims}:
            raise CascadeProtocolError("claim_unknown")
        source_snapshot = next(snapshot for snapshot in round_snapshot if snapshot.agent_id == entry.actor_id)
        visible_nodes = set(source_snapshot.visible_provenance_node_ids)
        visible_content = set(source_snapshot.visible_content_ids)
        direct_parent_ids = self._direct_parent_nodes(scenario.graph, node.node_id)
        event_parent_content_id: str | None = None
        if node.node_id in visible_nodes:
            if node.content_id not in visible_content:
                raise CascadeProtocolError("visible_content_mismatch")
            if entry.parent_content_id is not None:
                raise CascadeProtocolError("parent_content_not_allowed_for_visible_content")
        else:
            if not direct_parent_ids:
                raise CascadeProtocolError("content_not_exposed")
            graph_nodes = {graph_node.node_id: graph_node for graph_node in scenario.graph.nodes}
            direct_parent_content_ids = {graph_nodes[parent_id].content_id for parent_id in direct_parent_ids}
            if not direct_parent_ids.issubset(visible_nodes) or not direct_parent_content_ids.issubset(visible_content):
                raise CascadeProtocolError("provenance_parent_not_exposed")
            if entry.parent_content_id is None:
                raise CascadeProtocolError("parent_content_required")
            if entry.parent_content_id not in direct_parent_content_ids or entry.parent_content_id not in visible_content:
                raise CascadeProtocolError("parent_content_not_exposed")
            event_parent_content_id = entry.parent_content_id
        if not set(entry.evidence_card_ids).issubset(source_snapshot.visible_evidence_ids):
            raise CascadeProtocolError("evidence_not_exposed")
        channel = ExposureChannel.TIMELINE if entry.action is ActorAction.POST else ExposureChannel(entry.action.value)
        reason = {
            ExposureChannel.REPLY: VisibilityReason.REPLY_VISIBILITY,
            ExposureChannel.REPOST: VisibilityReason.REPOST_VISIBILITY,
            ExposureChannel.QUOTED_EVIDENCE: VisibilityReason.EVIDENCE_DELIVERY,
            ExposureChannel.TIMELINE: VisibilityReason.TIMELINE_DELIVERY,
            ExposureChannel.RECOMMENDATION: VisibilityReason.RECOMMENDATION_DELIVERY,
        }[channel]
        return ExposureEvent(
            event_id=event_id,
            target_agent_id=target_agent_id,
            source_actor_id=entry.actor_id,
            round_id=entry.round_id,
            content_id=node.content_id,
            claim_id=node.claim_id,
            provenance_node_id=node.node_id,
            channel=channel,
            visibility_reason=reason,
            evidence_card_ids=tuple(entry.evidence_card_ids),
            parent_content_id=event_parent_content_id,
        )

    @staticmethod
    def _direct_parent_nodes(graph: ProvenanceGraph, node_id: str) -> set[str]:
        return {edge.source_node_id for edge in graph.edges if edge.target_node_id == node_id}

    @staticmethod
    def _append_initial_content(scenario: CascadeScenarioSpec, round_id: int, events: list[ExposureEvent]) -> None:
        nodes = {node.node_id: node for node in scenario.graph.nodes}
        for index, item in enumerate(scenario.initial_public_content):
            if item.round_id != round_id:
                continue
            node = nodes[item.provenance_node_id]
            channel = ExposureChannel.QUOTED_EVIDENCE if item.evidence_card_ids else ExposureChannel.TIMELINE
            reason = VisibilityReason.EVIDENCE_DELIVERY if item.evidence_card_ids else VisibilityReason.TIMELINE_DELIVERY
            for target_index, target in enumerate(item.target_agent_ids):
                events.append(ExposureEvent(
                    event_id=f"{scenario.scenario_id}-initial-r{round_id}-i{index}-t{target_index}",
                    target_agent_id=target,
                    source_actor_id="system",
                    round_id=round_id,
                    content_id=node.content_id,
                    claim_id=node.claim_id,
                    provenance_node_id=node.node_id,
                    channel=channel,
                    visibility_reason=reason,
                    evidence_card_ids=item.evidence_card_ids,
                ))


CascadeScenario = CascadeScenarioSpec


def trajectory_sha256(record: CascadeRunRecord) -> str:
    return record.trajectory_sha256


__all__ = [
    "ActorAction", "ActorScheduleEntry", "CascadeProtocolError", "CascadeProtocolRunner", "CascadeRunRecord",
    "CascadeScenarioFile", "CascadeScenarioLoader", "CascadeScenarioSpec", "CascadeScenario", "InitialPublicContent", "trajectory_sha256",
]
