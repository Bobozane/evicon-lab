"""Public provenance graphs and evaluator-private truth contracts.

The public models in this module are intentionally independent from evaluator
annotations.  A private truth file can be joined offline by stable IDs, but
it cannot be nested in a claim, source root, graph, exposure, prompt, or
controller view.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .models import EvidenceCard
from .models._validation import identifier_list, normalized_text
from .provenance_cascade_preregistration import CascadeScenario


class VerificationStatus(str, Enum):
    """Public status visible to agents and audit tooling.

    ``unverified`` means no public verification is established;
    ``supported`` and ``refuted`` are public evidence states; ``contested``
    means public evidence points in incompatible directions.
    """

    UNVERIFIED = "unverified"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    CONTESTED = "contested"


class SourceCategory(str, Enum):
    PRIMARY_RECORD = "primary_record"
    INDEPENDENT_REPORT = "independent_report"
    REPOST = "repost"
    COMMUNITY_SUMMARY = "community_summary"


class ProvenanceRelation(str, Enum):
    ORIGINATES = "originates"
    REPOST = "repost"
    REPLY = "reply"
    QUOTES_EVIDENCE = "quotes_evidence"


class Claim(BaseModel):
    """Public claim summary; evaluator truth is deliberately absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    public_summary: str = Field(min_length=1)
    verification_status: VerificationStatus
    evidence_card_ids: list[str] = Field(default_factory=list)

    @field_validator("claim_id", "public_summary")
    @classmethod
    def clean_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("evidence_card_ids")
    @classmethod
    def evidence_ids(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "evidence_card_ids")


class SourceRoot(BaseModel):
    """Public source identity and its allowed evidence cards."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_root_id: str = Field(min_length=1)
    public_source_category: SourceCategory
    evidence_card_ids: list[str] = Field(default_factory=list)

    @field_validator("source_root_id")
    @classmethod
    def clean_root_id(cls, value: str) -> str:
        return normalized_text(value, "source_root_id")

    @field_validator("evidence_card_ids")
    @classmethod
    def root_evidence_ids(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "evidence_card_ids")


class ProvenanceNode(BaseModel):
    """One public content node in a scenario graph."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(min_length=1)
    content_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    source_root_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    public_statement: str | None = None

    @field_validator("public_statement")
    @classmethod
    def clean_public_statement(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = normalized_text(value, "public_statement")
        lowered = cleaned.lower()
        forbidden = (
            "ground_truth_label",
            "source_independence_label",
            "evaluator_truth",
            "private_fixture",
            "api_key",
        )
        if any(token in lowered for token in forbidden):
            raise ValueError("public statement contains a private field marker")
        return cleaned


    @field_validator("node_id", "content_id", "scenario_id", "claim_id", "source_root_id")
    @classmethod
    def clean_ids(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "id"))


class ProvenanceEdge(BaseModel):
    """A directed public relation from an upstream node to a downstream node."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    edge_id: str = Field(min_length=1)
    source_node_id: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    relation: ProvenanceRelation
    evidence_card_id: str | None = None

    @field_validator("edge_id", "source_node_id", "target_node_id")
    @classmethod
    def clean_edge_ids(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "id"))

    @field_validator("evidence_card_id")
    @classmethod
    def clean_optional_evidence(cls, value: str | None) -> str | None:
        return None if value is None else normalized_text(value, "evidence_card_id")

    @model_validator(mode="after")
    def evidence_reference_shape(self) -> "ProvenanceEdge":
        if self.relation is ProvenanceRelation.QUOTES_EVIDENCE and not self.evidence_card_id:
            raise ValueError("quotes_evidence edges require an evidence_card_id")
        if self.relation is not ProvenanceRelation.QUOTES_EVIDENCE and self.evidence_card_id is not None:
            raise ValueError("only quotes_evidence edges may carry an evidence_card_id")
        return self


class ProvenanceGraph(BaseModel):
    """Public, replayable provenance graph with root-resolution helpers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    scenario_type: CascadeScenario
    claims: list[Claim] = Field(min_length=1)
    source_roots: list[SourceRoot] = Field(min_length=1)
    evidence_cards: list[EvidenceCard] = Field(min_length=1)
    nodes: list[ProvenanceNode] = Field(min_length=1)
    edges: list[ProvenanceEdge] = Field(default_factory=list)

    @field_validator("scenario_id")
    @classmethod
    def clean_scenario_id(cls, value: str) -> str:
        return normalized_text(value, "scenario_id")

    @model_validator(mode="after")
    def validate_graph(self) -> "ProvenanceGraph":
        claim_ids = [claim.claim_id for claim in self.claims]
        root_ids = [root.source_root_id for root in self.source_roots]
        evidence_ids = [card.evidence_id for card in self.evidence_cards]
        node_ids = [node.node_id for node in self.nodes]
        content_ids = [node.content_id for node in self.nodes]
        edge_ids = [edge.edge_id for edge in self.edges]
        for name, values in (("claim", claim_ids), ("source root", root_ids), ("evidence", evidence_ids), ("node", node_ids), ("content", content_ids), ("edge", edge_ids)):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {name} ID")
        claims = {claim.claim_id: claim for claim in self.claims}
        roots = {root.source_root_id: root for root in self.source_roots}
        cards = {card.evidence_id: card for card in self.evidence_cards}
        nodes = {node.node_id: node for node in self.nodes}
        if any(node.scenario_id != self.scenario_id for node in self.nodes):
            raise ValueError("node belongs to a different scenario")
        if any(card.introduced_round < 0 for card in self.evidence_cards):
            raise ValueError("evidence introduction round must be non-negative")
        for claim in self.claims:
            if any(evidence_id not in cards for evidence_id in claim.evidence_card_ids):
                raise ValueError("claim references an unknown evidence card")
        for root in self.source_roots:
            if any(evidence_id not in cards for evidence_id in root.evidence_card_ids):
                raise ValueError("source root references an unknown evidence card")
        for node in self.nodes:
            if node.claim_id not in claims or node.source_root_id not in roots:
                raise ValueError("node references an unknown claim or source root")
        outgoing: dict[str, list[str]] = defaultdict(list)
        incoming: dict[str, list[str]] = defaultdict(list)
        for edge in self.edges:
            if edge.source_node_id not in nodes or edge.target_node_id not in nodes:
                raise ValueError("edge references an unknown node")
            source = nodes[edge.source_node_id]
            target = nodes[edge.target_node_id]
            if source.scenario_id != target.scenario_id:
                raise ValueError("cross-scenario edge is not allowed")
            if source.claim_id != target.claim_id:
                raise ValueError("cross-claim propagation is not allowed")
            if target.round_id < source.round_id:
                raise ValueError("provenance edge reverses time")
            if edge.relation is ProvenanceRelation.QUOTES_EVIDENCE:
                if edge.evidence_card_id not in cards or edge.evidence_card_id not in claims[target.claim_id].evidence_card_ids:
                    raise ValueError("quotes_evidence references an unavailable card")
                if cards[edge.evidence_card_id].introduced_round > target.round_id:
                    raise ValueError("quotes_evidence references evidence from the future")
            elif source.source_root_id != target.source_root_id:
                raise ValueError("nodes with a shared upstream edge cannot claim independent roots")
            outgoing[edge.source_node_id].append(edge.target_node_id)
            incoming[edge.target_node_id].append(edge.source_node_id)
        self._assert_acyclic(nodes, outgoing)
        return self

    @staticmethod
    def _assert_acyclic(nodes: dict[str, ProvenanceNode], outgoing: dict[str, list[str]]) -> None:
        indegree = {node_id: 0 for node_id in nodes}
        for children in outgoing.values():
            for child in children:
                indegree[child] += 1
        queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
        visited = 0
        while queue:
            node_id = queue.popleft()
            visited += 1
            for child in outgoing.get(node_id, []):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if visited != len(nodes):
            raise ValueError("provenance graph contains a cycle")

    def root_sources_for_node(self, node_id: str) -> frozenset[str]:
        nodes = {node.node_id: node for node in self.nodes}
        if node_id not in nodes:
            raise ValueError("unknown node")
        incoming: dict[str, list[str]] = defaultdict(list)
        for edge in self.edges:
            incoming[edge.target_node_id].append(edge.source_node_id)
        visiting: set[str] = set()
        cache: dict[str, frozenset[str]] = {}

        def resolve(current: str) -> frozenset[str]:
            if current in cache:
                return cache[current]
            if current in visiting:
                raise ValueError("provenance graph contains a cycle")
            visiting.add(current)
            roots = {nodes[current].source_root_id}
            for parent in incoming.get(current, []):
                roots.update(resolve(parent))
            visiting.remove(current)
            resolved = frozenset(roots)
            cache[current] = resolved
            return resolved

        return resolve(node_id)

    def root_sources_for_claim(self, claim_id: str) -> frozenset[str]:
        if claim_id not in {claim.claim_id for claim in self.claims}:
            raise ValueError("unknown claim")
        roots: set[str] = set()
        for node in self.nodes:
            if node.claim_id == claim_id:
                roots.update(self.root_sources_for_node(node.node_id))
        return frozenset(roots)

    def public_summary(self) -> dict[str, object]:
        """Return safe graph metadata without claim text, labels, or prompts."""
        return {
            "scenario_id": self.scenario_id,
            "scenario_type": self.scenario_type.value,
            "claim_count": len(self.claims),
            "source_root_count": len(self.source_roots),
            "evidence_card_count": len(self.evidence_cards),
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "claim_root_counts": {
                claim.claim_id: len(self.root_sources_for_claim(claim.claim_id)) for claim in self.claims
            },
        }


class GroundTruthLabel(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNCERTAIN = "uncertain"


class SourceIndependenceLabel(str, Enum):
    INDEPENDENT = "independent"
    DEPENDENT = "dependent"
    UNKNOWN = "unknown"


class EvaluatorTruthRecord(BaseModel):
    """Private evaluator annotation; never nested in public graph objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    ground_truth_label: GroundTruthLabel
    source_independence_label: SourceIndependenceLabel
    review_status: Literal["reviewed", "pending"]
    reviewer_id: str = Field(min_length=1)
    reviewed_on: str = Field(min_length=1)
    rationale: str = Field(min_length=1)

    @field_validator("claim_id", "reviewer_id", "reviewed_on", "rationale")
    @classmethod
    def private_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))


class EvaluatorTruthFixture(BaseModel):
    """Private file wrapper kept separate from the public scenario loader."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    records: list[EvaluatorTruthRecord] = Field(min_length=1)

    @field_validator("scenario_id")
    @classmethod
    def private_scenario_id(cls, value: str) -> str:
        return normalized_text(value, "scenario_id")

    @model_validator(mode="after")
    def unique_claims(self) -> "EvaluatorTruthFixture":
        ids = [record.claim_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate evaluator claim ID")
        return self


class ProvenanceFixtureError(ValueError):
    """Stable error for local public/private fixture validation."""


class PublicScenarioLoader:
    @staticmethod
    def load(path: str | Path) -> ProvenanceGraph:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            return ProvenanceGraph.model_validate(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise ProvenanceFixtureError("public_provenance_fixture_invalid") from exc


class EvaluatorTruthLoader:
    @staticmethod
    def load(path: str | Path) -> EvaluatorTruthFixture:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            return EvaluatorTruthFixture.model_validate(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise ProvenanceFixtureError("evaluator_truth_fixture_invalid") from exc


def validate_fixture_pair(public_graph: ProvenanceGraph, private_fixture: EvaluatorTruthFixture) -> dict[str, object]:
    """Join only stable IDs for offline validation; never expose labels."""
    claim_ids = {claim.claim_id for claim in public_graph.claims}
    private_ids = {record.claim_id for record in private_fixture.records}
    if public_graph.scenario_id != private_fixture.scenario_id:
        raise ProvenanceFixtureError("scenario_id_mismatch")
    if private_ids != claim_ids:
        raise ProvenanceFixtureError("evaluator_claim_set_mismatch")
    summary = public_graph.public_summary()
    summary.update({
        "status": "validated",
        "evaluator_record_count": len(private_fixture.records),
        "evaluator_labels_excluded_from_public": True,
    })
    return summary


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "Claim", "EvaluatorTruthFixture", "EvaluatorTruthLoader", "EvaluatorTruthRecord", "GroundTruthLabel",
    "ProvenanceEdge", "ProvenanceFixtureError", "ProvenanceGraph", "ProvenanceNode", "ProvenanceRelation",
    "PublicScenarioLoader", "SourceCategory", "SourceIndependenceLabel", "SourceRoot", "VerificationStatus",
    "file_sha256", "validate_fixture_pair",
]
