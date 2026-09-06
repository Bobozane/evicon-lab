"""Pairwise and MST structural diversity over normalized profile distances."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import combinations

from ..models import ValueProfile
from .distances import profile_distance
from .errors import InsufficientDataError, MetricError, MissingPairError
from .models import (
    MstEdge,
    PairDistance,
    PairwiseDiversityResult,
    SkippedPair,
    StructuralDiversityResult,
)


def pairwise_diversity(
    profiles: Sequence[ValueProfile],
    *,
    strict: bool = True,
) -> PairwiseDiversityResult:
    """Average all valid profile-pair distances, preserving skip reasons in non-strict mode."""
    profile_by_agent = _profile_by_agent(profiles)
    agent_ids = sorted(profile_by_agent)
    distances: list[PairDistance] = []
    skipped: list[SkippedPair] = []
    for agent_a, agent_b in combinations(agent_ids, 2):
        try:
            distance = profile_distance(profile_by_agent[agent_a], profile_by_agent[agent_b])
        except MetricError as exc:
            if strict:
                raise
            skipped.append(SkippedPair(agent_a=agent_a, agent_b=agent_b, reason=str(exc)))
            continue
        distances.append(PairDistance(agent_a=agent_a, agent_b=agent_b, distance=distance))
    if not distances:
        raise InsufficientDataError("pairwise_diversity has no valid agent pairs")
    return PairwiseDiversityResult(
        mean_distance=sum(pair.distance for pair in distances) / len(distances),
        pair_distances=distances,
        agent_ids=agent_ids,
        valid_pair_count=len(distances),
        skipped_pair_count=len(skipped),
        skipped_pairs=skipped,
    )


def structural_diversity(
    pair_distances: Sequence[PairDistance] | Mapping[tuple[str, str], float],
    agent_ids: Sequence[str],
    *,
    strict: bool = True,
) -> StructuralDiversityResult:
    """Compute the Kruskal MST span from a complete or explicitly partial distance graph."""
    nodes = _normalize_agent_ids(agent_ids)
    if len(nodes) < 2:
        raise InsufficientDataError("structural_diversity requires at least two agents")
    edge_by_pair = _edge_by_pair(pair_distances, known_agents=set(nodes))
    expected_pairs = list(combinations(nodes, 2))
    missing_pairs = [pair for pair in expected_pairs if pair not in edge_by_pair]
    if missing_pairs and strict:
        raise MissingPairError(f"structural_diversity is missing pair distances: {missing_pairs}")

    warnings = [f"missing pair distances: {missing_pairs}"] if missing_pairs else []
    selected: list[MstEdge] = []
    union_find = _UnionFind(nodes)
    sorted_edges = sorted(
        (
            MstEdge(agent_a=agent_a, agent_b=agent_b, distance=distance)
            for (agent_a, agent_b), distance in edge_by_pair.items()
        ),
        key=lambda edge: (edge.distance, edge.agent_a, edge.agent_b),
    )
    for edge in sorted_edges:
        if union_find.union(edge.agent_a, edge.agent_b):
            selected.append(edge)
            if len(selected) == len(nodes) - 1:
                break
    if len(selected) != len(nodes) - 1:
        if strict:
            raise MissingPairError("structural_diversity graph is disconnected")
        return StructuralDiversityResult(
            mst_span=None,
            mst_total_length=None,
            mst_edges=selected,
            agent_ids=nodes,
            warnings=warnings + ["distance graph is disconnected"],
            valid=False,
        )
    total_length = sum(edge.distance for edge in selected)
    return StructuralDiversityResult(
        mst_span=total_length / (len(nodes) - 1),
        mst_total_length=total_length,
        mst_edges=selected,
        agent_ids=nodes,
        warnings=warnings,
        valid=True,
    )


def _profile_by_agent(profiles: Sequence[ValueProfile]) -> dict[str, ValueProfile]:
    if len(profiles) < 2:
        raise InsufficientDataError("pairwise_diversity requires profiles for at least two agents")
    profile_by_agent = {profile.agent_id: profile for profile in profiles}
    if len(profile_by_agent) != len(profiles):
        raise MetricError("profiles must contain at most one ValueProfile per agent_id")
    return profile_by_agent


def _normalize_agent_ids(agent_ids: Sequence[str]) -> list[str]:
    if len(agent_ids) < 2:
        raise InsufficientDataError("structural_diversity requires at least two agents")
    if any(not isinstance(agent_id, str) or not agent_id.strip() for agent_id in agent_ids):
        raise MetricError("agent_ids must be non-blank strings")
    normalized = sorted(agent_id.strip() for agent_id in agent_ids)
    if len(set(normalized)) != len(normalized):
        raise MetricError("agent_ids must be unique")
    return normalized


def _edge_by_pair(
    pair_distances: Sequence[PairDistance] | Mapping[tuple[str, str], float],
    *,
    known_agents: set[str],
) -> dict[tuple[str, str], float]:
    raw_edges: list[tuple[str, str, float]] = []
    if isinstance(pair_distances, Mapping):
        for pair, distance in pair_distances.items():
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise MetricError("pair distance mappings must use (agent_a, agent_b) tuple keys")
            raw_edges.append((pair[0], pair[1], distance))
    else:
        for pair in pair_distances:
            raw_edges.append((pair.agent_a, pair.agent_b, pair.distance))

    edges: dict[tuple[str, str], float] = {}
    for agent_a, agent_b, distance in raw_edges:
        if agent_a not in known_agents or agent_b not in known_agents:
            raise MetricError("pair distance names an agent absent from agent_ids")
        if agent_a == agent_b:
            if distance != 0.0:
                raise MetricError("diagonal pair distances must be zero")
            continue
        value = _distance_value(distance)
        key = tuple(sorted((agent_a, agent_b)))
        if key in edges and not math.isclose(edges[key], value, abs_tol=1e-12):
            raise MetricError(f"conflicting symmetric pair distances for {key}")
        edges[key] = value
    return edges


def _distance_value(value: float) -> float:
    if isinstance(value, bool):
        raise MetricError("pair distance must be a finite float in [0, 1]")
    distance = float(value)
    if not math.isfinite(distance) or distance < 0.0 or distance > 1.0:
        raise MetricError("pair distance must be a finite float in [0, 1]")
    return distance


class _UnionFind:
    """Small deterministic disjoint-set implementation local to this metric module."""

    def __init__(self, nodes: Sequence[str]) -> None:
        self._parent = {node: node for node in nodes}

    def find(self, node: str) -> str:
        parent = self._parent[node]
        while parent != self._parent[parent]:
            parent = self._parent[parent]
        self._parent[node] = parent
        return parent

    def union(self, left: str, right: str) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if left_root < right_root:
            self._parent[right_root] = left_root
        else:
            self._parent[left_root] = right_root
        return True
