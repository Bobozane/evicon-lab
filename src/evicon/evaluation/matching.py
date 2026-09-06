"""Explicit matched-run pairing for offline counterfactual evaluation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .loaders import LoadedEvaluationRun
from .models import EvaluationCondition, MatchedRunPair, UnmatchedEntry


@dataclass(frozen=True)
class MatchingResult:
    """All valid pairs and all declared entries that could not be paired."""

    matched_pairs: tuple[MatchedRunPair, ...]
    unmatched_entries: tuple[UnmatchedEntry, ...]


def match_runs(runs: Iterable[LoadedEvaluationRun]) -> MatchingResult:
    """Match only entries with identical declared comparison keys.

    The manifest's counterfactual group identifies intended companions. Scenario,
    agents, model, seed, and probe set must also match. The protocol is not a
    key because it is the intentionally varied treatment in cross-condition
    comparisons. Ambiguous groups are reported rather than resolved by order.
    """
    ordered_runs = tuple(sorted(runs, key=lambda item: item.entry.run_id))
    pairs: list[MatchedRunPair] = []
    unmatched: list[UnmatchedEntry] = []

    for comparison, left, right, scope in (
        (
            "independent_social",
            EvaluationCondition.INDEPENDENT,
            EvaluationCondition.SOCIAL_ONLY,
            "standard",
        ),
        (
            "evidence_social",
            EvaluationCondition.EVIDENCE_ONLY,
            EvaluationCondition.EVIDENCE_SOCIAL,
            "standard",
        ),
        (
            "holdout_evidence_social",
            EvaluationCondition.EVIDENCE_ONLY,
            EvaluationCondition.EVIDENCE_SOCIAL,
            "holdout",
        ),
        (
            "baseline_evicon",
            EvaluationCondition.GENERIC_MEDIATOR,
            EvaluationCondition.EVICON,
            "standard",
        ),
    ):
        new_pairs, new_unmatched = _match_condition_pair(
            ordered_runs,
            comparison=comparison,
            left_condition=left,
            right_condition=right,
            scope=scope,
        )
        pairs.extend(new_pairs)
        unmatched.extend(new_unmatched)

    for scope in ("standard", "holdout"):
        new_pairs, new_unmatched = _match_initial_final(ordered_runs, scope=scope)
        pairs.extend(new_pairs)
        unmatched.extend(new_unmatched)

    return MatchingResult(
        matched_pairs=tuple(
            sorted(
                pairs,
                key=lambda pair: (
                    pair.comparison,
                    pair.scope,
                    pair.left_run_id,
                    pair.right_run_id,
                ),
            )
        ),
        unmatched_entries=tuple(
            sorted(
                _unique_unmatched(unmatched),
                key=lambda item: (item.comparison, item.run_id, item.reason),
            )
        ),
    )


def _match_condition_pair(
    runs: tuple[LoadedEvaluationRun, ...],
    *,
    comparison: str,
    left_condition: EvaluationCondition,
    right_condition: EvaluationCondition,
    scope: str,
) -> tuple[list[MatchedRunPair], list[UnmatchedEntry]]:
    candidates = [
        item
        for item in runs
        if item.entry.condition in {left_condition, right_condition}
        and item.entry.role == "final"
        and _has_scope(item, scope)
    ]
    groups: dict[tuple[object, ...], list[LoadedEvaluationRun]] = defaultdict(list)
    for item in candidates:
        groups[_condition_key(item, scope)].append(item)

    pairs: list[MatchedRunPair] = []
    unmatched: list[UnmatchedEntry] = []
    for key in sorted(groups, key=repr):
        group = groups[key]
        left_items = [item for item in group if item.entry.condition is left_condition]
        right_items = [item for item in group if item.entry.condition is right_condition]
        if len(left_items) == 1 and len(right_items) == 1:
            pairs.append(
                MatchedRunPair(
                    comparison=comparison,
                    scope=scope,
                    left_run_id=left_items[0].entry.evaluation_entry_id,
                    right_run_id=right_items[0].entry.evaluation_entry_id,
                    probe_set_id=left_items[0].probe_set_id,
                    matching_key=_key_payload(key, scope=scope),
                )
            )
            continue
        reason = _pairing_reason(left_items, right_items, left_condition, right_condition)
        unmatched.extend(
            UnmatchedEntry(
                run_id=item.entry.run_id,
                comparison=comparison,
                reason=reason,
            )
            for item in group
        )
    return pairs, unmatched


def _match_initial_final(
    runs: tuple[LoadedEvaluationRun, ...],
    *,
    scope: str,
) -> tuple[list[MatchedRunPair], list[UnmatchedEntry]]:
    candidates = [
        item
        for item in runs
        if item.entry.role in {"initial", "final"} and _has_scope(item, scope)
    ]
    groups: dict[tuple[object, ...], list[LoadedEvaluationRun]] = defaultdict(list)
    for item in candidates:
        groups[_initial_final_key(item, scope)].append(item)

    pairs: list[MatchedRunPair] = []
    unmatched: list[UnmatchedEntry] = []
    for key in sorted(groups, key=repr):
        group = groups[key]
        initial = [item for item in group if item.entry.role == "initial"]
        final = [item for item in group if item.entry.role == "final"]
        if len(initial) == 1 and len(final) == 1:
            pairs.append(
                MatchedRunPair(
                    comparison="initial_final",
                    scope=scope,
                    left_run_id=initial[0].entry.evaluation_entry_id,
                    right_run_id=final[0].entry.evaluation_entry_id,
                    probe_set_id=initial[0].probe_set_id,
                    matching_key=_key_payload(key, scope=scope),
                )
            )
            continue
        reason = _role_pairing_reason(initial, final)
        unmatched.extend(
            UnmatchedEntry(
                run_id=item.entry.evaluation_entry_id,
                comparison="initial_final",
                reason=reason,
            )
            for item in group
        )
    return pairs, unmatched


def _has_scope(item: LoadedEvaluationRun, scope: str) -> bool:
    if not item.probe_results:
        return False
    is_holdout = item.probe_results[0].is_holdout
    return (scope == "holdout") is is_holdout


def _condition_key(item: LoadedEvaluationRun, scope: str) -> tuple[object, ...]:
    return (
        item.record.config.scenario_id,
        item.entry.counterfactual_group_id,
        item.record.config.model_name,
        item.agent_ids,
        item.record.config.seed,
        item.probe_set_id,
        scope,
    )


def _initial_final_key(item: LoadedEvaluationRun, scope: str) -> tuple[object, ...]:
    return _condition_key(item, scope) + (item.entry.condition.value,)


def _key_payload(key: tuple[object, ...], *, scope: str) -> dict[str, object]:
    (
        scenario_id,
        group_id,
        model_name,
        agent_ids,
        seed,
        probe_set_id,
        _,
        *tail,
    ) = key
    payload: dict[str, object] = {
        "scenario_id": scenario_id,
        "counterfactual_group_id": group_id,
        "model_name": model_name,
        "agent_ids": list(agent_ids) if isinstance(agent_ids, tuple) else agent_ids,
        "seed": seed,
        "probe_set_id": probe_set_id,
        "scope": scope,
    }
    if tail:
        payload["condition"] = tail[0]
    return payload


def _pairing_reason(
    left: list[LoadedEvaluationRun],
    right: list[LoadedEvaluationRun],
    left_condition: EvaluationCondition,
    right_condition: EvaluationCondition,
) -> str:
    if not left:
        return f"missing required {left_condition.value} companion with the same matching key"
    if not right:
        return f"missing required {right_condition.value} companion with the same matching key"
    return "ambiguous matching key: each condition must occur exactly once"


def _role_pairing_reason(initial: list[LoadedEvaluationRun], final: list[LoadedEvaluationRun]) -> str:
    if not initial:
        return "missing required initial companion with the same matching key"
    if not final:
        return "missing required final companion with the same matching key"
    return "ambiguous matching key: initial and final must each occur exactly once"


def _unique_unmatched(entries: list[UnmatchedEntry]) -> list[UnmatchedEntry]:
    unique: dict[tuple[str, str, str], UnmatchedEntry] = {}
    for entry in entries:
        unique[(entry.run_id, entry.comparison, entry.reason)] = entry
    return list(unique.values())
