"""Offline smoke for pure provenance-cascade controller proposals.

This module constructs public-only views from local synthetic scenarios.  It
never applies proposals, creates a provider, writes a result, or reads
private evaluator fixtures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cascade_controller import CascadeControllerPolicyLoader, propose
from .cascade_protocol import CascadeProtocolRunner, CascadeScenarioLoader
from .provenance_cascade_exposure import (
    ControllerPublicView,
    ExposureChannel,
    ExposureEvent,
    VisibilityReason,
)
from .provenance_cascade_preregistration import CascadeCondition

DEFAULT_POLICY = Path("configs/provenance_cascade/cascade_controller_policy.v1.toml")
DEFAULT_SCENARIO_DIR = Path("configs/provenance_cascade/scenarios")


def _system_event(scenario: object, *, event_id: str, node_id: str, round_id: int, evidence_ids: tuple[str, ...] = ()) -> ExposureEvent:
    graph = getattr(scenario, "graph")
    node = next(node for node in graph.nodes if node.node_id == node_id)
    return ExposureEvent(
        event_id=event_id,
        target_agent_id="network-agent-01",
        source_actor_id="system",
        round_id=round_id,
        content_id=node.content_id,
        claim_id=node.claim_id,
        provenance_node_id=node.node_id,
        channel=ExposureChannel.QUOTED_EVIDENCE if evidence_ids else ExposureChannel.TIMELINE,
        visibility_reason=VisibilityReason.EVIDENCE_DELIVERY if evidence_ids else VisibilityReason.TIMELINE_DELIVERY,
        evidence_card_ids=evidence_ids,
    )


def fixture_view(scenario_name: str, scenario_dir: Path = DEFAULT_SCENARIO_DIR) -> ControllerPublicView:
    """Build one safe, public-only view for a named synthetic scenario."""

    scenario = CascadeScenarioLoader.load(scenario_dir / f"{scenario_name}.toml")
    deliveries: dict[str, tuple[int, tuple[tuple[str, int, tuple[str, ...]], ...]]] = {
        "false_majority": (
            2,
            (("fm-root-node", 0, ("evidence-fm",)), ("fm-repost-1", 1, ())),
        ),
        "true_minority_correction": (
            2,
            (("tmc-correction", 1, ("evidence-tmc-support",)),),
        ),
        "independent_true_consensus": (
            1,
            (("itc-a", 0, ("evidence-itc-a",)), ("itc-b", 0, ("evidence-itc-b",))),
        ),
        "unresolved_disagreement": (
            2,
            (("ud-a", 1, ("evidence-ud-a",)), ("ud-b", 1, ("evidence-ud-b",))),
        ),
    }
    try:
        snapshot_round, entries = deliveries[scenario_name]
    except KeyError as exc:
        raise ValueError("unknown_public_fixture") from exc
    events = tuple(
        _system_event(
            scenario,
            event_id=f"controller-smoke-{scenario_name}-{index}",
            node_id=node_id,
            round_id=round_id,
            evidence_ids=evidence_ids,
        )
        for index, (node_id, round_id, evidence_ids) in enumerate(entries, start=1)
    )
    snapshots = CascadeProtocolRunner().build_round_snapshot(scenario, events, snapshot_round)
    snapshot = next(item for item in snapshots if item.agent_id == "network-agent-01")
    return ControllerPublicView.from_snapshot(snapshot, scenario.graph)


def run_smoke(
    policy_path: Path = DEFAULT_POLICY,
    scenario_dir: Path = DEFAULT_SCENARIO_DIR,
) -> list[dict[str, object]]:
    config = CascadeControllerPolicyLoader.load(policy_path)
    records: list[dict[str, object]] = []
    for scenario_name in (
        "false_majority",
        "true_minority_correction",
        "independent_true_consensus",
        "unresolved_disagreement",
    ):
        view = fixture_view(scenario_name, scenario_dir)
        for condition_id in config.conditions:
            proposal = propose(condition_id, view, config)
            records.append(
                {
                    "scenario_id": proposal.scenario_id,
                    "condition_id": proposal.condition_id.value,
                    "action": proposal.action.value,
                    "reason_codes": [code.value for code in proposal.reason_codes],
                    "used_content_count": len(proposal.used_content_ids),
                    "used_evidence_count": len(proposal.used_evidence_ids),
                    "used_root_count": len(proposal.used_source_root_ids),
                    "valid": proposal.valid,
                    "status": "proposal_only",
                }
            )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create local, non-executing cascade controller proposals")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    args = parser.parse_args(argv)
    records = run_smoke(args.policy, args.scenario_dir)
    print(json.dumps({"status": "offline_proposal_smoke", "proposals": records}, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
