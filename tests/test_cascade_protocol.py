from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evicon.cascade_protocol import (
    ActorAction,
    ActorScheduleEntry,
    CascadeProtocolError,
    CascadeProtocolRunner,
    CascadeScenarioLoader,
)
from evicon.cascade_protocol_smoke import main as smoke_main
from evicon.cascade_replay import CascadeReplayValidationError, CascadeReplayValidator
from evicon.provenance_cascade_exposure import (
    ControllerPublicView,
    ExposureChannel,
    ExposureEvent,
    ExposureLedger,
    VisibilityReason,
)

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "configs/provenance_cascade/scenarios"
NAMES = ("false_majority", "true_minority_correction", "independent_true_consensus", "unresolved_disagreement")
SEEDS = (20260901, 20260902, 20260903)


def load(name: str):
    return CascadeScenarioLoader.load(SCENARIOS / f"{name}.toml")


def test_four_scenarios_run_with_fixed_public_shape_and_replay() -> None:
    runner = CascadeProtocolRunner()
    for name in NAMES:
        scenario = load(name)
        record = runner.run_scenario(scenario, SEEDS[0])
        assert record.replay.status.value == "passed"
        assert record.agent_ids == tuple(f"network-agent-{i:02d}" for i in range(1, 7))
        assert record.round_count == 3
        assert record.snapshot_count == 18
        assert len(record.ledger.agent_ids) == 6


def test_same_seed_schedule_and_topology_are_bitwise_deterministic() -> None:
    scenario = load("false_majority")
    runner = CascadeProtocolRunner()
    first = runner.run_scenario(scenario, 20260901)
    second = runner.run_scenario(scenario, 20260901)
    assert first.trajectory_sha256 == second.trajectory_sha256
    assert first.ledger.model_dump(mode="json") == second.ledger.model_dump(mode="json")
    with pytest.raises(CascadeProtocolError) as exc:
        runner.run_scenario(scenario, 20260901, policy="generic_dissent")
    assert exc.value.code == "policy_not_implemented"


def test_round_start_snapshot_excludes_same_round_events() -> None:
    record = CascadeProtocolRunner().run_scenario(load("false_majority"), 20260901)
    by_coord = {(snapshot.agent_id, snapshot.round_id): snapshot for snapshot in record.ledger.snapshots}
    assert by_coord[("network-agent-02", 1)].captured_event_ids == ()
    assert by_coord[("network-agent-02", 2)].captured_event_ids == ("cascade-false-majority-s20260901-r1-e0-t0",)
    assert by_coord[("network-agent-03", 2)].captured_event_ids == ()
    assert not any(snapshot.agent_id == "network-agent-03" and snapshot.round_id == 3 for snapshot in record.ledger.snapshots)


def test_root_resolution_preserves_repetition_and_independence() -> None:
    runner = CascadeProtocolRunner()
    assert runner.run_scenario(load("false_majority"), 1).independent_root_count_by_claim == {"claim-fm": 1}
    assert runner.run_scenario(load("independent_true_consensus"), 1).independent_root_count_by_claim == {"claim-itc": 2}
    assert runner.run_scenario(load("unresolved_disagreement"), 1).independent_root_count_by_claim == {"claim-ud": 2}


def test_minority_correction_is_publicly_forwarded_without_identity_filter() -> None:
    record = CascadeProtocolRunner().run_scenario(load("true_minority_correction"), 1)
    assert any(event.provenance_node_id == "tmc-correction-reply" and event.target_agent_id == "network-agent-03" for event in record.ledger.events)
    assert all("ground_truth_label" not in json.dumps(event.model_dump(mode="json")) for event in record.ledger.events)


def test_unresolved_scenario_keeps_public_contestation_without_truth_field() -> None:
    record = CascadeProtocolRunner().run_scenario(load("unresolved_disagreement"), 1)
    serialized = json.dumps(record.model_dump(mode="json"), sort_keys=True)
    assert "ground_truth_label" not in serialized
    assert "source_independence_label" not in serialized
    assert "contested" not in serialized  # verification status is graph metadata, not an event field
    assert {event.provenance_node_id for event in record.ledger.events} == {"ud-a", "ud-b"}


def test_invalid_schedule_and_snapshot_are_rejected() -> None:
    scenario = load("false_majority")
    invalid_entry = ActorScheduleEntry(
        round_id=1,
        actor_id="network-agent-01",
        target_agent_ids=("network-agent-02",),
        action=ActorAction.REPOST,
        provenance_node_id="missing-node",
    )
    with pytest.raises(ValidationError):
        scenario.__class__.model_validate({**scenario.model_dump(mode="python"), "actor_schedule": (invalid_entry,)})

    record = CascadeProtocolRunner().run_scenario(scenario, 1)
    payload = record.ledger.model_dump(mode="json")
    payload["snapshots"][7]["visible_content_ids"] = ["fm-content-0"]
    with pytest.raises(CascadeReplayValidationError):
        CascadeReplayValidator.validate(scenario.graph, ExposureLedger.model_validate(payload))


def test_actor_cannot_forward_content_before_exposure() -> None:
    scenario = load("false_majority")
    entry = ActorScheduleEntry(
        round_id=1,
        actor_id="network-agent-04",
        target_agent_ids=("network-agent-05",),
        action=ActorAction.REPOST,
        provenance_node_id="fm-repost-1",
    )
    with pytest.raises(CascadeProtocolError) as exc:
        CascadeProtocolRunner().append_public_event(
            scenario,
            events=(),
            snapshots=CascadeProtocolRunner().build_round_snapshot(scenario, (), 1),
            entry=entry,
            target_agent_id="network-agent-05",
            event_id="bad",
            round_snapshot=CascadeProtocolRunner().build_round_snapshot(scenario, (), 1),
        )
    assert exc.value.code == "provenance_parent_not_exposed"


def test_smoke_cli_is_offline_and_safe(capsys: pytest.CaptureFixture[str]) -> None:
    assert smoke_main(["--scenario-dir", str(SCENARIOS), "--seeds", "20260901"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["status"] == "offline_smoke"
    assert len(payload["scenarios"]) == 4
    assert all(item["replay_status"] == "passed" for item in payload["scenarios"])
    assert "ground_truth_label" not in output
    assert "source_independence_label" not in output
    assert "results" not in output
    assert "prompt" not in output


def test_visible_child_does_not_authorize_its_unexposed_parent_for_any_actor_action() -> None:
    scenario = load("false_majority")
    child_delivery = ExposureEvent(
        event_id="visible-child",
        target_agent_id="network-agent-04",
        source_actor_id="system",
        round_id=1,
        content_id="fm-content-1",
        claim_id="claim-fm",
        provenance_node_id="fm-repost-1",
        channel=ExposureChannel.TIMELINE,
        visibility_reason=VisibilityReason.TIMELINE_DELIVERY,
    )
    runner = CascadeProtocolRunner()
    snapshot = runner.build_round_snapshot(scenario, (child_delivery,), 2)
    for action in (ActorAction.POST, ActorAction.REPOST, ActorAction.REPLY, ActorAction.QUOTED_EVIDENCE):
        entry = ActorScheduleEntry(
            round_id=2,
            actor_id="network-agent-04",
            target_agent_ids=("network-agent-05",),
            action=action,
            provenance_node_id="fm-root-node",
        )
        with pytest.raises(CascadeProtocolError) as exc:
            runner.append_public_event(
                scenario,
                events=(child_delivery,),
                snapshots=snapshot,
                entry=entry,
                target_agent_id="network-agent-05",
                event_id=f"hidden-parent-{action.value}",
                round_snapshot=snapshot,
            )
        assert exc.value.code == "content_not_exposed"


def test_root_relation_remains_a_safe_summary_for_visible_child_only() -> None:
    scenario = load("false_majority")
    child_delivery = ExposureEvent(
        event_id="visible-child",
        target_agent_id="network-agent-04",
        source_actor_id="system",
        round_id=1,
        content_id="fm-content-1",
        claim_id="claim-fm",
        provenance_node_id="fm-repost-1",
        channel=ExposureChannel.TIMELINE,
        visibility_reason=VisibilityReason.TIMELINE_DELIVERY,
    )
    snapshot = next(
        item
        for item in CascadeProtocolRunner().build_round_snapshot(scenario, (child_delivery,), 2)
        if item.agent_id == "network-agent-04"
    )
    view = ControllerPublicView.from_snapshot(snapshot, scenario.graph)
    serialized = json.dumps(view.model_dump(mode="json"), sort_keys=True)
    assert view.root_count_for_claim("claim-fm") == 1
    assert [node.node_id for node in view.provenance_nodes] == ["fm-repost-1"]
    assert "fm-root-node" not in serialized
    assert "fm-content-0" not in serialized
    assert "evidence-fm" not in serialized
    with pytest.raises(ValidationError):
        ControllerPublicView.model_validate({**view.model_dump(mode="json"), "parent_content_id": "fm-content-0"})


def test_replay_rejects_hidden_direct_parent_or_mismatched_parent_reference() -> None:
    scenario = load("false_majority")
    runner = CascadeProtocolRunner()
    root_delivery = ExposureEvent(
        event_id="root-visible",
        target_agent_id="network-agent-04",
        source_actor_id="system",
        round_id=0,
        content_id="fm-content-0",
        claim_id="claim-fm",
        provenance_node_id="fm-root-node",
        channel=ExposureChannel.TIMELINE,
        visibility_reason=VisibilityReason.TIMELINE_DELIVERY,
    )
    snapshots = tuple(
        item
        for round_id in range(3)
        for item in runner.build_round_snapshot(scenario, (root_delivery,), round_id)
    )
    illegal_child = ExposureEvent(
        event_id="hidden-parent-child",
        target_agent_id="network-agent-05",
        source_actor_id="network-agent-04",
        round_id=2,
        content_id="fm-content-2",
        claim_id="claim-fm",
        provenance_node_id="fm-repost-2",
        channel=ExposureChannel.REPOST,
        visibility_reason=VisibilityReason.REPOST_VISIBILITY,
        parent_content_id="fm-content-1",
    )
    with pytest.raises(CascadeReplayValidationError):
        CascadeReplayValidator.validate(
            scenario.graph,
            ExposureLedger(
                scenario_id=scenario.scenario_id,
                agent_ids=scenario.agent_ids,
                events=(root_delivery, illegal_child),
                snapshots=snapshots,
            ),
        )

    record = runner.run_scenario(scenario, 20260901)
    payload = record.ledger.model_dump(mode="json")
    payload["events"][2]["parent_content_id"] = "fm-content-0"
    with pytest.raises(CascadeReplayValidationError):
        CascadeReplayValidator.validate(scenario.graph, ExposureLedger.model_validate(payload))
