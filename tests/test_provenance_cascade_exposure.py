from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evicon.cascade_exposure_smoke import main
from evicon.cascade_replay import CascadeReplayValidationError, CascadeReplayValidator, ExposureLedgerLoader
from evicon.provenance_cascade import PublicScenarioLoader
from evicon.provenance_cascade_exposure import (
    ControllerPublicView,
    ExposureLedger,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "configs/provenance_cascade/fixtures"


def load_pair(name: str) -> tuple[object, object]:
    graph = PublicScenarioLoader.load(FIXTURES / f"{name}.public.json")
    ledger = ExposureLedgerLoader.load(FIXTURES / f"{name}.exposure_ledger.json")
    return graph, ledger


def ledger_payload(name: str = "false_majority") -> dict[str, object]:
    return json.loads((FIXTURES / f"{name}.exposure_ledger.json").read_text(encoding="utf-8"))


def test_all_four_exposure_ledgers_replay_without_private_inputs() -> None:
    for name in ("false_majority", "true_minority_correction", "independent_true_consensus", "unresolved_disagreement"):
        graph, ledger = load_pair(name)
        report = CascadeReplayValidator.validate(graph, ledger)  # type: ignore[arg-type]
        assert report.status.value == "passed"


def test_controller_view_contains_only_one_snapshot_and_no_private_labels() -> None:
    graph, ledger = load_pair("false_majority")
    snapshot = ledger.timeline_for("network-agent-02").snapshots[0]  # type: ignore[union-attr]
    view = ControllerPublicView.from_snapshot(snapshot, graph)  # type: ignore[arg-type]
    assert [node.node_id for node in view.provenance_nodes] == ["fm-repost-1"]
    assert view.evidence_cards == ()
    assert view.root_count_for_claim("claim-fm") == 1
    serialized = json.dumps(view.model_dump(mode="json"), sort_keys=True)
    assert "fm-root-node" not in serialized
    assert "ground_truth_label" not in serialized
    assert "source_independence_label" not in serialized
    with pytest.raises(ValidationError):
        ControllerPublicView.model_validate({**view.model_dump(mode="json"), "ground_truth_label": "false"})


def test_true_independent_roots_remain_two_in_controller_view() -> None:
    graph, ledger = load_pair("independent_true_consensus")
    snapshot = ledger.timeline_for("network-agent-01").snapshots[0]  # type: ignore[union-attr]
    view = ControllerPublicView.from_snapshot(snapshot, graph)  # type: ignore[arg-type]
    assert view.root_count_for_claim("claim-itc") == 2


def test_unexposed_content_or_evidence_cannot_enter_snapshot() -> None:
    graph, _ledger = load_pair("false_majority")
    data = ledger_payload()
    data["snapshots"][1]["visible_content_ids"] = ["fm-content-0", "fm-content-1"]  # type: ignore[index]
    with pytest.raises(CascadeReplayValidationError):
        CascadeReplayValidator.validate(graph, ExposureLedger.model_validate(data))  # type: ignore[arg-type]

    data = ledger_payload()
    data["snapshots"][1]["visible_evidence_ids"] = ["evidence-fm"]  # type: ignore[index]
    with pytest.raises(CascadeReplayValidationError):
        CascadeReplayValidator.validate(graph, ExposureLedger.model_validate(data))  # type: ignore[arg-type]


def test_future_content_and_cross_agent_retransmission_are_rejected() -> None:
    graph, _ledger = load_pair("false_majority")
    future = ledger_payload()
    future["events"][0]["content_id"] = "fm-content-2"  # type: ignore[index]
    future["events"][0]["provenance_node_id"] = "fm-repost-2"  # type: ignore[index]
    with pytest.raises(CascadeReplayValidationError):
        CascadeReplayValidator.validate(graph, ExposureLedger.model_validate(future))  # type: ignore[arg-type]

    unauthorized = ledger_payload()
    unauthorized["events"][1]["source_actor_id"] = "network-agent-03"  # type: ignore[index]
    with pytest.raises(CascadeReplayValidationError):
        CascadeReplayValidator.validate(graph, ExposureLedger.model_validate(unauthorized))  # type: ignore[arg-type]


def test_duplicate_event_bad_channel_bad_provenance_and_time_reversal_are_rejected() -> None:
    graph, _ledger = load_pair("false_majority")
    duplicate = ledger_payload()
    duplicate["events"][1]["event_id"] = "fm-exposure-0"  # type: ignore[index]
    with pytest.raises(ValueError):
        ExposureLedger.model_validate(duplicate)

    bad_channel = ledger_payload()
    bad_channel["events"][0]["visibility_reason"] = "repost_visibility"  # type: ignore[index]
    with pytest.raises(ValueError):
        ExposureLedger.model_validate(bad_channel)

    bad_provenance = ledger_payload()
    bad_provenance["events"][0]["claim_id"] = "not-a-claim"  # type: ignore[index]
    with pytest.raises(CascadeReplayValidationError):
        CascadeReplayValidator.validate(graph, ExposureLedger.model_validate(bad_provenance))  # type: ignore[arg-type]

    reversal = ledger_payload()
    reversal["events"][2]["round_id"] = 1  # type: ignore[index]
    with pytest.raises(CascadeReplayValidationError):
        CascadeReplayValidator.validate(graph, ExposureLedger.model_validate(reversal))  # type: ignore[arg-type]


def test_snapshot_cannot_be_modified_by_later_historical_event_or_mutated_in_place() -> None:
    graph, ledger = load_pair("false_majority")
    with pytest.raises(ValidationError):
        ledger.snapshots = ()  # type: ignore[misc]

    data = ledger_payload()
    data["events"].append({"event_id": "fm-late-historical", "target_agent_id": "network-agent-01", "source_actor_id": "system", "round_id": 0, "content_id": "fm-content-0", "claim_id": "claim-fm", "provenance_node_id": "fm-root-node", "channel": "timeline", "visibility_reason": "timeline_delivery", "evidence_card_ids": []})  # type: ignore[index]
    with pytest.raises(CascadeReplayValidationError):
        CascadeReplayValidator.validate(graph, ExposureLedger.model_validate(data))  # type: ignore[arg-type]

    same_round = ledger_payload()
    same_round["events"].append({"event_id": "fm-same-round-later", "target_agent_id": "network-agent-01", "source_actor_id": "system", "round_id": 1, "content_id": "fm-content-1", "claim_id": "claim-fm", "provenance_node_id": "fm-repost-1", "channel": "timeline", "visibility_reason": "timeline_delivery", "evidence_card_ids": []})  # type: ignore[index]
    CascadeReplayValidator.validate(graph, ExposureLedger.model_validate(same_round))  # type: ignore[arg-type]


def test_unknown_agent_and_scenario_mismatch_are_rejected() -> None:
    graph, _ledger = load_pair("false_majority")
    unknown = ledger_payload()
    unknown["events"][0]["target_agent_id"] = "unknown-agent"  # type: ignore[index]
    with pytest.raises(ValueError):
        ExposureLedger.model_validate(unknown)

    mismatched = ledger_payload()
    mismatched["scenario_id"] = "different-scenario"
    with pytest.raises(ValueError):
        ExposureLedger.model_validate(mismatched)


def test_smoke_cli_is_safe_and_does_not_report_private_content(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--fixture-dir", str(FIXTURES)]) == 0
    output = capsys.readouterr().out
    summary = json.loads(output)
    assert summary["status"] == "passed"
    assert summary["fixture_count"] == 4
    lowered = output.lower()
    assert "ground_truth_label" not in lowered
    assert "source_independence_label" not in lowered
    assert "prompt" not in lowered
    assert "api_key" not in lowered
    assert "results" not in lowered
