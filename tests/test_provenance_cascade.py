from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evicon.provenance_cascade import (
    EvaluatorTruthFixture,
    EvaluatorTruthLoader,
    ProvenanceFixtureError,
    ProvenanceGraph,
    PublicScenarioLoader,
    file_sha256,
    validate_fixture_pair,
)
from evicon.validate_provenance_cascade import main


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "configs/provenance_cascade/fixtures"


def public_payload(name: str = "false_majority") -> dict[str, object]:
    return json.loads((FIXTURES / f"{name}.public.json").read_text(encoding="utf-8"))


def test_all_four_public_private_fixture_pairs_validate() -> None:
    for name in ("false_majority", "true_minority_correction", "independent_true_consensus", "unresolved_disagreement"):
        public_path = FIXTURES / f"{name}.public.json"
        private_path = FIXTURES / f"{name}.evaluator_private.json"
        public = PublicScenarioLoader.load(public_path)
        private = EvaluatorTruthLoader.load(private_path)
        summary = validate_fixture_pair(public, private)
        assert summary["status"] == "validated"
        assert summary["evaluator_labels_excluded_from_public"] is True


def test_repeated_reposts_resolve_to_one_root_and_independent_roots_to_two() -> None:
    false_graph = PublicScenarioLoader.load(FIXTURES / "false_majority.public.json")
    consensus_graph = PublicScenarioLoader.load(FIXTURES / "independent_true_consensus.public.json")
    assert len(false_graph.root_sources_for_claim("claim-fm")) == 1
    assert len(consensus_graph.root_sources_for_claim("claim-itc")) == 2


def test_public_json_and_safe_summary_do_not_expose_private_labels() -> None:
    graph = PublicScenarioLoader.load(FIXTURES / "true_minority_correction.public.json")
    serialized = json.dumps(graph.model_dump(mode="json"), sort_keys=True)
    summary = json.dumps(graph.public_summary(), sort_keys=True)
    assert "ground_truth_label" not in serialized
    assert "source_independence_label" not in serialized
    assert "ground_truth_label" not in summary
    assert "source_independence_label" not in summary


def test_public_claim_rejects_evaluator_truth_field() -> None:
    payload = public_payload()
    payload["claims"][0]["ground_truth_label"] = "false"  # type: ignore[index]
    with pytest.raises(ValueError):
        ProvenanceGraph.model_validate(payload)


def test_duplicate_ids_unknown_nodes_and_cross_claim_edges_are_rejected() -> None:
    duplicate = public_payload()
    duplicate["nodes"].append(copy.deepcopy(duplicate["nodes"][0]))  # type: ignore[index]
    with pytest.raises(ValueError):
        ProvenanceGraph.model_validate(duplicate)

    unknown_parent = public_payload()
    unknown_parent["edges"][0]["source_node_id"] = "missing-node"  # type: ignore[index]
    with pytest.raises(ValueError):
        ProvenanceGraph.model_validate(unknown_parent)

    cross_claim = public_payload()
    cross_claim["claims"].append({"claim_id": "claim-other", "public_summary": "Another public claim.", "verification_status": "unverified", "evidence_card_ids": []})  # type: ignore[index]
    cross_claim["nodes"].append({"node_id": "other-node", "content_id": "other-content", "scenario_id": "cascade-false-majority", "claim_id": "claim-other", "source_root_id": "root-fm", "round_id": 1})  # type: ignore[index]
    cross_claim["edges"][0]["target_node_id"] = "other-node"  # type: ignore[index]
    with pytest.raises(ValueError):
        ProvenanceGraph.model_validate(cross_claim)


def test_cycle_and_time_reversal_are_rejected() -> None:
    cycle = public_payload()
    cycle["edges"].append({"edge_id": "fm-cycle", "source_node_id": "fm-repost-3", "target_node_id": "fm-root-node", "relation": "reply"})  # type: ignore[index]
    with pytest.raises(ValueError):
        ProvenanceGraph.model_validate(cycle)

    reversal = public_payload()
    reversal["nodes"][1]["round_id"] = 1  # type: ignore[index]
    reversal["nodes"][2]["round_id"] = 2  # type: ignore[index]
    reversal["nodes"][3]["round_id"] = 3  # type: ignore[index]
    reversal["edges"][1]["source_node_id"] = "fm-repost-2"  # type: ignore[index]
    reversal["edges"][1]["target_node_id"] = "fm-repost-1"  # type: ignore[index]
    with pytest.raises(ValueError):
        ProvenanceGraph.model_validate(reversal)


def test_shared_upstream_parent_cannot_claim_a_different_root() -> None:
    payload = public_payload()
    payload["source_roots"].append({"source_root_id": "root-fm-fake", "public_source_category": "independent_report", "evidence_card_ids": []})  # type: ignore[index]
    payload["nodes"][1]["source_root_id"] = "root-fm-fake"  # type: ignore[index]
    with pytest.raises(ValueError, match="independent roots"):
        ProvenanceGraph.model_validate(payload)


def test_quotes_evidence_must_be_allowed_and_not_future() -> None:
    payload = public_payload("true_minority_correction")
    payload["edges"][0]["evidence_card_id"] = "evidence-not-allowed"  # type: ignore[index]
    with pytest.raises(ValueError):
        ProvenanceGraph.model_validate(payload)

    future = public_payload("true_minority_correction")
    future["nodes"][2]["round_id"] = 0  # type: ignore[index]
    with pytest.raises(ValueError):
        ProvenanceGraph.model_validate(future)


def test_private_fixture_is_separate_and_claim_set_must_match() -> None:
    private = EvaluatorTruthLoader.load(FIXTURES / "false_majority.evaluator_private.json")
    assert "ground_truth_label" in private.model_dump(mode="json")["records"][0]
    graph = PublicScenarioLoader.load(FIXTURES / "false_majority.public.json")
    altered = EvaluatorTruthFixture.model_validate({"scenario_id": private.scenario_id, "records": [{**private.records[0].model_dump(mode="json"), "claim_id": "unknown-claim"}]})
    with pytest.raises(ProvenanceFixtureError, match="claim_set"):
        validate_fixture_pair(graph, altered)


def test_cli_is_local_and_safe(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--fixture-dir", str(FIXTURES)]) == 0
    output = capsys.readouterr().out
    parsed = json.loads(output)
    assert parsed["fixture_count"] == 4
    assert parsed["status"] == "validated"
    assert "ground_truth_label" not in output
    assert "source_independence_label" not in output
    assert "prompt" not in output.lower()
    assert "api_key" not in output.lower()
    assert "results" not in output.lower()


def test_file_hash_is_local_and_stable() -> None:
    path = FIXTURES / "false_majority.public.json"
    assert file_sha256(path) == file_sha256(path)
