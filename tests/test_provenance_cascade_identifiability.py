from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evicon.cascade_agent_prompts import CascadeAgentPromptContext, CascadeAgentRuntimeConfig
from evicon.cascade_agent_protocol_hg import (
    HG_TEMPLATE_VERSION,
    build_identifiable_prompt_context,
    render_cascade_agent_turn_hg,
)
from evicon.cascade_intervention_application import (
    CascadeDirectiveType,
    StructuredCascadeDirective,
)
from evicon.cascade_protocol import CascadeProtocolRunner
from evicon.provenance_cascade import ProvenanceNode
from evicon.provenance_cascade_exposure import ControllerPublicView
from evicon.provenance_cascade_identifiability import (
    HGFakeProvider,
    hg_preflight,
    load_hg_config,
    run_hg_fake_smoke,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg.v1.toml"
OLD_CONFIG = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd21.v1.toml"
OLD_RECEIPT = ROOT / "results/provenance-cascade-pilot-hd21-v1/pilot_receipt.json"
OLD_ANALYSIS = ROOT / "results/analyses/provenance-cascade-hd21-pilot-hf-v1/analysis_report.json"
SAFE_RECEIPTS = (
    ROOT / "outputs/study-locks/provenance_cascade_hd21_non_identifiable_receipt.json",
    ROOT / "outputs/study-locks/provenance_cascade_hg_identifiability_amendment_receipt.json",
)


def _view(scenario_id: str, agent_id: str, round_id: int):
    _, scenarios, _ = load_hg_config(CONFIG)
    scenario = scenarios[scenario_id]
    run = CascadeProtocolRunner().run_scenario(scenario, 20260911)
    snapshot = next(
        item for item in run.ledger.snapshots
        if item.agent_id == agent_id and item.round_id == round_id
    )
    return scenario, ControllerPublicView.from_snapshot(snapshot, scenario.graph)


def test_audit_confirms_old_builder_used_ids_but_hg_uses_public_text():
    _, view = _view("cascade-hg-false-majority", "network-agent-01", 1)
    old = CascadeAgentPromptContext.from_public_view(
        view,
        claim_id="claim-hg-fm",
        runtime_config=CascadeAgentRuntimeConfig(model_name="fake", max_tokens=512),
    )
    assert old.claims[0].public_summary == old.claims[0].claim_id
    assert old.contents[0].public_summary == old.contents[0].content_id
    new = build_identifiable_prompt_context(
        view,
        claim_id="claim-hg-fm",
        runtime_config=CascadeAgentRuntimeConfig(model_name="fake", max_tokens=512),
    )
    assert new.claims[0].public_summary != new.claims[0].claim_id
    assert all(item.public_summary != item.content_id for item in new.contents)
    rendered = json.loads(render_cascade_agent_turn_hg(new).user_prompt)
    assert rendered["template_version"] == HG_TEMPLATE_VERSION
    assert "HarborLink" in rendered["visible_claims"][0]["public_summary"]


def test_same_root_repetition_has_distinct_public_statements_and_one_root():
    _, view = _view("cascade-hg-false-majority", "network-agent-01", 1)
    context = build_identifiable_prompt_context(view, claim_id="claim-hg-fm")
    assert len(context.contents) == 2
    assert len({item.public_summary for item in context.contents}) == 2
    assert len({root for item in context.contents for root in item.source_root_ids}) == 1
    assert view.root_count_for_claim("claim-hg-fm") == 1


def test_independent_public_sources_are_distinguishable_without_private_label():
    _, view = _view("cascade-hg-independent-true-consensus", "network-agent-01", 1)
    context = build_identifiable_prompt_context(view, claim_id="claim-hg-itc")
    assert len(context.contents) == 2
    assert len({item.public_summary for item in context.contents}) == 2
    assert view.root_count_for_claim("claim-hg-itc") == 2
    payload = render_cascade_agent_turn_hg(context).user_prompt.lower()
    assert "source_independence_label" not in payload
    assert "ground_truth_label" not in payload
    assert "evaluator" not in payload


def test_unexposed_global_content_is_not_projected():
    scenario, view = _view("cascade-hg-true-minority-correction", "network-agent-02", 1)
    context = build_identifiable_prompt_context(view, claim_id="claim-hg-tmc-rumor")
    visible = {item.content_id for item in context.contents}
    graph_content = {item.content_id for item in scenario.graph.nodes}
    assert visible == {"hg-tmc-content-rumor"}
    assert graph_content - visible
    assert not any("Saturday" in item.public_summary for item in context.contents)


def test_content_sensitive_fake_provider_changes_stance_after_public_correction():
    _, before_view = _view("cascade-hg-true-minority-correction", "network-agent-01", 1)
    before = build_identifiable_prompt_context(before_view, claim_id="claim-hg-tmc-rumor")
    before_request = render_cascade_agent_turn_hg(before)
    before_response = json.loads(HGFakeProvider().complete(before_request).content)
    assert before_response["stance"] == "endorses"

    _, after_view = _view("cascade-hg-true-minority-correction", "network-agent-01", 2)
    after = build_identifiable_prompt_context(after_view, claim_id="claim-hg-tmc-rumor")
    after_request = render_cascade_agent_turn_hg(after)
    after_response = json.loads(HGFakeProvider().complete(after_request).content)
    assert after_response["stance"] == "rejects"
    assert after_response["evidence_ids_used"] == ["evidence-hg-tmc-schedule"]


def test_structured_directive_changes_behavior_without_adding_public_material():
    scenario, view = _view("cascade-hg-false-majority", "network-agent-01", 2)
    baseline = build_identifiable_prompt_context(view, claim_id="claim-hg-fm")
    directive = StructuredCascadeDirective(
        directive_id="directive-hg-fm-agent01-round2",
        directive_type=CascadeDirectiveType.VERIFICATION_REQUEST,
        scenario_id=scenario.scenario_id,
        target_agent_id=view.agent_id,
        claim_id="claim-hg-fm",
        effective_round_id=view.round_id,
        visible_content_ids=tuple(item.content_id for item in baseline.contents),
        visible_evidence_ids=tuple(item.evidence_id for item in baseline.evidence),
        visible_same_root_repetition_count=2,
        visible_root_count=1,
        reason_codes=("visible_unverified_same_root_repetition",),
    )
    directed = build_identifiable_prompt_context(
        view,
        claim_id="claim-hg-fm",
        directive=directive,
    )

    provider = HGFakeProvider()
    baseline_response = json.loads(provider.complete(render_cascade_agent_turn_hg(baseline)).content)
    directed_response = json.loads(provider.complete(render_cascade_agent_turn_hg(directed)).content)

    assert baseline_response["stance"] == "endorses"
    assert directed_response["stance"] == "uncertain"
    assert directed.claims == baseline.claims
    assert directed.contents == baseline.contents
    assert directed.evidence == baseline.evidence
    assert set(directive.visible_content_ids) == {item.content_id for item in baseline.contents}
    assert set(directive.visible_evidence_ids) == {item.evidence_id for item in baseline.evidence}


def test_public_statement_rejects_private_markers():
    with pytest.raises(ValidationError):
        ProvenanceNode(
            node_id="n",
            content_id="c",
            scenario_id="cascade-x",
            claim_id="claim-x",
            source_root_id="root-x",
            round_id=0,
            public_statement="ground_truth_label=false",
        )


def test_full_48_run_fake_smoke_is_replayable_and_identifiable():
    summary = run_hg_fake_smoke(CONFIG)
    assert summary["status"] == "fake_smoke_passed"
    assert summary["run_count"] == 48
    assert summary["matched_group_count"] == 12
    assert summary["logical_request_count"] == 864
    assert summary["provider_call_count"] == 864
    assert summary["replay_passed_count"] == 48
    assert summary["beneficial_receptivity_eligible"] > 0
    assert summary["supported_correction_eligible"] > 0
    assert summary["condition_operations"]["provenance_aware_controller"]["scheduled_count"] == 3
    assert summary["condition_operations"]["provenance_aware_controller"]["applied_count"] == 3
    assert summary["scenario_condition_operations"]["cascade-hg-false-majority|provenance_aware_controller"]["applied_count"] == 3
    assert summary["private_truth_exposed"] is False
    assert summary["results_written"] is False


def test_preflight_is_offline_and_blocks_overwrite_after_pilot_start():
    report = hg_preflight(CONFIG)
    assert report["status"] == "blocked"
    assert report["blocking_reasons"] == ["output_root_exists"]
    assert report["approval_status"] == "accepted"
    assert report["ready_for_real_pilot"] is False
    assert report["run_count"] == 48
    assert report["logical_request_cap"] == 864
    assert report["completion_reservation_cap"] == 442368
    assert report["network"] == "disabled"
    assert report["provider_constructed"] is False
    assert report["api_key_read"] is False
    assert report["results_written"] is False


def test_hg_receipts_do_not_store_prompts_responses_or_private_labels():
    serialized = " ".join(path.read_text(encoding="utf-8").lower() for path in SAFE_RECEIPTS)
    for forbidden in (
        "system_prompt",
        "user_prompt",
        "model_response",
        "provider_metadata",
        "api_key",
        "ground_truth_label",
        "source_independence_label",
    ):
        assert forbidden not in serialized


def test_parent_pilot_and_analysis_hashes_remain_bound_and_untouched():
    config, _, _ = load_hg_config(CONFIG)
    assert sha256_file(OLD_CONFIG) == config.parent_config_sha256
    assert sha256_file(OLD_RECEIPT) == config.parent_pilot_receipt_sha256
    assert sha256_file(OLD_ANALYSIS) == config.parent_analysis_report_sha256
    assert config.output_root == "results/provenance-cascade-pilot-hg-v1"
    assert config.output_root != "results/provenance-cascade-pilot-hd21-v1"
