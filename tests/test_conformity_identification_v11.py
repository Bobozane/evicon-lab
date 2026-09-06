from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.conformity_identification import (
    IdentificationCondition,
    IdentificationStage,
    load_identification_config,
    sha256_file,
)
from evicon.conformity_identification_analysis_plan import ContrastId, load_analysis_plan, safe_analysis_preflight
from evicon.conformity_identification_protocol_v11 import BranchPromptContext, render_branch_turn
from evicon.conformity_identification_smoke import _context
from evicon.conformity_identification_v11 import load_v11_config, validate_branch_plan
from evicon.conformity_identification_v11_smoke import run_fake_smoke
from evicon.conformity_identification_stability import (
    load_stability_config,
    preflight as stability_preflight,
    run_fake_smoke as run_stability_smoke,
)
from evicon.conformity_source_manipulation import (
    RootAssignment,
    SourceProjection,
    parse_manipulation_response,
    safe_preflight as manipulation_preflight,
)
from evicon.conformity_source_manipulation_smoke import build_cases, run_fake_smoke as run_manipulation_smoke
from evicon.validate_conformity_identification_v11 import preflight

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "configs/provenance_cascade/identification/conformity_identification_study.v1.toml"
V11 = ROOT / "configs/provenance_cascade/identification/conformity_identification_study.v1_1.toml"


def test_v1_remains_immutable_and_v11_is_independent():
    assert sha256_file(V1) == "17fdaac5c794aac507a2b5e7dad36ab5695e2f4fb7af8556b88260985114aaec"
    config, parent = load_v11_config(V11)
    assert config.parent_config_sha256 == sha256_file(V1)
    assert config.study_id != parent.study_id
    assert config.output_root != parent.output_root


def test_shared_t0_plan_has_one_checkpoint_and_six_branches_per_group():
    config, _ = load_v11_config(V11)
    validate_branch_plan(config)
    assert len(config.shared_t0_specs) == 12
    assert len(config.branch_runs) == 72
    for checkpoint in config.shared_t0_specs:
        branches = [item for item in config.branch_runs if item.matched_group_id == checkpoint.matched_group_id]
        assert len(branches) == 6
        assert {item.condition for item in branches} == set(IdentificationCondition)
        assert {item.shared_t0_checkpoint_id for item in branches} == {checkpoint.checkpoint_id}


def test_shared_t0_budget_is_exact_and_lower_than_v1():
    config, parent = load_v11_config(V11)
    assert config.logical_request_cap == 12 * 6 + 72 * 24 == 1800
    assert config.completion_reservation_cap == 1800 * 512 == 921_600
    assert config.logical_request_cap < parent.logical_request_cap
    assert config.completion_reservation_cap < parent.completion_reservation_cap


def test_branch_identity_changes_fingerprint_not_agent_prompt():
    config, _ = load_v11_config(V11)
    _, scenarios = load_identification_config(config.parent_config_path)
    scenario = scenarios[0]
    common = _context(
        scenario, IdentificationCondition.PRIVATE_BASELINE,
        config.seeds[0], config.agent_ids[0], IdentificationStage.SELF_REFLECTION,
    )
    shared_hash = "1" * 64
    first = render_branch_turn(BranchPromptContext(
        public_context=common, matched_group_id="group", branch_id="branch-a",
        shared_t0_observation_sha256=shared_hash,
    ))
    second = render_branch_turn(BranchPromptContext(
        public_context=common, matched_group_id="group", branch_id="branch-b",
        shared_t0_observation_sha256=shared_hash,
    ))
    assert first.user_prompt == second.user_prompt
    assert first.system_prompt == second.system_prompt
    assert first.request_id != second.request_id
    assert "branch-a" not in first.user_prompt
    assert "private_baseline" not in first.user_prompt


def test_shared_t0_fake_smoke_has_no_replayed_provider_calls():
    report = run_fake_smoke(str(V11))
    assert report["status"] == "shared_t0_fake_smoke_passed"
    assert report["shared_t0_request_count"] == 72
    assert report["continuation_request_count"] == 1728
    assert report["logical_request_count"] == 1800
    assert report["unique_request_fingerprint_count"] == 1800
    assert report["projected_initial_observation_count"] == 432
    assert report["initial_substantive_count"] == 48
    assert report["correction_transition_eligible_count"] == 24
    assert report["effect_estimated"] is False


def test_manipulation_cases_are_text_matched_and_have_zero_one_two_roots():
    cases = build_cases()
    assert len(cases) == 12
    by_scenario: dict[str, list] = {}
    for case in cases:
        by_scenario.setdefault(case.scenario_id, []).append(case)
    for scenario_cases in by_scenario.values():
        assert len({item.visible_content_ids for item in scenario_cases}) == 1
        assert len({item.public_summaries for item in scenario_cases}) == 1
        counts = {
            item.projection: len({root.source_root_id for root in item.public_root_assignments if root.source_root_id})
            for item in scenario_cases
        }
        assert counts == {
            SourceProjection.SOURCE_FREE: 0,
            SourceProjection.SAME_ROOT: 1,
            SourceProjection.INDEPENDENT_ROOTS: 2,
        }


def test_manipulation_parser_rejects_wrong_root_assignment_and_extra_field():
    context = next(item for item in build_cases() if item.projection is SourceProjection.SAME_ROOT)
    valid = {
        "visible_message_count": 2,
        "visible_source_root_count": 1,
        "root_assignments": [item.model_dump(mode="json") for item in context.public_root_assignments],
    }
    assert parse_manipulation_response(json.dumps(valid), context).visible_source_root_count == 1
    wrong = dict(valid)
    wrong["root_assignments"] = [
        RootAssignment(content_id=item.content_id, source_root_id="fabricated-root").model_dump(mode="json")
        for item in context.public_root_assignments
    ]
    with pytest.raises(ValueError, match="root_assignment_mismatch"):
        parse_manipulation_response(json.dumps(wrong), context)
    with pytest.raises(ValueError, match="invalid_schema"):
        parse_manipulation_response(json.dumps({**valid, "truth": "unknown"}), context)


def test_manipulation_fake_smoke_only_checks_structure():
    report = run_manipulation_smoke()
    assert report["passed_case_count"] == 12
    assert report["logical_request_count"] == 12
    assert report["root_count_totals"] == {"independent_roots": 8, "same_root": 4, "source_free": 0}
    assert report["behavior_effect_estimated"] is False
    assert report["joined_to_behavior_study"] is False
    assert report["private_truth_exposed"] is False


def test_analysis_plan_freezes_all_five_mechanism_contrasts():
    plan = load_analysis_plan()
    assert [item.contrast_id for item in plan.primary_contrasts] == list(ContrastId)
    independent = next(
        item for item in plan.primary_contrasts
        if item.contrast_id is ContrastId.INDEPENDENT_CORROBORATION_INCREMENT
    )
    assert independent.minuend is IdentificationCondition.INDEPENDENT_ROOTS
    assert independent.subtrahend is IdentificationCondition.SAME_ROOT_SOCIAL
    assert plan.agent_observations_are_independent_replicates is False
    assert plan.aggregate_agents_before_inference is True
    assert plan.diagnostic_metric_is_primary is False


def test_analysis_and_manipulation_preflights_are_network_free():
    analysis = safe_analysis_preflight()
    manipulation = manipulation_preflight()
    assert analysis["status"] == "analysis_plan_ready"
    assert manipulation["status"] == "offline_gate_ready"
    for report in (analysis, manipulation):
        assert report["network"] == "disabled"
        assert report["provider_constructed"] is False
        assert report["api_key_read"] is False
        assert report["results_written"] is False
        assert report["private_truth_exposed"] is False


def test_final_preflight_stops_before_network_and_real_results():
    report = preflight(str(V11))
    assert report["status"] == "offline_design_ready"
    assert report["ready_for_network"] is False
    assert report["blocking_reasons"] == [
        "exact_hash_human_approval_required",
        "real_source_manipulation_check_required",
        "protocol_stability_probe_required",
        "provider_compatibility_check_required",
    ]
    assert report["fake_logical_request_count"] == 1800
    assert report["fake_unique_fingerprint_count"] == 1800
    assert report["manipulation_fake_passed_count"] == 12
    assert report["output_root_exists"] is False


def test_stability_probe_is_independent_pending_and_exactly_thirty_requests():
    config, approval = load_stability_config()
    assert approval.acceptance_status == "pending"
    assert config.logical_request_cap == 30
    assert config.completion_reservation_cap == 15_360
    assert config.merge_with_behavior_study is False
    assert not (ROOT / config.output_root).exists()
    smoke = run_stability_smoke()
    assert smoke["status"] == "stability_fake_smoke_passed"
    assert smoke["logical_request_count"] == 30
    assert smoke["unique_request_fingerprint_count"] == 30
    assert smoke["stage_counts"] == {
        "initial_private": 6,
        "self_reflection": 6,
        "social_exposure": 6,
        "correction_evidence": 6,
        "final_judgment": 6,
    }
    assert smoke["parser_invalid_count"] == 0


def test_stability_preflight_is_offline_and_requires_separate_approval():
    report = stability_preflight()
    assert report["status"] == "offline_probe_ready"
    assert report["ready_for_network"] is False
    assert report["blocking_reasons"] == [
        "exact_hash_human_approval_required",
        "provider_compatibility_check_required",
        "network_authorization_required",
    ]
    assert report["network"] == "disabled"
    assert report["provider_constructed"] is False
    assert report["api_key_read"] is False
    assert report["results_written"] is False


def test_new_safe_outputs_contain_no_private_or_provider_material():
    reports = [
        run_fake_smoke(str(V11)),
        run_manipulation_smoke(),
        safe_analysis_preflight(),
        preflight(str(V11)),
        run_stability_smoke(),
        stability_preflight(),
    ]
    serialized = json.dumps(reports, sort_keys=True).lower()
    for forbidden in (
        "system_prompt", "user_prompt", "model_response", "api_key_value", "authorization_header",
        "provider_metadata", "ground_truth_label", "source_independence_label", "evaluator_private",
    ):
        assert forbidden not in serialized
