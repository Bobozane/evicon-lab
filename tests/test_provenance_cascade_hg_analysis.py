from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.provenance_cascade_hg_analysis import (
    HGEvaluatorAnalysisError,
    analyze_hg_pilot,
    run_hg_analysis,
)
from evicon.provenance_cascade_hg_analysis_plan import (
    EXPECTED_HG_REPLAY_CONTRACT,
    load_hg_analysis_plan,
)
from evicon.provenance_cascade_pilot_analysis import HFMetricStatus
from evicon.validate_provenance_cascade_hg_analysis import main as validator_main

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg_analysis.v1.toml"


@pytest.fixture(scope="module")
def plan():
    return load_hg_analysis_plan(PLAN_PATH)


@pytest.fixture(scope="module")
def completed_analysis(plan):
    return analyze_hg_pilot(plan)


def test_completed_hg_pilot_integrity_and_hash_chain(plan, completed_analysis) -> None:
    manifest, report = completed_analysis
    assert manifest.run_count == 48
    assert manifest.matched_group_count == 12
    assert manifest.logical_request_count == 864
    assert manifest.transport_attempt_count == 864
    assert len(manifest.run_record_sha256) == 48
    assert manifest.outcome_replay_contract_version == EXPECTED_HG_REPLAY_CONTRACT
    assert manifest.technical_replay_amendment_sha256 == plan.technical_replay_amendment_sha256
    assert manifest.final_batch_record_sha256 == plan.batch_record_sha256
    assert manifest.pre_resume_batch_record_sha256 != manifest.final_batch_record_sha256
    assert report.data_integrity_status == "passed"
    assert report.decision_status == "thresholds_not_pre_registered"
    assert len(report.run_metrics) == 48 * 9
    assert len(report.scenario_condition_summaries) == 4 * 4 * 9
    assert len(report.matched_differences) == 4 * 3 * 9 * 4


def test_hg_primary_behavior_metrics_preserve_denominators_and_nonseparation(completed_analysis) -> None:
    _, report = completed_analysis
    summaries = {(item.condition.value, item.metric_name): item for item in report.condition_summaries}
    for condition in (
        "no_intervention", "generic_dissent", "source_blind_controller", "provenance_aware_controller"
    ):
        assert summaries[(condition, "false_cascade_adoption_rate")].mean == 0.0
        assert summaries[(condition, "supported_correction_retention")].mean == pytest.approx(2 / 3)
        assert summaries[(condition, "beneficial_receptivity")].applicable_count == 0
        assert summaries[(condition, "harmful_conformity")].applicable_count == 0
    separation = {item.metric_name: item for item in report.separation_summaries}
    for metric in (
        "false_cascade_adoption_rate",
        "supported_correction_retention",
        "intervention_false_positive_rate",
        "provenance_diversity",
        "replay_audit_pass_rate",
    ):
        assert separation[metric].condition_mean_difference_observed is False
    assert separation["beneficial_receptivity"].range_across_condition_means is None
    assert separation["harmful_conformity"].range_across_condition_means is None


def test_scenario_level_denominators_and_supported_correction_are_retained(completed_analysis) -> None:
    _, report = completed_analysis
    rows = {
        (item.scenario_id, item.condition, item.metric_name): item
        for item in report.scenario_condition_summaries
    }
    for condition in (
        "no_intervention", "generic_dissent", "source_blind_controller", "provenance_aware_controller"
    ):
        correction = rows[(
            "cascade-hg-true-minority-correction",
            condition,
            "supported_correction_retention",
        )]
        assert correction.denominator_sum == 9
        assert correction.numerator_sum == 3.0
        assert correction.pooled_value == pytest.approx(1 / 3)
        consensus = rows[(
            "cascade-hg-independent-true-consensus",
            condition,
            "supported_correction_retention",
        )]
        assert consensus.denominator_sum == 3
        assert consensus.pooled_value == 1.0


def test_operation_audit_matches_receipt_and_has_no_transport_recovery(completed_analysis) -> None:
    _, report = completed_analysis
    operations = {item.condition.value: item for item in report.condition_operations}
    assert sum(item.logical_request_count for item in operations.values()) == 864
    assert sum(item.transport_attempt_count for item in operations.values()) == 864
    assert sum(item.total_tokens or 0 for item in operations.values()) == 334286
    assert sum(item.applied_count for item in operations.values()) == 9
    assert operations["no_intervention"].applied_count == 0
    assert operations["generic_dissent"].applied_count == 3
    assert operations["source_blind_controller"].applied_count == 3
    assert operations["provenance_aware_controller"].applied_count == 3
    assert all(item.connection_failure_count == 0 for item in operations.values())
    assert all(item.connection_recovery_count == 0 for item in operations.values())


def test_intervention_false_positive_is_zero_when_applicable_and_na_without_intervention(completed_analysis) -> None:
    _, report = completed_analysis
    metrics = [item for item in report.run_metrics if item.metric_name == "intervention_false_positive_rate"]
    no_intervention = [item for item in metrics if item.condition.value == "no_intervention"]
    assert all(item.status is HFMetricStatus.NOT_APPLICABLE for item in no_intervention)
    applied = [item for item in metrics if item.status is HFMetricStatus.VALUE]
    assert len(applied) == 9
    assert all(item.value == 0.0 and item.denominator == 1 for item in applied)


def test_hg_analysis_excludes_old_pilot_and_sensitive_data(completed_analysis) -> None:
    manifest, report = completed_analysis
    assert manifest.h_d21_pilot_excluded is True
    assert manifest.h_f_analysis_excluded is True
    assert manifest.wvs_and_calibration_excluded is True
    payload = json.dumps(
        {"manifest": manifest.model_dump(mode="json"), "report": report.model_dump(mode="json")},
        ensure_ascii=True,
        sort_keys=True,
    ).lower()
    for forbidden in (
        "ground_truth_label",
        "source_independence_label",
        "system_prompt",
        "user_prompt",
        "provider_metadata",
        "api_key",
        "authorization",
        "model_response",
        "public_statement",
    ):
        assert forbidden not in payload
    assert "results/provenance-cascade-pilot-hd21-v1" not in payload


def test_hash_tampering_blocks_analysis(plan) -> None:
    with pytest.raises(HGEvaluatorAnalysisError, match="pilot_receipt_hash_mismatch"):
        analyze_hg_pilot(plan.model_copy(update={"pilot_receipt_sha256": "0" * 64}))
    with pytest.raises(HGEvaluatorAnalysisError, match="technical_replay_amendment_hash_mismatch"):
        analyze_hg_pilot(plan.model_copy(update={"technical_replay_amendment_sha256": "0" * 64}))
    with pytest.raises(HGEvaluatorAnalysisError, match="new_validator_hash_mismatch"):
        analyze_hg_pilot(plan.model_copy(update={"new_validator_sha256": "0" * 64}))


def test_analysis_writes_only_new_directory_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "hg-analysis"
    first = run_hg_analysis(PLAN_PATH, write_outputs=True, output_dir=output)
    assert first.status == "completed"
    assert (output / "analysis_manifest.json").is_file()
    assert (output / "analysis_report.json").is_file()
    second = run_hg_analysis(PLAN_PATH, write_outputs=True, output_dir=output)
    assert second.status == "blocked"
    assert second.blocking_reasons == ("analysis_output_exists",)


def test_validator_cli_is_offline_and_safe(capsys) -> None:
    assert validator_main(["--config", str(PLAN_PATH)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "validated"
    assert payload["network"] == "disabled"
    assert payload["provider_constructed"] is False
    assert payload["private_truth_exposed"] is False
    assert payload["output_dir"] is None
