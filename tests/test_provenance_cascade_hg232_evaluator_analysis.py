import json

import pytest
from pydantic import ValidationError

from evicon.provenance_cascade_hg232_evaluator_analysis import (
    HG232AnalysisManifest, analyze_hg232, run_analysis,
)
from evicon.provenance_cascade_hg232_integrity_audit import (
    FAILED_FINGERPRINT, audit_hg232_integrity, write_audit_receipt,
)
from evicon.provenance_cascade_hg232_eligibility_funnel import diagnose_hg232, write_diagnostic


def test_completed_hg232_integrity_is_reconstructable() -> None:
    receipt = audit_hg232_integrity()
    assert receipt.run_count == 16
    assert receipt.unique_logical_request_count == 288
    assert receipt.transport_attempt_count == 289
    assert receipt.approved_timeout_fingerprint == FAILED_FINGERPRINT
    assert receipt.approved_fingerprint_transport_attempt_count == 2
    assert receipt.duplicate_completed_fingerprint_count == 0
    assert receipt.cascade_replay_passed_count == 16
    assert receipt.application_replay_passed_count == 16
    assert receipt.outcome_replay_passed_count == 16


def test_audit_receipt_can_only_be_written_identically(tmp_path) -> None:
    target = tmp_path / "audit.json"
    first = write_audit_receipt(target)
    second = write_audit_receipt(target)
    assert first == second


def test_analysis_uses_adoption_decision_not_epistemic_stance() -> None:
    manifest, report = analyze_hg232()
    assert manifest.false_adoption_source == "adoption_decision_only"
    assert manifest.epistemic_stance_used_for_false_adoption is False
    false_metrics = [item for item in report.run_metrics if item.metric_name == "false_cascade_adoption_rate"]
    assert len(false_metrics) == 16
    assert all("adoption_decision_operationalization" in item.warnings for item in false_metrics)


def test_analysis_reports_all_scenario_condition_denominators() -> None:
    _, report = analyze_hg232()
    coordinates = {(item.scenario_id, item.condition.value) for item in report.eligibility_summaries}
    assert len(coordinates) == 16
    assert len(report.scenario_condition_metrics) == 144
    assert report.decision_status == "thresholds_not_pre_registered"
    assert report.effect_evidence_status == "not_assessed"
    harmful = [item for item in report.condition_summaries if item.metric_name == "harmful_conformity"]
    assert all(item.denominator_sum == 12 and item.pooled_value == 0.0 for item in harmful)


def test_analysis_output_is_safe_and_writes_only_new_directory(tmp_path) -> None:
    target = tmp_path / "analysis"
    result = run_analysis(write_outputs=True, output_dir=target)
    assert result.status == "completed"
    payload = (target / "analysis_report.json").read_text(encoding="utf-8").lower()
    for marker in ("ground_truth_label", "source_independence_label", "raw_response", "api_key", "provider_metadata"):
        assert marker not in payload
    assert (target / "analysis_manifest.json").is_file()


def test_analysis_manifest_rejects_unknown_fields() -> None:
    manifest, _ = analyze_hg232()
    payload = manifest.model_dump(mode="json")
    payload["prompt"] = "forbidden"
    with pytest.raises(ValidationError):
        HG232AnalysisManifest.model_validate(payload)


def test_eligibility_funnel_identifies_floor_and_corrected_join() -> None:
    _, report = diagnose_hg232()
    assert report.attribution == "mixed_or_indeterminate"
    assert report.primary_blocker_after_join_fix == "behavioral_floor"
    assert report.analyzer_defect_found is True
    assert report.join_audit.decision_outcome_agent_round_match_count == 288
    assert report.join_audit.decision_outcome_agent_round_mismatch_count == 0
    conditions = {item.value: item for item in report.aggregates if item.dimension == "condition"}
    assert all(item.round0_false_claim_observation_count == 12 for item in conditions.values())
    assert all(item.round0_false_adoption_count == 0 for item in conditions.values())
    assert all(item.harmful_conformity_eligible_count == 12 for item in conditions.values())


def test_correction_exposure_exists_after_round0_but_no_initial_adopter() -> None:
    _, report = diagnose_hg232()
    scenarios = {item.value: item for item in report.aggregates if item.dimension == "scenario"}
    correction = scenarios["cascade-hg1-true-minority-correction"]
    assert correction.round0_false_claim_observation_count == 24
    assert correction.later_correction_exposure_count == 24
    assert correction.round0_false_adoption_count == 0
    assert correction.correction_after_false_adoption_count == 0
    assert correction.beneficial_receptivity_eligible_count == 0


def test_funnel_writes_only_safe_new_outputs(tmp_path) -> None:
    target = tmp_path / "funnel"
    write_diagnostic(target)
    payload = (target / "eligibility_funnel_report.json").read_text(encoding="utf-8").lower()
    for marker in ("ground_truth_label", "source_independence_label", "raw_response", "api_key", "provider_metadata"):
        assert marker not in payload
