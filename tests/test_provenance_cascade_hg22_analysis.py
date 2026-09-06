from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.provenance_cascade_hg22_analysis import (
    DEFAULT_PLAN, analyze_hg22, run_analysis,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def completed_analysis():
    return analyze_hg22(DEFAULT_PLAN)


def test_hg22_analysis_integrity_and_coordinates(completed_analysis) -> None:
    manifest, report = completed_analysis
    assert manifest.run_count == 16 and manifest.matched_group_count == 4
    assert manifest.logical_request_count == 288 and manifest.transport_attempt_count == 292
    assert len(manifest.run_record_sha256) == 16
    assert len(report.run_metrics) == 16 * 9
    assert len(report.scenario_condition_summaries) == 4 * 4 * 9
    assert len(report.behavior_distributions) == 16
    assert len(report.eligibility_summaries) == 16
    assert report.data_integrity_status == "passed"
    assert report.effect_evidence_status == "not_assessed"
    assert report.decision_status == "thresholds_not_pre_registered"


def test_hg22_behavior_eligibility_and_recovery_are_auditable(completed_analysis) -> None:
    _, report = completed_analysis
    assert sum(item.observation_count for item in report.behavior_distributions) == 288
    assert sum(item.share_decision_count for item in report.behavior_distributions) == 30
    assert sum(item.round0_non_defer_behavior_count for item in report.behavior_distributions) == 69
    assert report.recovery_audit.unique_logical_request_count == 288
    assert report.recovery_audit.transport_attempt_count == 292
    assert report.recovery_audit.failed_attempt_count == report.recovery_audit.recovered_fingerprint_count == 4
    assert report.recovery_audit.parser_recovery_count == 0
    assert all(item.correction_transition_eligible_count == 0 for item in report.eligibility_summaries)
    assert all(item.harmful_conformity_eligible_count == 0 for item in report.eligibility_summaries)
    beneficial = [item for item in report.run_metrics if item.metric_name == "beneficial_receptivity"]
    harmful = [item for item in report.run_metrics if item.metric_name == "harmful_conformity"]
    assert all(item.status.value == "not_applicable" for item in beneficial + harmful)


def test_hg22_analysis_output_is_safe(completed_analysis) -> None:
    manifest, report = completed_analysis
    serialized = json.dumps({"manifest": manifest.model_dump(mode="json"), "report": report.model_dump(mode="json")}, sort_keys=True).lower()
    for forbidden in (
        "ground_truth_label", "source_independence_label", "system_prompt", "user_prompt",
        "provider_metadata", "api_key", "authorization", "raw_response", "public_statement", "private_fixture",
    ):
        assert forbidden not in serialized


def test_hg22_hash_tampering_blocks_before_analysis(tmp_path: Path) -> None:
    raw = (ROOT / DEFAULT_PLAN).read_text(encoding="utf-8")
    candidate = tmp_path / "tampered.toml"
    candidate.write_text(raw.replace(
        'technical_audit_receipt_sha256 = "43fae0b2e75645bad6eaa8d5cf706849e757ea6a7fdd91df1a188ed176c79ef0"',
        'technical_audit_receipt_sha256 = "' + "0" * 64 + '"',
    ), encoding="utf-8")
    result = run_analysis(candidate)
    assert result.status == "blocked"
    assert result.blocking_reasons == ("technical_audit_receipt_hash_mismatch",)


def test_hg22_analysis_writes_new_directory_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "analysis"
    first = run_analysis(DEFAULT_PLAN, write_outputs=True, output_dir=output)
    assert first.status == "completed"
    assert (output / "analysis_manifest.json").is_file()
    assert (output / "analysis_report.json").is_file()
    second = run_analysis(DEFAULT_PLAN, write_outputs=True, output_dir=output)
    assert second.status == "blocked"
    assert second.blocking_reasons == ("analysis_output_exists",)
