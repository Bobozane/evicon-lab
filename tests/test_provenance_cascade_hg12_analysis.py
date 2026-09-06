from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.provenance_cascade_hg12_analysis import HG12EvaluatorAnalysisError, analyze_hg12_pilot, run_hg12_analysis
from evicon.provenance_cascade_hg12_analysis_plan import load_hg12_analysis_plan
from evicon.validate_provenance_cascade_hg12_analysis import main as validator_main

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg12_analysis.v1.toml"


@pytest.fixture(scope="module")
def plan():
    return load_hg12_analysis_plan(PLAN_PATH)


@pytest.fixture(scope="module")
def completed_analysis(plan):
    return analyze_hg12_pilot(plan)


def test_hg12_completed_pilot_audit_and_metric_coordinates(completed_analysis) -> None:
    manifest, report = completed_analysis
    assert manifest.run_count == 48
    assert manifest.matched_group_count == 12
    assert manifest.logical_request_count == 864
    assert manifest.transport_attempt_count >= 864
    assert len(manifest.run_record_sha256) == 48
    assert len(report.run_metrics) == 48 * 9
    assert len(report.scenario_condition_summaries) == 4 * 4 * 9
    assert report.data_integrity_status == "passed"
    assert report.decision_status == "thresholds_not_pre_registered"


def test_hg12_report_is_safe_and_excludes_legacy_assets(completed_analysis) -> None:
    manifest, report = completed_analysis
    assert manifest.hg11_hg1_hd21_excluded is True
    assert manifest.wvs_002_003_calibration_excluded is True
    serialized = json.dumps({"manifest": manifest.model_dump(mode="json"), "report": report.model_dump(mode="json")}, sort_keys=True).lower()
    for marker in ("ground_truth_label", "source_independence_label", "system_prompt", "user_prompt", "provider_metadata", "api_key", "authorization", "model_response", "public_statement", "private_fixture"):
        assert marker not in serialized


def test_hg12_hash_tampering_blocks_analysis(plan) -> None:
    with pytest.raises(HG12EvaluatorAnalysisError, match="pilot_receipt_hash_mismatch"):
        analyze_hg12_pilot(plan.model_copy(update={"pilot_receipt_sha256": "0" * 64}))
    with pytest.raises(HG12EvaluatorAnalysisError, match="compatibility_receipt_hash_mismatch"):
        analyze_hg12_pilot(plan.model_copy(update={"compatibility_receipt_sha256": "0" * 64}))


def test_hg12_analysis_writes_only_new_directory_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "hg12-analysis"
    first = run_hg12_analysis(PLAN_PATH, write_outputs=True, output_dir=output)
    assert first.status == "completed"
    assert (output / "analysis_manifest.json").is_file()
    assert (output / "analysis_report.json").is_file()
    second = run_hg12_analysis(PLAN_PATH, write_outputs=True, output_dir=output)
    assert second.status == "blocked"
    assert second.blocking_reasons == ("analysis_output_exists",)


def test_hg12_validator_cli_is_offline_and_safe(capsys) -> None:
    assert validator_main(["--config", str(PLAN_PATH)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "validated"
    assert payload["network"] == "disabled"
    assert payload["provider_constructed"] is False
    assert payload["private_truth_exposed"] is False
