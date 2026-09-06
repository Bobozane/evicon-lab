from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.cascade_agent_protocol_v21 import load_hd21_config
from evicon.cascade_agent_protocol_v21_pilot import (
    HD21PilotBatchRecord,
    HD21PilotRunStatus,
)
from evicon.provenance_cascade_pilot_analysis import (
    HFPilotAnalysisError,
    HFMetricStatus,
    analyze_provenance_cascade_pilot,
    run_hf_analysis,
    validate_hf_run_coordinates,
)
from evicon.provenance_cascade_pilot_analysis_plan import (
    EXPECTED_CONDITIONS,
    EXPECTED_METRICS,
    load_provenance_cascade_pilot_analysis_plan,
)
from evicon.validate_provenance_cascade_pilot_analysis import main as validator_main

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd21_analysis.v1.toml"
PILOT_ROOT = ROOT / "results/provenance-cascade-pilot-hd21-v1"


@pytest.fixture(scope="module")
def plan():
    return load_provenance_cascade_pilot_analysis_plan(PLAN_PATH)


@pytest.fixture(scope="module")
def completed_analysis(plan):
    return analyze_provenance_cascade_pilot(plan)


def _config_and_batch(plan):
    config, _ = load_hd21_config(ROOT / plan.config_path)
    batch = HD21PilotBatchRecord.model_validate_json(
        (ROOT / plan.batch_record_path).read_text(encoding="utf-8")
    )
    return config, batch


def test_bound_plan_and_complete_pilot_are_valid(plan, completed_analysis):
    manifest, report = completed_analysis
    assert manifest.run_count == 48
    assert manifest.matched_group_count == 12
    assert manifest.primary_metrics == EXPECTED_METRICS
    assert len(report.run_metrics) == 48 * 9
    assert len(report.matched_differences) == 4 * 3 * 9 * 4
    assert report.data_integrity_status == "passed"
    assert report.effect_evidence_status == "descriptive_pilot_only"
    assert report.decision_status == "thresholds_not_pre_registered"
    assert report.private_truth_exposed is False


def test_metrics_preserve_denominators_not_applicable_and_fixed_conditions(completed_analysis):
    _, report = completed_analysis
    assert {item.condition for item in report.condition_summaries} == set(EXPECTED_CONDITIONS)
    beneficial = [
        item for item in report.condition_summaries
        if item.metric_name == "beneficial_receptivity"
    ]
    assert all(item.applicable_count == 0 for item in beneficial)
    assert all(item.not_applicable_count == 12 for item in beneficial)
    intervention = [
        item for item in report.run_metrics
        if item.metric_name == "intervention_false_positive_rate"
    ]
    assert any(item.status is HFMetricStatus.NOT_APPLICABLE for item in intervention)
    assert all(item.denominator >= 0 for item in report.run_metrics)


def test_operation_audit_separates_logical_requests_and_transport_recovery(completed_analysis):
    _, report = completed_analysis
    assert sum(item.logical_request_count for item in report.run_operations) == 864
    assert sum(item.transport_attempt_count for item in report.run_operations) == 866
    assert sum(item.connection_recovery_count for item in report.run_operations) == 2
    assert sum(item.applied_count for item in report.run_operations) == 9
    assert sum(item.rejected_count for item in report.run_operations) == 0
    assert sum(item.abstain_count for item in report.run_operations) == 18


def test_missing_run_and_duplicate_run_are_rejected(plan):
    config, batch = _config_and_batch(plan)
    with pytest.raises(HFPilotAnalysisError, match="run_set_incomplete"):
        validate_hf_run_coordinates(config.runs, batch.runs[:-1])
    duplicate = batch.runs[:-1] + (batch.runs[0],)
    with pytest.raises(HFPilotAnalysisError, match="run_set_incomplete"):
        validate_hf_run_coordinates(config.runs, duplicate)


def test_incomplete_matched_group_is_rejected(plan):
    config, batch = _config_and_batch(plan)
    changed = config.runs[0].model_copy(update={"matched_group_id": "hd21-unregistered-group"})
    specs = (changed,) + config.runs[1:]
    with pytest.raises(HFPilotAnalysisError, match="matched_group_incomplete"):
        validate_hf_run_coordinates(specs, batch.runs)


def test_failed_replay_and_coordinate_mismatch_are_rejected(plan):
    config, batch = _config_and_batch(plan)
    replay_failed = batch.runs[0].model_copy(update={"outcome_replay_status": "blocked"})
    with pytest.raises(HFPilotAnalysisError, match="replay_not_passed"):
        validate_hf_run_coordinates(config.runs, (replay_failed,) + batch.runs[1:])
    mismatched = batch.runs[0].model_copy(update={"seed": 1})
    with pytest.raises(HFPilotAnalysisError, match="run_coordinate_mismatch"):
        validate_hf_run_coordinates(config.runs, (mismatched,) + batch.runs[1:])



def test_truth_fixture_scenario_mismatch_is_rejected(plan):
    first, second, *rest = plan.truth_fixtures
    swapped = (
        first.model_copy(update={"path": second.path, "sha256": second.sha256}),
        second,
        *rest,
    )
    changed = plan.model_copy(update={"truth_fixtures": tuple(swapped)})
    with pytest.raises(HFPilotAnalysisError, match="evaluator_fixture_binding_mismatch"):
        analyze_provenance_cascade_pilot(changed)


def test_legacy_or_foreign_pilot_root_is_rejected_by_plan(plan):
    payload = plan.model_dump(mode="json")
    payload["pilot_root"] = "results/provenance-cascade-pilot-hd2-v1"
    with pytest.raises(Exception, match="H-D.2.1 Pilot root"):
        type(plan).model_validate(payload)

def test_hash_tampering_blocks_before_analysis(plan):
    tampered = plan.model_copy(update={"pilot_receipt_sha256": "0" * 64})
    with pytest.raises(HFPilotAnalysisError, match="pilot_receipt_hash_mismatch"):
        analyze_provenance_cascade_pilot(tampered)


def test_reports_are_safe_and_do_not_include_private_or_model_content(completed_analysis):
    manifest, report = completed_analysis
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
        "question_text",
        "model_response",
    ):
        assert forbidden not in payload
    assert str(PILOT_ROOT) not in payload


def test_analysis_writes_only_new_directory_and_refuses_overwrite(tmp_path):
    output = tmp_path / "hf-analysis"
    first = run_hf_analysis(PLAN_PATH, write_outputs=True, output_dir=output)
    assert first.status == "completed"
    assert (output / "analysis_manifest.json").is_file()
    assert (output / "analysis_report.json").is_file()
    second = run_hf_analysis(PLAN_PATH, write_outputs=True, output_dir=output)
    assert second.status == "blocked"
    assert second.blocking_reasons == ("analysis_output_exists",)


def test_validator_cli_is_offline_and_safe(capsys):
    assert validator_main(["--config", str(PLAN_PATH)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "validated"
    assert payload["network"] == "disabled"
    assert payload["provider_constructed"] is False
    assert payload["private_truth_exposed"] is False
    assert payload["output_dir"] is None
