from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from evicon.batch_experiment import expected_artifact_paths
from evicon.validate_wvs7_confirmatory_analysis import main
from evicon.wvs7_confirmatory_analysis import (
    WVS7ConfirmatoryAnalysisError,
    exact_paired_sign_flip,
    run_confirmatory_analysis,
    validate_confirmatory_analysis,
)
from evicon.wvs7_confirmatory_analysis_plan import ConfirmatoryAnalysisInputs, load_confirmatory_analysis_plan

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/studies/wvs7_english_protocol_blind_v2_confirmatory_analysis.toml"
BATCH = ROOT / "results/batches/wvs7-protocol-blind-v2-confirmatory-execution"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def copied_plan(tmp_path: Path):
    """Copy only allowed safe receipt/audit/report/manifest/record artifacts."""
    safe = tmp_path / "safe"
    safe.mkdir()
    for name in (
        "confirmatory_batch_receipt.json",
        "confirmatory_completeness_audit.json",
        "offline_evaluation_report_initial_final_v2.json",
    ):
        shutil.copyfile(BATCH / name, safe / name)
    manifest = _json(BATCH / "batch_manifest.json")
    run_root = safe / "runs"
    manifest["batch_output_dir"] = str(safe / "batches")
    for run in manifest["runs"]:
        source = Path(run["artifacts"]["run_record_path"])
        run["output_dir"] = str(run_root)
        artifacts = expected_artifact_paths(run["output_dir"], run["run_id"]).model_dump(mode="json")
        destination = Path(artifacts["run_record_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        run["artifacts"] = artifacts
    manifest_path = safe / "batch_manifest.json"
    _write_json(manifest_path, manifest)
    plan = load_confirmatory_analysis_plan(CONFIG)
    return plan.model_copy(update={
        "inputs": ConfirmatoryAnalysisInputs(
            batch_receipt_path=str(safe / "confirmatory_batch_receipt.json"),
            completeness_audit_path=str(safe / "confirmatory_completeness_audit.json"),
            offline_evaluation_report_path=str(safe / "offline_evaluation_report_initial_final_v2.json"),
            batch_manifest_path=str(manifest_path),
        ),
        "output_dir": str(tmp_path / "analysis"),
    })


def test_complete_twelve_run_data_is_analyzable(tmp_path: Path) -> None:
    plan = copied_plan(tmp_path)
    result = validate_confirmatory_analysis(plan)
    assert result.status == "completed"
    assert result.seed_count == 3
    assert result.condition_count == 4
    assert result.primary_metrics == ["pairwise_diversity", "social_influence_loss", "profile_drift"]


def test_missing_run_is_rejected(tmp_path: Path) -> None:
    plan = copied_plan(tmp_path)
    manifest_path = Path(plan.inputs.batch_manifest_path)
    manifest = _json(manifest_path)
    removed = manifest["runs"].pop()
    manifest["run_order"].remove(removed["run_id"])
    _write_json(manifest_path, manifest)
    with pytest.raises(WVS7ConfirmatoryAnalysisError):
        validate_confirmatory_analysis(plan)


def test_replay_failure_is_rejected(tmp_path: Path) -> None:
    plan = copied_plan(tmp_path)
    receipt_path = Path(plan.inputs.batch_receipt_path)
    receipt = _json(receipt_path)
    receipt["replay_statuses"][next(iter(receipt["replay_statuses"]))] = "failed"
    _write_json(receipt_path, receipt)
    with pytest.raises(WVS7ConfirmatoryAnalysisError, match="analysis_replay_not_passed"):
        validate_confirmatory_analysis(plan)


def test_receipt_hash_tampering_is_rejected(tmp_path: Path) -> None:
    plan = copied_plan(tmp_path)
    receipt_path = Path(plan.inputs.batch_receipt_path)
    receipt = _json(receipt_path)
    receipt["frozen_probe_set_sha256"] = "0" * 64
    _write_json(receipt_path, receipt)
    with pytest.raises(WVS7ConfirmatoryAnalysisError, match="analysis_receipt_hash_mismatch"):
        validate_confirmatory_analysis(plan)


def test_v1_or_calibration_run_is_rejected(tmp_path: Path) -> None:
    plan = copied_plan(tmp_path)
    manifest_path = Path(plan.inputs.batch_manifest_path)
    manifest = _json(manifest_path)
    manifest["runs"][0]["run_id"] = "wvs7-real-baseline-pilot-seed-002"
    _write_json(manifest_path, manifest)
    with pytest.raises(WVS7ConfirmatoryAnalysisError):
        validate_confirmatory_analysis(plan)


def test_invalid_social_metric_pairing_is_rejected(tmp_path: Path) -> None:
    plan = copied_plan(tmp_path)
    report_path = Path(plan.inputs.offline_evaluation_report_path)
    report = _json(report_path)
    for metric in report["metrics"]:
        if metric["metric_name"] == "social_influence_loss":
            metric["input_ids"] = ["wvs7-v2-confirmatory-20260820-independent::final", metric["input_ids"][1]]
            break
    _write_json(report_path, report)
    with pytest.raises(WVS7ConfirmatoryAnalysisError, match="analysis_metric_condition_pairing_invalid"):
        validate_confirmatory_analysis(plan)


def test_exact_sign_flip_and_leave_one_out_are_reproducible(tmp_path: Path) -> None:
    expected = exact_paired_sign_flip([0.1, -0.2, 0.3])
    assert expected == exact_paired_sign_flip([0.1, -0.2, 0.3])
    assert expected.permutation_count == 8
    assert expected.n == 3
    first = run_confirmatory_analysis(copied_plan(tmp_path))
    report = _json(Path(first.output_dir) / "analysis_report.json")
    robustness = _json(Path(first.output_dir) / "robustness_report.json")
    assert len(report["raw_metric_values"]) == 24
    assert len(report["social_influence_loss_values"]) == 3
    assert len(robustness["leave_one_seed_out"]) == 12
    assert all(item["remaining_seed_count"] == 2 for item in robustness["leave_one_seed_out"])


def test_report_is_content_free_and_marks_limits(tmp_path: Path) -> None:
    result = run_confirmatory_analysis(copied_plan(tmp_path))
    combined = "\n".join((Path(result.output_dir) / name).read_text(encoding="utf-8") for name in (
        "analysis_manifest.json", "analysis_report.json", "robustness_report.json",
    )).lower()
    for forbidden in ("prompt", "raw_response", "question_text", "response_options", "api_key", "provider_metadata"):
        assert forbidden not in combined
    report = _json(Path(result.output_dir) / "analysis_report.json")
    assert report["confirmatory"] is True
    assert report["statistical_analysis"] is True
    assert report["not_causal_conclusion"] is True
    assert report["v1_002_excluded"] is True
    assert report["v2_003_excluded"] is True
    assert report["interpretation_limit"] == "limited_sample_confirmatory_pilot_no_causal_conclusion"
    assert len(report["source_evaluation_warnings"]) == 3


def test_cli_default_is_offline_and_writes_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(CONFIG)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "completed"
    assert output["output_dir"] is None
