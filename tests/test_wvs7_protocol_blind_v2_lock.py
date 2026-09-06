import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evicon.wvs7_protocol_blind_v2_lock import (
    CalibrationLockError,
    lock_protocol_blind_v2_calibration,
)
from pydantic import ValidationError

from evicon.wvs7_confirmatory_study import (
    ConfirmatoryStudyError,
    load_confirmatory_manifest,
    validate_confirmatory_manifest,
)


CALIBRATION = Path("results/batches/wvs7-real-baseline-pilot-seed-003")
TEMPLATE = Path("configs/studies/wvs7_english_protocol_blind_v2_confirmatory_template.toml")

def copy_calibration(tmp_path: Path) -> Path:
    target = tmp_path / "calibration"
    target.mkdir()
    for name in (
        "protocol_blind_v2_calibration_receipt.json",
        "batch_manifest.json",
        "batch_record.json",
        "request_ledger.jsonl",
        "offline_evaluation_report_initial_final_v2.json",
        "protocol_blind_v2_initial_final_report.json",
    ):
        (target / name).write_bytes((CALIBRATION / name).read_bytes())
    return target

def test_completed_calibration_locks_and_accepts_recovered_failure(tmp_path):
    result = lock_protocol_blind_v2_calibration(CALIBRATION, output_path=tmp_path / "lock.json")
    assert result.status == "locked"
    assert result.logical_request_count == 384
    assert result.recovered_failure_attempt_count == 1
    payload = json.loads((tmp_path / "lock.json").read_text())
    assert payload["protocol_blind_audit"]["pre_probe_coordinate_count"] == 46
    assert payload["protocol_blind_audit"]["pre_probe_hash_mismatch_count"] == 0

def test_lock_rejects_tampered_receipt_hash(tmp_path):
    directory = copy_calibration(tmp_path)
    receipt = json.loads((directory / "protocol_blind_v2_calibration_receipt.json").read_text())
    receipt["frozen_probe_set_sha256"] = "0" * 64
    (directory / "protocol_blind_v2_calibration_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(CalibrationLockError, match="frozen_probe_set_hash_mismatch"):
        lock_protocol_blind_v2_calibration(directory, output_path=tmp_path / "lock.json")

def test_lock_rejects_missing_condition(tmp_path):
    directory = copy_calibration(tmp_path)
    manifest = json.loads((directory / "batch_manifest.json").read_text())
    manifest["conditions"] = manifest["conditions"][:-1]
    (directory / "batch_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(CalibrationLockError, match="calibration_conditions_mismatch"):
        lock_protocol_blind_v2_calibration(directory, output_path=tmp_path / "lock.json")

def test_lock_rejects_pre_probe_hash_mismatch(tmp_path):
    directory = copy_calibration(tmp_path)
    rows = [json.loads(line) for line in (directory / "request_ledger.jsonl").read_text().splitlines()]
    for row in rows:
        if row["status"] == "completed" and row["phase"] == "probe" and row["round_id"] == 0:
            row["public_content_sha256"] = "f" * 64
            break
    (directory / "request_ledger.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(CalibrationLockError, match="pre_probe_public_hash_mismatch"):
        lock_protocol_blind_v2_calibration(directory, output_path=tmp_path / "lock.json")

def test_lock_rejects_replay_failure(monkeypatch, tmp_path):
    import evicon.wvs7_protocol_blind_v2_lock as module
    monkeypatch.setattr(module, "validate_replay", lambda *_args: SimpleNamespace(validation="failed"))
    with pytest.raises(CalibrationLockError, match="calibration_replay_validation_failed"):
        lock_protocol_blind_v2_calibration(CALIBRATION, output_path=tmp_path / "lock.json")

def test_confirmatory_template_is_blocked_and_does_not_create_results():
    manifest = load_confirmatory_manifest(TEMPLATE)
    report = validate_confirmatory_manifest(manifest)
    assert report.status == "blocked"
    assert "independent_test_scenario_ids_required" in report.blocking_reasons
    assert "complete_four_condition_groups_required" in report.blocking_reasons

def test_confirmatory_rejects_v1_run_mixing():
    manifest = load_confirmatory_manifest(TEMPLATE)
    runs = []
    for seed in manifest.seeds:
        for condition in manifest.conditions:
            runs.append({
                "run_id": f"confirm-{seed}-{condition.value}",
                "matched_group_id": f"group-{seed}",
                "condition": condition.value,
                "scenario_id": "test-scenario",
                "seed": seed,
                "agent_ids": manifest.agent_ids,
                "max_rounds": 2,
                "model_name": manifest.model_name,
                "probe_set_id": manifest.probe_set_id,
                "probe_template_version": "contextual_value_probe.v1" if seed == manifest.seeds[0] and condition.value == "independent" else "contextual_value_probe.v2",
            })
    complete = manifest.model_copy(update={
        "test_scenario_ids": ["test-scenario"],
        "test_scenario_review_status": "accepted",
        "runs": runs,
    })
    report = validate_confirmatory_manifest(complete, verify_frozen_file=False)
    assert report.status == "blocked"
    assert "v1_or_non_v2_run_rejected" in report.blocking_reasons

def test_confirmatory_unknown_field_is_rejected():
    payload = json.loads(json.dumps(load_confirmatory_manifest(TEMPLATE).model_dump(mode="json")))
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        from evicon.wvs7_confirmatory_study import ConfirmatoryStudyManifest
        ConfirmatoryStudyManifest.model_validate(payload)
