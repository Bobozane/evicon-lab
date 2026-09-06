import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evicon.wvs7_confirmatory_plan_lock import (
    ConfirmatoryPlanLockError,
    lock_confirmatory_plan,
)
from evicon.wvs7_confirmatory_study import (
    ConfirmatoryStudyManifest,
    load_confirmatory_manifest,
    validate_confirmatory_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/studies/wvs7_english_protocol_blind_v2_confirmatory_plan.toml"
TEMPLATE = ROOT / "configs/studies/wvs7_english_protocol_blind_v2_confirmatory_template.toml"


def test_formal_plan_is_ready_and_has_twelve_explicit_runs() -> None:
    manifest = load_confirmatory_manifest(PLAN)
    report = validate_confirmatory_manifest(manifest)
    assert report.status == "ready_for_confirmatory"
    assert len(manifest.runs) == 12
    assert len({run.run_id for run in manifest.runs}) == 12
    assert len({run.matched_group_id for run in manifest.runs}) == 3
    assert manifest.seeds == [20260820, 20260821, 20260822]


def test_cost_plan_is_explicit_and_separates_reservation_from_observation() -> None:
    manifest = load_confirmatory_manifest(PLAN)
    assert manifest.logical_requests_per_matched_group == 384
    assert manifest.completion_reservation_per_matched_group == 51_200
    assert manifest.total_logical_requests == 1_152
    assert manifest.total_completion_reservation == 153_600
    assert manifest.observational_total_token_reference_per_matched_group == 190_602
    assert manifest.observational_total_token_reference == 571_806
    assert manifest.total_completion_reservation != manifest.observational_total_token_reference


def test_formal_plan_lock_writes_safe_receipt_without_prompts(tmp_path: Path) -> None:
    result = lock_confirmatory_plan(PLAN, output_path=tmp_path / "plan-lock.json")
    assert result.status == "locked"
    payload = json.loads((tmp_path / "plan-lock.json").read_text(encoding="utf-8"))
    assert payload["run_count"] == 12
    assert payload["matched_group_count"] == 3
    assert payload["total_logical_requests"] == 1_152
    assert payload["total_completion_reservation"] == 153_600
    assert payload["observed_total_token_reference"] == 571_806
    serialized = json.dumps(payload)
    for forbidden in ("prompt", "question_text", "response", "api_key", "authorization"):
        assert forbidden not in serialized.lower()


def test_lock_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "lock.json"
    lock_confirmatory_plan(PLAN, output_path=output)
    with pytest.raises(ConfirmatoryPlanLockError, match="lock_output_already_exists"):
        lock_confirmatory_plan(PLAN, output_path=output)


def test_lock_rejects_manifest_hash_tampering(tmp_path: Path) -> None:
    tampered = tmp_path / "tampered.toml"
    text = PLAN.read_text(encoding="utf-8").replace(
        "68a17e54e50e4d5f27276104d02b9e9d5390334643b2a33ec6dcd00f0456149e",
        "0" * 64,
        1,
    )
    tampered.write_text(text, encoding="utf-8")
    with pytest.raises(ConfirmatoryPlanLockError, match="confirmatory_manifest_invalid|test_scenario_hash_mismatch"):
        lock_confirmatory_plan(tampered, output_path=tmp_path / "lock.json")


def test_duplicate_run_id_and_shared_group_are_blocked() -> None:
    manifest = load_confirmatory_manifest(PLAN)
    duplicate = manifest.model_copy(update={"runs": [manifest.runs[0].model_copy(update={"run_id": manifest.runs[1].run_id}), *manifest.runs[1:]]})
    report = validate_confirmatory_manifest(duplicate, verify_frozen_file=False)
    assert "duplicate_run_id" in report.blocking_reasons

    shared = manifest.model_copy(update={"runs": [run.model_copy(update={"matched_group_id": "one-group"}) for run in manifest.runs]})
    report = validate_confirmatory_manifest(shared, verify_frozen_file=False)
    assert "matched_group_id_must_be_unique" in report.blocking_reasons


def test_wrong_seed_or_condition_is_rejected_by_strict_contract() -> None:
    payload = load_confirmatory_manifest(PLAN).model_dump(mode="json")
    payload["seeds"] = [20260820, 20260821, 20260823]
    with pytest.raises(ValidationError):
        ConfirmatoryStudyManifest.model_validate(payload)
    payload = load_confirmatory_manifest(PLAN).model_dump(mode="json")
    payload["conditions"] = ["independent", "social_only", "evidence_only"]
    with pytest.raises(ValidationError, match="fixed order"):
        ConfirmatoryStudyManifest.model_validate(payload)


def test_original_template_remains_blocked() -> None:
    manifest = load_confirmatory_manifest(TEMPLATE)
    report = validate_confirmatory_manifest(manifest)
    assert report.status == "blocked"
    assert "independent_test_scenario_ids_required" in report.blocking_reasons
    assert "complete_four_condition_groups_required" in report.blocking_reasons
