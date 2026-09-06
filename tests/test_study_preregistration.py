from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from pathlib import Path

import pytest

from evicon.study_preregistration import (
    StudyPreregistration,
    StudyPreregistrationError,
    StudyPreregistrationLoader,
    StudyStatus,
    load_and_validate_study_preregistration,
    validate_study_preregistration,
)
from evicon.validate_study_preregistration import main


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/studies/wvs7_english_protocol_blind_v2_calibration.toml"
FROZEN = ROOT / "outputs/wvs7-frozen/english_core_wvs7_23_frozen.json"
FROZEN_MANIFEST = ROOT / "outputs/wvs7-frozen/english_core_wvs7_23_frozen_manifest.json"


def _payload() -> dict[str, object]:
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))


def _valid_model() -> StudyPreregistration:
    return StudyPreregistrationLoader.load(CONFIG)


def test_calibration_config_is_strict_and_validates_frozen_receipt() -> None:
    summary = load_and_validate_study_preregistration(CONFIG)
    assert summary.status is StudyStatus.CALIBRATION
    assert summary.condition_count == 4
    assert summary.seed_count == 1
    assert summary.expected_provider_requests_per_matched_group == 384
    assert summary.completion_reservation_cap_per_matched_group == 51_200
    assert summary.estimated_total_tokens_per_matched_group == 190_602
    assert summary.calibration_runnable is True


def test_unknown_fields_are_rejected() -> None:
    payload = _payload()
    payload["unexpected_prompt"] = "must not be accepted"
    with pytest.raises(ValueError):
        StudyPreregistration.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("probe_template_version", "contextual_value_probe.v1"),
        ("frozen_probe_set_sha256", "0" * 64),
        ("conditions", ["independent", "social_only", "evidence_only"]),
        ("primary_metrics", ["structural_diversity", "profile_drift", "pairwise_diversity"]),
    ],
)
def test_fixed_v2_contract_rejects_drift(field: str, value: object) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(ValueError):
        StudyPreregistration.model_validate(payload)


def test_calibration_cannot_be_marked_confirmatory_without_confirmatory_requirements() -> None:
    payload = _payload()
    payload.update({"status": "confirmatory", "development_only": True, "not_paper_result": True})
    with pytest.raises(ValueError):
        StudyPreregistration.model_validate(payload)


def test_confirmatory_requires_three_seeds() -> None:
    payload = _payload()
    payload.update({"status": "confirmatory", "development_only": False, "not_paper_result": False})
    with pytest.raises(ValueError, match="fewer seeds"):
        StudyPreregistration.model_validate(payload)


def test_confirmatory_development_and_test_splits_must_be_disjoint() -> None:
    payload = _payload()
    payload["future_confirmatory_requirements"]["test_scenario_ids"] = ["wvs7-english-baseline-scenario"]  # type: ignore[index]
    with pytest.raises(ValueError, match="must not overlap"):
        StudyPreregistration.model_validate(payload)


def test_v1_002_is_explicitly_excluded_and_cost_reference_is_not_a_cap() -> None:
    model = _valid_model()
    assert model.v1_002_exclusion.excluded_from_v2_main_analysis is True
    assert model.v1_002_exclusion.observed_total_tokens == 190_602
    assert model.cost_plan.completion_reservation_cap == 51_200
    assert model.cost_plan.estimated_total_tokens_per_matched_group == 190_602
    assert model.cost_plan.reference_is_hard_cap is False


def test_missing_future_003_declaration_is_rejected() -> None:
    payload = _payload()
    payload["planned_calibration_run_id"] = "wvs7-real-baseline-pilot-seed-002"
    with pytest.raises(ValueError):
        StudyPreregistration.model_validate(payload)


def test_validator_does_not_modify_frozen_material_or_results(tmp_path: Path) -> None:
    before_probe = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    before_manifest = FROZEN_MANIFEST.read_bytes()
    summary = validate_study_preregistration(_valid_model())
    assert summary.calibration_runnable is True
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == before_probe
    assert FROZEN_MANIFEST.read_bytes() == before_manifest
    assert not (ROOT / "results" / "wvs7-real-baseline-pilot-seed-003").exists()


def test_cli_is_safe_summary_only(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(CONFIG)]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["study_id"] == "wvs7-english-protocol-blind-v2-calibration"
    assert payload["calibration_runnable"] is True
    assert "prompt" not in output.lower()
    assert "api_key" not in output.lower()
    assert "question" not in output.lower()


def test_missing_config_is_blocked(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(ROOT / "missing-preregistration.toml")]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"blocking_reasons": ["preregistration_validation_failed"], "status": "blocked"}
