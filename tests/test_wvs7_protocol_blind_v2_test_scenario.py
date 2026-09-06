import hashlib
import json
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from evicon.wvs7_confirmatory_study import (
    ConfirmatoryRunSpec,
    ConfirmatoryStudyManifest,
    load_confirmatory_manifest,
    validate_confirmatory_manifest,
)
from evicon.wvs7_confirmatory_authoring import main as authoring_main
from evicon.wvs7_test_scenario import (
    DEVELOPMENT_SCENARIO_ID,
    TEST_SCENARIO_ID,
    TestScenarioAuthoring,
    TestScenarioReview,
    load_authoring,
    load_review,
    validate_test_scenario,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORING = ROOT / "configs/studies/wvs7_english_protocol_blind_v2_test_scenario_authoring.toml"
REVIEW = ROOT / "configs/studies/wvs7_english_protocol_blind_v2_test_scenario_review.toml"
TEMPLATE = ROOT / "configs/studies/wvs7_english_protocol_blind_v2_confirmatory_template.toml"
DEVELOPMENT_SCENARIO = ROOT / "configs/scenarios/wvs7/english-baseline-scenario.toml"


def _accepted_review(tmp_path: Path) -> Path:
    payload = tomllib.loads(REVIEW.read_text(encoding="utf-8"))
    payload.update(
        {
            "status": "accepted",
            "reviewer_id": "researcher-001",
            "reviewed_on": "2026-08-19",
            "approve_distinct_from_development": True,
            "approve_public_synthetic_content": True,
            "approve_protocol_blind_use": True,
        }
    )
    path = tmp_path / "accepted-review.toml"
    lines = []
    for key, value in payload.items():
        if isinstance(value, str):
            lines.append(f"{key} = {json.dumps(value)}")
        elif isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _complete_manifest(tmp_path: Path) -> ConfirmatoryStudyManifest:
    manifest = load_confirmatory_manifest(TEMPLATE)
    review_path = _accepted_review(tmp_path)
    runs = [
        ConfirmatoryRunSpec(
            run_id=f"wvs7-v2-confirmatory-{seed}-{condition.value}",
            matched_group_id=f"wvs7-v2-confirmatory-group-{seed}",
            condition=condition,
            scenario_id=TEST_SCENARIO_ID,
            seed=seed,
            agent_ids=manifest.agent_ids,
            max_rounds=manifest.max_rounds,
            model_name=manifest.model_name,
            probe_set_id=manifest.probe_set_id,
            probe_template_version=manifest.probe_template_version,
        )
        for seed in manifest.seeds
        for condition in manifest.conditions
    ]
    return manifest.model_copy(
        update={
            "test_scenario_ids": [TEST_SCENARIO_ID],
            "test_scenario_review_status": "accepted",
            "test_scenario_review_path": str(review_path),
            "runs": runs,
        }
    )


def test_pending_review_blocks_the_new_test_scenario() -> None:
    pending = load_review(REVIEW).model_copy(
        update={
            "status": "pending_manual_review",
            "reviewer_id": "TODO_RESEARCHER_REVIEW",
            "reviewed_on": "TODO_REVIEW_DATE",
            "approve_distinct_from_development": False,
            "approve_public_synthetic_content": False,
            "approve_protocol_blind_use": False,
        }
    )
    report = validate_test_scenario(load_authoring(AUTHORING), pending)
    assert report.status == "blocked"
    assert report.blocking_reasons == ["test_scenario_manual_review_required"]


def test_development_scenario_cannot_be_reused_as_test_scenario() -> None:
    authoring = load_authoring(AUTHORING)
    development_hash = hashlib.sha256(DEVELOPMENT_SCENARIO.read_bytes()).hexdigest()
    reused = authoring.model_copy(
        update={
            "scenario_path": "configs/scenarios/wvs7/english-baseline-scenario.toml",
            "scenario_sha256": development_hash,
        }
    )
    review = load_review(REVIEW).model_copy(update={"scenario_sha256": development_hash})
    report = validate_test_scenario(reused, review, require_review=False)
    assert report.status == "blocked"
    assert "test_scenario_reuses_development_structure" in report.blocking_reasons


def test_complete_reviewed_manifest_is_ready_without_provider_use(tmp_path: Path) -> None:
    report = validate_confirmatory_manifest(_complete_manifest(tmp_path))
    assert report.status == "ready_for_confirmatory"
    assert report.blocking_reasons == []


def test_less_than_three_seeds_is_blocked(tmp_path: Path) -> None:
    manifest = _complete_manifest(tmp_path)
    two_seeds = manifest.seeds[:2]
    reduced = manifest.model_copy(
        update={"seeds": two_seeds, "runs": [run for run in manifest.runs if run.seed in two_seeds]}
    )
    report = validate_confirmatory_manifest(reduced)
    assert "at_least_three_new_seeds_required" in report.blocking_reasons


def test_calibration_seed_is_rejected_by_strict_model() -> None:
    payload = load_confirmatory_manifest(TEMPLATE).model_dump(mode="json")
    payload["seeds"] = [20260819, 20260820, 20260821]
    with pytest.raises(ValidationError, match="calibration seed"):
        ConfirmatoryStudyManifest.model_validate(payload)


def test_development_scenario_id_is_rejected_by_strict_manifest() -> None:
    payload = load_confirmatory_manifest(TEMPLATE).model_dump(mode="json")
    payload["test_scenario_ids"] = [DEVELOPMENT_SCENARIO_ID]
    with pytest.raises(ValidationError, match="independently authored"):
        ConfirmatoryStudyManifest.model_validate(payload)


def test_authoring_cli_is_offline_and_does_not_expose_scenario_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = authoring_main(
        ["--authoring", str(AUTHORING), "--review", str(REVIEW), "--manifest", str(TEMPLATE)]
    )
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert status == 1
    assert payload["status"] == "blocked"
    assert "initial_context" not in output
    assert "Option A" not in output
    assert "prompt" not in output


def test_missing_condition_is_blocked(tmp_path: Path) -> None:
    manifest = _complete_manifest(tmp_path)
    missing = manifest.model_copy(update={"runs": manifest.runs[:-1]})
    report = validate_confirmatory_manifest(missing)
    assert "complete_four_condition_groups_required" in report.blocking_reasons


def test_wrong_frozen_hash_is_rejected_by_strict_model() -> None:
    payload = load_confirmatory_manifest(TEMPLATE).model_dump(mode="json")
    payload["frozen_probe_set_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="audited hash"):
        ConfirmatoryStudyManifest.model_validate(payload)


@pytest.mark.parametrize(
    "development_run_id",
    ["wvs7-real-baseline-pilot-seed-002", "wvs7-real-baseline-pilot-seed-003"],
)
def test_development_runs_cannot_enter_confirmatory_manifest(
    tmp_path: Path, development_run_id: str
) -> None:
    manifest = _complete_manifest(tmp_path)
    contaminated = manifest.model_copy(
        update={
            "runs": [
                manifest.runs[0].model_copy(update={"run_id": development_run_id}),
                *manifest.runs[1:],
            ]
        }
    )
    report = validate_confirmatory_manifest(contaminated)
    assert "development_run_cannot_enter_confirmatory_manifest" in report.blocking_reasons


def test_manifest_path_must_match_reviewed_authoring_path(tmp_path: Path) -> None:
    manifest = _complete_manifest(tmp_path)
    mismatched = manifest.model_copy(
        update={"test_scenario_path": "configs/scenarios/wvs7/english-baseline-scenario.toml"}
    )
    report = validate_confirmatory_manifest(mismatched)
    assert "test_scenario_path_mismatch" in report.blocking_reasons


def test_authoring_and_review_reject_unknown_fields() -> None:
    authoring = load_authoring(AUTHORING).model_dump(mode="json")
    review = load_review(REVIEW).model_dump(mode="json")
    authoring["prompt"] = "forbidden"
    review["question_text"] = "forbidden"
    with pytest.raises(ValidationError):
        TestScenarioAuthoring.model_validate(authoring)
    with pytest.raises(ValidationError):
        TestScenarioReview.model_validate(review)


def test_test_scenario_constants_bind_new_and_development_ids() -> None:
    assert TEST_SCENARIO_ID != DEVELOPMENT_SCENARIO_ID
