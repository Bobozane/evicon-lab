"""Tests for the explicit local WVS 7 ProbeSet freeze gate."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from evicon import freeze_wvs7_probe_set as freeze_cli
from evicon.frozen_wvs7_manifest import load_frozen_manifest
from evicon.models import ProbeSet
from evicon.probe_loader import ProbeSetLoader
from evicon.wvs7_authoring import (
    WVS7AuthoringAcceptanceLoader,
    WVS7LanguageArm,
    WVS7ManualResponseOption,
    WVS7ManualTranscriptionSet,
    build_wvs7_authoring_plan,
)
from evicon.wvs7_candidates import WVS7CandidateLoader
from evicon.wvs7_review import WVS7ReviewLoader
from evicon.freeze_wvs7_probe_set import (
    EXPECTED_VARIABLE_IDS,
    WVS7FreezeError,
    build_frozen_wvs7_probe_set,
    freeze_wvs7_probe_set,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "configs/studies/wvs7_candidate_24.toml"
REVIEW = ROOT / "configs/studies/wvs7_candidate_24_codex_preliminary_review.toml"
ACCEPTANCE = ROOT / "configs/studies/wvs7_authoring_acceptance_researcher_2026-08-18.toml"
PENDING = ROOT / "configs/studies/wvs7_authoring_acceptance_template.toml"


def _inputs():
    candidate = WVS7CandidateLoader.load(CANDIDATE)
    review = WVS7ReviewLoader.load(REVIEW)
    acceptance = WVS7AuthoringAcceptanceLoader.load(ACCEPTANCE)
    plan = build_wvs7_authoring_plan(candidate, review, acceptance, WVS7LanguageArm.ENGLISH_CORE)
    return plan, acceptance


def _synthetic(plan):
    rows = []
    for item in plan.items:
        options = [WVS7ManualResponseOption(code=code, label=f"synthetic {item.variable_id} {code}") for code in item.response_option_codes]
        rows.append({
            "variable_id": item.variable_id,
            "language": "english_core",
            "source_review_id": plan.source_review_id,
            "source_questionnaire_file": item.source_questionnaire_file,
            "source_codebook_file": item.source_codebook_file,
            "source_questionnaire_pages": item.source_questionnaire_pages,
            "source_codebook_pages": item.source_codebook_pages,
            "source_sha256_prefixes": item.source_sha256_prefixes,
            "question_text": f"Synthetic fixture item {item.variable_id}",
            "response_options": [option.model_dump(mode="json") for option in options],
            "scale_min": item.scale_min,
            "scale_max": item.scale_max,
            "high_score_meaning": item.high_score_meaning,
            "reverse_scored": item.reverse_scored,
            "endpoint_labels": {code: f"endpoint {code}" for code in item.endpoint_codes},
            "special_response_codes": item.special_response_codes,
            "manual_transcriber_id": "synthetic-researcher",
            "transcribed_on": "2026-08-18",
            "human_verified": True,
        })
    return WVS7ManualTranscriptionSet(
        language="english_core", source_review_id=plan.source_review_id, synthetic_fixture=True, items=rows
    )


def test_default_real_draft_is_blocked_and_no_output_is_written(tmp_path: Path) -> None:
    plan, acceptance = _inputs()
    draft_path = ROOT / "private_wvs_transcriptions/english_core_wvs7_23.toml"
    from evicon.wvs7_authoring import WVS7ManualTranscriptionLoader
    completed = WVS7ManualTranscriptionLoader.load(draft_path)
    first = completed.items[0].model_copy(update={
        "human_verified": False,
        "transcription_origin": "local_pdf_text_extraction_draft",
    })
    draft = completed.model_copy(update={"items": [first, *completed.items[1:]]})
    with pytest.raises(WVS7FreezeError, match="human_verification_required"):
        freeze_wvs7_probe_set(plan, acceptance, draft, tmp_path / "outputs/wvs7-frozen/frozen.json", confirm_freeze=True)
    assert not list(tmp_path.rglob("*.json"))


def test_without_confirm_freeze_writes_nothing(tmp_path: Path) -> None:
    plan, acceptance = _inputs()
    data = _synthetic(plan)
    output = tmp_path / "outputs/wvs7-frozen/frozen.json"
    with pytest.raises(WVS7FreezeError, match="confirm_freeze_required"):
        freeze_wvs7_probe_set(plan, acceptance, data, output, confirm_freeze=False)
    assert not output.exists()


def test_complete_synthetic_fixture_writes_probe_set_and_manifest(tmp_path: Path) -> None:
    plan, acceptance = _inputs()
    data = _synthetic(plan)
    output = tmp_path / "outputs/wvs7-frozen/english_core.json"
    probe_set, manifest = freeze_wvs7_probe_set(plan, acceptance, data, output, confirm_freeze=True)
    assert len(probe_set.items) == 23
    assert [item.probe_id for item in probe_set.items] == [f"wvs7-english-core-{x}" for x in EXPECTED_VARIABLE_IDS]
    assert ProbeSetLoader.load(output) == probe_set
    manifest_path = output.with_name("english_core_manifest.json")
    assert load_frozen_manifest(manifest_path) == manifest
    assert manifest.probe_set_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest.variable_ids == list(EXPECTED_VARIABLE_IDS)
    assert probe_set.metadata["wvs_official_factor_structure"] is False
    assert not (tmp_path / "results").exists()


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    plan, acceptance = _inputs()
    data = _synthetic(plan)
    output = tmp_path / "outputs/wvs7-frozen/frozen.json"
    output.parent.mkdir(parents=True)
    output.write_text("sentinel", encoding="utf-8")
    with pytest.raises(WVS7FreezeError, match="already exists"):
        freeze_wvs7_probe_set(plan, acceptance, data, output, confirm_freeze=True)
    assert output.read_text(encoding="utf-8") == "sentinel"


def test_special_code_and_metadata_contracts_are_kept() -> None:
    plan, acceptance = _inputs()
    probe_set = build_frozen_wvs7_probe_set(plan, acceptance, _synthetic(plan))
    metadata = probe_set.metadata["variable_metadata"]
    assert metadata["Q111"]["special_response_codes"]["3"].startswith("volunteered_other")
    assert metadata["Q149"]["scoring_mode"] == "binary_categorical_not_continuous"
    assert metadata["Q150"]["scoring_mode"] == "binary_categorical_not_continuous"
    for variable_id in ("Q241", "Q243", "Q246", "Q247", "Q248", "Q249"):
        assert "0" in metadata[variable_id]["special_response_codes"]
    for variable_id in ("Q158", "Q159", "Q160", "Q161", "Q162", "Q163"):
        assert "secondary_science_technology_attitude" in " ".join(metadata[variable_id]["authoring_notes"])
    assert "semantic_scope" in " ".join(metadata["Q246"]["authoring_notes"])


@pytest.mark.parametrize("field", ["question_text", "response_options", "endpoint_labels", "source_questionnaire_pages", "source_sha256_prefixes"])
def test_incomplete_transcription_metadata_is_rejected(field: str) -> None:
    plan, acceptance = _inputs()
    data = _synthetic(plan)
    first = data.items[0]
    replacement = {
        "question_text": "TODO_MANUAL_ENTRY",
        "response_options": [],
        "endpoint_labels": {},
        "source_questionnaire_pages": [999],
        "source_sha256_prefixes": {"wrong.pdf": "0" * 12},
    }[field]
    broken = data.model_copy(update={"items": [first.model_copy(update={field: replacement}), *data.items[1:]]})
    with pytest.raises(WVS7FreezeError):
        build_frozen_wvs7_probe_set(plan, acceptance, broken)


def test_missing_item_q48_and_special_code_contracts_are_rejected() -> None:
    plan, acceptance = _inputs()
    data = _synthetic(plan)
    missing = data.model_copy(update={"items": data.items[:-1]})
    with pytest.raises(WVS7FreezeError):
        build_frozen_wvs7_probe_set(plan, acceptance, missing)

    q48_plan = plan.model_copy(update={"items": [plan.items[0].model_copy(update={"variable_id": "Q48"}), *plan.items[1:]]})
    with pytest.raises(WVS7FreezeError, match="23"):
        build_frozen_wvs7_probe_set(q48_plan, acceptance, data)

    q111 = next(item for item in data.items if item.variable_id == "Q111")
    broken_q111 = data.model_copy(update={"items": [
        item.model_copy(update={"special_response_codes": {}}) if item.variable_id == "Q111" else item
        for item in data.items
    ]})
    with pytest.raises(WVS7FreezeError):
        build_frozen_wvs7_probe_set(plan, acceptance, broken_q111)
    assert q111.special_response_codes["3"].startswith("volunteered_other")


def test_formal_outputs_are_limited_to_frozen_output_directory(tmp_path: Path) -> None:
    plan, acceptance = _inputs()
    with pytest.raises(WVS7FreezeError, match="outputs/wvs7-frozen"):
        freeze_wvs7_probe_set(plan, acceptance, _synthetic(plan), tmp_path / "elsewhere.json", confirm_freeze=True)


def test_cli_without_confirm_is_blocked_and_does_not_print_source(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "outputs/wvs7-frozen/frozen.json"
    status = freeze_cli.main([
        "--candidate", str(CANDIDATE), "--review", str(REVIEW), "--acceptance", str(ACCEPTANCE),
        "--transcriptions", str(ROOT / "private_wvs_transcriptions/english_core_wvs7_23.toml"),
        "--output", str(output),
    ])
    assert status == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"blocking_reasons": ["confirm_freeze_required"], "status": "blocked"}
    assert not output.exists()
    assert "Science and technology" not in json.dumps(payload)


def test_pending_acceptance_is_rejected(tmp_path: Path) -> None:
    from evicon.wvs7_authoring import WVS7AuthoringAcceptanceLoader
    candidate = WVS7CandidateLoader.load(CANDIDATE)
    review = WVS7ReviewLoader.load(REVIEW)
    pending = WVS7AuthoringAcceptanceLoader.load(PENDING)
    plan = build_wvs7_authoring_plan(candidate, review, pending, WVS7LanguageArm.ENGLISH_CORE)
    with pytest.raises(WVS7FreezeError, match="acceptance_required"):
        build_frozen_wvs7_probe_set(plan, pending, _synthetic(plan))
