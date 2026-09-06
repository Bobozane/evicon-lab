"""Tests for the ignored English-core WVS transcription template workflow."""

from __future__ import annotations

import json
import socket
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from evicon import transcription_template_smoke
from evicon import validate_wvs7_authoring as authoring_cli
from evicon.wvs7_authoring import (
    AcceptanceStatus,
    WVS7AuthoringAcceptanceLoader,
    WVS7AuthoringError,
    WVS7LanguageArm,
    WVS7ManualResponseOption,
    WVS7ManualTranscriptionSet,
    build_wvs7_authoring_plan,
    validate_manual_transcriptions,
)
from evicon.wvs7_candidates import WVS7CandidateLoader
from evicon.wvs7_review import WVS7ReviewLoader
from evicon.wvs7_transcription_template import (
    ENGLISH_CORE_VARIABLE_IDS,
    WVS7BlankTranscriptionTemplate,
    WVS7TranscriptionTemplateError,
    build_english_core_template,
    load_blank_template,
    make_template_from_files,
    validate_completed_english_transcription,
    write_english_core_template,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "configs/studies/wvs7_candidate_24.toml"
REVIEW = ROOT / "configs/studies/wvs7_candidate_24_codex_preliminary_review.toml"
ACCEPTANCE = ROOT / "configs/studies/wvs7_authoring_acceptance_researcher_2026-08-18.toml"
PENDING_ACCEPTANCE = ROOT / "configs/studies/wvs7_authoring_acceptance_template.toml"


def _inputs():
    return (
        WVS7CandidateLoader.load(CANDIDATE),
        WVS7ReviewLoader.load(REVIEW),
        WVS7AuthoringAcceptanceLoader.load(ACCEPTANCE),
    )


def _plan():
    candidate, review, acceptance = _inputs()
    return build_wvs7_authoring_plan(candidate, review, acceptance, WVS7LanguageArm.ENGLISH_CORE)


def _complete_transcriptions(plan) -> WVS7ManualTranscriptionSet:
    rows = []
    for item in plan.items:
        rows.append(
            {
                "variable_id": item.variable_id,
                "language": "english_core",
                "source_review_id": plan.source_review_id,
                "source_questionnaire_file": item.source_questionnaire_file,
                "source_codebook_file": item.source_codebook_file,
                "source_questionnaire_pages": item.source_questionnaire_pages,
                "source_codebook_pages": item.source_codebook_pages,
                "source_sha256_prefixes": item.source_sha256_prefixes,
                "question_text": f"synthetic question for {item.variable_id}",
                "response_options": [
                    {"code": code, "label": f"synthetic label {item.variable_id} {code}"}
                    for code in item.response_option_codes
                ],
                "scale_min": item.scale_min,
                "scale_max": item.scale_max,
                "high_score_meaning": item.high_score_meaning,
                "reverse_scored": item.reverse_scored,
                "endpoint_labels": {code: f"synthetic endpoint {code}" for code in item.endpoint_codes},
                "special_response_codes": item.special_response_codes,
                "manual_transcriber_id": "synthetic-transcriber",
                "transcribed_on": "2026-08-18",
            }
        )
    return WVS7ManualTranscriptionSet(
        language="english_core",
        source_review_id=plan.source_review_id,
        synthetic_fixture=True,
        items=rows,
    )


def test_template_has_exact_23_ids_and_excludes_q48(tmp_path: Path) -> None:
    plan = _plan()
    template = write_english_core_template(plan, _inputs()[2], tmp_path / "private_wvs_transcriptions" / "english.toml")
    assert tuple(item.variable_id for item in template.items) == ENGLISH_CORE_VARIABLE_IDS
    assert len(template.items) == 23
    assert "Q48" not in {item.variable_id for item in template.items}
    loaded = load_blank_template(tmp_path / "private_wvs_transcriptions" / "english.toml")
    assert loaded == template
    serialized = (tmp_path / "private_wvs_transcriptions" / "english.toml").read_text(encoding="utf-8")
    assert "question_text = \"TODO_MANUAL_ENTRY\"" in serialized
    assert "response_options = []" in serialized
    assert "high_score_meaning" in serialized
    assert "reverse_scored" in serialized


def test_generator_uses_accepted_english_scope_only(tmp_path: Path) -> None:
    candidate, review, accepted = _inputs()
    pending = WVS7AuthoringAcceptanceLoader.load(PENDING_ACCEPTANCE)
    with pytest.raises(WVS7TranscriptionTemplateError, match="acceptance"):
        build_english_core_template(
            build_wvs7_authoring_plan(candidate, review, pending, WVS7LanguageArm.ENGLISH_CORE), pending
        )
    chinese_plan = build_wvs7_authoring_plan(candidate, review, accepted, WVS7LanguageArm.CHINESE_APPLIED)
    with pytest.raises(WVS7TranscriptionTemplateError, match="English"):
        build_english_core_template(chinese_plan, accepted)


def test_generator_rejects_q48_or_wrong_variable_set(tmp_path: Path) -> None:
    plan = _plan()
    mutated_item = plan.items[0].model_copy(update={"variable_id": "Q48"})
    mutated_plan = plan.model_copy(update={"items": [mutated_item, *plan.items[1:]]})
    with pytest.raises(WVS7TranscriptionTemplateError, match="Q48"):
        build_english_core_template(mutated_plan, _inputs()[2])
    with pytest.raises(WVS7TranscriptionTemplateError, match="private_wvs_transcriptions"):
        write_english_core_template(plan, _inputs()[2], tmp_path / "not-private" / "english.toml")


def test_blank_template_is_blocked_by_existing_authoring_validator(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plan = _plan()
    path = tmp_path / "private_wvs_transcriptions" / "english.toml"
    write_english_core_template(plan, _inputs()[2], path)
    arguments = [
        "--candidate", str(CANDIDATE), "--review", str(REVIEW), "--acceptance", str(ACCEPTANCE),
        "--language", "english", "--transcriptions", str(path),
    ]
    assert authoring_cli.main(arguments) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"blocking_reasons": ["manual_transcription_invalid"], "status": "blocked"}


def test_complete_synthetic_transcription_passes_without_writing_probe_set(tmp_path: Path) -> None:
    plan = _plan()
    transcriptions = _complete_transcriptions(plan)
    assert validate_manual_transcriptions(plan, transcriptions) == 23
    assert not list(tmp_path.iterdir())
    assert not (tmp_path / "frozen_probe_set.json").exists()


@pytest.mark.parametrize("field", ["question_text", "manual_transcriber_id", "transcribed_on"])
def test_todo_text_or_transcriber_is_rejected(field: str) -> None:
    plan = _plan()
    data = _complete_transcriptions(plan)
    changed = []
    for item in data.items:
        if item.variable_id == plan.items[0].variable_id:
            changed.append(item.model_copy(update={field: None if field == "transcribed_on" else "TODO_MANUAL_ENTRY"}))
        else:
            changed.append(item)
    with pytest.raises(WVS7AuthoringError, match="TODO|missing"):
        validate_manual_transcriptions(plan, data.model_copy(update={"items": changed}))


@pytest.mark.parametrize("field", ["source_questionnaire_pages", "source_codebook_pages", "source_sha256_prefixes", "endpoint_labels"])
def test_locators_hashes_and_endpoints_must_match_plan(field: str) -> None:
    plan = _plan()
    data = _complete_transcriptions(plan)
    item = data.items[0]
    replacement = {
        "source_questionnaire_pages": [999],
        "source_codebook_pages": [999],
        "source_sha256_prefixes": {"wrong": "0" * 12, "wrong2": "1" * 12},
        "endpoint_labels": {"999": "synthetic"},
    }[field]
    changed = [item.model_copy(update={field: replacement}), *data.items[1:]]
    with pytest.raises(WVS7AuthoringError, match="does not match"):
        validate_manual_transcriptions(plan, data.model_copy(update={"items": changed}))


def test_special_code_rules_are_required() -> None:
    plan = _plan()
    for variable_id, special in (("Q111", {"3": "missing"}), ("Q241", {"0": "missing"}), ("Q149", {})):
        changed = [item.model_copy(update={"special_response_codes": special}) if item.variable_id == variable_id else item for item in plan.items]
        mutated = plan.model_copy(update={"items": changed})
        with pytest.raises(WVS7TranscriptionTemplateError, match="special-code"):
            build_english_core_template(mutated, _inputs()[2])


def test_q149_and_q150_cannot_be_continuous() -> None:
    plan = _plan()
    item = next(item for item in plan.items if item.variable_id == "Q149")
    mutated = plan.model_copy(update={"items": [item.model_copy(update={"scoring_mode": "continuous"}) if row.variable_id == "Q149" else row for row in plan.items]})
    with pytest.raises(WVS7TranscriptionTemplateError, match="special-code"):
        build_english_core_template(mutated, _inputs()[2])


def test_unknown_fields_and_missing_required_transcription_shapes_rejected() -> None:
    plan = _plan()
    complete = _complete_transcriptions(plan)
    with pytest.raises(ValidationError, match="Extra inputs"):
        type(complete.items[0]).model_validate({**complete.items[0].model_dump(mode="json"), "question_text_extra": "x"})
    with pytest.raises(ValidationError):
        type(complete.items[0]).model_validate({**complete.items[0].model_dump(mode="json"), "response_options": []})
    with pytest.raises(ValidationError):
        type(complete.items[0]).model_validate({**complete.items[0].model_dump(mode="json"), "endpoint_labels": {}})


def test_smoke_is_local_only_and_reports_needs_manual_text(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def refuse_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("template smoke attempted network")

    monkeypatch.setattr(socket, "create_connection", refuse_network)
    assert transcription_template_smoke.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "needs_manual_text"
    assert payload["included_count"] == 23
    assert len(payload["blocking_candidate_ids"]) == 23
    assert "synthetic question" not in json.dumps(payload)
    assert "synthetic label" not in json.dumps(payload)


def test_completed_validation_does_not_create_probe_set_or_results(tmp_path: Path) -> None:
    plan = _plan()
    path = tmp_path / "private_wvs_transcriptions" / "complete.toml"
    path.parent.mkdir()
    # The in-memory synthetic fixture is sufficient to prove validation; no file
    # writer is invoked and no frozen artifact is produced.
    assert validate_manual_transcriptions(plan, _complete_transcriptions(plan)) == 23
    assert not (tmp_path / "results").exists()
