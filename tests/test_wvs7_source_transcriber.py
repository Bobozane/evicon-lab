"""Tests for the local-only English WVS source extraction draft."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
from pydantic import ValidationError

from evicon import source_transcription_smoke
from evicon import validate_wvs7_authoring as authoring_cli
from evicon.wvs7_authoring import (
    WVS7AuthoringAcceptanceLoader,
    WVS7LanguageArm,
    build_wvs7_authoring_plan,
)
from evicon.wvs7_candidates import WVS7CandidateLoader
from evicon.wvs7_review import WVS7ReviewLoader
from evicon.wvs7_source_transcriber import (
    DeterministicPdfTextExtractor,
    PdfPageText,
    WVS7SourceTranscriptionError,
    extract_english_core_source_draft,
    safe_source_draft_summary,
    write_english_core_source_draft,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "configs/studies/wvs7_candidate_24.toml"
REVIEW = ROOT / "configs/studies/wvs7_candidate_24_codex_preliminary_review.toml"
ACCEPTANCE = ROOT / "configs/studies/wvs7_authoring_acceptance_researcher_2026-08-18.toml"
PDF = ROOT / "date/F00008646-WVS7_Questionnaire_USA_2017_English.pdf"


def _plan():
    return build_wvs7_authoring_plan(
        WVS7CandidateLoader.load(CANDIDATE),
        WVS7ReviewLoader.load(REVIEW),
        WVS7AuthoringAcceptanceLoader.load(ACCEPTANCE),
        WVS7LanguageArm.ENGLISH_CORE,
    )


def test_local_pdf_extracts_exactly_23_items_without_q48() -> None:
    plan = _plan()
    draft = extract_english_core_source_draft(plan, PDF)
    assert len(draft.items) == 23
    assert "Q48" not in {item.variable_id for item in draft.items}
    assert all(item.human_verified is False for item in draft.items)
    assert all(item.transcription_origin == "local_pdf_text_extraction_draft" for item in draft.items)
    assert draft.items[5].special_response_codes["3"].startswith("volunteered_other")
    assert draft.items[6].special_response_codes["_scoring_mode"] == "binary_categorical_not_continuous"
    assert draft.items[-1].special_response_codes["0"].startswith("volunteered_against_democracy")
    plan_special_codes = {item.variable_id: item.special_response_codes for item in plan.items}
    assert all(item.special_response_codes == plan_special_codes[item.variable_id] for item in draft.items)


def test_source_draft_writes_only_private_path_and_preserves_metadata(tmp_path: Path) -> None:
    output = tmp_path / "private_wvs_transcriptions" / "english.toml"
    draft = write_english_core_source_draft(_plan(), PDF, output)
    assert output.exists()
    assert len(draft.items) == 23
    assert 'transcription_origin = "local_pdf_text_extraction_draft"' in output.read_text(encoding="utf-8")
    assert "human_verified = false" in output.read_text(encoding="utf-8")
    with pytest.raises(WVS7SourceTranscriptionError, match="private"):
        write_english_core_source_draft(_plan(), PDF, tmp_path / "results" / "bad.toml")


def test_q48_plan_and_non_english_plan_are_rejected() -> None:
    plan = _plan()
    broken = plan.model_copy(update={"items": tuple(plan.items[:-1])})
    with pytest.raises(WVS7SourceTranscriptionError, match="23"):
        extract_english_core_source_draft(broken, PDF)
    chinese = build_wvs7_authoring_plan(
        WVS7CandidateLoader.load(CANDIDATE),
        WVS7ReviewLoader.load(REVIEW),
        WVS7AuthoringAcceptanceLoader.load(ACCEPTANCE),
        WVS7LanguageArm.CHINESE_APPLIED,
    )
    with pytest.raises(WVS7SourceTranscriptionError, match="English"):
        extract_english_core_source_draft(chinese, PDF)


def test_missing_and_invalid_pages_are_rejected() -> None:
    with pytest.raises(WVS7SourceTranscriptionError, match="outside"):
        DeterministicPdfTextExtractor(PDF).extract_pages([999])
    with pytest.raises(WVS7SourceTranscriptionError, match="invalid"):
        DeterministicPdfTextExtractor(PDF).extract_pages([])
    plan = _plan().model_copy(deep=True)
    plan.items[0].source_questionnaire_pages[:] = [999]  # type: ignore[index]
    with pytest.raises(WVS7SourceTranscriptionError, match="outside"):
        extract_english_core_source_draft(plan, PDF)


def test_missing_pdf_text_creates_warning_and_never_guesses(monkeypatch: pytest.MonkeyPatch) -> None:
    original = DeterministicPdfTextExtractor.extract_pages

    def sparse(self: DeterministicPdfTextExtractor, pages: object) -> dict[int, PdfPageText]:
        return {int(page): PdfPageText(int(page), ("no variable marker",)) for page in pages}  # type: ignore[arg-type]

    monkeypatch.setattr(DeterministicPdfTextExtractor, "extract_pages", sparse)
    draft = extract_english_core_source_draft(_plan(), PDF)
    summary = safe_source_draft_summary(draft)
    assert summary["extracted_count"] == 23
    assert all("question_marker_not_found" in row["warnings"] for row in summary["items"])
    monkeypatch.setattr(DeterministicPdfTextExtractor, "extract_pages", original)


def test_human_verification_false_blocks_existing_authoring_gate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "private_wvs_transcriptions" / "english.toml"
    write_english_core_source_draft(_plan(), PDF, output)
    status = authoring_cli.main([
        "--candidate", str(CANDIDATE), "--review", str(REVIEW), "--acceptance", str(ACCEPTANCE),
        "--language", "english", "--transcriptions", str(output),
    ])
    assert status == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"blocking_reasons": ["human_verification_required"], "status": "blocked"}


def test_unknown_machine_or_llm_fields_are_rejected() -> None:
    draft = extract_english_core_source_draft(_plan(), PDF)
    payload = draft.items[0].model_dump(mode="json")
    with pytest.raises(ValidationError):
        type(draft.items[0]).model_validate({**payload, "llm_generated_text": "x"})
    with pytest.raises(ValidationError):
        type(draft.items[0]).model_validate({**payload, "transcription_origin": "machine_translation"})


def test_smoke_is_local_and_does_not_print_source_text(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    output_path = tmp_path / "private_wvs_transcriptions" / "english_core_wvs7_23.toml"
    assert source_transcription_smoke.main(["--output", str(output_path)]) == 0
    output = capsys.readouterr().out
    assert "human_verification_required" in output
    assert "Protecting the environment" not in output
