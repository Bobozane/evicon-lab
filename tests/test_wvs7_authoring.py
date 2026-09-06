"""Tests for the local-only WVS Wave 7 pre-freeze authoring gate."""

from __future__ import annotations

import json
import socket
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from evicon import validate_wvs7_authoring as authoring_cli
from evicon import wvs7_authoring_smoke as authoring_smoke
from evicon.wvs7_authoring import (
    AcceptanceStatus,
    AuthoringGateStatus,
    AuthoringPlanStatus,
    WVS7AuthoringAcceptance,
    WVS7AuthoringAcceptanceLoader,
    WVS7AuthoringError,
    WVS7LanguageArm,
    WVS7ManualResponseOption,
    WVS7ManualTranscriptionItem,
    WVS7ManualTranscriptionSet,
    build_wvs7_authoring_plan,
    evaluate_wvs7_authoring_gate,
    freeze_synthetic_probe_set,
    validate_manual_transcriptions,
)
from evicon.wvs7_candidates import WVS7CandidateLoader
from evicon.wvs7_review import WVS7ReviewLoader


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = ROOT / "configs/studies/wvs7_candidate_24.toml"
REVIEW_PATH = ROOT / "configs/studies/wvs7_candidate_24_codex_preliminary_review.toml"
ACCEPTANCE_PATH = ROOT / "configs/studies/wvs7_authoring_acceptance_template.toml"


def _candidate():
    return WVS7CandidateLoader.load(CANDIDATE_PATH)


def _review():
    return WVS7ReviewLoader.load(REVIEW_PATH)


def _pending_acceptance():
    return WVS7AuthoringAcceptanceLoader.load(ACCEPTANCE_PATH)


def _accepted_acceptance(*, scope: list[str] | None = None) -> WVS7AuthoringAcceptance:
    payload = _pending_acceptance().model_dump(mode="json")
    payload.update(
        {
            "acceptance_status": "accepted",
            "accepted_by": "researcher-local",
            "accepted_on": "2026-08-18",
            "acceptance_scope": scope or ["english_core", "chinese_applied"],
            "confirm_preliminary_decisions": True,
            "confirm_q48_exclusion": True,
            "confirm_special_code_handling": True,
            "confirm_cross_language_independence": True,
        }
    )
    return WVS7AuthoringAcceptance.model_validate(payload)


def _plan(language: WVS7LanguageArm, *, accepted: bool = True):
    return build_wvs7_authoring_plan(_candidate(), _review(), _accepted_acceptance() if accepted else _pending_acceptance(), language)


def _synthetic_transcriptions(plan, *, language: WVS7LanguageArm | None = None) -> WVS7ManualTranscriptionSet:
    selected_language = language or plan.language_arm
    rows = []
    for item in plan.items:
        rows.append(
            WVS7ManualTranscriptionItem(
                variable_id=item.variable_id,
                language=selected_language,
                source_review_id=plan.source_review_id,
                source_questionnaire_file=item.source_questionnaire_file,
                source_codebook_file=item.source_codebook_file,
                source_questionnaire_pages=item.source_questionnaire_pages,
                source_codebook_pages=item.source_codebook_pages,
                source_sha256_prefixes=item.source_sha256_prefixes,
                question_text=f"synthetic fixture item {item.variable_id}",
                response_options=[
                    WVS7ManualResponseOption(code=code, label=f"synthetic option {item.variable_id}-{code}")
                    for code in item.response_option_codes
                ],
                scale_min=item.scale_min,
                scale_max=item.scale_max,
                endpoint_labels={code: f"synthetic endpoint {item.variable_id}-{code}" for code in item.endpoint_codes},
                special_response_codes=item.special_response_codes,
                manual_transcriber_id="synthetic-transcriber",
                transcribed_on=date(2026, 8, 18),
            )
        )
    return WVS7ManualTranscriptionSet(
        language=selected_language,
        source_review_id=plan.source_review_id,
        synthetic_fixture=True,
        items=rows,
    )


def test_default_acceptance_is_pending_and_blocks_both_arms() -> None:
    acceptance = _pending_acceptance()
    assert acceptance.acceptance_status is AcceptanceStatus.PENDING
    for language in (WVS7LanguageArm.ENGLISH_CORE, WVS7LanguageArm.CHINESE_APPLIED):
        plan, report = evaluate_wvs7_authoring_gate(_candidate(), _review(), acceptance, language)
        assert plan.status is AuthoringPlanStatus.BLOCKED
        assert report.status is AuthoringGateStatus.BLOCKED
        assert report.blocking_reasons == ["acceptance_pending", "manual_transcription_required"]
        assert report.transcribed_count == 0


@pytest.mark.parametrize("field", ["accepted_by", "accepted_on"])
def test_acceptance_requires_identity_date_and_all_confirmations(field: str) -> None:
    payload = _accepted_acceptance().model_dump(mode="json")
    payload[field] = ""
    with pytest.raises(ValidationError, match="accepted"):
        WVS7AuthoringAcceptance.model_validate(payload)

    payload = _accepted_acceptance().model_dump(mode="json")
    payload["confirm_special_code_handling"] = False
    with pytest.raises(ValidationError, match="confirmation"):
        WVS7AuthoringAcceptance.model_validate(payload)


def test_ready_preliminary_review_cannot_bypass_researcher_acceptance() -> None:
    plan = _plan(WVS7LanguageArm.ENGLISH_CORE, accepted=False)
    assert plan.status is AuthoringPlanStatus.BLOCKED
    assert _review().items[1].english_core.status.value == "included"


def test_q48_is_excluded_and_each_language_plan_has_23_distinct_items() -> None:
    english = _plan(WVS7LanguageArm.ENGLISH_CORE)
    chinese = _plan(WVS7LanguageArm.CHINESE_APPLIED)
    assert english.candidate_count == chinese.candidate_count == 24
    assert english.included_count == chinese.included_count == 23
    assert english.excluded_count == chinese.excluded_count == 1
    assert "Q48" not in {item.variable_id for item in english.items}
    assert {item.variable_id for item in english.items} == {item.variable_id for item in chinese.items}
    assert english.items[0].source_questionnaire_file != chinese.items[0].source_questionnaire_file


def test_special_code_and_operational_notes_are_preserved() -> None:
    plan = _plan(WVS7LanguageArm.ENGLISH_CORE)
    items = {item.variable_id: item for item in plan.items}
    assert items["Q111"].special_response_codes["3"].startswith("volunteered_other")
    assert items["Q149"].scoring_mode == "binary_categorical_not_continuous"
    assert items["Q150"].scoring_mode == "binary_categorical_not_continuous"
    for variable_id in ("Q241", "Q243", "Q246", "Q247", "Q248", "Q249"):
        assert items[variable_id].special_response_codes["0"].startswith("volunteered_against_democracy")
    for variable_id in ("Q158", "Q159", "Q160", "Q161", "Q162", "Q163"):
        assert "secondary_science_technology_attitude_analysis_not_pure_value_factor" in items[variable_id].authoring_notes
    assert "english_chinese_semantic_scope_requires_manual_cross_language_review" in items["Q246"].authoring_notes


def test_manual_transcription_rejects_missing_text_options_endpoints_pages_and_hashes() -> None:
    plan = _plan(WVS7LanguageArm.ENGLISH_CORE)
    item = _synthetic_transcriptions(plan).items[0]
    payload = item.model_dump(mode="json")
    for field, replacement in (
        ("question_text", ""),
        ("response_options", []),
        ("source_questionnaire_pages", []),
        ("source_sha256_prefixes", {}),
    ):
        broken = dict(payload)
        broken[field] = replacement
        with pytest.raises(ValidationError):
            WVS7ManualTranscriptionItem.model_validate(broken)
    broken = dict(payload)
    broken["endpoint_labels"] = {str(item.scale_min): "synthetic endpoint"}
    with pytest.raises(ValidationError, match="endpoint_labels"):
        WVS7ManualTranscriptionItem.model_validate(broken)


def test_manual_transcription_rejects_language_unknown_duplicate_and_missing_variables() -> None:
    english_plan = _plan(WVS7LanguageArm.ENGLISH_CORE)
    english = _synthetic_transcriptions(english_plan)
    chinese_plan = _plan(WVS7LanguageArm.CHINESE_APPLIED)
    chinese = _synthetic_transcriptions(chinese_plan)
    with pytest.raises(WVS7AuthoringError, match="language or review mismatch"):
        validate_manual_transcriptions(english_plan, chinese)

    unknown = english.model_copy(update={"items": [english.items[0].model_copy(update={"variable_id": "Q999"}), *english.items[1:]]})
    with pytest.raises(WVS7AuthoringError, match="exactly cover"):
        validate_manual_transcriptions(english_plan, unknown)
    missing = english.model_copy(update={"items": english.items[:-1]})
    with pytest.raises(WVS7AuthoringError, match="exactly cover"):
        validate_manual_transcriptions(english_plan, missing)
    with pytest.raises(ValidationError, match="unique"):
        WVS7ManualTranscriptionSet.model_validate({**english.model_dump(mode="json"), "items": [english.items[0].model_dump(mode="json"), english.items[0].model_dump(mode="json")]})


def test_transcription_must_preserve_special_code_rules_and_noncontinuous_binary_handling() -> None:
    plan = _plan(WVS7LanguageArm.ENGLISH_CORE)
    transcriptions = _synthetic_transcriptions(plan)
    for variable_id in ("Q111", "Q149", "Q241"):
        changed = []
        for item in transcriptions.items:
            changed.append(item.model_copy(update={"special_response_codes": {}}) if item.variable_id == variable_id else item)
        broken = transcriptions.model_copy(update={"items": changed})
        with pytest.raises(WVS7AuthoringError, match="special response codes"):
            validate_manual_transcriptions(plan, broken)


def test_complete_synthetic_arms_can_be_ready_but_no_write_happens_without_confirmation(tmp_path: Path) -> None:
    acceptance = _accepted_acceptance()
    for language in (WVS7LanguageArm.ENGLISH_CORE, WVS7LanguageArm.CHINESE_APPLIED):
        plan = build_wvs7_authoring_plan(_candidate(), _review(), acceptance, language)
        transcriptions = _synthetic_transcriptions(plan)
        _, report = evaluate_wvs7_authoring_gate(_candidate(), _review(), acceptance, language, transcriptions)
        assert report.status is AuthoringGateStatus.READY_FOR_FREEZE
        output = tmp_path / f"{language.value}.json"
        with pytest.raises(WVS7AuthoringError, match="confirm_freeze"):
            freeze_synthetic_probe_set(plan, report, transcriptions, output, confirm_freeze=False)
        assert not output.exists()


def test_confirmed_synthetic_freeze_creates_two_independent_safe_test_probe_sets(tmp_path: Path) -> None:
    acceptance = _accepted_acceptance()
    outputs = []
    for language in (WVS7LanguageArm.ENGLISH_CORE, WVS7LanguageArm.CHINESE_APPLIED):
        plan = build_wvs7_authoring_plan(_candidate(), _review(), acceptance, language)
        transcriptions = _synthetic_transcriptions(plan)
        _, report = evaluate_wvs7_authoring_gate(_candidate(), _review(), acceptance, language, transcriptions)
        output = tmp_path / f"synthetic-{language.value}.json"
        probe_set = freeze_synthetic_probe_set(plan, report, transcriptions, output, confirm_freeze=True)
        assert output.exists()
        assert probe_set.metadata["synthetic_fixture"] is True
        assert probe_set.metadata["not_wvs_probe_set"] is True
        outputs.append(output)
    assert outputs[0].read_bytes() != outputs[1].read_bytes()
    combined = "".join(path.read_text(encoding="utf-8") for path in outputs)
    assert "hidden" not in combined
    assert "api_key" not in combined
    assert "provider_metadata" not in combined


def test_unknown_fields_and_content_fields_are_rejected_from_plans_and_acceptance() -> None:
    plan = _plan(WVS7LanguageArm.ENGLISH_CORE)
    payload = plan.items[0].model_dump(mode="json")
    for field in ("question_text", "options", "prompt"):
        with pytest.raises(ValidationError, match="Extra inputs"):
            type(plan.items[0]).model_validate({**payload, field: "synthetic"})
    acceptance_payload = _pending_acceptance().model_dump(mode="json")
    with pytest.raises(ValidationError, match="Extra inputs"):
        WVS7AuthoringAcceptance.model_validate({**acceptance_payload, "question_text": "synthetic"})


def test_cli_and_smoke_are_safe_and_do_not_read_pdfs_or_use_network(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    original_read_bytes = Path.read_bytes

    def refuse_pdf(path: Path) -> bytes:
        if path.suffix == ".pdf":
            raise AssertionError("authoring gate attempted to read a PDF")
        return original_read_bytes(path)

    def refuse_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("authoring gate attempted a network call")

    monkeypatch.setattr(Path, "read_bytes", refuse_pdf)
    monkeypatch.setattr(socket, "create_connection", refuse_network)
    arguments = ["--candidate", str(CANDIDATE_PATH), "--review", str(REVIEW_PATH), "--acceptance", str(ACCEPTANCE_PATH), "--language", "english"]
    assert authoring_cli.main(arguments) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "blocked"
    assert output["blocking_reasons"] == ["acceptance_pending", "manual_transcription_required"]
    serialized = json.dumps(output)
    assert "question_text" not in serialized
    assert "response_options" not in serialized
    assert "prompt" not in serialized

    assert authoring_smoke.main() == 0
    smoke_output = json.loads(capsys.readouterr().out)
    assert smoke_output["mode"] == "local_only"
    assert all(report["status"] == "blocked" for report in smoke_output["reports"])
