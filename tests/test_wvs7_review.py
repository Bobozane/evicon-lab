"""Tests for draft-only WVS Wave 7 manual review signoff contracts."""

from __future__ import annotations

import json
import socket
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from evicon import validate_wvs7_review as review_cli
from evicon.wvs7_candidates import InstrumentDomain, WVS7CandidateLoader
from evicon.wvs7_review import (
    AuthoringStatus,
    LanguageArmStatus,
    WVS7LanguageReviewArm,
    WVS7ReviewError,
    WVS7ReviewLoader,
    WVS7ReviewManifest,
    validate_wvs7_review,
)


def _candidate_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs/studies/wvs7_candidate_24.toml"


def _review_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs/studies/wvs7_candidate_24_review_template.toml"


def _candidate():
    return WVS7CandidateLoader.load(_candidate_path())


def _review():
    return WVS7ReviewLoader.load(_review_path())


def _completed_arm(arm: WVS7LanguageReviewArm, *, status: LanguageArmStatus = LanguageArmStatus.INCLUDED) -> WVS7LanguageReviewArm:
    values: dict[str, object] = {
        "status": status,
        "reviewer_id": "reviewer-local",
        "reviewed_on": date(2026, 8, 17),
        "codebook_pages": [1],
        "rationale": "manual review fixture",
    }
    if status is LanguageArmStatus.INCLUDED:
        values.update(
            {
                "scale_min": 1,
                "scale_max": 10,
                "high_score_meaning": "documented higher endpoint",
                "reverse_scored": False,
            }
        )
    return arm.model_copy(update=values)


def _review_with_completed_language(language_arm: str):
    review = _review()
    items = []
    for item in review.items:
        items.append(item.model_copy(update={language_arm: _completed_arm(getattr(item, language_arm))}))
    return review.model_copy(update={"items": items})


def test_default_template_covers_24_candidates_and_blocks_both_language_arms() -> None:
    report = validate_wvs7_review(_candidate(), _review())

    assert report.candidate_count == 24
    assert report.status is AuthoringStatus.BLOCKED
    assert report.english_core.blocked is True
    assert report.chinese_applied.blocked is True
    assert report.english_core.pending_count == 24
    assert report.chinese_applied.pending_count == 24


def test_review_validator_rejects_unknown_duplicate_missing_group_hash_and_page_mismatches() -> None:
    candidate = _candidate()
    review = _review()

    unknown_item = review.items[0].model_copy(update={"variable_id": "Q999"})
    unknown = review.model_copy(update={"items": [unknown_item, *review.items[1:]]})
    with pytest.raises(WVS7ReviewError, match="cover exactly"):
        validate_wvs7_review(candidate, unknown)

    duplicate = review.model_copy(update={"items": [review.items[0], review.items[0], *review.items[2:]]})
    with pytest.raises(WVS7ReviewError, match="duplicate variable_id"):
        validate_wvs7_review(candidate, duplicate)

    missing = review.model_copy(update={"items": review.items[:-1]})
    with pytest.raises(WVS7ReviewError, match="cover exactly"):
        validate_wvs7_review(candidate, missing)

    wrong_group_item = review.items[0].model_copy(update={"instrument_domain": InstrumentDomain.DEMOCRATIC_ORDER})
    wrong_group = review.model_copy(update={"items": [wrong_group_item, *review.items[1:]]})
    with pytest.raises(WVS7ReviewError, match="operational group"):
        validate_wvs7_review(candidate, wrong_group)

    wrong_hashes = dict(review.source_hashes)
    first_source = next(iter(wrong_hashes))
    wrong_hashes[first_source] = "0" * 12
    wrong_hash = review.model_copy(update={"source_hashes": wrong_hashes})
    with pytest.raises(WVS7ReviewError, match="source hash"):
        validate_wvs7_review(candidate, wrong_hash)

    wrong_page_arm = review.items[0].english_core.model_copy(update={"questionnaire_pages": [999]})
    wrong_page_item = review.items[0].model_copy(update={"english_core": wrong_page_arm})
    wrong_page = review.model_copy(update={"items": [wrong_page_item, *review.items[1:]]})
    with pytest.raises(WVS7ReviewError, match="questionnaire pages"):
        validate_wvs7_review(candidate, wrong_page)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("scale_min", "", "scale_min"),
        ("scale_max", "", "scale_min"),
        ("high_score_meaning", "", "high_score_meaning"),
        ("reverse_scored", "", "reverse_scored"),
        ("reviewer_id", "", "reviewer_id"),
        ("reviewed_on", "", "reviewer_id"),
        ("rationale", "", "reviewer_id"),
    ],
)
def test_included_arm_requires_all_manual_signoff_fields(field: str, value: object, expected: str) -> None:
    data = _review().items[0].english_core.model_dump(mode="python")
    data.update(
        {
            "status": "included",
            "reviewer_id": "reviewer-local",
            "reviewed_on": "2026-08-17",
            "codebook_pages": [1],
            "scale_min": 1,
            "scale_max": 10,
            "high_score_meaning": "documented higher endpoint",
            "reverse_scored": False,
            "rationale": "manual review fixture",
        }
    )
    data[field] = value
    with pytest.raises(ValidationError, match=expected):
        WVS7LanguageReviewArm.model_validate(data)


def test_excluded_arm_requires_rationale() -> None:
    data = _review().items[0].english_core.model_dump(mode="python")
    data.update(
        {
            "status": "excluded",
            "reviewer_id": "reviewer-local",
            "reviewed_on": "2026-08-17",
            "codebook_pages": [1],
            "rationale": "",
        }
    )
    with pytest.raises(ValidationError, match="rationale"):
        WVS7LanguageReviewArm.model_validate(data)


def test_english_can_be_ready_while_chinese_remains_independently_blocked() -> None:
    report = validate_wvs7_review(_candidate(), _review_with_completed_language("english_core"))

    assert report.status is AuthoringStatus.BLOCKED
    assert report.english_core.ready_for_authoring is True
    assert report.english_core.included_count == 24
    assert report.chinese_applied.blocked is True
    assert report.chinese_applied.pending_count == 24


def test_completed_synthetic_review_is_ready_without_creating_a_probe_set(tmp_path: Path) -> None:
    review = _review_with_completed_language("english_core")
    completed_items = [
        item.model_copy(update={"chinese_applied": _completed_arm(item.chinese_applied)}) for item in review.items
    ]
    completed_review = review.model_copy(update={"items": completed_items})

    before = list(tmp_path.iterdir())
    report = validate_wvs7_review(_candidate(), completed_review)
    after = list(tmp_path.iterdir())

    assert report.status is AuthoringStatus.READY_FOR_AUTHORING
    assert report.english_core.ready_for_authoring is True
    assert report.chinese_applied.ready_for_authoring is True
    assert before == after == []
    assert not (tmp_path / "wvs7_probe_set.toml").exists()


@pytest.mark.parametrize("field", ["question_text", "options", "prompt"])
def test_unknown_and_content_leakage_fields_are_rejected(field: str) -> None:
    data = _review().model_dump(mode="python")
    data["items"][0]["english_core"][field] = "redacted"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        WVS7ReviewManifest.model_validate(data)


def test_validator_does_not_read_pdf_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = _candidate()
    review = _review()

    def reject_pdf_read(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("review validator attempted to read a PDF")

    monkeypatch.setattr(Path, "read_bytes", reject_pdf_read)
    assert validate_wvs7_review(candidate, review).status is AuthoringStatus.BLOCKED


def test_cli_output_is_safe_and_strict_mode_reports_blocked(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def reject_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("review validation attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", reject_connection)
    arguments = ["--candidate", str(_candidate_path()), "--review", str(_review_path())]
    assert review_cli.main(arguments) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["status"] == "blocked"
    assert output["english_core"]["pending_count"] == 24
    assert output["chinese_applied"]["pending_count"] == 24
    assert "question_text" not in json.dumps(output)
    assert "prompt" not in json.dumps(output)
    assert review_cli.main([*arguments, "--strict"]) == 1
