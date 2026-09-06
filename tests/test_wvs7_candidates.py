"""Tests for the draft-only local WVS Wave 7 candidate checklist."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
from pydantic import ValidationError

from evicon import validate_wvs7_candidates as candidates_cli
from evicon.wvs7_candidates import (
    CandidateMaterialStatus,
    WVS7CandidateError,
    WVS7CandidateLoader,
    WVS7CandidateManifest,
    validate_wvs7_candidates,
)


def _manifest_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs/studies/wvs7_candidate_24.toml"


def _manifest():
    return WVS7CandidateLoader.load(_manifest_path())


def test_checked_in_candidate_manifest_has_24_candidates_and_five_domains() -> None:
    report = validate_wvs7_candidates(_manifest())

    assert report.material_status is CandidateMaterialStatus.DRAFT
    assert report.candidate_count == 24
    assert report.domain_counts == {
        "autonomy_security": 3,
        "democratic_order": 6,
        "economic_distribution": 6,
        "science_knowledge": 6,
        "surveillance_privacy": 3,
    }
    assert len(report.source_hash_prefixes) == 4


def test_candidate_manifest_receipt_matches_external_source_document() -> None:
    manifest = _manifest()
    receipt = (Path(__file__).resolve().parents[1] / "docs/external-wvs7-sources.md").read_text(encoding="utf-8")

    for source in manifest.source_files:
        assert source.filename in receipt
        assert source.sha256 in receipt


def test_validator_rejects_duplicate_variable_wrong_hash_wrong_status_and_missing_presence() -> None:
    manifest = _manifest()
    duplicate = manifest.model_copy(
        update={"candidates": [manifest.candidates[0], manifest.candidates[0], *manifest.candidates[2:]]}
    )
    with pytest.raises(WVS7CandidateError, match="duplicate variable_id"):
        validate_wvs7_candidates(duplicate)

    wrong_hash_source = manifest.source_files[0].model_copy(update={"sha256": "0" * 64})
    wrong_hash = manifest.model_copy(update={"source_files": [wrong_hash_source, *manifest.source_files[1:]]})
    with pytest.raises(WVS7CandidateError, match="source hash"):
        validate_wvs7_candidates(wrong_hash)

    wrong_status = manifest.model_copy(update={"material_status": CandidateMaterialStatus.PILOT})
    with pytest.raises(WVS7CandidateError, match="status must be draft"):
        validate_wvs7_candidates(wrong_status)

    missing_presence_candidate = manifest.candidates[0].model_copy(
        update={"china": manifest.candidates[0].china.model_copy(update={"present": False})}
    )
    missing_presence = manifest.model_copy(
        update={"candidates": [missing_presence_candidate, *manifest.candidates[1:]]}
    )
    with pytest.raises(WVS7CandidateError, match="USA and China source presence"):
        validate_wvs7_candidates(missing_presence)


def test_manifest_rejects_illegal_domain_and_question_text_leakage_fields() -> None:
    data = _manifest().model_dump(mode="python")
    data["candidates"][0]["instrument_domain"] = "unsupported_domain"
    with pytest.raises(ValidationError, match="instrument_domain"):
        WVS7CandidateManifest.model_validate(data)

    data = _manifest().model_dump(mode="python")
    data["candidates"][0]["text"] = "redacted"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        WVS7CandidateManifest.model_validate(data)

    data = _manifest().model_dump(mode="python")
    data["metadata"]["question"] = "redacted"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        WVS7CandidateManifest.model_validate(data)


def test_validator_does_not_read_source_pdfs(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _manifest()

    def reject_pdf_read(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("candidate validator attempted to read a PDF")

    monkeypatch.setattr(Path, "read_bytes", reject_pdf_read)
    assert validate_wvs7_candidates(manifest).candidate_count == 24


def test_cli_output_is_safe_and_never_connects(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def reject_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("candidate validation attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", reject_connection)
    assert candidates_cli.main(["--manifest", str(_manifest_path())]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["candidate_count"] == 24
    assert set(output) == {
        "study_id",
        "manifest_version",
        "material_status",
        "candidate_count",
        "domain_counts",
        "source_hash_prefixes",
    }
