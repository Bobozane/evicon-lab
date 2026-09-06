from __future__ import annotations

from pathlib import Path

from evicon.conformity_identification import sha256_file
from evicon.conformity_source_relation_comprehension_calibration_approval_v1 import (
    DEFAULT_APPROVAL,
    approval_sha256,
    load_approval,
    safe_preflight,
)
from evicon.conformity_source_relation_comprehension_calibration_v1 import (
    DEFAULT_CONFIG,
    RESPONSE_SCHEMA_SHA256,
    load_config,
)


def _accepted_approval(tmp_path: Path) -> Path:
    raw = (Path(__file__).resolve().parents[1] / DEFAULT_APPROVAL).read_text(
        encoding="utf-8"
    )
    raw = raw.replace('acceptance_status = "pending"', 'acceptance_status = "accepted"')
    raw = raw.replace('accepted_by = ""', 'accepted_by = "test-reviewer"')
    raw = raw.replace('accepted_on = ""', 'accepted_on = "2026-08-29"')
    path = tmp_path / "accepted-approval.toml"
    path.write_text(raw, encoding="utf-8")
    return path


def test_pending_default_approval_is_bound_and_keeps_network_unapproved() -> None:
    approval = load_approval()
    config = load_config()
    assert approval.acceptance_status == "pending"
    assert approval.network_execution_authorized is False
    assert approval.config_sha256 == sha256_file(DEFAULT_CONFIG)
    assert approval.protocol_sha256 == config.protocol_sha256
    assert approval.response_schema_sha256 == RESPONSE_SCHEMA_SHA256
    assert approval_sha256()


def test_preflight_is_non_network_and_reports_both_review_boundaries() -> None:
    report = safe_preflight()
    assert report["status"] == "source_relation_calibration_approval_ready"
    assert report["approval_status"] == "pending"
    assert report["blocking_reasons"] == [
        "human_approval_required",
        "provider_compatibility_not_requested",
    ]
    assert report["network"] == "disabled"
    assert report["api_key_read"] is False
    assert report["provider_constructed"] is False
    assert report["results_written"] is False


def test_accepted_approval_loads_only_when_identity_and_date_are_present(
    tmp_path: Path,
) -> None:
    path = _accepted_approval(tmp_path)
    approval = load_approval(path)
    assert approval.acceptance_status == "accepted"
    assert approval.accepted_by == "test-reviewer"
    assert approval.accepted_on == "2026-08-29"


def test_hash_tampering_blocks_preflight_without_exposing_details(tmp_path: Path) -> None:
    path = _accepted_approval(tmp_path)
    raw = path.read_text(encoding="utf-8")
    path.write_text(
        raw.replace(
            "response_schema_sha256 = \"7f02f9d703c9d614f6e4e162e25903c7154991602d8f101a3e6e09c9439eb6ed\"",
            "response_schema_sha256 = \"0000000000000000000000000000000000000000000000000000000000000000\"",
        ),
        encoding="utf-8",
    )
    report = safe_preflight(path)
    assert report == {
        "status": "blocked",
        "blocking_reasons": ["source_relation_calibration_approval_schema_hash_mismatch"],
        "network": "disabled",
        "provider_constructed": False,
        "api_key_read": False,
        "results_written": False,
    }
