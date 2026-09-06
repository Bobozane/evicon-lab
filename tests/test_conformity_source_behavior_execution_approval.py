from __future__ import annotations

from pathlib import Path

import pytest

from evicon.conformity_source_behavior_execution_approval import (
    DEFAULT_EXECUTION_APPROVAL,
    load_execution_approval,
    safe_preflight,
)


ROOT = Path(__file__).resolve().parents[1]


def test_accepted_execution_approval_binds_completed_compatibility_receipt() -> None:
    approval = load_execution_approval()
    report = safe_preflight()
    assert approval.acceptance_status == "accepted"
    assert report["status"] == "source_behavior_execution_gate_offline_ready"
    assert report["blocking_reasons"] == [
        "behavior_qualification_network_authorization_required"
    ]
    assert report["ready_for_network"] is True
    assert report["compatibility_receipt_sha256"] == (
        "196e5fb3ecc0faae2fe435ffe8fffa973dfacfe16d9f445fdbc043ede8aacd22"
    )
    assert report["network"] == "disabled"
    assert report["provider_constructed"] is False
    assert report["api_key_read"] is False
    assert report["outputs_written"] is False


def test_execution_approval_rejects_unknown_field(tmp_path: Path) -> None:
    raw = (ROOT / DEFAULT_EXECUTION_APPROVAL).read_text(encoding="utf-8")
    path = tmp_path / "approval.toml"
    path.write_text(raw + "\nraw_prompt = \"forbidden\"\n", encoding="utf-8")
    with pytest.raises(Exception, match="source_behavior_execution_approval_invalid"):
        load_execution_approval(path)


def test_execution_approval_rejects_binding_drift(tmp_path: Path) -> None:
    raw = (ROOT / DEFAULT_EXECUTION_APPROVAL).read_text(encoding="utf-8")
    path = tmp_path / "approval.toml"
    path.write_text(
        raw.replace("logical_request_cap = 12", "logical_request_cap = 11"),
        encoding="utf-8",
    )
    with pytest.raises(Exception):
        load_execution_approval(path)


def test_accepted_execution_approval_requires_all_confirmations(tmp_path: Path) -> None:
    raw = (ROOT / DEFAULT_EXECUTION_APPROVAL).read_text(encoding="utf-8")
    raw = raw.replace(
        "confirm_safe_case_audit_only = true",
        "confirm_safe_case_audit_only = false",
    )
    path = tmp_path / "approval.toml"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(Exception, match="source_behavior_execution_approval_invalid"):
        load_execution_approval(path)


def test_accepted_execution_approval_still_requires_per_command_network_authorization(
    tmp_path: Path,
) -> None:
    raw = (ROOT / DEFAULT_EXECUTION_APPROVAL).read_text(encoding="utf-8")
    path = tmp_path / "approval.toml"
    path.write_text(raw, encoding="utf-8")
    report = safe_preflight(path)
    assert report["approval_status"] == "accepted"
    assert report["ready_for_network"] is True
    assert report["blocking_reasons"] == [
        "behavior_qualification_network_authorization_required"
    ]
    assert report["network"] == "disabled"
    assert report["api_key_read"] is False
