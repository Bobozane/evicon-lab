from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.provenance_cascade_hg22_integrity_audit import (
    HG22IntegrityAuditError, audit_hg22_integrity, safe_summary, write_audit_receipt,
)


def test_completed_hg22_batch_reconstructs_from_append_only_ledgers() -> None:
    receipt = audit_hg22_integrity()
    assert receipt.status == "passed_development_analysis_eligible"
    assert receipt.fix_occurred_after_batch_start and receipt.fix_occurred_before_batch_completion
    assert receipt.unique_logical_request_count == receipt.completed_logical_request_count == 288
    assert receipt.transport_attempt_count == 292
    assert receipt.failed_transport_attempt_count == receipt.recovered_failed_fingerprint_count == 4
    assert receipt.duplicate_completed_fingerprint_count == receipt.orphaned_started_attempt_count == 0
    assert receipt.replay_passed_count == 16
    assert receipt.prompt_or_request_content_changed is False
    assert receipt.fingerprint_algorithm_changed is False
    assert receipt.parser_or_output_contract_changed is False
    assert receipt.behavior_or_mechanism_conclusion_permitted is False


def test_audit_receipt_is_idempotent_and_safe(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    first = write_audit_receipt(path)
    second = write_audit_receipt(path)
    assert first == second
    serialized = path.read_text(encoding="utf-8").lower()
    for forbidden in ("ground_truth_label", "source_independence_label", "system_prompt", "user_prompt", "raw_response", "provider_metadata", "api_key", "authorization"):
        assert forbidden not in serialized
    summary = safe_summary(first, receipt_path=path)
    assert summary["network"] == "disabled" and summary["private_truth_loaded"] is False


def test_tampered_post_fix_hash_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("evicon.provenance_cascade_hg22_integrity_audit.POST_FIX_REQUEST_LEDGER_SHA256", "0" * 64)
    with pytest.raises(HG22IntegrityAuditError, match="request_ledger_post_fix_hash_mismatch"):
        audit_hg22_integrity()
