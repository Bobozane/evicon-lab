from __future__ import annotations

import json
import re
import shutil
import hashlib
from pathlib import Path

from evicon.conformity_source_manipulation_resume import (
    DEFAULT_AMENDMENT,
    DEFAULT_LEDGER,
    audit_initial_failure,
    load_amendment,
    preflight,
    run_resume,
)
from evicon.conformity_source_manipulation_smoke import ManipulationFakeProvider
from evicon.request_ledger import RequestLedger, RequestLedgerStatus

ROOT = Path(__file__).resolve().parents[1]


def _accepted_amendment(tmp_path: Path) -> Path:
    raw = (ROOT / DEFAULT_AMENDMENT).read_text(encoding="utf-8")
    raw = raw.replace('acceptance_status = "pending"', 'acceptance_status = "accepted"')
    raw = raw.replace('accepted_by = ""', 'accepted_by = "test-reviewer"')
    raw = raw.replace('accepted_on = ""', 'accepted_on = "2026-08-28"')
    raw = re.sub(r"(confirm_[a-z0-9_]+) = false", r"\1 = true", raw)
    target = tmp_path / "amendment.toml"
    target.write_text(raw, encoding="utf-8")
    return target


def _initial_ledger_copy(tmp_path: Path) -> Path:
    # The real ledger is append-only and now has a later local configuration
    # failure. v1 tests reproduce only its immutable initial 14-line prefix.
    target = tmp_path / "initial_request_ledger.jsonl"
    lines = (ROOT / DEFAULT_LEDGER).read_text(encoding="utf-8").splitlines()
    target.write_text("\n".join(lines[:14]) + "\n", encoding="utf-8")
    return target


def _accepted_amendment_for_ledger(tmp_path: Path, ledger: Path) -> Path:
    target = _accepted_amendment(tmp_path)
    raw = target.read_text(encoding="utf-8")
    raw = raw.replace(
        'initial_ledger_sha256 = "844dac246707bfa88a0dee32bf8ce84a773d01cc9f6c8563cbc5925b7190ef9b"',
        f'initial_ledger_sha256 = "{hashlib.sha256(ledger.read_bytes()).hexdigest()}"',
    )
    target.write_text(raw, encoding="utf-8")
    return target


def test_live_v1_ledger_refuses_later_append_only_history() -> None:
    try:
        audit_initial_failure(load_amendment())
    except Exception as exc:
        assert getattr(exc, "code", None) == "timeout_resume_ledger_hash_mismatch"
    else:
        raise AssertionError("v1 may not reinterpret a later append-only ledger")


def test_default_resume_preflight_refuses_later_history_without_network() -> None:
    result = preflight()
    assert result["status"] == "blocked"
    assert result["blocking_reasons"] == ["timeout_resume_ledger_hash_mismatch"]
    assert result["network"] == "disabled"
    assert result["provider_constructed"] is False


def test_fake_resume_preserves_completed_and_finishes(tmp_path: Path) -> None:
    ledger = _initial_ledger_copy(tmp_path)
    original_lines = ledger.read_text(encoding="utf-8").splitlines()
    receipt = tmp_path / "receipt.json"
    result = run_resume(
        ManipulationFakeProvider(), amendment_path=_accepted_amendment_for_ledger(tmp_path, ledger),
        ledger_path=ledger, receipt_path=receipt,
    )
    assert result["status"] == "completed"
    assert result["logical_request_count"] == 12
    assert result["transport_attempt_count"] == 13
    assert result["timeout_resume_transport_attempt_count"] == 1
    assert result["completed_fingerprint_replay_count"] == 0
    assert result["root_count_totals"] == {
        "independent_roots": 8, "same_root": 4, "source_free": 0,
    }
    final_lines = ledger.read_text(encoding="utf-8").splitlines()
    assert final_lines[:len(original_lines)] == original_lines
    entries = RequestLedger(ledger).entries()
    started = [item for item in entries if item.status is RequestLedgerStatus.STARTED]
    assert len(started) == 13
    attempts = [item for item in started if item.fingerprint == result.get("failed_fingerprint", "")]
    # The receipt omits the fingerprint; the ledger remains the source of truth.
    timeout_fingerprint = load_amendment(_accepted_amendment_for_ledger(tmp_path, ledger)).failed_fingerprint
    assert sum(item.fingerprint == timeout_fingerprint for item in started) == 2
    safe = receipt.read_text(encoding="utf-8").lower()
    for forbidden in ("system_prompt", "user_prompt", "model_reply", "api_key", "authorization", "ground_truth_label"):
        assert forbidden not in safe


def test_resume_rejects_changed_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "request_ledger.jsonl"
    shutil.copyfile(ROOT / DEFAULT_LEDGER, ledger)
    ledger.write_text(ledger.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    amendment = load_amendment()
    try:
        audit_initial_failure(amendment, ledger_path=ledger)
    except Exception as exc:
        assert getattr(exc, "code", None) == "timeout_resume_ledger_hash_mismatch"
    else:
        raise AssertionError("changed ledger must be rejected")


def test_amendment_contains_no_sensitive_material() -> None:
    safe = (ROOT / DEFAULT_AMENDMENT).read_text(encoding="utf-8").lower()
    for forbidden in ("system_prompt", "user_prompt", "model_reply", "api_key", "authorization", "ground_truth_label"):
        assert forbidden not in safe
