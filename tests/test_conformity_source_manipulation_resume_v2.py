from __future__ import annotations

import re
import shutil
from hashlib import sha256
from pathlib import Path

from evicon.conformity_source_manipulation_resume_v2 import (
    DEFAULT_AMENDMENT, DEFAULT_LEDGER, audit_pre_resume_ledger, load_amendment, preflight, run_resume,
)
from evicon.conformity_source_manipulation_smoke import ManipulationFakeProvider
from evicon.request_ledger import RequestLedger, RequestLedgerStatus

ROOT = Path(__file__).resolve().parents[1]


def _v2_history_prefix(tmp_path: Path) -> Path:
    ledger = tmp_path / "v2_history.jsonl"
    lines = (ROOT / DEFAULT_LEDGER).read_text(encoding="utf-8").splitlines()
    ledger.write_text("\n".join(lines[:16]) + "\n", encoding="utf-8")
    return ledger


def _accepted_amendment(tmp_path: Path, ledger: Path) -> Path:
    raw = (ROOT / DEFAULT_AMENDMENT).read_text(encoding="utf-8")
    raw = raw.replace('acceptance_status = "pending"', 'acceptance_status = "accepted"')
    raw = raw.replace('accepted_by = ""', 'accepted_by = "test-reviewer"')
    raw = raw.replace('accepted_on = ""', 'accepted_on = "2026-08-28"')
    raw = re.sub(r"(confirm_[a-z0-9_]+) = false", r"\1 = true", raw)
    raw = re.sub(
        r'pre_resume_ledger_sha256 = "[0-9a-f]+"',
        f'pre_resume_ledger_sha256 = "{sha256(ledger.read_bytes()).hexdigest()}"',
        raw,
    )
    target = tmp_path / "amendment.toml"
    target.write_text(raw, encoding="utf-8")
    return target


def test_v2_audits_local_configuration_failure_without_http_claim() -> None:
    try:
        audit_pre_resume_ledger(load_amendment())
    except Exception as exc:
        assert getattr(exc, "code", None) == "timeout_resume_v2_ledger_hash_mismatch"
    else:
        raise AssertionError("v2 must reject post-v2 append-only ledger history")


def test_v2_preflight_is_network_free_and_requires_approval() -> None:
    result = preflight()
    assert result["blocking_reasons"] == ["timeout_resume_v2_ledger_hash_mismatch"]
    assert result["network"] == "disabled"
    assert result["provider_constructed"] is False


def test_v2_fake_resume_preserves_prefix_and_completes(tmp_path: Path) -> None:
    ledger = _v2_history_prefix(tmp_path)
    prefix = ledger.read_text(encoding="utf-8")
    amendment = _accepted_amendment(tmp_path, ledger)
    result = run_resume(ManipulationFakeProvider(), amendment_path=amendment,
                        ledger_path=ledger, receipt_path=tmp_path / "receipt.json")
    assert result["status"] == "completed"
    assert result["logical_request_count"] == 12
    assert result["transport_attempt_count"] == 14
    assert result["http_transport_retry_count"] == 1
    assert result["completed_fingerprint_replay_count"] == 0
    assert ledger.read_text(encoding="utf-8").startswith(prefix)
    entries = RequestLedger(ledger).entries()
    started = [item for item in entries if item.status is RequestLedgerStatus.STARTED]
    assert len(started) == 14
    target = load_amendment(amendment).failed_fingerprint
    assert sum(item.fingerprint == target for item in started) == 3


def test_v2_rejects_any_ledger_change(tmp_path: Path) -> None:
    ledger = _v2_history_prefix(tmp_path)
    ledger.write_text(ledger.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    try:
        audit_pre_resume_ledger(load_amendment(), ledger_path=ledger)
    except Exception as exc:
        assert getattr(exc, "code", None) == "timeout_resume_v2_ledger_hash_mismatch"
    else:
        raise AssertionError("mutated ledger must be rejected")
