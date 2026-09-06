from __future__ import annotations

import os
import re
import shutil
import hashlib
from pathlib import Path

from evicon.conformity_source_manipulation_resume_v3 import DEFAULT_AMENDMENT, DEFAULT_LEDGER, audit_pre_resume_ledger, load_amendment, require_provider_environment, run_resume
from evicon.conformity_source_manipulation_smoke import ManipulationFakeProvider
from evicon.request_ledger import RequestLedger, RequestLedgerStatus

ROOT = Path(__file__).resolve().parents[1]


def _accepted(tmp_path: Path) -> Path:
    raw = (ROOT / DEFAULT_AMENDMENT).read_text(encoding="utf-8")
    raw = raw.replace('acceptance_status = "pending"', 'acceptance_status = "accepted"')
    raw = raw.replace('accepted_by = ""', 'accepted_by = "test-reviewer"').replace('accepted_on = ""', 'accepted_on = "2026-08-28"')
    raw = re.sub(r"(confirm_[a-z0-9_]+) = false", r"\1 = true", raw)
    target = tmp_path / "amendment.toml"
    target.write_text(raw, encoding="utf-8")
    return target


def _pre_resume_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Reconstruct the immutable 18-line pre-v3 history in a temp location."""
    ledger = tmp_path / "pre_resume_ledger.jsonl"
    lines = (ROOT / DEFAULT_LEDGER).read_text(encoding="utf-8").splitlines()
    ledger.write_text("\n".join(lines[:18]) + "\n", encoding="utf-8")
    amendment = _accepted(tmp_path)
    raw = amendment.read_text(encoding="utf-8")
    raw = re.sub(
        r'pre_resume_ledger_sha256 = "[0-9a-f]{64}"',
        f'pre_resume_ledger_sha256 = "{hashlib.sha256(ledger.read_bytes()).hexdigest()}"',
        raw,
    )
    amendment.write_text(raw, encoding="utf-8")
    return ledger, amendment


def test_v3_requires_environment_before_ledger_write(tmp_path: Path, monkeypatch) -> None:
    ledger = tmp_path / "ledger.jsonl"
    shutil.copyfile(ROOT / DEFAULT_LEDGER, ledger)
    before = ledger.read_bytes()
    monkeypatch.delenv("EVICON_LLM_BASE_URL", raising=False)
    try:
        require_provider_environment()
    except Exception as exc:
        assert getattr(exc, "code", None) == "missing_base_url"
    else:
        raise AssertionError("missing environment must be rejected")
    assert ledger.read_bytes() == before


def test_v3_audits_pre_resume_history_in_isolation(tmp_path: Path) -> None:
    ledger, amendment = _pre_resume_fixture(tmp_path)
    state = audit_pre_resume_ledger(load_amendment(amendment), ledger_path=ledger)
    assert state["completed_count"] == 6
    assert state["transport_attempt_count"] == 9
    assert state["local_configuration_failure_count"] == 2


def test_v3_fake_resume_preserves_history_and_completes(tmp_path: Path) -> None:
    ledger, amendment = _pre_resume_fixture(tmp_path)
    prefix = ledger.read_text(encoding="utf-8")
    result = run_resume(ManipulationFakeProvider(), amendment_path=amendment, ledger_path=ledger, receipt_path=tmp_path / "receipt.json")
    assert result["logical_request_count"] == 12
    assert result["transport_attempt_count"] == 15
    assert result["http_transport_retry_count"] == 1
    assert ledger.read_text(encoding="utf-8").startswith(prefix)
    entries = RequestLedger(ledger).entries()
    target = load_amendment(amendment).failed_fingerprint
    assert sum(item.fingerprint == target and item.status is RequestLedgerStatus.STARTED for item in entries) == 4
