from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from evicon.provenance_cascade_hg_pilot import HGPilotBatchRecord, HGPilotRunner
from evicon.provenance_cascade_hg_replay_amendment import (
    DEFAULT_RECEIPT_PATH,
    FAILED_RUN_ID,
    HGReplayAmendmentError,
    audit_technical_amendment,
    safe_summary,
    validate_technical_amendment_receipt,
    write_technical_amendment_receipt,
)
from evicon.provenance_cascade_identifiability import hg_preflight, sha256_file

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/provenance-cascade-pilot-hg-v1"
BATCH_PATH = RESULT_ROOT / "pilot_batch_record.json"
FAILED_RUN_DIR = RESULT_ROOT / FAILED_RUN_ID


def _protected_hashes() -> dict[str, str]:
    paths = [
        BATCH_PATH,
        FAILED_RUN_DIR / "agent_checkpoint.json",
        FAILED_RUN_DIR / "request_ledger.jsonl",
    ]
    paths.extend(sorted(RESULT_ROOT.glob("*/run_record.json")))
    return {str(path): sha256_file(path) for path in paths}


def test_actual_batch_audit_is_read_only_and_complete() -> None:
    before = _protected_hashes()
    receipt, _ = validate_technical_amendment_receipt(DEFAULT_RECEIPT_PATH)
    after = _protected_hashes()

    assert before == after
    assert receipt.completed_run_count == 12
    assert receipt.completed_run_cascade_replay_passed_count == 12
    assert receipt.completed_run_application_replay_passed_count == 12
    assert receipt.completed_run_outcome_replay_passed_count == 12
    assert receipt.failed_run_logical_request_count == 18
    assert receipt.failed_run_completed_fingerprint_count == 18
    assert receipt.failed_run_checkpoint_entry_count == 18
    assert receipt.failed_run_transport_attempt_count == 18
    assert receipt.failed_run_resume_provider_calls_required == 0
    assert receipt.failed_run_cascade_replay_status == "passed"
    assert receipt.failed_run_application_replay_status == "passed"
    assert receipt.failed_run_outcome_replay_status == "passed"
    assert receipt.completed_fingerprints_replayed is False


def test_receipt_round_trip_no_overwrite_and_safe_summary(tmp_path: Path) -> None:
    path = tmp_path / "technical_amendment.json"
    path.write_bytes(DEFAULT_RECEIPT_PATH.read_bytes())
    stored, stored_digest = validate_technical_amendment_receipt(path)
    digest = sha256_file(path)

    assert stored_digest == digest
    with pytest.raises(HGReplayAmendmentError) as error:
        write_technical_amendment_receipt(path)
    assert error.value.code == "technical_amendment_receipt_exists"

    serialized = json.dumps(safe_summary(stored, digest), sort_keys=True).lower()
    for forbidden in (
        "system_prompt",
        "user_prompt",
        "api_key",
        "provider_metadata",
        "ground_truth_label",
        "source_independence_label",
        "model_response",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "field_name",
    ("batch_record_sha256", "old_validator_sha256", "new_validator_sha256"),
)
def test_tampered_receipt_binding_is_rejected(tmp_path: Path, field_name: str) -> None:
    receipt, _ = validate_technical_amendment_receipt(DEFAULT_RECEIPT_PATH)
    path = tmp_path / "tampered.json"
    path.write_text(
        receipt.model_copy(update={field_name: "0" * 64}).model_dump_json(indent=2),
        encoding="utf-8",
    )
    with pytest.raises(HGReplayAmendmentError) as error:
        validate_technical_amendment_receipt(path)
    assert error.value.code == "technical_amendment_receipt_hash_mismatch"


def test_resume_preflight_requires_receipt_and_validator_hash(monkeypatch) -> None:
    ready = hg_preflight(allow_existing_output=True)
    stored, digest = validate_technical_amendment_receipt(DEFAULT_RECEIPT_PATH)
    assert ready["status"] == "ready_for_real_pilot"
    assert ready["technical_amendment_receipt_sha256"] == digest
    assert ready["outcome_replay_validator_sha256"] == stored.new_validator_sha256
    assert ready["outcome_replay_contract_version"] == stored.outcome_replay_contract_version
    assert ready["provider_constructed"] is False
    assert ready["network"] == "disabled"

    import evicon.provenance_cascade_hg_replay_amendment as amendment

    def reject(_path):
        raise HGReplayAmendmentError("technical_amendment_receipt_hash_mismatch")

    monkeypatch.setattr(amendment, "validate_technical_amendment_receipt", reject)
    blocked = hg_preflight(allow_existing_output=True)
    assert blocked["status"] == "blocked"
    assert "technical_amendment_receipt_hash_mismatch" in blocked["blocking_reasons"]


class _ProviderMustNotRun:
    calls = 0

    def complete(self, request):
        self.calls += 1
        raise AssertionError("completed checkpoint fingerprints must not call the Provider")


def test_failed_run_checkpoint_reconstructs_with_zero_provider_calls(tmp_path: Path) -> None:
    batch = HGPilotBatchRecord.model_validate_json(BATCH_PATH.read_text(encoding="utf-8"))
    runner = HGPilotRunner()
    spec = next(item for item in runner.config.runs if item.run_id == FAILED_RUN_ID)
    root = tmp_path / "isolated-resume"
    shutil.copytree(FAILED_RUN_DIR, root / FAILED_RUN_ID)
    provider = _ProviderMustNotRun()

    record = runner.run_one(
        spec,
        provider=provider,
        root=root,
        model_name=batch.model_name,
        resume=True,
    )

    assert provider.calls == 0
    assert record.replay is not None
    assert record.replay.status.value == "passed"
    assert record.replay.cascade_replay.status.value == "passed"
    assert record.replay.application_replay.status.value == "passed"
