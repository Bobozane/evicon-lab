from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.provenance_cascade_hg2 import sha256_file
from evicon.provenance_cascade_hg231_failure_diagnostic import (
    FINISH_REASON_UNAVAILABLE,
    HG231FailureAssessment,
    HG231FailureDiagnosticError,
    classify_failure,
    diagnose,
    write_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/provenance-cascade-hg231-adoption-identifiability-v1"
INPUTS = tuple(sorted(RESULT_ROOT.rglob("*")))


def _input_hashes() -> dict[Path, str]:
    return {path: sha256_file(path) for path in INPUTS if path.is_file()}


def test_actual_failure_is_truncation_consistent_and_reconstructable() -> None:
    before = _input_hashes()
    receipt = diagnose()
    assert receipt.assessment is HG231FailureAssessment.TRUNCATION_MORE_CONSISTENT
    assert receipt.completion_tokens == receipt.configured_max_tokens == 1024
    assert receipt.completion_token_limit_reached is True
    assert receipt.finish_reason == FINISH_REASON_UNAVAILABLE
    assert receipt.finish_reason_available is False
    assert receipt.transport_completed is True
    assert receipt.http_status_available is False
    assert receipt.completed_run_count == receipt.completed_run_replay_passed_count == 5
    assert receipt.unique_logical_request_count == 106
    assert receipt.transport_attempt_count == 107
    assert receipt.append_only_counts_reconstructed is True
    assert receipt.failed_fingerprint_checkpointed is False
    assert receipt.current_batch_must_not_resume is True
    assert receipt.narrow_recovery_amendment_drafted is False
    assert before == _input_hashes()


def test_classifier_requires_non_limit_stop_for_sporadic_assessment() -> None:
    assessment, _ = classify_failure(
        parser_error="malformed_json", finish_reason="stop",
        completion_tokens=40, configured_max_tokens=1024,
    )
    assert assessment is HG231FailureAssessment.SPORADIC_MALFORMED_MORE_CONSISTENT
    unknown, reasons = classify_failure(
        parser_error="malformed_json", finish_reason=None,
        completion_tokens=40, configured_max_tokens=1024,
    )
    assert unknown is HG231FailureAssessment.INDETERMINATE
    assert "finish_reason_unavailable" in reasons


def test_receipt_is_content_free_and_no_overwrite(tmp_path: Path) -> None:
    receipt = diagnose()
    output = tmp_path / "diagnostic.json"
    digest = write_receipt(receipt, output)
    assert digest == sha256_file(output)
    assert write_receipt(receipt, output) == digest
    raw = output.read_text(encoding="utf-8").lower()
    for forbidden in (
        "system_prompt", "user_prompt", "raw_response", "model_response",
        "api_key", "authorization", "provider_metadata", "ground_truth_label",
        "source_independence_label",
    ):
        assert forbidden not in raw
    changed = receipt.model_copy(update={"fingerprint": "f" * 64})
    with pytest.raises(HG231FailureDiagnosticError, match="diagnostic_receipt_mismatch"):
        write_receipt(changed, output)


def test_safe_json_contains_no_response_or_private_fields() -> None:
    payload = json.loads(diagnose().model_dump_json())
    assert payload["network_used_for_diagnostic"] is False
    assert payload["provider_constructed"] is False
    assert payload["response_repair_performed"] is False
    serialized = json.dumps(payload).lower()
    assert "prompt" not in serialized or "prompt_tokens" in serialized
    for forbidden in ("raw_response", "api_key", "provider_metadata", "ground_truth_label"):
        assert forbidden not in serialized
