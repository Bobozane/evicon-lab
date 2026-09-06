from __future__ import annotations

from pathlib import Path

import pytest

from evicon.llm_contract import LLMProviderError, LLMRequest, LLMResponse
from evicon.provenance_cascade_hg2 import sha256_file
from evicon.provenance_cascade_hg232_timeout_resume_v4 import (
    COMPLETED_RUN_RECORD_SHA256,
    DEFAULT_RECEIPT,
    FAILED_FINGERPRINT,
    FAILED_RUN_ID,
    ROOT,
    HG232TimeoutResumeReceipt,
    TimeoutResumeFingerprintGuard,
    build_receipt,
    execute_resume,
    preflight,
)


class RecordingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content="{}",
            finish_reason="stop",
            latency_ms=1.0,
        )


def request() -> LLMRequest:
    return LLMRequest(
        request_id="safe-request",
        model_name="safe-model",
        system_prompt="public system",
        user_prompt="public input",
        temperature=0.2,
        max_tokens=2048,
        seed=20261031,
    )


def test_receipt_locks_historical_timeout_and_immutable_artifacts() -> None:
    # The one-shot builder intentionally rejects once the batch has resumed.
    # Validate the immutable historical receipt instead of reconstructing the
    # obsolete failure state from the now-completed live directory.
    receipt = HG232TimeoutResumeReceipt.model_validate_json(Path(DEFAULT_RECEIPT).read_text(encoding="utf-8"))
    assert receipt.failed_fingerprint == FAILED_FINGERPRINT
    assert receipt.failure_code == "timeout"
    assert receipt.completed_run_count == 7
    assert receipt.failed_run_checkpoint_count == 8
    assert receipt.unique_logical_request_count == 135
    assert receipt.total_logical_request_cap == 288
    assert receipt.completion_reservation_cap == 589824
    assert receipt.parser_recovery_enabled is False
    assert receipt.completed_run_record_sha256 == COMPLETED_RUN_RECORD_SHA256
    for run_id, expected in COMPLETED_RUN_RECORD_SHA256.items():
        assert sha256_file(Path(ROOT) / run_id / "run_record.json") == expected
    assert Path(DEFAULT_RECEIPT).exists()


def test_historical_offline_preflight_refuses_reuse_after_completion(monkeypatch) -> None:
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg232_timeout_resume_v4.ProviderConfig.from_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("provider constructed")),
    )
    report = preflight()
    assert report["status"] == "blocked"
    assert report["blocking_reasons"] == ["timeout_resume_initial_batch_hash_mismatch"]
    assert report["provider_constructed"] is False
    assert report["api_key_read"] is False
    assert report["results_modified"] is False


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({}, "allow_network_required"),
        ({"allow_network": True}, "confirm_resume_required"),
        ({"allow_network": True, "confirm_resume": True, "confirm_request_cap": 287}, "confirm_request_cap_must_equal_288"),
        (
            {"allow_network": True, "confirm_resume": True, "confirm_request_cap": 288, "confirm_completion_reservation_cap": 1},
            "confirm_completion_reservation_cap_must_equal_589824",
        ),
    ],
)
def test_resume_gate_rejects_before_provider_construction(monkeypatch, kwargs, code: str) -> None:
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg232_timeout_resume_v4.ProviderConfig.from_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("provider constructed")),
    )
    result = execute_resume(**kwargs)
    assert result.status == "blocked"
    assert result.error_code == code
    assert result.provider_constructed_count == 0


def test_guard_allows_only_failed_fingerprint_first(monkeypatch) -> None:
    provider = RecordingProvider()
    values = iter((FAILED_FINGERPRINT, "new-logical-fingerprint"))
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg232_timeout_resume_v4.request_fingerprint_facts",
        lambda _request: {"fingerprint": next(values)},
    )
    guard = TimeoutResumeFingerprintGuard(provider, preexisting_fingerprints={"completed-fingerprint"})
    guard.complete(request())
    guard.complete(request())
    assert provider.calls == 2
    assert guard.timeout_retry_consumed is True


def test_guard_rejects_wrong_first_or_completed_fingerprint(monkeypatch) -> None:
    provider = RecordingProvider()
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg232_timeout_resume_v4.request_fingerprint_facts",
        lambda _request: {"fingerprint": "completed-fingerprint"},
    )
    guard = TimeoutResumeFingerprintGuard(provider, preexisting_fingerprints={"completed-fingerprint"})
    with pytest.raises(LLMProviderError, match="timeout_resume_first_fingerprint_mismatch"):
        guard.complete(request())
    assert provider.calls == 0


def test_guard_rejects_failed_fingerprint_second_time(monkeypatch) -> None:
    provider = RecordingProvider()
    values = iter((FAILED_FINGERPRINT, FAILED_FINGERPRINT))
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg232_timeout_resume_v4.request_fingerprint_facts",
        lambda _request: {"fingerprint": next(values)},
    )
    guard = TimeoutResumeFingerprintGuard(provider, preexisting_fingerprints=set())
    guard.complete(request())
    with pytest.raises(LLMProviderError, match="timeout_resume_preexisting_fingerprint_rejected"):
        guard.complete(request())
    assert provider.calls == 1


def test_public_contract_contains_no_sensitive_payload_fields() -> None:
    payload = Path(DEFAULT_RECEIPT).read_text(encoding="utf-8").lower()
    for forbidden in (
        "system_prompt",
        "user_prompt",
        "raw_response",
        "api_key",
        "authorization:",
        "provider_metadata",
        "ground_truth_label",
        "source_independence_label",
    ):
        assert forbidden not in payload
    assert FAILED_RUN_ID in payload
