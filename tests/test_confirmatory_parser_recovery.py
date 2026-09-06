from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.llm_contract import LLMRequest, LLMResponse
from evicon.request_ledger import (
    LedgeredProvider,
    RequestLedger,
    RequestLedgerError,
    request_fingerprint_facts,
)


class CountingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            request_id=request.request_id, model_name=request.model_name,
            content='{"choice":"1"}', finish_reason='stop',
            prompt_tokens=2, completion_tokens=1, total_tokens=3, latency_ms=1.0,
        )


def probe_request() -> LLMRequest:
    payload = {
        "measurement_context": {"protocol": "evidence_only"},
        "probe_item": {"probe_id": "probe-Q158"},
    }
    return LLMRequest(
        request_id="probe-request", model_name="test-model",
        system_prompt="return json", user_prompt=json.dumps(payload),
        temperature=0.0, max_tokens=128, seed=7,
        metadata={
            "template_version": "contextual_value_probe.v2",
            "agent_id": "access-harm", "round_id": 1, "protocol": "evidence_only",
        },
    )


def agent_request() -> LLMRequest:
    return LLMRequest(
        request_id="agent-request", model_name="test-model",
        system_prompt="return json", user_prompt="{}", temperature=0.2, max_tokens=256, seed=7,
        metadata={"template_version": "agent_turn.v1", "agent_id": "access-harm", "round_id": 1, "protocol": "evidence_only"},
    )


def test_completed_fingerprint_remains_rejected(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "ledger.jsonl")
    provider = CountingProvider()
    request = probe_request()
    LedgeredProvider(provider, ledger=ledger, request_cap=1, completion_reservation_cap=128).complete(request)
    with pytest.raises(RequestLedgerError, match="completed_request_fingerprint_exists"):
        LedgeredProvider(provider, ledger=ledger, request_cap=1, completion_reservation_cap=128, resume=True).complete(request)
    assert provider.calls == 1


def test_unallowlisted_completed_fingerprint_is_rejected_on_resume(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "ledger.jsonl")
    provider = CountingProvider()
    request = probe_request()
    LedgeredProvider(provider, ledger=ledger, request_cap=1, completion_reservation_cap=128).complete(request)
    with pytest.raises(RequestLedgerError, match="completed_request_fingerprint_exists"):
        LedgeredProvider(
            provider, ledger=ledger, request_cap=1, completion_reservation_cap=128, resume=True,
            recovery_fingerprints={"0" * 64}, recovery_kind="parser_invalid_probe.v1",
        ).complete(request)
    assert provider.calls == 1


def test_single_parser_invalid_recovery_is_append_only_and_bounded(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "ledger.jsonl")
    provider = CountingProvider()
    request = probe_request()
    first = LedgeredProvider(provider, ledger=ledger, request_cap=1, completion_reservation_cap=128)
    first.complete(request)
    fingerprint = str(request_fingerprint_facts(request)["fingerprint"])
    recovery = LedgeredProvider(
        provider, ledger=ledger, request_cap=1, completion_reservation_cap=128, resume=True,
        recovery_fingerprints={fingerprint}, recovery_kind="parser_invalid_probe.v1",
    )
    recovery.complete(request)
    with pytest.raises(RequestLedgerError, match="parser_recovery_attempt_limit_reached"):
        recovery.complete(request)
    entries = ledger.entries()
    assert len(entries) == 4
    assert [entry.attempt_count for entry in entries] == [1, 1, 2, 2]
    summary = ledger.summary(request_cap=1, completion_reservation_cap=128)
    assert summary.unique_logical_request_count == 1
    assert summary.transport_attempt_count == 2
    assert summary.parser_recovery_transport_attempt_count == 1
    assert summary.actual_total_token_count == 6
    assert provider.calls == 2


def test_recovery_allowlist_cannot_be_used_for_agent_requests(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "ledger.jsonl")
    provider = CountingProvider()
    request = agent_request()
    facts = request_fingerprint_facts(request)
    LedgeredProvider(provider, ledger=ledger, request_cap=1, completion_reservation_cap=256).complete(request)
    with pytest.raises(RequestLedgerError, match="completed_request_fingerprint_exists"):
        # A pre-existing completed Agent request is never eligible for semantic probe recovery.
        LedgeredProvider(
            provider, ledger=ledger, request_cap=1, completion_reservation_cap=256, resume=True,
            recovery_fingerprints={str(facts["fingerprint"])}, recovery_kind="parser_invalid_probe.v1",
        ).complete(request)


def test_recovery_rejects_changed_first_fingerprint(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "ledger.jsonl")
    provider = CountingProvider()
    request = probe_request()
    LedgeredProvider(provider, ledger=ledger, request_cap=2, completion_reservation_cap=256).complete(request)
    fingerprint = str(request_fingerprint_facts(request)["fingerprint"])
    changed = request.model_copy(
        update={
            "user_prompt": json.dumps({
                "measurement_context": {"protocol": "evidence_only"},
                "probe_item": {"probe_id": "probe-Q159"},
            })
        }
    )
    recovery = LedgeredProvider(
        provider, ledger=ledger, request_cap=2, completion_reservation_cap=256, resume=True,
        recovery_fingerprints={fingerprint}, recovery_kind="parser_invalid_probe.v1",
    )
    with pytest.raises(RequestLedgerError, match="parser_recovery_fingerprint_changed"):
        recovery.complete(changed)
    assert provider.calls == 1
    recovery.complete(request)
    assert provider.calls == 2
