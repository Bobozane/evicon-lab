from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.cascade_agent_integration_smoke import _FakeProvider
from evicon.cascade_agent_response import CascadeAgentResponseError
from evicon.cascade_real_agent_runner import CascadeRealAgentRunError, CascadeRealAgentRunner
from evicon.cascade_protocol import CascadeScenarioLoader
from evicon.llm_contract import LLMResponse
from evicon.provenance_cascade_preregistration import CascadeCondition
from evicon.request_ledger import RequestLedger

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "configs/provenance_cascade/scenarios/false_majority.toml"


class InvalidSchemaProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content=json.dumps({"unexpected": True}),
            finish_reason="stop",
            prompt_tokens=3,
            completion_tokens=2,
            total_tokens=5,
            latency_ms=1.0,
        )


def _run(tmp_path: Path, provider, *, recovery_fingerprint: str | None = None):
    return CascadeRealAgentRunner().run_scenario(
        CascadeScenarioLoader.load(SCENARIO),
        20260901,
        CascadeCondition.NO_INTERVENTION,
        provider=provider,
        run_id="hc-agent-recovery",
        ledger_path=tmp_path / "request_ledger.jsonl",
        checkpoint_path=tmp_path / "checkpoint.json",
        request_cap=18,
        completion_reservation_cap=4608,
        resume=recovery_fingerprint is not None,
        recovery_fingerprints={recovery_fingerprint} if recovery_fingerprint else None,
        recovery_kind="parser_invalid_agent.v1" if recovery_fingerprint else None,
        max_recovery_attempts=1 if recovery_fingerprint else 0,
    )


def test_agent_completed_fingerprint_requires_explicit_single_recovery(tmp_path: Path) -> None:
    with pytest.raises(CascadeRealAgentRunError) as first:
        _run(tmp_path, InvalidSchemaProvider())
    assert first.value.code == CascadeAgentResponseError.INVALID_SCHEMA.value
    ledger = RequestLedger(tmp_path / "request_ledger.jsonl")
    fingerprint = ledger.entries()[-1].fingerprint

    recovered_provider = _FakeProvider()
    record = _run(tmp_path, recovered_provider, recovery_fingerprint=fingerprint)
    summary = ledger.summary(request_cap=18, completion_reservation_cap=4608)
    assert record.replay is not None and record.replay.status.value == "passed"
    assert recovered_provider.calls == 18
    assert summary.unique_logical_request_count == 18
    assert summary.transport_attempt_count == 19
    assert summary.parser_recovery_transport_attempt_count == 1


def test_second_agent_parser_invalid_recovery_is_rejected_without_transport(tmp_path: Path) -> None:
    with pytest.raises(CascadeRealAgentRunError):
        _run(tmp_path, InvalidSchemaProvider())
    ledger = RequestLedger(tmp_path / "request_ledger.jsonl")
    fingerprint = ledger.entries()[-1].fingerprint

    second_invalid = InvalidSchemaProvider()
    with pytest.raises(CascadeRealAgentRunError) as second:
        _run(tmp_path, second_invalid, recovery_fingerprint=fingerprint)
    assert second.value.code == CascadeAgentResponseError.INVALID_SCHEMA.value
    assert second_invalid.calls == 1

    forbidden_provider = _FakeProvider()
    with pytest.raises(CascadeRealAgentRunError) as third:
        _run(tmp_path, forbidden_provider, recovery_fingerprint=fingerprint)
    assert third.value.code == "parser_recovery_attempt_limit_reached"
    assert forbidden_provider.calls == 0
