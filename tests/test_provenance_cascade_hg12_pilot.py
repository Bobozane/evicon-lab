from __future__ import annotations

from pathlib import Path

import pytest

from evicon.cascade_real_agent_runner import CascadeRealAgentRunError
from evicon.llm_contract import LLMResponse
from evicon.provenance_cascade_hg1 import HG1FakeProvider
from evicon.provenance_cascade_hg12_pilot import HG12PilotRunner, execute_real, run_fake_smoke


def test_real_gate_rejects_before_environment_or_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg12_pilot.ProviderConfig.from_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("provider must not be constructed")),
    )
    result = execute_real()
    assert result.status == "blocked"
    assert result.error_code == "allow_network_required"
    assert result.provider_constructed_count == 0
    assert result.network == "disabled"


def test_fake_smoke_completes_full_matched_plan() -> None:
    result = run_fake_smoke()
    assert result["status"] == "fake_smoke_passed"
    assert result["run_count"] == result["replay_passed_count"] == 48
    assert result["matched_group_count"] == 12
    assert result["logical_request_count"] == result["provider_call_count"] == 864
    assert result["transport_attempt_count"] == 864
    assert result["completion_reservation_cap"] == 884736
    assert result["network"] == "disabled"
    assert result["results_written"] is False


def test_no_overwrite_and_completed_resume_does_not_call_provider(tmp_path: Path) -> None:
    runner = HG12PilotRunner()
    root = tmp_path / "pilot"
    records, receipt = runner.run_all(
        provider_factory=lambda _spec: HG1FakeProvider(), root=root,
        model_name="hg12-fake", write_receipt=False, network="disabled",
    )
    assert len(records) == 48 and receipt.logical_request_count == 864
    with pytest.raises(CascadeRealAgentRunError, match="output_root_exists"):
        runner.run_all(
            provider_factory=lambda _spec: HG1FakeProvider(), root=root,
            model_name="hg12-fake", write_receipt=False, network="disabled",
        )
    calls = 0
    def fail_if_called(_spec):
        nonlocal calls
        calls += 1
        raise AssertionError("completed request must not replay")
    resumed, resumed_receipt = runner.run_all(
        provider_factory=fail_if_called, root=root, model_name="hg12-fake",
        resume=True, write_receipt=False, network="disabled",
    )
    assert calls == 0
    assert len(resumed) == 48 and resumed_receipt.logical_request_count == 864


class MalformedProvider:
    def complete(self, request):
        return LLMResponse(
            request_id=request.request_id, model_name=request.model_name,
            content='{"stance":"uncertain"', finish_reason="length",
            prompt_tokens=20, completion_tokens=1024, total_tokens=1044,
            latency_ms=1.0,
        )


def test_parser_invalid_stops_without_recovery(tmp_path: Path) -> None:
    runner = HG12PilotRunner()
    spec = runner.runs[0]
    with pytest.raises(CascadeRealAgentRunError, match="malformed_json"):
        runner.run_one(spec, provider=MalformedProvider(), root=tmp_path, model_name="hg12-fake")
    ledger = (tmp_path / spec.run_id / "request_ledger.jsonl").read_text(encoding="utf-8")
    assert "parser_recovery" not in ledger
    assert "system_prompt" not in ledger and "user_prompt" not in ledger
