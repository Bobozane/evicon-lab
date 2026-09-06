import json
from pathlib import Path

import pytest

from evicon.cascade_agent_integration_smoke import _FakeProvider, run_smoke
from evicon.cascade_intervention_calibration import CascadeInterventionCalibrationRunner
from evicon.cascade_real_agent_runner import CascadeRealAgentRunError, CascadeRealAgentRunner
from evicon.cascade_protocol import CascadeScenarioLoader
from evicon.cascade_agent_response import CascadeAgentResponseError
from evicon.llm_contract import LLMProviderError, LLMResponse, ProviderErrorCode
from evicon.openai_provider import OpenAICompatibleProvider, ProviderConfig, TransportResponse
from evicon.provenance_cascade_preregistration import CascadeCondition

ROOT = Path(__file__).resolve().parents[1]
HB = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hb.v1.toml"
SCENARIO = ROOT / "configs/provenance_cascade/scenarios/false_majority.toml"


class FailingProvider(_FakeProvider):
    def __init__(self, fail_at=None, content=None):
        super().__init__()
        self.fail_at = fail_at
        self.content_override = content

    def complete(self, request):
        if self.fail_at is not None and self.calls + 1 == self.fail_at:
            self.calls += 1
            raise LLMProviderError(ProviderErrorCode.TIMEOUT, "redacted")
        return super().complete(request)


def _scenario():
    return CascadeScenarioLoader.load(SCENARIO)


def _run(tmp_path, provider, *, resume=False, checkpoint=True, run_id="hc-test"):
    return CascadeRealAgentRunner().run_scenario(
        _scenario(), 20260901, CascadeCondition.NO_INTERVENTION, provider=provider,
        run_id=run_id, ledger_path=tmp_path / "request_ledger.jsonl",
        checkpoint_path=(tmp_path / "checkpoint.json") if checkpoint else None,
        request_cap=18, completion_reservation_cap=4608, resume=resume,
    )


def test_hc_run_uses_fake_provider_and_public_only(tmp_path):
    provider = _FakeProvider()
    record = _run(tmp_path, provider)
    assert record.logical_request_count == 18
    assert provider.calls == 18
    assert record.replay is not None and record.replay.status.value == "passed"
    raw = (tmp_path / "request_ledger.jsonl").read_text()
    assert "system_prompt" not in raw and "user_prompt" not in raw
    assert "api_key" not in raw and "provider_metadata" not in raw
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert len(checkpoint["entries"]) == 18
    assert "prompt" not in (tmp_path / "checkpoint.json").read_text().lower()


def test_resume_reuses_successful_public_checkpoint_and_retries_failed_coordinate(tmp_path):
    with pytest.raises(CascadeRealAgentRunError) as first:
        _run(tmp_path, FailingProvider(fail_at=5))
    assert first.value.code == ProviderErrorCode.TIMEOUT.value
    second_provider = _FakeProvider()
    record = _run(tmp_path, second_provider, resume=True)
    assert record.replay is not None and record.replay.status.value == "passed"
    assert second_provider.calls == 14
    entries = (tmp_path / "request_ledger.jsonl").read_text().splitlines()
    assert len(entries) == 18 * 2 + 2


def test_completed_ledger_cannot_be_replayed_without_resume(tmp_path):
    _run(tmp_path, _FakeProvider())
    replacement = _FakeProvider()
    with pytest.raises(CascadeRealAgentRunError) as error:
        _run(tmp_path, replacement, run_id="hc-test", resume=False)
    assert error.value.code == "completed_request_fingerprint_exists"
    assert replacement.calls == 0


def test_completed_run_resume_rebuilds_from_checkpoint_without_provider_calls(tmp_path):
    _run(tmp_path, _FakeProvider())
    resumed_provider = _FakeProvider()
    resumed = _run(tmp_path, resumed_provider, resume=True)
    assert resumed_provider.calls == 0
    assert resumed.replay is not None and resumed.replay.status.value == "passed"
    assert resumed.logical_request_count == 18


@pytest.mark.parametrize("content,code", [
    ("{", CascadeAgentResponseError.MALFORMED_JSON.value),
    (json.dumps({"stance": "uncertain", "content_ids_used": ["not-visible"]}), CascadeAgentResponseError.UNAVAILABLE_CONTENT.value),
])
def test_parser_invalid_transport_response_is_ledgered_but_not_checkpointed(tmp_path, content, code):
    class InvalidResponseProvider:
        calls = 0

        def complete(self, request):
            self.calls += 1
            return LLMResponse(
                request_id=request.request_id, model_name=request.model_name, content=content,
                finish_reason="stop", prompt_tokens=1, completion_tokens=1, total_tokens=2, latency_ms=1.0,
            )

    provider = InvalidResponseProvider()
    with pytest.raises(CascadeRealAgentRunError) as error:
        _run(tmp_path, provider)
    assert error.value.code == code
    assert provider.calls == 1
    ledger = (tmp_path / "request_ledger.jsonl").read_text()
    assert '"status":"completed"' in ledger
    assert not (tmp_path / "checkpoint.json").exists()
    with pytest.raises(CascadeRealAgentRunError) as resumed:
        _run(tmp_path, _FakeProvider(), resume=True)
    assert resumed.value.code == "completed_request_fingerprint_exists"


def test_binding_change_rejects_checkpoint_reuse(tmp_path):
    _run(tmp_path, _FakeProvider())
    with pytest.raises(CascadeRealAgentRunError) as error:
        CascadeRealAgentRunner().run_scenario(
            _scenario(), 20260902, CascadeCondition.NO_INTERVENTION, provider=_FakeProvider(),
            run_id="hc-test", ledger_path=tmp_path / "other-ledger.jsonl",
            checkpoint_path=tmp_path / "checkpoint.json", request_cap=18,
            completion_reservation_cap=4608, resume=True,
        )
    assert error.value.code == "checkpoint_binding_mismatch"


@pytest.mark.parametrize("provider_code", [
    ProviderErrorCode.TIMEOUT,
    ProviderErrorCode.RATE_LIMITED,
    ProviderErrorCode.HTTP_SERVER_ERROR,
    ProviderErrorCode.AUTHENTICATION_FAILED,
])
def test_provider_failures_are_ledgered_and_redacted(tmp_path, provider_code):
    class ErrorProvider:
        def complete(self, request):
            raise LLMProviderError(provider_code, "system_prompt api_key raw-response")

    with pytest.raises(CascadeRealAgentRunError) as error:
        _run(tmp_path, ErrorProvider(), checkpoint=False)
    assert error.value.code == provider_code.value
    ledger = (tmp_path / "request_ledger.jsonl").read_text()
    assert provider_code.value in ledger
    assert "system_prompt" not in ledger and "api_key" not in ledger
    assert "raw-response" not in ledger


@pytest.mark.parametrize("first_result", [429, 500, "timeout"])
def test_injected_provider_retries_transport_once_without_expanding_logical_requests(tmp_path, first_result):
    response_body = json.dumps({
        "model": "fake-cascade-agent",
        "choices": [{
            "message": {"content": json.dumps({
                "stance": "uncertain",
                "content_ids_used": [],
                "evidence_ids_used": [],
                "share_content_id": None,
            })},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })

    class SequenceTransport:
        def __init__(self):
            initial = TimeoutError() if first_result == "timeout" else TransportResponse(first_result, "{}")
            self.responses = [initial, *[TransportResponse(200, response_body) for _ in range(18)]]
            self.calls = 0

        def post(self, url, headers, payload, timeout_seconds):
            self.calls += 1
            result = self.responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

    transport = SequenceTransport()
    provider = OpenAICompatibleProvider(
        ProviderConfig(
            base_url="https://offline.invalid/v1",
            model_name="fake-cascade-agent",
            timeout_seconds=0.1,
            max_retries=1,
            allow_network=True,
            temperature=0.2,
            max_tokens=256,
            seed=20260901,
        ),
        transport=transport,
        environment={"EVICON_LLM_API_KEY": "unit-test-placeholder"},
    )
    record = _run(tmp_path, provider, checkpoint=False)
    summary = __import__("evicon.request_ledger", fromlist=["RequestLedger"]).RequestLedger(
        tmp_path / "request_ledger.jsonl"
    ).summary(request_cap=18, completion_reservation_cap=4608)
    assert transport.calls == 19
    assert record.logical_request_count == summary.unique_logical_request_count == 18
    assert summary.transport_attempt_count == 18
    assert summary.completed_count == 18
    assert "provider_metadata" not in (tmp_path / "request_ledger.jsonl").read_text()


def test_calibration_fixture_has_real_next_round_directive(tmp_path):
    calibration = CascadeInterventionCalibrationRunner.from_file()
    scenario = next(item.scenario for item in calibration.scenarios if item.scenario.scenario_type.value == "false_majority")
    record = CascadeRealAgentRunner(policy_config=calibration.policy).run_scenario(
        scenario, 20260901, CascadeCondition.PROVENANCE_AWARE_CONTROLLER, provider=_FakeProvider(),
        run_id="hc-calibration", ledger_path=tmp_path / "ledger.jsonl", request_cap=18,
        completion_reservation_cap=4608, policy_config=calibration.policy,
    )
    assert record.directive_applied_count >= 1
    assert record.replay is not None and record.replay.status.value == "passed"
    scheduled = [item for item in record.application_ledger.schedules if item.status.value == "applied"]
    assert scheduled and all(item.effective_round_id == item.created_round_id + 1 for item in scheduled)
    directive_contexts = [context for context in record.round_contexts if context.directives]
    assert directive_contexts and all(
        context.snapshot.round_id == context.directives[0].effective_round_id
        for context in directive_contexts
    )


def test_48_run_smoke_is_safe_and_replayable():
    summary = run_smoke()
    assert summary["run_count"] == 48
    assert summary["matched_group_count"] == 12
    assert summary["logical_request_count"] == 864
    assert summary["provider_call_count"] == 864
    assert summary["replay_status"] == "passed"
    assert summary["network"] == "disabled"
    assert summary["private_truth_exposed"] is False
    assert summary["not_paper_result"] is True
    assert summary["calibration_directive_applied_count"] >= 1
