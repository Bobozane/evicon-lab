"""Offline tests for the explicitly confirmed real WVS baseline pilot boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon import wvs7_real_pilot as real_cli
from evicon import wvs7_real_preflight as preflight_cli
from evicon import wvs7_re_evaluate as reevaluate_cli
from evicon.evaluation import OfflineEvaluator
from evicon.llm_contract import LLMProviderError, LLMRequest, LLMResponse, ProviderErrorCode
from evicon.openai_provider import OpenAICompatibleProvider, ProviderConfig, TransportResponse
from evicon.request_ledger import LedgeredProvider, RequestLedger, RequestLedgerEntry, RequestLedgerStatus, request_fingerprint_facts
from evicon.wvs7_baseline_pilot import (
    PreparedWVS7BaselinePilot,
    WVS7BaselineFakeLLMProvider,
    WVS7BaselinePilotConfig,
    _build_batch_manifest,
    prepare_wvs7_baseline_pilot,
)
from evicon.wvs7_real_pilot import WVS7RealPilotRunner, build_wvs7_evaluation_manifest
from evicon.wvs7_real_preflight import (
    REAL_REQUEST_CAP,
    REAL_COMPLETION_RESERVATION_CAP,
    connection_check,
    preflight_wvs7_real_pilot,
    _validate_fixed_real_scope,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/studies/wvs7_english_baseline_pilot.toml"


class _LocalProvider:
    def __init__(self, outcomes: list[LLMResponse | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome.model_copy(update={"request_id": request.request_id, "model_name": request.model_name})


class _ScriptedTransport:
    def __init__(self, outcomes: list[TransportResponse | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, headers: dict[str, str], payload: dict[str, object], timeout_seconds: float) -> TransportResponse:
        self.calls.append({"url": url, "payload": payload, "timeout_seconds": timeout_seconds})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FailOnceProvider:
    def __init__(self, *, fail_at: int) -> None:
        self.fail_at = fail_at
        self.calls = 0
        self.delegate = WVS7BaselineFakeLLMProvider()

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        if self.calls == self.fail_at:
            raise LLMProviderError(ProviderErrorCode.TIMEOUT, "synthetic timeout")
        return self.delegate.complete(request)


def _response() -> LLMResponse:
    return LLMResponse(
        request_id="placeholder", model_name="placeholder", content='{"message":"ok","evidence_ids_used":[]}',
        finish_reason="stop", prompt_tokens=1, completion_tokens=1, total_tokens=2,
        latency_ms=0.0, provider_metadata={"provider_name": "test"},
    )


def _request(*, content: str = "local-content", max_tokens: int = 8, agent_id: str = "agent-a") -> LLMRequest:
    return LLMRequest(
        request_id="local-request", model_name="local-model", system_prompt="Return JSON.", user_prompt=content,
        temperature=0.0, max_tokens=max_tokens, seed=7,
        metadata={"template_version": "agent_turn.v1", "agent_id": agent_id, "round_id": 0, "protocol": "independent"},
    )


def _prepared(tmp_path: Path) -> PreparedWVS7BaselinePilot:
    source = prepare_wvs7_baseline_pilot(CONFIG)
    raw = source.config.model_dump(mode="python")
    raw["run_output_dir"] = str(tmp_path / "results")
    raw["batch_output_dir"] = str(tmp_path / "results/batches")
    config = WVS7BaselinePilotConfig.model_validate(raw)
    return PreparedWVS7BaselinePilot(
        config=config, scenario=source.scenario, probe_set=source.probe_set,
        frozen_manifest=source.frozen_manifest,
        batch_manifest=_build_batch_manifest(config, source.scenario, source.probe_set),
    )


def _provider_from_transport(outcomes: list[TransportResponse | Exception]) -> tuple[OpenAICompatibleProvider, _ScriptedTransport]:
    transport = _ScriptedTransport(outcomes)
    provider = OpenAICompatibleProvider(
        ProviderConfig(base_url="https://local.invalid/v1", model_name="local-model", allow_network=True, max_retries=1),
        transport=transport,
        environment={"EVICON_LLM_API_KEY": "test-secret-not-logged"},
    )
    return provider, transport


def _success_transport_response() -> TransportResponse:
    return TransportResponse(200, json.dumps({
        "model": "local-model",
        "choices": [{"message": {"content": '{"message":"ok","evidence_ids_used":[]}'}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }))


@pytest.mark.parametrize(
    ("outcomes", "expected_calls", "error"),
    [
        ([_success_transport_response()], 1, None),
        ([TransportResponse(429, "{}"), _success_transport_response()], 2, None),
        ([TransportResponse(500, "{}"), _success_transport_response()], 2, None),
        ([TransportResponse(401, "{}")], 1, ProviderErrorCode.AUTHENTICATION_FAILED),
        ([TimeoutError(), _success_transport_response()], 2, None),
        ([ConnectionError(), _success_transport_response()], 2, None),
    ],
)
def test_allowed_transport_retries_are_finite_and_4xx_is_not_retried(
    tmp_path: Path, outcomes: list[TransportResponse | Exception], expected_calls: int, error: ProviderErrorCode | None,
) -> None:
    provider, transport = _provider_from_transport(outcomes)
    ledgered = LedgeredProvider(provider, ledger=RequestLedger(tmp_path / "ledger.jsonl"), request_cap=4, completion_reservation_cap=64)
    if error is None:
        assert ledgered.complete(_request()).finish_reason == "stop"
    else:
        with pytest.raises(LLMProviderError) as caught:
            ledgered.complete(_request())
        assert caught.value.code is error
    assert len(transport.calls) == expected_calls


def test_ledger_redacts_content_enforces_caps_and_detects_fingerprint_reuse(tmp_path: Path) -> None:
    provider = _LocalProvider([_response(), _response()])
    ledger = RequestLedger(tmp_path / "ledger.jsonl")
    bounded = LedgeredProvider(provider, ledger=ledger, request_cap=1, completion_reservation_cap=8)
    request = _request(content="do-not-persist-this-local-content", max_tokens=8)
    bounded.complete(request)
    summary = ledger.summary(request_cap=1, completion_reservation_cap=8)
    assert summary.completion_reservation_cap == 8
    assert summary.completion_reserved_token_count == 8
    assert summary.actual_prompt_token_count == 1
    assert summary.actual_completion_token_count == 1
    assert summary.actual_total_token_count == 2
    with pytest.raises(LLMProviderError, match="completed_request_fingerprint_exists"):
        bounded.complete(request)
    assert len(provider.calls) == 1
    with pytest.raises(LLMProviderError, match="request_fingerprint_mismatch"):
        bounded.complete(_request(content="different-content", max_tokens=8))
    text = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    for forbidden in ("do-not-persist-this-local-content", "system_prompt", "user_prompt", "test-secret", "message"):
        assert forbidden not in text

    capped = LedgeredProvider(_LocalProvider([_response(), _response()]), ledger=RequestLedger(tmp_path / "capped.jsonl"), request_cap=2, completion_reservation_cap=8)
    capped.complete(_request(max_tokens=8))
    with pytest.raises(LLMProviderError, match="completion_reservation_cap_reached"):
        capped.complete(_request(content="coordinate-change", max_tokens=1, agent_id="agent-b"))

    request_limited_provider = _LocalProvider([_response(), _response()])
    request_limited = LedgeredProvider(
        request_limited_provider, ledger=RequestLedger(tmp_path / "request-capped.jsonl"), request_cap=1, completion_reservation_cap=64,
    )
    request_limited.complete(_request())
    with pytest.raises(LLMProviderError, match="request_cap_reached"):
        request_limited.complete(_request(content="new-logical-request", agent_id="agent-b"))


def test_failed_fingerprint_retries_only_after_explicit_resume(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "ledger.jsonl")
    provider = _LocalProvider([LLMProviderError(ProviderErrorCode.TIMEOUT, "timeout"), _response()])
    request = _request()
    with pytest.raises(LLMProviderError) as first:
        LedgeredProvider(provider, ledger=ledger, request_cap=4, completion_reservation_cap=64).complete(request)
    assert first.value.code is ProviderErrorCode.TIMEOUT
    with pytest.raises(LLMProviderError, match="failed_request_requires_resume"):
        LedgeredProvider(provider, ledger=ledger, request_cap=4, completion_reservation_cap=64).complete(request)
    assert LedgeredProvider(provider, ledger=ledger, request_cap=4, completion_reservation_cap=64, resume=True).complete(request).finish_reason == "stop"
    assert len(provider.calls) == 2


def test_failed_fingerprint_resume_does_not_consume_a_second_logical_slot(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "ledger.jsonl")
    provider = _LocalProvider([LLMProviderError(ProviderErrorCode.TIMEOUT, "timeout"), _response()])
    request = _request(max_tokens=8)
    with pytest.raises(LLMProviderError) as first:
        LedgeredProvider(
            provider,
            ledger=ledger,
            request_cap=1,
            completion_reservation_cap=8,
        ).complete(request)
    assert first.value.code is ProviderErrorCode.TIMEOUT

    resumed = LedgeredProvider(
        provider,
        ledger=ledger,
        request_cap=1,
        completion_reservation_cap=8,
        resume=True,
    )
    assert resumed.complete(request).finish_reason == "stop"
    summary = ledger.summary(request_cap=1, completion_reservation_cap=8)
    assert summary.unique_logical_request_count == 1
    assert summary.transport_attempt_count == 2
    assert summary.completion_reserved_token_count == 8

    with pytest.raises(LLMProviderError, match="request_cap_reached"):
        resumed.complete(_request(content="new-coordinate", max_tokens=8, agent_id="agent-b"))


def test_incomplete_started_receipt_requires_explicit_resume(tmp_path: Path) -> None:
    request = _request()
    facts = request_fingerprint_facts(request)
    ledger = RequestLedger(tmp_path / "ledger.jsonl")
    ledger.append(RequestLedgerEntry(
        request_id=request.request_id, fingerprint=str(facts["fingerprint"]), request_key=str(facts["request_key"]),
        model_name=request.model_name, template_version=str(facts["template_version"]),
        public_content_sha256=str(facts["public_content_sha256"]), condition=str(facts["condition"]),
        phase=str(facts["phase"]), agent_id="agent-a", round_id=0, status=RequestLedgerStatus.STARTED,
        attempt_count=1, reserved_tokens=request.max_tokens,
    ))
    provider = _LocalProvider([_response()])
    with pytest.raises(LLMProviderError, match="incomplete_request_requires_resume"):
        LedgeredProvider(provider, ledger=ledger, request_cap=4, completion_reservation_cap=64).complete(request)
    assert LedgeredProvider(provider, ledger=ledger, request_cap=4, completion_reservation_cap=64, resume=True).complete(request).finish_reason == "stop"


def test_preflight_is_default_offline_and_reports_only_configuration_presence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVICON_LLM_MODEL", raising=False)
    monkeypatch.delenv("EVICON_LLM_BASE_URL", raising=False)
    prepared, report = preflight_wvs7_real_pilot(CONFIG, run_id="preflight-test-no-network")
    assert prepared.probe_set.probe_set_id == "wvs7-english-core-23"
    assert report.status == "ready"
    assert report.request_cap == REAL_REQUEST_CAP
    assert report.completion_reservation_cap == REAL_COMPLETION_RESERVATION_CAP
    assert report.model_configured is False
    assert report.base_url_configured is False


def test_fixed_real_scope_rejects_budget_or_agent_drift(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    changed = prepared.config.model_copy(update={"agent_max_tokens": 257})
    drifted = PreparedWVS7BaselinePilot(
        config=changed, scenario=prepared.scenario, probe_set=prepared.probe_set,
        frozen_manifest=prepared.frozen_manifest, batch_manifest=prepared.batch_manifest,
    )
    with pytest.raises(ValueError, match="scope_or_budget_mismatch"):
        _validate_fixed_real_scope(drifted)


def test_missing_confirmations_are_blocked_before_provider_creation(capsys: pytest.CaptureFixture[str]) -> None:
    status = real_cli.main([
        "--config", str(CONFIG), "--run-id", "blocked-without-confirmation-test",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert status == 1
    assert payload["status"] == "blocked"
    assert payload["provider_call_count"] == 0
    assert "allow_network_required" in payload["blocking_reasons"]

    status = real_cli.main([
        "--config", str(CONFIG), "--run-id", "wrong-caps", "--allow-network", "--confirm-run",
        "--confirm-request-cap", "1", "--confirm-token-cap", "2",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert status == 1
    assert payload["provider_call_count"] == 0
    assert set(payload["blocking_reasons"]) == {"confirm_request_cap_must_equal_384", "confirm_completion_reservation_cap_must_equal_51200"}


def test_connection_check_uses_one_injected_fake_request_and_creates_no_output(tmp_path: Path) -> None:
    provider = _LocalProvider([_response()])
    result = connection_check(provider, model_name="local-model")
    assert result["status"] == "completed"
    assert len(provider.calls) == 1
    assert provider.calls[0].max_tokens == 256
    assert not list(tmp_path.iterdir())


def test_connection_check_provider_error_is_safe_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class _ProviderFailure:
        def __init__(self, config: ProviderConfig) -> None:
            self.config = config

        def complete(self, request: LLMRequest) -> LLMResponse:
            raise LLMProviderError(ProviderErrorCode.HTTP_CLIENT_ERROR, "synthetic provider failure")

    monkeypatch.setattr(preflight_cli, "OpenAICompatibleProvider", _ProviderFailure)
    monkeypatch.setenv("EVICON_LLM_BASE_URL", "https://local.invalid/v1")
    monkeypatch.setenv("EVICON_LLM_MODEL", "local-model")
    monkeypatch.setenv("EVICON_LLM_API_KEY", "test-secret-not-logged")
    status = preflight_cli.main([
        "--config", str(CONFIG), "--run-id", "connection-check-provider-error",
        "--allow-network", "--connection-check-only",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert status == 2
    assert payload == {"error_code": "provider_http_client_error", "status": "provider_error"}


def test_real_pilot_provider_configs_disable_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[ProviderConfig] = []

    class _StopAfterProviderConstruction:
        def __init__(self, _prepared: object, **kwargs: object) -> None:
            for provider in (kwargs["agent_provider"], kwargs["probe_provider"]):
                assert isinstance(provider, OpenAICompatibleProvider)
                captured.append(provider.config)
            raise real_cli.WVS7RealPilotError("stop_after_provider_configuration")

    prepared, report = preflight_wvs7_real_pilot(CONFIG, run_id="provider-config-reasoning-test")
    monkeypatch.setattr(real_cli, "preflight_wvs7_real_pilot", lambda *args, **kwargs: (prepared, report))
    monkeypatch.setattr(real_cli, "WVS7RealPilotRunner", _StopAfterProviderConstruction)
    monkeypatch.setenv("EVICON_LLM_BASE_URL", "https://local.invalid/v1")
    monkeypatch.setenv("EVICON_LLM_MODEL", "local-model")
    monkeypatch.setenv("EVICON_LLM_API_KEY", "test-secret-not-logged")

    status = real_cli.main([
        "--config", str(CONFIG), "--run-id", "provider-config-reasoning-test", "--allow-network", "--confirm-run",
        "--confirm-request-cap", "384", "--confirm-token-cap", "51200",
    ])

    assert status == 1
    assert len(captured) == 2
    assert [config.reasoning_effort.value if config.reasoning_effort else None for config in captured] == ["none", "none"]


def test_fake_real_pilot_completes_replays_evaluates_and_resume_does_not_rerun_completed_conditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(tmp_path)
    provider = WVS7BaselineFakeLLMProvider()
    runner = WVS7RealPilotRunner(
        prepared, batch_run_id="complete-real-pilot", model_name="fake-real-model",
        agent_provider=provider, probe_provider=provider, max_retries=1,
    )
    summary = runner.run()
    assert summary.status == "completed"
    assert summary.completed_run_count == 4
    assert summary.request_ledger.provider_call_count == REAL_REQUEST_CAP
    assert summary.request_ledger.completion_reserved_token_count == REAL_COMPLETION_RESERVATION_CAP
    assert summary.request_ledger.actual_total_token_count is not None
    assert set(summary.replay_statuses.values()) == {"passed"}
    assert {"pairwise_diversity", "structural_diversity", "value_dimension_coverage", "social_influence_loss"}.issubset(summary.offline_metric_names)
    first_call_count = provider.request_count
    resumed = WVS7RealPilotRunner(
        prepared, batch_run_id="complete-real-pilot", model_name="fake-real-model",
        agent_provider=provider, probe_provider=provider, max_retries=1,
    ).run(resume=True)
    assert resumed.status == "completed"
    assert provider.request_count == first_call_count
    assert (runner.batch_directory / "offline_evaluation_report.json").is_file()
    evaluation_manifest = build_wvs7_evaluation_manifest(
        runner.manifest,
        evaluation_id="complete-real-pilot-initial-final",
        coverage_threshold=prepared.config.coverage_threshold,
    )
    evaluation = OfflineEvaluator(evaluation_manifest, manifest_directory=runner.batch_directory).evaluate()
    assert len(evaluation_manifest.runs) == 8
    assert [metric.metric_name for metric in evaluation.metrics].count("profile_drift") == 4
    assert len([pair for pair in evaluation.matched_pairs if pair.comparison == "initial_final"]) == 4

    monkeypatch.setattr(reevaluate_cli, "prepare_wvs7_baseline_pilot", lambda _path: prepared)
    receipt = reevaluate_cli.re_evaluate_wvs7_batch("local-config", batch_id="complete-real-pilot")
    assert receipt["status"] == "completed_with_warnings"
    assert receipt["metric_names"].count("profile_drift") == 4
    assert Path(str(receipt["report_path"])).is_file()


def test_failed_pre_probe_can_resume_without_overwriting_and_incomplete_group_skips_metrics(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    failed = _FailOnceProvider(fail_at=3)
    runner = WVS7RealPilotRunner(
        prepared, batch_run_id="resume-real-pilot", model_name="fake-real-model",
        agent_provider=failed, probe_provider=failed, max_retries=1,
    )
    first = runner.run()
    assert first.status == "failed"
    assert first.completed_run_count == 0
    assert not (runner.batch_directory / "offline_evaluation_report.json").exists()
    assert not Path(runner.manifest.runs[0].artifacts.run_directory).exists()
    assert (runner.batch_directory / "private_probe_recovery" / runner.manifest.runs[0].run_id / "pre_responses.json").is_file()

    resumed_provider = WVS7BaselineFakeLLMProvider()
    resumed = WVS7RealPilotRunner(
        prepared, batch_run_id="resume-real-pilot", model_name="fake-real-model",
        agent_provider=resumed_provider, probe_provider=resumed_provider, max_retries=1,
    ).run(resume=True)
    assert resumed.status == "completed"
    assert set(resumed.replay_statuses.values()) == {"passed"}
    assert not (runner.batch_directory / "private_probe_recovery").exists()
