"""Offline tests for the opt-in contextual value-probe smoke command."""

from __future__ import annotations

import json
import socket

import pytest

from evicon import real_probe_smoke
from evicon.llm_contract import LLMRequest, LLMResponse
from evicon.openai_provider import ProviderConfig, ReasoningEffort


class LocalSmokeProbeProvider:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content='{"choice":"4"}',
            finish_reason="stop",
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
            latency_ms=1.0,
            provider_metadata={"provider_name": "local_probe_smoke"},
        )


def test_default_real_probe_smoke_is_network_disabled_and_creates_no_files(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)

    def reject_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("default real probe smoke attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", reject_connection)
    assert real_probe_smoke.main([]) == 0
    assert capsys.readouterr().out.strip() == "network_disabled"
    assert list(tmp_path.iterdir()) == []


def test_opt_in_smoke_uses_one_local_call_with_fixed_safe_parameters(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_config = ProviderConfig(
        base_url="https://local.test/v1",
        model_name="probe-smoke-model",
        timeout_seconds=1.0,
        max_retries=2,
        allow_network=True,
        temperature=1.0,
        max_tokens=1,
        seed=13,
    )
    provider = LocalSmokeProbeProvider()
    configs: list[ProviderConfig] = []

    def fake_from_env(cls: type[ProviderConfig], *, allow_network: bool | None = None) -> ProviderConfig:
        assert allow_network is True
        return source_config

    def fake_provider(config: ProviderConfig) -> LocalSmokeProbeProvider:
        configs.append(config)
        return provider

    monkeypatch.setattr(ProviderConfig, "from_env", classmethod(fake_from_env))
    monkeypatch.setattr(real_probe_smoke, "OpenAICompatibleProvider", fake_provider)

    assert real_probe_smoke.main(["--allow-network"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert len(configs) == 1
    assert configs[0].max_retries == 0
    assert configs[0].max_tokens == 128
    assert configs[0].temperature == 0.0
    assert configs[0].reasoning_effort is ReasoningEffort.NONE
    assert len(provider.requests) == 1
    assert output["status"] == "completed"
    assert output["parser_valid"] is True
    for forbidden in ("choice", "probe", "prompt", "message", "api_key"):
        assert forbidden not in output


def test_real_probe_smoke_rejects_response_printing_flags() -> None:
    with pytest.raises(SystemExit):
        real_probe_smoke.main(["--print-response"])
