"""Offline tests for the opt-in real mediator pilot command."""

from __future__ import annotations

import json
import socket

import pytest

from evicon.fake_provider_adapter import FakeLLMProviderAdapter
from evicon.openai_provider import ProviderConfig, ReasoningEffort
from evicon import real_mediator_smoke


def test_default_smoke_is_network_disabled(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def reject_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("default real mediator smoke attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", reject_connection)

    assert real_mediator_smoke.main([]) == 0
    assert capsys.readouterr().out.strip() == "network_disabled"


def test_opt_in_smoke_uses_one_local_fake_provider_call_and_safe_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = ProviderConfig(
        base_url="https://local.test/v1",
        model_name="local-smoke-model",
        timeout_seconds=5.0,
        max_retries=3,
        allow_network=True,
        temperature=0.0,
        max_tokens=16,
        seed=5,
    )
    provider = FakeLLMProviderAdapter()
    received_configs: list[ProviderConfig] = []

    def fake_from_env(cls: type[ProviderConfig], *, allow_network: bool | None = None) -> ProviderConfig:
        assert allow_network is True
        return config

    def fake_provider(provider_config: ProviderConfig) -> FakeLLMProviderAdapter:
        received_configs.append(provider_config)
        return provider

    monkeypatch.setattr(ProviderConfig, "from_env", classmethod(fake_from_env))
    monkeypatch.setattr(real_mediator_smoke, "OpenAICompatibleProvider", fake_provider)

    assert real_mediator_smoke.main(["--allow-network"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert len(received_configs) == 1
    assert received_configs[0].max_retries == 0
    assert received_configs[0].max_tokens == 256
    assert received_configs[0].reasoning_effort is ReasoningEffort.NONE
    assert len(provider._response_modes) == 0
    assert output == {
        "action": "solicit_dissent",
        "error_code": None,
        "finish_reason": "stop",
        "latency_ms": 1.0,
        "model": "local-smoke-model",
        "parser_valid": True,
        "request_id": output["request_id"],
        "status": "completed",
        "template_version": "solicit_dissent.v1",
        "token_usage": output["token_usage"],
    }
    assert output["token_usage"]["total_tokens"] == (
        output["token_usage"]["prompt_tokens"] + output["token_usage"]["completion_tokens"]
    )


def test_smoke_rejects_response_printing_flags() -> None:
    with pytest.raises(SystemExit):
        real_mediator_smoke.main(["--print-response"])
