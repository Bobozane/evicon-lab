"""Offline tests for the opt-in real Agent smoke command."""

from __future__ import annotations

import json
import socket

import pytest

from evicon import real_agent_smoke
from evicon.llm_contract import LLMRequest, LLMResponse
from evicon.openai_provider import ProviderConfig, ReasoningEffort


class LocalSmokeProvider:
    """A local Agent provider substitute that cannot access a transport."""

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content=json.dumps({"message": "A local smoke response.", "evidence_ids_used": []}),
            finish_reason="stop",
            prompt_tokens=3,
            completion_tokens=5,
            total_tokens=8,
            latency_ms=1.0,
            provider_metadata={"provider_name": "local_smoke"},
        )


def test_default_real_agent_smoke_is_network_disabled(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def reject_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("default real Agent smoke attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", reject_connection)

    assert real_agent_smoke.main([]) == 0
    assert capsys.readouterr().out.strip() == "network_disabled"


def test_opt_in_smoke_uses_one_local_provider_call_with_controlled_parameters(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_config = ProviderConfig(
        base_url="https://local.test/v1",
        model_name="agent-smoke-model",
        timeout_seconds=1.0,
        max_retries=3,
        allow_network=True,
        temperature=0.0,
        max_tokens=1,
        seed=7,
    )
    provider = LocalSmokeProvider()
    received_configs: list[ProviderConfig] = []

    def fake_from_env(cls: type[ProviderConfig], *, allow_network: bool | None = None) -> ProviderConfig:
        assert allow_network is True
        return source_config

    def fake_provider(config: ProviderConfig) -> LocalSmokeProvider:
        received_configs.append(config)
        return provider

    monkeypatch.setattr(ProviderConfig, "from_env", classmethod(fake_from_env))
    monkeypatch.setattr(real_agent_smoke, "OpenAICompatibleProvider", fake_provider)

    assert real_agent_smoke.main(["--allow-network"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert len(received_configs) == 1
    assert received_configs[0].max_retries == 0
    assert received_configs[0].max_tokens == 256
    assert received_configs[0].temperature == 0.2
    assert received_configs[0].reasoning_effort is ReasoningEffort.NONE
    assert len(provider.requests) == 1
    assert output["status"] == "completed"
    assert output["parser_valid"] is True
    assert output["evidence_ids_used"] == []
    assert "message" not in output


def test_real_agent_smoke_rejects_response_printing_flags() -> None:
    with pytest.raises(SystemExit):
        real_agent_smoke.main(["--print-response"])
