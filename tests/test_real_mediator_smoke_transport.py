"""Verify the real mediator smoke request parameters with an injected fake transport."""

from __future__ import annotations

import json

import pytest

from provider_cases import FakeTransport, response_body

from evicon import real_mediator_smoke
from evicon.openai_provider import OpenAICompatibleProvider, ProviderConfig, TransportResponse


def test_opt_in_smoke_uses_one_fake_transport_call_with_controlled_json_parameters(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_config = ProviderConfig(
        base_url="https://provider.example/v1",
        model_name="smoke-test-model",
        timeout_seconds=1.0,
        max_retries=3,
        allow_network=True,
        temperature=0.0,
        max_tokens=1,
        seed=23,
    )
    transport = FakeTransport([TransportResponse(200, response_body())])
    received_configs: list[ProviderConfig] = []

    def fake_from_env(cls: type[ProviderConfig], *, allow_network: bool | None = None) -> ProviderConfig:
        assert allow_network is True
        return source_config

    def fake_provider(config: ProviderConfig) -> OpenAICompatibleProvider:
        received_configs.append(config)
        return OpenAICompatibleProvider(
            config,
            transport=transport,
            environment={"EVICON_LLM_API_KEY": "test-secret-key"},
        )

    monkeypatch.setattr(ProviderConfig, "from_env", classmethod(fake_from_env))
    monkeypatch.setattr(real_mediator_smoke, "OpenAICompatibleProvider", fake_provider)

    assert real_mediator_smoke.main(["--allow-network"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert len(received_configs) == 1
    assert received_configs[0].max_retries == 0
    assert received_configs[0].max_tokens == 256
    assert received_configs[0].reasoning_effort.value == "none"
    assert len(transport.calls) == 1
    assert transport.calls[0]["payload"]["max_tokens"] == 256
    assert transport.calls[0]["payload"]["reasoning_effort"] == "none"
    assert output["status"] == "completed"
    assert output["parser_valid"] is True
