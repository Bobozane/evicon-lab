"""Offline tests for the opt-in real Agent protocol smoke command."""

from __future__ import annotations

import json
import socket

import pytest

from agent_cases import LocalAgentProvider, agent_response_content

from evicon import real_protocol_smoke
from evicon.openai_provider import ProviderConfig, ReasoningEffort


def test_default_smoke_is_network_disabled_and_creates_no_output_directory(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def reject_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("default real protocol smoke attempted a network connection")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(socket, "create_connection", reject_connection)

    assert real_protocol_smoke.main([]) == 0
    assert capsys.readouterr().out.strip() == "network_disabled"
    assert not (tmp_path / "results").exists()


def test_opt_in_smoke_uses_one_controlled_config_and_four_local_provider_calls(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_config = ProviderConfig(
        base_url="https://local.test/v1",
        model_name="protocol-smoke-model",
        timeout_seconds=1.0,
        max_retries=3,
        allow_network=True,
        temperature=0.0,
        max_tokens=1,
        seed=43,
    )
    provider = LocalAgentProvider(content=agent_response_content())
    received_configs: list[ProviderConfig] = []

    def fake_from_env(cls: type[ProviderConfig], *, allow_network: bool | None = None) -> ProviderConfig:
        assert allow_network is True
        return source_config

    def fake_provider(config: ProviderConfig) -> LocalAgentProvider:
        received_configs.append(config)
        return provider

    monkeypatch.setattr(ProviderConfig, "from_env", classmethod(fake_from_env))
    monkeypatch.setattr(real_protocol_smoke, "OpenAICompatibleProvider", fake_provider)
    output_dir = tmp_path / "output"

    assert real_protocol_smoke.main(
        ["--allow-network", "--run-id", "protocol-smoke", "--output-dir", str(output_dir)]
    ) == 0
    output = json.loads(capsys.readouterr().out)

    assert len(received_configs) == 1
    assert received_configs[0].max_retries == 0
    assert received_configs[0].max_tokens == 256
    assert received_configs[0].reasoning_effort is ReasoningEffort.NONE
    assert len(provider.requests) == 4
    assert output["protocol"] == "social_only"
    assert output["turn_count"] == 4
    assert output["replay_status"] == "passed"
    assert "message" not in output
    assert (output_dir / "protocol-smoke" / "events.jsonl").exists()

    with pytest.raises(FileExistsError):
        real_protocol_smoke.main(
            ["--allow-network", "--run-id", "protocol-smoke", "--output-dir", str(output_dir)]
        )
