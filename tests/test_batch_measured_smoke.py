"""Offline tests for the batch measured smoke entry point."""

from __future__ import annotations

import json
import socket

import pytest

from evicon import batch_measured_smoke
from evicon.llm_contract import LLMRequest, LLMResponse
from evicon.openai_provider import ProviderConfig, ReasoningEffort


class LocalBatchSmokeProvider:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        content = (
            json.dumps({"message": "A public batch smoke turn.", "evidence_ids_used": []})
            if self.kind == "agent"
            else json.dumps({"choice": "4"})
        )
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content=content,
            finish_reason="stop",
            prompt_tokens=3,
            completion_tokens=2,
            total_tokens=5,
            latency_ms=1.0,
            provider_metadata={"provider_name": f"local_{self.kind}_batch_smoke"},
        )


def test_default_smoke_is_network_disabled_and_plan_only_creates_no_files(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)

    def reject_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("batch smoke attempted a network connection")

    def forbid_env(*args: object, **kwargs: object) -> object:
        raise AssertionError("plan-only smoke attempted to read provider settings")

    monkeypatch.setattr(socket, "create_connection", reject_connection)
    monkeypatch.setattr(ProviderConfig, "from_env", classmethod(forbid_env))
    assert batch_measured_smoke.main([]) == 0
    assert capsys.readouterr().out.strip() == "network_disabled"
    assert batch_measured_smoke.main(["--plan-only", "--batch-id", "local-plan"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["batch_id"] == "local-plan"
    assert output["condition_count"] == 4
    assert output["planned_run_count"] == 4
    assert output["expected_request_upper_bound"] == 32
    assert list(tmp_path.iterdir()) == []


def test_opt_in_branch_uses_fake_providers_and_fixed_request_settings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    source_config = ProviderConfig(
        base_url="https://local.test/v1",
        model_name="batch-smoke-model",
        timeout_seconds=1.0,
        max_retries=2,
        allow_network=True,
        temperature=1.0,
        max_tokens=1,
        seed=0,
    )
    configs: list[ProviderConfig] = []
    providers: list[LocalBatchSmokeProvider] = []

    def fake_from_env(cls: type[ProviderConfig], *, allow_network: bool | None = None) -> ProviderConfig:
        assert allow_network is True
        return source_config

    def fake_provider(config: ProviderConfig) -> LocalBatchSmokeProvider:
        configs.append(config)
        provider = LocalBatchSmokeProvider("agent" if config.temperature == 0.2 else "probe")
        providers.append(provider)
        return provider

    monkeypatch.setattr(ProviderConfig, "from_env", classmethod(fake_from_env))
    monkeypatch.setattr(batch_measured_smoke, "OpenAICompatibleProvider", fake_provider)
    assert batch_measured_smoke.main([
        "--allow-network",
        "--batch-id",
        "local-batch",
        "--output-dir",
        str(tmp_path / "results"),
        "--batch-output-dir",
        str(tmp_path / "results" / "batches"),
    ]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(lines) == 2
    assert lines[0]["expected_request_upper_bound"] == 32
    assert lines[1]["batch_status"] == "completed"
    assert lines[1]["actual_request_count"] == 32
    assert sum(len(provider.requests) for provider in providers) == 32
    agent_config = next(config for config in configs if config.temperature == 0.2)
    probe_config = next(config for config in configs if config.temperature == 0.0)
    assert (agent_config.max_retries, agent_config.max_tokens, agent_config.reasoning_effort) == (0, 256, ReasoningEffort.NONE)
    assert (probe_config.max_retries, probe_config.max_tokens, probe_config.reasoning_effort) == (0, 128, ReasoningEffort.NONE)
