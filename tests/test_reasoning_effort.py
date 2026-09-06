"""Offline coverage for optional GPT-5.6 reasoning effort request configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from provider_cases import FakeTransport, request, response_body

from evicon.mediator_runtime import MediatorRequestSettings
from evicon.openai_provider import OpenAICompatibleProvider, ProviderConfig, ReasoningEffort, TransportResponse


_ENV = {"EVICON_LLM_API_KEY": "test-secret-key"}


def _config(*, reasoning_effort: ReasoningEffort | str | None = None) -> ProviderConfig:
    return ProviderConfig(
        base_url="https://provider.example/v1",
        model_name="provider-test-model",
        timeout_seconds=0.1,
        max_retries=0,
        allow_network=True,
        temperature=0.0,
        max_tokens=16,
        seed=3,
        reasoning_effort=reasoning_effort,
    )


def test_none_reasoning_effort_is_omitted_from_compatible_payload() -> None:
    _, _, rendered = request()
    transport = FakeTransport([TransportResponse(200, response_body())])

    OpenAICompatibleProvider(_config(), transport=transport, environment=_ENV).complete(rendered)

    assert "reasoning_effort" not in transport.calls[0]["payload"]


@pytest.mark.parametrize("effort", list(ReasoningEffort))
def test_valid_reasoning_effort_is_sent_in_compatible_payload(effort: ReasoningEffort) -> None:
    _, _, rendered = request()
    transport = FakeTransport([TransportResponse(200, response_body())])

    OpenAICompatibleProvider(
        _config(reasoning_effort=effort), transport=transport, environment=_ENV
    ).complete(rendered)

    assert transport.calls[0]["payload"]["reasoning_effort"] == effort.value


@pytest.mark.parametrize("effort", ["minimal", "auto", "", "ultra", 1])
def test_invalid_reasoning_effort_is_rejected(effort: object) -> None:
    with pytest.raises(ValidationError, match="reasoning_effort"):
        _config(reasoning_effort=effort)  # type: ignore[arg-type]


def test_mediator_runtime_default_budget_is_raised_without_changing_explicit_tests() -> None:
    assert MediatorRequestSettings().max_tokens == 256
    assert MediatorRequestSettings(max_tokens=32).max_tokens == 32
