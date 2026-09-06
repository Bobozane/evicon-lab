"""Additional explicit provider prerequisite and unknown-usage coverage."""

from __future__ import annotations

import pytest

from provider_cases import FakeTransport, request, response_body

from evicon.llm_contract import LLMProviderError, LLMResponse, ProviderErrorCode
from evicon.openai_provider import OpenAICompatibleProvider, ProviderConfig, TransportResponse


@pytest.mark.parametrize(
    "config,code",
    [
        (ProviderConfig(model_name="model", allow_network=True), ProviderErrorCode.MISSING_BASE_URL),
        (ProviderConfig(base_url="https://provider.example/v1", allow_network=True), ProviderErrorCode.MISSING_MODEL),
    ],
)
def test_missing_nonsecret_provider_settings_fail_without_transport(
    config: ProviderConfig,
    code: ProviderErrorCode,
) -> None:
    _, _, rendered = request()
    transport = FakeTransport([TransportResponse(200, response_body())])

    with pytest.raises(LLMProviderError) as error:
        OpenAICompatibleProvider(
            config,
            transport=transport,
            environment={"EVICON_LLM_API_KEY": "secret"},
        ).complete(rendered)

    assert error.value.code is code
    assert transport.calls == []


def test_unknown_usage_round_trips_without_fabricated_token_counts() -> None:
    response = LLMResponse(
        request_id="unknown-usage",
        model_name="model",
        content="{}",
        finish_reason="stop",
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        latency_ms=0.0,
        provider_metadata={"provider_name": "fake"},
    )

    assert LLMResponse.model_validate_json(response.model_dump_json()) == response
