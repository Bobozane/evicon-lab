"""Provider transport may receive the key, but outputs must never retain it."""

from __future__ import annotations

import pytest

from provider_cases import FakeTransport, request, response_body

from evicon.llm_contract import LLMProviderError, ProviderErrorCode
from evicon.openai_provider import OpenAICompatibleProvider, ProviderConfig, TransportResponse


def _provider(transport: FakeTransport, environment: dict[str, str]) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        ProviderConfig(
            base_url="https://provider.example/v1",
            model_name="provider-test-model",
            timeout_seconds=0.1,
            max_retries=0,
            allow_network=True,
            temperature=0.0,
            max_tokens=16,
            seed=3,
        ),
        transport=transport,
        environment=environment,
    )


def test_key_is_header_only_and_absent_from_response_metadata_and_errors() -> None:
    secret = "redaction-test-secret"
    _, _, rendered = request()
    transport = FakeTransport([TransportResponse(200, response_body())])

    response = _provider(transport, {"EVICON_LLM_API_KEY": secret}).complete(rendered)

    assert transport.calls[0]["headers"]["Authorization"] == f"Bearer {secret}"
    assert secret not in response.model_dump_json()
    assert rendered.system_prompt not in response.provider_metadata.values()
    assert rendered.user_prompt not in response.provider_metadata.values()

    with pytest.raises(LLMProviderError) as error:
        _provider(FakeTransport([TransportResponse(401, "{}")]), {"EVICON_LLM_API_KEY": secret}).complete(rendered)
    assert secret not in str(error.value)
    assert error.value.code is ProviderErrorCode.AUTHENTICATION_FAILED


def test_provider_payload_does_not_mutate_the_intervention_plan() -> None:
    plan, _, rendered = request()
    before = plan.model_dump(mode="json")

    _provider(FakeTransport([TransportResponse(200, response_body())]), {"EVICON_LLM_API_KEY": "secret"}).complete(rendered)

    assert plan.model_dump(mode="json") == before
