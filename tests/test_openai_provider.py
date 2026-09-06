"""OpenAI-compatible Provider tests using only an injected fake transport."""

from __future__ import annotations

import json

import pytest

from provider_cases import FakeTransport, request, response_body

from evicon.llm_contract import LLMProviderError, ProviderErrorCode
from evicon.mediator_response import parse_mediator_response
from evicon.openai_provider import OpenAICompatibleProvider, ProviderConfig, TransportResponse


_ENV = {"EVICON_LLM_API_KEY": "test-secret-key"}


def _config(*, retries: int = 1, allow_network: bool = True) -> ProviderConfig:
    return ProviderConfig(
        base_url="https://provider.example/v1",
        model_name="provider-test-model",
        timeout_seconds=0.1,
        max_retries=retries,
        allow_network=allow_network,
        temperature=0.0,
        max_tokens=16,
        seed=3,
    )


def _provider(transport: FakeTransport, *, retries: int = 1, allow_network: bool = True, environment: dict[str, str] | None = None) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        _config(retries=retries, allow_network=allow_network),
        transport=transport,
        environment=_ENV if environment is None else environment,
    )


def test_network_disabled_never_calls_the_transport() -> None:
    _, _, rendered = request()
    transport = FakeTransport([TransportResponse(200, response_body())])

    with pytest.raises(LLMProviderError) as error:
        _provider(transport, allow_network=False).complete(rendered)

    assert error.value.code is ProviderErrorCode.NETWORK_DISABLED
    assert transport.calls == []


def test_missing_key_is_structured_before_transport_access() -> None:
    _, _, rendered = request()
    transport = FakeTransport([TransportResponse(200, response_body())])

    with pytest.raises(LLMProviderError) as error:
        _provider(transport, environment={}).complete(rendered)

    assert error.value.code is ProviderErrorCode.MISSING_API_KEY
    assert transport.calls == []


def test_normal_fake_transport_response_parses_and_preserves_public_parser_boundary() -> None:
    plan, context, rendered = request()
    transport = FakeTransport([TransportResponse(200, response_body())])

    response = _provider(transport).complete(rendered)
    parsed = parse_mediator_response(response.content, plan, context)

    assert parsed.valid is True
    assert response.total_tokens == 7
    assert response.provider_metadata["retry_count"] == 0
    assert transport.calls[0]["url"] == "https://provider.example/v1/chat/completions"


@pytest.mark.parametrize("status", [429, 500])
def test_rate_limit_and_server_errors_retry_by_configuration(status: int) -> None:
    _, _, rendered = request()
    transport = FakeTransport(
        [TransportResponse(status, "{}"), TransportResponse(200, response_body())]
    )

    response = _provider(transport, retries=1).complete(rendered)

    assert len(transport.calls) == 2
    assert response.provider_metadata["retry_count"] == 1


def test_401_does_not_retry() -> None:
    _, _, rendered = request()
    transport = FakeTransport([TransportResponse(401, "{}")])

    with pytest.raises(LLMProviderError) as error:
        _provider(transport, retries=3).complete(rendered)

    assert error.value.code is ProviderErrorCode.AUTHENTICATION_FAILED
    assert len(transport.calls) == 1


def test_timeout_retries_by_configuration() -> None:
    _, _, rendered = request()
    transport = FakeTransport([TimeoutError(), TransportResponse(200, response_body())])

    response = _provider(transport, retries=1).complete(rendered)

    assert len(transport.calls) == 2
    assert response.provider_metadata["retry_count"] == 1


@pytest.mark.parametrize(
    "body,code",
    [
        ("not-json", ProviderErrorCode.MALFORMED_RESPONSE),
        (json.dumps({"choices": []}), ProviderErrorCode.INVALID_RESPONSE),
        (json.dumps({"choices": [{"message": {}, "finish_reason": "stop"}]}), ProviderErrorCode.INVALID_RESPONSE),
    ],
)
def test_malformed_or_incomplete_response_is_rejected(body: str, code: ProviderErrorCode) -> None:
    _, _, rendered = request()
    transport = FakeTransport([TransportResponse(200, body)])

    with pytest.raises(LLMProviderError) as error:
        _provider(transport).complete(rendered)

    assert error.value.code is code


def test_missing_usage_is_unknown_and_inconsistent_usage_is_rejected() -> None:
    _, _, rendered = request()
    no_usage = json.dumps(
        {
            "model": "provider-test-model",
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        }
    )
    response = _provider(FakeTransport([TransportResponse(200, no_usage)])).complete(rendered)
    assert (response.prompt_tokens, response.completion_tokens, response.total_tokens) == (None, None, None)

    inconsistent = response_body(usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 99})
    with pytest.raises(LLMProviderError) as error:
        _provider(FakeTransport([TransportResponse(200, inconsistent)])).complete(rendered)
    assert error.value.code is ProviderErrorCode.USAGE_MISMATCH
