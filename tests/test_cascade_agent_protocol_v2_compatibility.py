from __future__ import annotations

import json
import socket

import pytest

from evicon.cascade_agent_protocol_v2 import render_cascade_agent_turn_v2
from evicon.cascade_agent_protocol_v2_compatibility import (
    COMPATIBILITY_MAX_TOKENS,
    COMPATIBILITY_REASONING_EFFORT,
    COMPATIBILITY_SEED,
    COMPATIBILITY_TEMPERATURE,
    V2_SCHEMA_NAME,
    build_minimal_public_context,
    check_provider,
    run_compatibility_check,
)
from evicon.llm_contract import ProviderErrorCode
from evicon.openai_provider import (
    OpenAICompatibleProvider,
    ProviderConfig,
    ResponseFormatMode,
    TransportResponse,
)


class FakeTransport:
    def __init__(self, outcomes: list[TransportResponse]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def post(self, url, headers, payload, timeout_seconds):
        self.calls.append({"url": url, "headers": dict(headers), "payload": payload})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def strict_response(*, content: str | None = None, finish_reason: str = "stop", completion_tokens: int = 5) -> str:
    content = content or json.dumps(
        {
            "stance": "uncertain",
            "content_ids_used": ["compatibility-content-01"],
            "evidence_ids_used": ["compatibility-evidence-01"],
            "share_content_id": "compatibility-content-01",
        },
    )
    return json.dumps(
        {
            "model": "compat-model",
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 4, "completion_tokens": completion_tokens, "total_tokens": 4 + completion_tokens},
        },
    )


def provider(transport: FakeTransport, *, retries: int = 0) -> OpenAICompatibleProvider:
    config = ProviderConfig(
        base_url="https://provider.invalid/v1",
        model_name="compat-model",
        allow_network=True,
        timeout_seconds=5.0,
        max_retries=retries,
        max_tokens=COMPATIBILITY_MAX_TOKENS,
        temperature=COMPATIBILITY_TEMPERATURE,
        seed=COMPATIBILITY_SEED,
        response_format=ResponseFormatMode.JSON_SCHEMA,
        response_schema_name=V2_SCHEMA_NAME,
    )
    return OpenAICompatibleProvider(config, transport=transport, environment={"EVICON_LLM_API_KEY": "test-secret"})


def test_strict_schema_is_injected_and_only_four_fields_are_declared():
    context = build_minimal_public_context(model_name="compat-model")
    request = render_cascade_agent_turn_v2(context)
    transport = FakeTransport([TransportResponse(200, strict_response())])
    payload = provider(transport)._payload(request)
    schema = payload["response_format"]["json_schema"]
    assert payload["response_format"]["type"] == "json_schema"
    assert schema["name"] == V2_SCHEMA_NAME
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False
    assert set(schema["schema"]["properties"]) == {"stance", "content_ids_used", "evidence_ids_used", "share_content_id"}
    assert payload["max_tokens"] == 256
    assert payload["temperature"] == 0.2
    assert payload["seed"] == 20260911
    assert COMPATIBILITY_MAX_TOKENS == 256
    assert COMPATIBILITY_TEMPERATURE == 0.2
    assert COMPATIBILITY_SEED == 20260911
    assert COMPATIBILITY_REASONING_EFFORT is None
    assert "reasoning_effort" not in payload


def test_fake_200_is_parsed_without_writing_files(tmp_path):
    context = build_minimal_public_context(model_name="compat-model")
    transport = FakeTransport([TransportResponse(200, strict_response())])
    result = check_provider(provider(transport), context=context)
    assert result.status == "completed"
    assert result.parser_valid is True
    assert result.request_attempt_count == 1
    assert result.total_tokens == 9
    assert not list(tmp_path.iterdir())
    assert "test-secret" not in result.model_dump_json()


@pytest.mark.parametrize(
    "body,code",
    [
        ("{}", ProviderErrorCode.HTTP_CLIENT_ERROR.value),
        ("not-json", "malformed_provider_response"),
    ],
)
def test_unsupported_format_and_malformed_response_do_not_retry(body, code):
    context = build_minimal_public_context(model_name="compat-model")
    transport = FakeTransport([TransportResponse(400 if body == "{}" else 200, body), TransportResponse(200, strict_response())])
    result = check_provider(provider(transport, retries=1), context=context)
    assert result.status == "provider_error"
    assert result.provider_error_code == code
    assert result.request_attempt_count == 1
    assert len(transport.calls) == 1
    assert "test-secret" not in result.model_dump_json()



def test_h_d2_aligned_256_token_stop_response_is_parser_compatible():
    context = build_minimal_public_context(model_name="compat-model")
    transport = FakeTransport([TransportResponse(200, strict_response(finish_reason="stop", completion_tokens=12))])
    result = check_provider(provider(transport), context=context)
    assert result.status == "completed"
    assert result.finish_reason == "stop"
    assert result.parser_valid is True
    assert result.completion_tokens == 12
    request_payload = transport.calls[0]["payload"]
    assert request_payload["max_tokens"] == 256
    assert request_payload["temperature"] == 0.2
    assert request_payload["seed"] == 20260911


def test_length_response_is_reported_invalid_without_schema_failure():
    context = build_minimal_public_context(model_name="compat-model")
    transport = FakeTransport([
        TransportResponse(200, strict_response(content='{"stance":"uncertain"', finish_reason="length", completion_tokens=256)),
    ])
    result = check_provider(provider(transport), context=context)
    assert result.status == "invalid_response"
    assert result.finish_reason == "length"
    assert result.parser_valid is False
    assert result.provider_error_code is None
    assert result.http_status_class == "2xx"
    assert result.request_attempt_count == 1


def test_default_check_is_network_disabled_and_never_calls_transport():
    class FailTransport:
        def post(self, *args, **kwargs):
            raise AssertionError("network must remain disabled")

    result = run_compatibility_check(allow_network=False, transport=FailTransport())
    assert result.status == "network_disabled"
    assert result.request_attempt_count == 0
    assert result.network == "disabled"
    assert result.safety["results_written"] is False


@pytest.mark.parametrize(
    "outcome,category,status_class",
    [
        (TimeoutError("sensitive-timeout-marker"), "timeout", None),
        (socket.gaierror("sensitive-dns-marker"), "dns_or_url_failure", None),
        (ConnectionError("sensitive-connection-marker"), "connection_failure", None),
        (TransportResponse(400, '{"error":{"message":"invalid request"}}'), "http_client_error", "4xx"),
        (TransportResponse(400, '{"error":{"message":"response_format json_schema unsupported"}}'), "response_format_unsupported", "4xx"),
        (TransportResponse(401, '{"error":{"message":"secret"}}'), "authentication_failed", "4xx"),
        (TransportResponse(429, '{"error":{"message":"secret"}}'), "rate_limited", "4xx"),
        (TransportResponse(500, '{"error":{"message":"secret"}}'), "http_server_error", "5xx"),
        (TransportResponse(200, "not-json"), "malformed_provider_response", None),
        (TransportResponse(200, "{}"), "invalid_provider_response", None),
    ],
)
def test_transport_categories_are_stable_and_redacted(outcome, category, status_class):
    context = build_minimal_public_context(model_name="compat-model")
    transport = FakeTransport([outcome, TransportResponse(200, strict_response())])
    result = check_provider(provider(transport, retries=0), context=context)
    assert result.status == "provider_error"
    assert result.provider_error_code == category
    assert result.transport_category == category
    expected_status_class = (
        "2xx"
        if status_class is None and isinstance(outcome, TransportResponse) and outcome.status_code == 200
        else status_class
    )
    assert result.http_status_class == expected_status_class
    assert result.request_attempt_count == 1
    assert result.attempt_count == 1
    assert len(transport.calls) == 1
    dumped = result.model_dump_json()
    assert "sensitive-" not in dumped
    assert "response_format json_schema unsupported" not in dumped


def test_success_reports_safe_status_class_and_attempt_alias():
    context = build_minimal_public_context(model_name="compat-model")
    transport = FakeTransport([TransportResponse(200, strict_response())])
    result = check_provider(provider(transport), context=context)
    assert result.status == "completed"
    assert result.http_status_class == "2xx"
    assert result.transport_category is None
    assert result.response_format_requested == "json_schema"
    assert result.attempt_count == result.request_attempt_count == 1
