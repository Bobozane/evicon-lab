from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from evicon.conformity_identification import sha256_file
from evicon.conformity_source_behavior_compatibility import (
    MAX_RETRIES,
    MAX_TOKENS,
    REASONING_EFFORT,
    RESPONSE_SCHEMA,
    SCHEMA_VERSION,
    SEED,
    TEMPERATURE,
    TIMEOUT_SECONDS,
    CompatibilityReceipt,
    execute_compatibility,
    load_receipt,
    run_compatibility_check,
)
from evicon.conformity_source_behavior_qualification import (
    DEFAULT_CONFIG,
    PROTOCOL_VERSION,
    SCHEMA_NAME,
    TEMPLATE_VERSION,
    BehaviorQualificationResponse,
    load_config,
)
from evicon.conformity_source_behavior_qualification_approval import (
    DEFAULT_APPROVAL,
    approval_sha256,
)
from evicon.openai_provider import TransportResponse


ENVIRONMENT = {
    "EVICON_LLM_BASE_URL": "https://provider.invalid/v1",
    "EVICON_LLM_MODEL": "source-behavior-compat-model",
    "EVICON_LLM_API_KEY": "test-only-secret",
}


class FakeTransport:
    def __init__(self, outcomes: list[TransportResponse | BaseException]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def post(self, url, headers, payload, timeout_seconds):
        self.calls.append({
            "url": url,
            "headers": dict(headers),
            "payload": payload,
            "timeout_seconds": timeout_seconds,
        })
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _valid_content() -> dict[str, object]:
    return {
        "adoption_decision": "withhold",
        "sharing_decision": "do_not_share",
        "content_ids_used": [],
        "share_content_id": None,
    }


def _provider_body(
    *,
    content: str | None = None,
    finish_reason: str = "stop",
    completion_tokens: int = 17,
) -> str:
    if content is None:
        content = json.dumps(_valid_content(), ensure_ascii=True, sort_keys=True)
    return json.dumps({
        "model": "source-behavior-compat-model",
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": 31,
            "completion_tokens": completion_tokens,
            "total_tokens": 31 + completion_tokens,
        },
    }, ensure_ascii=True, sort_keys=True)


def _run(transport: FakeTransport):
    return run_compatibility_check(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=transport,
    )


def test_default_path_reads_no_environment_constructs_nothing_and_writes_nothing(
    tmp_path: Path,
) -> None:
    class ExplodingEnvironment(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            raise AssertionError("environment must not be read")

        def __iter__(self):
            raise AssertionError("environment must not be read")

        def __len__(self) -> int:
            raise AssertionError("environment must not be read")

        def get(self, key: str, default=None):
            raise AssertionError("environment must not be read")

    class FailTransport:
        def post(self, *args, **kwargs):
            raise AssertionError("transport must not be called")

    result = execute_compatibility(
        allow_network=False,
        environment=ExplodingEnvironment(),
        transport=FailTransport(),
        receipt_path=tmp_path / "must-not-exist.json",
    )
    assert result.status == "network_disabled"
    assert result.transport_category == "network_disabled"
    assert result.attempt_count == 0
    assert result.safety.network_enabled is False
    assert result.safety.provider_constructed is False
    assert result.safety.api_key_read is False
    assert result.safety.results_written is False
    assert not list(tmp_path.iterdir())
    assert set(result.model_dump()) == {
        "status", "model", "finish_reason", "parser_valid", "http_status_class",
        "transport_category", "token_usage", "latency_ms", "schema",
        "attempt_count", "safety",
    }


def test_valid_2xx_uses_exact_v2_schema_parameters_and_one_transport_call() -> None:
    transport = FakeTransport([TransportResponse(200, _provider_body())])
    result = _run(transport)
    assert result.status == "completed"
    assert result.parser_valid is True
    assert result.finish_reason == "stop"
    assert result.http_status_class == "2xx"
    assert result.token_usage.model_dump() == {
        "prompt_tokens": 31, "completion_tokens": 17, "total_tokens": 48,
    }
    assert result.attempt_count == 1
    assert len(transport.calls) == 1

    call = transport.calls[0]
    payload = call["payload"]
    assert payload["max_tokens"] == MAX_TOKENS == 128
    assert payload["temperature"] == TEMPERATURE == 0.0
    assert payload["seed"] == SEED == 20261110
    assert payload["reasoning_effort"] == REASONING_EFFORT.value == "none"
    assert call["timeout_seconds"] == TIMEOUT_SECONDS == 5.0
    assert MAX_RETRIES == 0
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"] == {
        "name": SCHEMA_NAME,
        "strict": True,
        "schema": RESPONSE_SCHEMA,
    }
    assert RESPONSE_SCHEMA == BehaviorQualificationResponse.model_json_schema(mode="validation")
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False
    assert set(response_format["json_schema"]["schema"]["required"]) == {
        "adoption_decision", "sharing_decision", "content_ids_used", "share_content_id",
    }
    assert set(response_format["json_schema"]["schema"]["properties"]) == {
        "adoption_decision", "sharing_decision", "content_ids_used", "share_content_id",
    }

    messages = payload["messages"]
    public_request = messages[1]["content"].lower()
    assert "fictional" in public_request
    for forbidden in ("wvs", "pilot", "evaluator_private", "ground_truth_label"):
        assert forbidden not in public_request


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        json.dumps(_valid_content() | {"unknown_field": "forbidden"}),
        json.dumps(_valid_content() | {"content_ids_used": ["not-exposed"]}),
    ],
    ids=["malformed-inner-json", "unknown-field", "unexposed-content-id"],
)
def test_semantically_invalid_outputs_stop_after_one_call(content: str) -> None:
    transport = FakeTransport([
        TransportResponse(200, _provider_body(content=content)),
        TransportResponse(200, _provider_body()),
    ])
    result = _run(transport)
    assert result.status == "invalid_response"
    assert result.parser_valid is False
    assert result.finish_reason == "stop"
    assert result.transport_category == "invalid_provider_response"
    assert result.attempt_count == 1
    assert len(transport.calls) == 1


def test_finish_reason_length_is_invalid_without_retry() -> None:
    transport = FakeTransport([
        TransportResponse(200, _provider_body(finish_reason="length")),
        TransportResponse(200, _provider_body()),
    ])
    result = _run(transport)
    assert result.status == "invalid_response"
    assert result.parser_valid is True
    assert result.finish_reason == "length"
    assert result.transport_category == "invalid_provider_response"
    assert result.attempt_count == 1
    assert len(transport.calls) == 1


def test_observed_at_limit_truncation_writes_no_receipt(tmp_path: Path) -> None:
    receipt_path = tmp_path / "must-not-exist.json"
    transport = FakeTransport([
        TransportResponse(200, _provider_body(
            content='{"adoption_decision":',
            finish_reason="length",
            completion_tokens=128,
        )),
        TransportResponse(200, _provider_body()),
    ])
    result = execute_compatibility(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=transport,
        receipt_path=receipt_path,
    )
    assert result.status == "invalid_response"
    assert result.finish_reason == "length"
    assert result.parser_valid is False
    assert result.token_usage.completion_tokens == 128
    assert result.attempt_count == 1
    assert len(transport.calls) == 1
    assert not receipt_path.exists()


@pytest.mark.parametrize(
    ("outcome", "category", "status_class"),
    [
        (TimeoutError("sensitive-timeout"), "timeout", None),
        (ConnectionError("sensitive-connection"), "connection_failure", None),
        (TransportResponse(401, '{"error":{"message":"sensitive-auth"}}'), "authentication_failed", "4xx"),
        (TransportResponse(429, '{"error":{"message":"sensitive-rate"}}'), "rate_limited", "4xx"),
        (TransportResponse(500, '{"error":{"message":"sensitive-server"}}'), "http_server_error", "5xx"),
        (
            TransportResponse(400, '{"error":{"message":"response_format json_schema unsupported secret"}}'),
            "response_format_unsupported",
            "4xx",
        ),
        (TransportResponse(200, "malformed-provider-envelope"), "malformed_provider_response", "2xx"),
    ],
)
def test_transport_and_http_failures_are_redacted_and_never_retried(
    outcome: TransportResponse | BaseException,
    category: str,
    status_class: str | None,
) -> None:
    transport = FakeTransport([outcome, TransportResponse(200, _provider_body())])
    result = _run(transport)
    assert result.status == "provider_error"
    assert result.transport_category == category
    assert result.http_status_class == status_class
    assert result.attempt_count == 1
    assert len(transport.calls) == 1
    serialized = result.model_dump_json().lower()
    for forbidden in (
        "sensitive-", "unsupported secret", "test-only-secret", "authorization",
        '"provider_metadata":', "system_prompt", "user_prompt",
    ):
        assert forbidden not in serialized


def test_successful_execution_writes_only_bound_redacted_receipt(tmp_path: Path) -> None:
    receipt_path = tmp_path / "compatibility-receipt.json"
    transport = FakeTransport([TransportResponse(200, _provider_body())])
    result = execute_compatibility(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=transport,
        receipt_path=receipt_path,
    )
    assert result.status == "completed"
    assert result.safety.compatibility_receipt_written is True
    assert len(transport.calls) == 1
    assert [item.name for item in tmp_path.iterdir()] == [receipt_path.name]

    receipt = load_receipt(receipt_path)
    config = load_config(DEFAULT_CONFIG)
    assert receipt.receipt_version == "conformity_source_behavior_compatibility_receipt.v2_1"
    assert receipt.compatibility_module_version == "conformity_source_behavior_compatibility.v2_1"
    assert receipt.config_sha256 == sha256_file(DEFAULT_CONFIG)
    assert receipt.protocol_sha256 == config.protocol_sha256
    assert receipt.approval_sha256 == approval_sha256(DEFAULT_APPROVAL)
    assert receipt.protocol_version == PROTOCOL_VERSION
    assert receipt.template_version == TEMPLATE_VERSION
    assert receipt.schema_contract.schema_name == SCHEMA_NAME
    assert receipt.schema_contract.schema_version == SCHEMA_VERSION
    assert receipt.generation_parameters.model_dump() == {
        "max_tokens": 128,
        "temperature": 0.0,
        "seed": 20261110,
        "max_retries": 0,
        "timeout_seconds": 5.0,
        "reasoning_effort": "none",
    }
    assert receipt.safety.behavior_qualification_executed is False
    assert receipt.safety.results_written is False

    serialized = receipt_path.read_text(encoding="utf-8").lower()
    for forbidden in (
        "test-only-secret", "authorization", "bearer", "system_prompt", "user_prompt",
        '"provider_metadata":', "compatibility-public-content", "fictional community room",
    ):
        assert forbidden not in serialized


def test_existing_receipt_blocks_before_environment_or_transport_access(tmp_path: Path) -> None:
    receipt_path = tmp_path / "existing.json"
    receipt_path.write_text("do-not-overwrite", encoding="utf-8")

    class ExplodingEnvironment(dict[str, str]):
        def get(self, key: str, default=None):
            raise AssertionError("environment must not be read")

    class FailTransport:
        def post(self, *args, **kwargs):
            raise AssertionError("transport must not be called")

    result = execute_compatibility(
        allow_network=True,
        environment=ExplodingEnvironment(),
        transport=FailTransport(),
        receipt_path=receipt_path,
    )
    assert result.status == "blocked"
    assert result.transport_category == "compatibility_receipt_exists"
    assert result.attempt_count == 0
    assert result.safety.api_key_read is False
    assert receipt_path.read_text(encoding="utf-8") == "do-not-overwrite"


def test_receipt_contract_rejects_unknown_fields(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    result = execute_compatibility(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=FakeTransport([TransportResponse(200, _provider_body())]),
        receipt_path=receipt_path,
    )
    assert result.status == "completed"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["raw_response"] = "forbidden"
    with pytest.raises(Exception):
        CompatibilityReceipt.model_validate(payload)
