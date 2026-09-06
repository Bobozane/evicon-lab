from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

import pytest

from evicon.conformity_source_behavior_compatibility_v3 import (
    DEFAULT_RECEIPT,
    DEFAULT_MODULE,
    MAX_RETRIES,
    MAX_TOKENS,
    RESPONSE_SCHEMA,
    RESPONSE_SCHEMA_SHA256,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    SEED,
    TEMPERATURE,
    TIMEOUT_SECONDS,
    CompatibilityReceiptV3,
    execute_compatibility_v3,
    load_receipt_v3,
    minimal_public_context_v3,
    run_compatibility_check_v3,
)
from evicon.conformity_source_behavior_qualification_v3 import (
    BehaviorQualificationResponseV3,
    DEFAULT_APPROVAL,
    PROTOCOL_VERSION,
    TEMPLATE_VERSION,
)
from evicon.conformity_source_behavior_qualification_v3 import load_config
from evicon.conformity_identification import sha256_file
from evicon.openai_provider import TransportResponse


ENVIRONMENT = {
    "EVICON_LLM_BASE_URL": "https://provider.invalid/v1",
    "EVICON_LLM_MODEL": "source-behavior-v3-compat-model",
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


def _accepted_approval(tmp_path: Path) -> Path:
    raw = (Path(__file__).resolve().parents[1] / DEFAULT_APPROVAL).read_text(encoding="utf-8")
    raw = raw.replace('acceptance_status = "pending"', 'acceptance_status = "accepted"')
    raw = raw.replace('accepted_by = ""', 'accepted_by = "test-reviewer"')
    raw = raw.replace('accepted_on = ""', 'accepted_on = "2026-08-28"')
    raw = re.sub(r"(confirm_[a-z0-9_]+) = false", r"\1 = true", raw)
    path = tmp_path / "compatibility-approval.toml"
    path.write_text(raw, encoding="utf-8")
    return path


def _valid_content() -> dict[str, object]:
    return {
        "ordinal_judgment": "uncertain",
        "public_action": "withhold",
        "share_decision": "do_not_share",
        "confidence": 3,
        "content_ids_used": ["content-03", "content-04"],
        "share_content_id": None,
    }


def _provider_body(*, content: str | None = None, finish_reason: str = "stop", completion_tokens: int = 24) -> str:
    if content is None:
        content = json.dumps(_valid_content(), ensure_ascii=True, sort_keys=True)
    return json.dumps({
        "model": ENVIRONMENT["EVICON_LLM_MODEL"],
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": 80,
            "completion_tokens": completion_tokens,
            "total_tokens": 80 + completion_tokens,
        },
    }, ensure_ascii=True, sort_keys=True)


def _provider_body_without_usage() -> str:
    return json.dumps({
        "model": ENVIRONMENT["EVICON_LLM_MODEL"],
        "choices": [{
            "message": {"content": json.dumps(_valid_content(), sort_keys=True)},
            "finish_reason": "stop",
        }],
    }, ensure_ascii=True, sort_keys=True)


def _provider_body_with_usage_mismatch() -> str:
    payload = json.loads(_provider_body())
    payload["usage"]["total_tokens"] = 999
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _run(transport: FakeTransport, *, approval_path: Path | None = None):
    return run_compatibility_check_v3(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=transport,
        approval_path=approval_path or DEFAULT_APPROVAL,
    )


def test_default_path_is_offline_and_does_not_read_environment_or_transport(tmp_path: Path) -> None:
    class ExplodingEnvironment(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            raise AssertionError("environment must not be read")

        def __iter__(self):
            raise AssertionError("environment must not be read")

        def __len__(self):
            raise AssertionError("environment must not be read")

        def get(self, key: str, default=None):
            raise AssertionError("environment must not be read")

    class FailTransport:
        def post(self, *args, **kwargs):
            raise AssertionError("transport must not be called")

    result = execute_compatibility_v3(
        allow_network=False,
        environment=ExplodingEnvironment(),
        transport=FailTransport(),
        receipt_path=tmp_path / "must-not-exist.json",
    )
    assert result.status == "network_disabled"
    assert result.transport_category == "network_disabled"
    assert result.attempt_count == 0
    assert result.safety.api_key_read is False
    assert result.safety.provider_constructed is False
    assert result.safety.qualification_executed is False
    assert not list(tmp_path.iterdir())
    assert set(result.model_dump(by_alias=True)) == {
        "status", "model", "finish_reason", "parser_valid", "http_status_class",
        "transport_category", "token_usage", "latency_ms", "schema", "attempt_count", "safety",
    }


def test_valid_2xx_uses_exact_v3_schema_parameters_and_one_call(tmp_path: Path) -> None:
    approval = _accepted_approval(tmp_path)
    transport = FakeTransport([TransportResponse(200, _provider_body())])
    result = _run(transport, approval_path=approval)
    assert result.status == "completed"
    assert result.parser_valid is True
    assert result.finish_reason == "stop"
    assert result.http_status_class == "2xx"
    assert result.token_usage.model_dump() == {
        "prompt_tokens": 80, "completion_tokens": 24, "total_tokens": 104,
    }
    assert result.attempt_count == 1
    assert len(transport.calls) == 1
    call = transport.calls[0]
    payload = call["payload"]
    assert payload["max_tokens"] == MAX_TOKENS == 256
    assert payload["temperature"] == TEMPERATURE == 0.2
    assert payload["seed"] == SEED == 20261201
    assert payload["reasoning_effort"] == "none"
    assert call["timeout_seconds"] == TIMEOUT_SECONDS == 5.0
    assert MAX_RETRIES == 0
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": RESPONSE_SCHEMA},
    }
    assert RESPONSE_SCHEMA == BehaviorQualificationResponseV3.model_json_schema(mode="validation")
    public_request = payload["messages"][1]["content"].lower()
    assert "target_claim" in public_request
    assert "decision_task" in public_request
    for forbidden in ("source-behavior-v3-q001", "source-behavior-v3-s01", "source_free", "20261201"):
        assert forbidden not in public_request


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        json.dumps(_valid_content() | {"unknown_field": "forbidden"}),
        json.dumps(_valid_content() | {"content_ids_used": ["not-exposed"]}),
        json.dumps(_valid_content() | {"ordinal_judgment": "lean_adopt", "public_action": "reject"}),
    ],
    ids=["malformed-inner-json", "unknown-field", "unexposed-content-id", "inconsistent-action"],
)
def test_invalid_v3_outputs_stop_after_one_call(content: str, tmp_path: Path) -> None:
    transport = FakeTransport([
        TransportResponse(200, _provider_body(content=content)),
        TransportResponse(200, _provider_body()),
    ])
    result = _run(transport, approval_path=_accepted_approval(tmp_path))
    assert result.status == "invalid_response"
    assert result.parser_valid is False
    assert result.transport_category == "invalid_provider_response"
    assert result.attempt_count == 1
    assert len(transport.calls) == 1


def test_finish_reason_length_is_invalid_without_retry(tmp_path: Path) -> None:
    transport = FakeTransport([
        TransportResponse(200, _provider_body(finish_reason="length", completion_tokens=256)),
        TransportResponse(200, _provider_body()),
    ])
    result = _run(transport, approval_path=_accepted_approval(tmp_path))
    assert result.status == "invalid_response"
    assert result.parser_valid is True
    assert result.finish_reason == "length"
    assert result.transport_category == "invalid_provider_response"
    assert result.attempt_count == 1
    assert len(transport.calls) == 1


def test_missing_token_usage_is_invalid_without_retry(tmp_path: Path) -> None:
    transport = FakeTransport([
        TransportResponse(200, _provider_body_without_usage()),
        TransportResponse(200, _provider_body()),
    ])
    result = _run(transport, approval_path=_accepted_approval(tmp_path))
    assert result.status == "invalid_response"
    assert result.parser_valid is True
    assert result.transport_category == "invalid_token_usage"
    assert result.attempt_count == 1
    assert len(transport.calls) == 1


def test_inconsistent_token_usage_is_redacted_without_retry(tmp_path: Path) -> None:
    transport = FakeTransport([
        TransportResponse(200, _provider_body_with_usage_mismatch()),
        TransportResponse(200, _provider_body()),
    ])
    result = _run(transport, approval_path=_accepted_approval(tmp_path))
    assert result.status == "provider_error"
    assert result.transport_category == "invalid_token_usage"
    assert result.attempt_count == 1
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    ("outcome", "category", "status_class"),
    [
        (TimeoutError("sensitive-timeout"), "timeout", None),
        (ConnectionError("sensitive-connection"), "connection_failure", None),
        (TransportResponse(401, '{"error":{"message":"sensitive-auth"}}'), "authentication_failed", "4xx"),
        (TransportResponse(429, '{"error":{"message":"sensitive-rate"}}'), "rate_limited", "4xx"),
        (TransportResponse(500, '{"error":{"message":"sensitive-server"}}'), "http_server_error", "5xx"),
        (TransportResponse(400, '{"error":{"message":"response_format json_schema unsupported secret"}}'), "response_format_unsupported", "4xx"),
    ],
)
def test_transport_failures_are_redacted_and_never_retried(
    outcome: TransportResponse | BaseException,
    category: str,
    status_class: str | None,
    tmp_path: Path,
) -> None:
    transport = FakeTransport([outcome, TransportResponse(200, _provider_body())])
    result = _run(transport, approval_path=_accepted_approval(tmp_path))
    assert result.status == "provider_error"
    assert result.transport_category == category
    assert result.http_status_class == status_class
    assert result.attempt_count == 1
    assert len(transport.calls) == 1
    serialized = result.model_dump_json().lower()
    for forbidden in ("sensitive-", "unsupported secret", "test-only-secret", "authorization", '"provider_metadata":', "system_prompt", "user_prompt"):
        assert forbidden not in serialized


def test_malformed_provider_envelope_is_redacted(tmp_path: Path) -> None:
    transport = FakeTransport([TransportResponse(200, "malformed-provider-envelope")])
    result = _run(transport, approval_path=_accepted_approval(tmp_path))
    assert result.status == "provider_error"
    assert result.transport_category == "malformed_provider_response"
    assert result.http_status_class == "2xx"
    assert result.attempt_count == 1


def test_success_writes_only_bound_v3_receipt(tmp_path: Path) -> None:
    approval = _accepted_approval(tmp_path)
    receipt_path = tmp_path / "compatibility-receipt.json"
    transport = FakeTransport([TransportResponse(200, _provider_body())])
    result = execute_compatibility_v3(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=transport,
        approval_path=approval,
        receipt_path=receipt_path,
    )
    assert result.status == "completed"
    assert result.safety.compatibility_receipt_written is True
    assert {item.name for item in tmp_path.iterdir() if item.is_file()} == {"compatibility-approval.toml", "compatibility-receipt.json"}
    receipt = load_receipt_v3(receipt_path, approval_path=approval)
    config, _ = load_config()
    assert receipt.receipt_version == "conformity_source_behavior_compatibility_receipt.v3"
    assert receipt.compatibility_module_version == "conformity_source_behavior_compatibility.v3"
    assert receipt.config_sha256
    assert receipt.corpus_sha256 == config.corpus_sha256
    assert receipt.protocol_sha256 == config.protocol_sha256
    assert receipt.compatibility_module_sha256 == sha256_file(DEFAULT_MODULE)
    assert receipt.protocol_version == PROTOCOL_VERSION
    assert receipt.template_version == TEMPLATE_VERSION
    assert receipt.schema_contract.schema_name == SCHEMA_NAME
    assert receipt.schema_contract.schema_version == SCHEMA_VERSION
    assert receipt.schema_contract.schema_sha256 == RESPONSE_SCHEMA_SHA256
    assert receipt.generation_parameters.model_dump() == {
        "max_tokens": 256, "temperature": 0.2, "seed": 20261201,
        "max_retries": 0, "timeout_seconds": 5.0, "reasoning_effort": "none",
    }
    serialized = receipt_path.read_text(encoding="utf-8").lower()
    for forbidden in (
        "test-only-secret", "authorization", "bearer", "system_prompt", "user_prompt",
        '"provider_metadata":', "target_claim", "decision_task", "fictional service update",
        "content-01",
    ):
        assert forbidden not in serialized


def test_existing_receipt_blocks_before_environment_or_transport_access(tmp_path: Path) -> None:
    approval = _accepted_approval(tmp_path)
    receipt_path = tmp_path / "existing.json"
    receipt_path.write_text("do-not-overwrite", encoding="utf-8")

    class ExplodingEnvironment(dict[str, str]):
        def get(self, key: str, default=None):
            raise AssertionError("environment must not be read")

    class FailTransport:
        def post(self, *args, **kwargs):
            raise AssertionError("transport must not be called")

    result = execute_compatibility_v3(
        allow_network=True,
        environment=ExplodingEnvironment(),
        transport=FailTransport(),
        approval_path=approval,
        receipt_path=receipt_path,
    )
    assert result.status == "blocked"
    assert result.transport_category == "compatibility_receipt_exists"
    assert result.attempt_count == 0
    assert result.safety.api_key_read is False
    assert receipt_path.read_text(encoding="utf-8") == "do-not-overwrite"


@pytest.mark.parametrize(
    ("missing", "category"),
    [
        ("EVICON_LLM_BASE_URL", "missing_base_url"),
        ("EVICON_LLM_MODEL", "missing_model"),
        ("EVICON_LLM_API_KEY", "missing_api_key"),
    ],
)
def test_missing_provider_environment_fails_without_transport_call(
    missing: str,
    category: str,
    tmp_path: Path,
) -> None:
    environment = dict(ENVIRONMENT)
    del environment[missing]
    transport = FakeTransport([TransportResponse(200, _provider_body())])
    result = run_compatibility_check_v3(
        allow_network=True,
        environment=environment,
        transport=transport,
        approval_path=_accepted_approval(tmp_path),
    )
    assert result.status == "provider_error"
    assert result.transport_category == category
    assert result.attempt_count == 0
    assert len(transport.calls) == 0


def test_v3_schema_rejects_unknown_receipt_fields(tmp_path: Path) -> None:
    approval = _accepted_approval(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    result = execute_compatibility_v3(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=FakeTransport([TransportResponse(200, _provider_body())]),
        approval_path=approval,
        receipt_path=receipt_path,
    )
    assert result.status == "completed"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["raw_response"] = "forbidden"
    with pytest.raises(Exception):
        CompatibilityReceiptV3.model_validate(payload)


def test_receipt_loader_rejects_duplicate_fields(tmp_path: Path) -> None:
    approval = _accepted_approval(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    result = execute_compatibility_v3(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=FakeTransport([TransportResponse(200, _provider_body())]),
        approval_path=approval,
        receipt_path=receipt_path,
    )
    assert result.status == "completed"
    serialized = receipt_path.read_text(encoding="utf-8")
    receipt_path.write_text(
        serialized.replace(
            '"status": "completed",',
            '"status": "completed",\n  "status": "completed",',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="compatibility_v3_receipt_invalid"):
        load_receipt_v3(receipt_path, approval_path=approval)


def test_minimal_context_is_v3_and_does_not_consult_corpus() -> None:
    context = minimal_public_context_v3()
    assert context.case_id == "source-behavior-v3-q001"
    assert context.scenario_id == "source-behavior-v3-s01"
    assert len(context.visible_content) == 4
    assert len(context.public_root_assignments) == 4
    assert context.public_root_assignments[2].source_root_id == "root-03"
