from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from evicon.conformity_identification import sha256_file
from evicon.conformity_source_relation_comprehension_calibration_v1 import (
    render_request,
)
from evicon.conformity_source_relation_comprehension_compatibility_provider_subset_short_name_v4 import (
    DEFAULT_MODULE,
    DEFAULT_RECEIPT,
    MAX_RETRIES,
    MAX_TOKENS,
    PROVIDER_RESPONSE_SCHEMA,
    PROVIDER_RESPONSE_SCHEMA_SHA256,
    RECEIPT_VERSION,
    SEED,
    SHORT_PROVIDER_SCHEMA_NAME,
    TEMPERATURE,
    TIMEOUT_SECONDS,
    CompatibilityReceiptV4,
    execute_compatibility_provider_subset_short_name_v4,
    load_receipt_v4,
)
from evicon.conformity_source_relation_comprehension_compatibility_provider_subset_timeout_v3 import (
    _nonsecret_provider_config as v3_provider_config,
)
from evicon.conformity_source_relation_comprehension_compatibility_provider_subset_v2 import (
    minimal_public_context_v2,
)
from evicon.conformity_source_relation_comprehension_provider_subset_short_name_v4_approval import (
    DEFAULT_APPROVAL,
    DEFAULT_CONFIG,
    V1_ATTEMPT_CLAIM,
    V1_RECEIPT,
    V2_ATTEMPT_CLAIM,
    V2_RECEIPT,
    V3_ATTEMPT_CLAIM,
    V3_RECEIPT,
)
from evicon.openai_provider import (
    OpenAICompatibleProvider,
    TransportResponse,
)


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = {
    "EVICON_LLM_BASE_URL": "https://provider.invalid/v1",
    "EVICON_LLM_MODEL": "source-relation-short-name-v4-test-model",
    "EVICON_LLM_API_KEY": "test-only-secret",
}


class FakeTransport:
    def __init__(self, outcomes: list[TransportResponse | BaseException]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def post(self, url, headers, payload, timeout_seconds):
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _accepted_approval(tmp_path: Path) -> Path:
    raw = (ROOT / DEFAULT_APPROVAL).read_text(encoding="utf-8")
    raw = raw.replace('accepted_by = "researcher_user"', 'accepted_by = "test-reviewer"')
    path = tmp_path / "accepted-short-name-v4.toml"
    path.write_text(raw, encoding="utf-8")
    return path


def _pending_approval(tmp_path: Path) -> Path:
    raw = (ROOT / DEFAULT_APPROVAL).read_text(encoding="utf-8")
    raw = raw.replace('acceptance_status = "accepted"', 'acceptance_status = "pending"')
    raw = raw.replace('accepted_by = "researcher_user"', 'accepted_by = ""')
    raw = raw.replace('accepted_on = "2026-08-30"', 'accepted_on = ""')
    path = tmp_path / "pending-short-name-v4.toml"
    path.write_text(raw, encoding="utf-8")
    return path


def _valid_content() -> dict[str, object]:
    return {
        "visible_social_message_count": 2,
        "source_relation": "distinct_shown_roots",
        "visible_social_root_count": 2,
        "social_root_assignments": [
            {"content_id": "message-a", "source_root_id": "lineage-k7"},
            {"content_id": "message-b", "source_root_id": "lineage-p4"},
        ],
        "direct_evidence_card_ids": ["evidence-card-01"],
        "support_kind": "direct_evidence_present",
        "source_structure_alone_decides_truth": False,
    }


def _provider_body(
    *,
    content: str | None = None,
    finish_reason: str = "stop",
    completion_tokens: int = 32,
    model_name: str | None = None,
) -> str:
    content = content or json.dumps(_valid_content(), sort_keys=True)
    return json.dumps(
        {
            "model": model_name or ENVIRONMENT["EVICON_LLM_MODEL"],
            "choices": [
                {"message": {"content": content}, "finish_reason": finish_reason}
            ],
            "usage": {
                "prompt_tokens": 90,
                "completion_tokens": completion_tokens,
                "total_tokens": 90 + completion_tokens,
            },
        },
        sort_keys=True,
    )


def _run(transport: FakeTransport, tmp_path: Path, receipt_name: str = "receipt.json"):
    return execute_compatibility_provider_subset_short_name_v4(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=transport,
        approval_path=_accepted_approval(tmp_path),
        receipt_path=tmp_path / receipt_name,
    )


def test_default_path_is_fully_offline_and_writes_nothing(tmp_path: Path) -> None:
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

    result = execute_compatibility_provider_subset_short_name_v4(
        environment=ExplodingEnvironment(),
        transport=FailTransport(),
        receipt_path=tmp_path / "must-not-exist.json",
    )
    assert result.status == "network_disabled"
    assert result.attempt_count == 0
    assert result.safety.api_key_read is False
    assert result.safety.provider_constructed is False
    assert not list(tmp_path.iterdir())


def test_pending_approval_blocks_before_environment_access(tmp_path: Path) -> None:
    class ExplodingEnvironment(dict[str, str]):
        def get(self, key: str, default=None):
            raise AssertionError("environment must not be read")

    result = execute_compatibility_provider_subset_short_name_v4(
        allow_network=True,
        environment=ExplodingEnvironment(),
        transport=FakeTransport([]),
        approval_path=_pending_approval(tmp_path),
        receipt_path=tmp_path / "blocked.json",
    )
    assert result.status == "blocked"
    assert result.transport_category == "compatibility_approval_not_accepted"
    assert result.attempt_count == 0
    assert not (tmp_path / "blocked.json").exists()
    assert not (tmp_path / ".blocked.json.attempt-claim").exists()


def test_valid_2xx_changes_only_the_v3_schema_name(tmp_path: Path) -> None:
    transport = FakeTransport([TransportResponse(200, _provider_body())])
    result = _run(transport, tmp_path)
    assert result.status == "completed"
    assert result.parser_valid is True
    assert result.attempt_count == 1
    assert len(transport.calls) == 1
    call = transport.calls[0]
    payload = call["payload"]
    assert call["timeout_seconds"] == TIMEOUT_SECONDS == 15.0
    assert payload["max_tokens"] == MAX_TOKENS == 128
    assert payload["temperature"] == TEMPERATURE == 0.0
    assert payload["seed"] == SEED == 20261401
    assert payload["reasoning_effort"] == "none"
    assert MAX_RETRIES == 0
    schema = payload["response_format"]["json_schema"]
    assert schema["name"] == SHORT_PROVIDER_SCHEMA_NAME
    assert len(schema["name"]) == 42 <= 64
    assert schema["strict"] is True
    assert schema["schema"] == PROVIDER_RESPONSE_SCHEMA

    # Compare against the immutable v3 request builder.  This checks the whole
    # wire payload, including prompts and generation settings, before changing
    # exactly the single designated field.
    v3_provider = OpenAICompatibleProvider(
        v3_provider_config(ENVIRONMENT),
        transport=FakeTransport([]),
        environment=ENVIRONMENT,
    )
    expected_v3_payload = v3_provider._payload(
        render_request(minimal_public_context_v2(ENVIRONMENT["EVICON_LLM_MODEL"]))
    )
    expected_v3_payload["response_format"]["json_schema"]["name"] = (
        SHORT_PROVIDER_SCHEMA_NAME
    )
    assert payload == expected_v3_payload


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        json.dumps(_valid_content() | {"unknown_field": "forbidden"}),
        json.dumps(
            _valid_content()
            | {
                "social_root_assignments": [
                    {"content_id": "message-a", "source_root_id": "lineage-k7"},
                    {"content_id": "message-x", "source_root_id": "lineage-p4"},
                ]
            }
        ),
    ],
    ids=["malformed-json", "unknown-field", "unexposed-content-id"],
)
def test_invalid_output_is_redacted_and_never_retried(
    content: str, tmp_path: Path
) -> None:
    transport = FakeTransport(
        [
            TransportResponse(200, _provider_body(content=content)),
            TransportResponse(200, _provider_body()),
        ]
    )
    result = _run(transport, tmp_path)
    assert result.status == "invalid_response"
    assert result.parser_valid is False
    assert result.attempt_count == 1
    assert len(transport.calls) == 1


def test_finish_reason_length_is_invalid_without_retry(tmp_path: Path) -> None:
    transport = FakeTransport(
        [
            TransportResponse(
                200, _provider_body(finish_reason="length", completion_tokens=128)
            ),
            TransportResponse(200, _provider_body()),
        ]
    )
    result = _run(transport, tmp_path)
    assert result.status == "invalid_response"
    assert result.finish_reason == "length"
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    ("outcome", "category", "status_class"),
    [
        (TimeoutError("sensitive-timeout"), "timeout", None),
        (ConnectionError("sensitive-connection"), "connection_failure", None),
        (TransportResponse(401, '{"error":{"message":"sensitive-auth"}}'), "authentication_failed", "4xx"),
        (TransportResponse(429, '{"error":{"message":"sensitive-rate"}}'), "rate_limited", "4xx"),
        (TransportResponse(500, '{"error":{"message":"sensitive-server"}}'), "http_server_error", "5xx"),
        (TransportResponse(400, '{"error":{"message":"ordinary client secret"}}'), "http_client_error", "4xx"),
        (
            TransportResponse(
                400,
                '{"error":{"message":"response_format json_schema unsupported secret"}}',
            ),
            "response_format_unsupported",
            "4xx",
        ),
        (TransportResponse(200, "not-provider-json"), "malformed_provider_response", "2xx"),
    ],
)
def test_transport_failures_are_redacted_and_never_retried(
    outcome: TransportResponse | BaseException,
    category: str,
    status_class: str | None,
    tmp_path: Path,
) -> None:
    transport = FakeTransport([outcome, TransportResponse(200, _provider_body())])
    result = _run(transport, tmp_path)
    assert result.status == "provider_error"
    assert result.transport_category == category
    assert result.http_status_class == status_class
    assert result.attempt_count == 1
    assert len(transport.calls) == 1
    serialized = result.model_dump_json(by_alias=True).lower()
    for forbidden in (
        "sensitive-",
        "ordinary client secret",
        "unsupported secret",
        "test-only-secret",
        "authorization",
        '"provider_metadata":',
        "system_prompt",
        "user_prompt",
    ):
        assert forbidden not in serialized


def test_success_writes_only_hash_bound_redacted_receipt_and_preserves_parents(
    tmp_path: Path,
) -> None:
    approval = _accepted_approval(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    parent_claims = (V1_ATTEMPT_CLAIM, V2_ATTEMPT_CLAIM, V3_ATTEMPT_CLAIM)
    parent_snapshots = {
        claim: (ROOT / claim).read_bytes() for claim in parent_claims
    }
    result = execute_compatibility_provider_subset_short_name_v4(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=FakeTransport([TransportResponse(200, _provider_body())]),
        approval_path=approval,
        receipt_path=receipt_path,
    )
    assert result.status == "completed"
    assert result.safety.compatibility_receipt_written is True
    receipt = load_receipt_v4(receipt_path, approval_path=approval)
    assert receipt.receipt_version == RECEIPT_VERSION
    assert receipt.execution_bindings.compatibility_config_sha256 == sha256_file(
        DEFAULT_CONFIG
    )
    assert receipt.execution_bindings.compatibility_module_sha256 == sha256_file(
        DEFAULT_MODULE
    )
    assert receipt.schema_contract.schema_sha256 == PROVIDER_RESPONSE_SCHEMA_SHA256
    assert receipt.execution_bindings.v3_failure_category == "http_client_error"
    assert receipt.execution_bindings.provider_schema_name == SHORT_PROVIDER_SCHEMA_NAME
    assert receipt.generation_parameters.timeout_seconds == 15.0
    assert {
        claim: (ROOT / claim).read_bytes() for claim in parent_claims
    } == parent_snapshots
    for parent_receipt in (V1_RECEIPT, V2_RECEIPT, V3_RECEIPT):
        assert not (ROOT / parent_receipt).exists()
    serialized = receipt_path.read_text(encoding="utf-8").lower()
    for forbidden in (
        "test-only-secret",
        "authorization",
        "bearer",
        "system_prompt",
        "user_prompt",
        '"provider_metadata":',
        "birchline",
        "message-a",
        "lineage-k7",
        "evidence-card-01",
    ):
        assert forbidden not in serialized


def test_claim_blocks_second_attempt(tmp_path: Path) -> None:
    receipt_path = tmp_path / "claimed.json"
    first = _run(
        FakeTransport([TransportResponse(200, _provider_body(content="not-json"))]),
        tmp_path,
        receipt_name=receipt_path.name,
    )
    assert first.status == "invalid_response"
    claim = receipt_path.with_name(f".{receipt_path.name}.attempt-claim")
    assert claim.exists()
    second_transport = FakeTransport([TransportResponse(200, _provider_body())])
    second = execute_compatibility_provider_subset_short_name_v4(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=second_transport,
        approval_path=_accepted_approval(tmp_path),
        receipt_path=receipt_path,
    )
    assert second.status == "blocked"
    assert second.transport_category == "compatibility_attempt_claim_exists"
    assert not second_transport.calls


def test_receipt_contract_has_no_raw_or_secret_fields() -> None:
    fields = CompatibilityReceiptV4.model_fields
    for forbidden in (
        "raw_response",
        "prompt",
        "headers",
        "api_key",
        "provider_metadata",
        "request_ledger",
    ):
        assert forbidden not in fields
    assert DEFAULT_RECEIPT.startswith("outputs/study-locks/")
