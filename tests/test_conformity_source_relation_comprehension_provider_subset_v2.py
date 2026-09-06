from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from evicon.conformity_identification import sha256_file
from evicon.conformity_source_relation_comprehension_calibration_v1 import (
    RESPONSE_SCHEMA as CANONICAL_SCHEMA,
    RESPONSE_SCHEMA_SHA256 as CANONICAL_SCHEMA_SHA256,
    SourceRelationComprehensionResponse,
)
import evicon.conformity_source_relation_comprehension_compatibility_provider_subset_v2 as compatibility_module
from evicon.conformity_source_relation_comprehension_compatibility_provider_subset_v2 import (
    DEFAULT_MODULE,
    DEFAULT_RECEIPT,
    MAX_RETRIES,
    MAX_TOKENS,
    PROVIDER_RESPONSE_SCHEMA,
    PROVIDER_RESPONSE_SCHEMA_SHA256,
    PROVIDER_SCHEMA_NAME,
    RECEIPT_VERSION,
    SEED,
    TEMPERATURE,
    TIMEOUT_SECONDS,
    CompatibilityReceiptV2,
    execute_compatibility_provider_subset_v2,
    load_receipt_v2,
    minimal_public_context_v2,
)
from evicon.conformity_source_relation_comprehension_provider_subset_v2 import (
    REMOVED_PROVIDER_KEYWORDS,
)
from evicon.conformity_source_relation_comprehension_provider_subset_v2_approval import (
    DEFAULT_APPROVAL,
    DEFAULT_CONFIG,
    load_compatibility_config,
)
from evicon.openai_provider import TransportResponse


ENVIRONMENT = {
    "EVICON_LLM_BASE_URL": "https://provider.invalid/v1",
    "EVICON_LLM_MODEL": "source-relation-subset-test-model",
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
    raw = (Path(__file__).resolve().parents[1] / DEFAULT_APPROVAL).read_text(
        encoding="utf-8"
    )
    raw = raw.replace('acceptance_status = "pending"', 'acceptance_status = "accepted"')
    raw = raw.replace('accepted_by = ""', 'accepted_by = "test-reviewer"')
    raw = raw.replace('accepted_by = "researcher_user"', 'accepted_by = "test-reviewer"')
    raw = raw.replace('accepted_on = ""', 'accepted_on = "2026-08-29"')
    path = tmp_path / "provider-subset-approval.toml"
    path.write_text(raw, encoding="utf-8")
    return path


def _pending_approval(tmp_path: Path) -> Path:
    raw = (Path(__file__).resolve().parents[1] / DEFAULT_APPROVAL).read_text(
        encoding="utf-8"
    )
    raw = raw.replace('acceptance_status = "accepted"', 'acceptance_status = "pending"')
    raw = raw.replace('accepted_by = "researcher_user"', 'accepted_by = ""')
    raw = raw.replace('accepted_on = "2026-08-29"', 'accepted_on = ""')
    path = tmp_path / "provider-subset-pending-approval.toml"
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
    if content is None:
        content = json.dumps(_valid_content(), ensure_ascii=True, sort_keys=True)
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
        ensure_ascii=True,
        sort_keys=True,
    )


def _run(
    transport: FakeTransport,
    tmp_path: Path,
    *,
    approval_path: Path | None = None,
    receipt_name: str = "receipt.json",
):
    return execute_compatibility_provider_subset_v2(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=transport,
        approval_path=approval_path or _accepted_approval(tmp_path),
        receipt_path=tmp_path / receipt_name,
    )


def test_default_path_is_offline_and_does_not_read_environment_or_write_files(
    tmp_path: Path,
) -> None:
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

    result = execute_compatibility_provider_subset_v2(
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
    assert not list(tmp_path.iterdir())


def test_pending_approval_blocks_before_environment_or_transport_access(
    tmp_path: Path,
) -> None:
    class ExplodingEnvironment(dict[str, str]):
        def get(self, key: str, default=None):
            raise AssertionError("environment must not be read")

    class FailTransport:
        def post(self, *args, **kwargs):
            raise AssertionError("transport must not be called")

    result = execute_compatibility_provider_subset_v2(
        allow_network=True,
        environment=ExplodingEnvironment(),
        transport=FailTransport(),
        approval_path=_pending_approval(tmp_path),
        receipt_path=tmp_path / "blocked.json",
    )
    assert result.status == "blocked"
    assert result.transport_category == "compatibility_approval_not_accepted"
    assert result.attempt_count == 0


def test_valid_2xx_uses_provider_subset_schema_and_exact_parameters(
    tmp_path: Path,
) -> None:
    transport = FakeTransport([TransportResponse(200, _provider_body())])
    result = _run(transport, tmp_path)
    assert result.status == "completed"
    assert result.parser_valid is True
    assert result.finish_reason == "stop"
    assert result.http_status_class == "2xx"
    assert result.attempt_count == 1
    assert len(transport.calls) == 1

    call = transport.calls[0]
    payload = call["payload"]
    assert payload["max_tokens"] == MAX_TOKENS == 128
    assert payload["temperature"] == TEMPERATURE == 0.0
    assert payload["seed"] == SEED == 20261401
    assert payload["reasoning_effort"] == "none"
    assert call["timeout_seconds"] == TIMEOUT_SECONDS == 5.0
    assert MAX_RETRIES == 0
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": PROVIDER_SCHEMA_NAME,
            "strict": True,
            "schema": PROVIDER_RESPONSE_SCHEMA,
        },
    }
    assert result.safety.provider_subset_schema_used is True
    assert result.safety.canonical_parser_used is True


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
    ids=["malformed-inner-json", "unknown-field", "unexposed-content-id"],
)
def test_invalid_structural_output_is_not_retried(
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
    assert result.transport_category == "invalid_provider_response"
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
    assert result.parser_valid is True
    assert result.finish_reason == "length"
    assert result.transport_category == "invalid_provider_response"
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    ("outcome", "category", "status_class"),
    [
        (TimeoutError("sensitive-timeout"), "timeout", None),
        (ConnectionError("sensitive-connection"), "connection_failure", None),
        (TransportResponse(401, '{"error":{"message":"sensitive-auth"}}'), "authentication_failed", "4xx"),
        (TransportResponse(429, '{"error":{"message":"sensitive-rate"}}'), "rate_limited", "4xx"),
        (TransportResponse(500, '{"error":{"message":"sensitive-server"}}'), "http_server_error", "5xx"),
        (
            TransportResponse(
                400,
                '{"error":{"message":"response_format json_schema unsupported secret"}}',
            ),
            "response_format_unsupported",
            "4xx",
        ),
    ],
    ids=["timeout", "connection", "401", "429", "500", "format-unsupported"],
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
        "unsupported secret",
        "test-only-secret",
        "authorization",
        '"provider_metadata":',
        "system_prompt",
        "user_prompt",
    ):
        assert forbidden not in serialized


def test_malformed_provider_envelope_is_redacted(tmp_path: Path) -> None:
    result = _run(
        FakeTransport([TransportResponse(200, "malformed-provider-envelope")]),
        tmp_path,
    )
    assert result.status == "provider_error"
    assert result.transport_category == "malformed_provider_response"
    assert result.http_status_class == "2xx"
    assert result.attempt_count == 1


def test_success_writes_only_bound_redacted_receipt(tmp_path: Path) -> None:
    approval = _accepted_approval(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    result = execute_compatibility_provider_subset_v2(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=FakeTransport([TransportResponse(200, _provider_body())]),
        approval_path=approval,
        receipt_path=receipt_path,
    )
    assert result.status == "completed"
    assert result.safety.compatibility_receipt_written is True
    receipt = load_receipt_v2(receipt_path, approval_path=approval)
    assert receipt.receipt_version == RECEIPT_VERSION
    assert receipt.compatibility_config_sha256 == sha256_file(DEFAULT_CONFIG)
    assert receipt.provider_schema_sha256 == PROVIDER_RESPONSE_SCHEMA_SHA256
    assert receipt.canonical_schema_sha256 == CANONICAL_SCHEMA_SHA256
    assert receipt.compatibility_module_sha256 == sha256_file(DEFAULT_MODULE)
    assert receipt.schema_contract.schema_name == PROVIDER_SCHEMA_NAME
    assert receipt.generation_parameters.max_tokens == 128
    assert receipt.safety.provider_subset_amendment is True
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


def test_model_reflection_is_not_persisted(tmp_path: Path) -> None:
    reflected = "provider-returned-untrusted-name"
    receipt_path = tmp_path / "receipt.json"
    result = _run(
        FakeTransport(
            [TransportResponse(200, _provider_body(model_name=reflected))]
        ),
        tmp_path,
        receipt_name=receipt_path.name,
    )
    assert result.status == "completed"
    assert result.model == ENVIRONMENT["EVICON_LLM_MODEL"]
    assert reflected not in result.model_dump_json(by_alias=True)
    assert reflected not in receipt_path.read_text(encoding="utf-8")


def test_claim_blocks_second_attempt_and_invalid_attempt_is_not_overwritten(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "claimed.json"
    first = _run(
        FakeTransport([TransportResponse(200, _provider_body(content="not-json"))]),
        tmp_path,
        receipt_name=receipt_path.name,
    )
    assert first.status == "invalid_response"
    claim = receipt_path.with_name(f".{receipt_path.name}.attempt-claim")
    assert claim.exists()

    class ExplodingEnvironment(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            raise AssertionError("environment must not be read after a claim")

        def __iter__(self):
            raise AssertionError("environment must not be read after a claim")

        def __len__(self):
            raise AssertionError("environment must not be read after a claim")

        def get(self, key: str, default=None):
            raise AssertionError("environment must not be read after a claim")

    second = execute_compatibility_provider_subset_v2(
        allow_network=True,
        environment=ExplodingEnvironment(),
        transport=FakeTransport([TransportResponse(200, _provider_body())]),
        approval_path=_accepted_approval(tmp_path),
        receipt_path=receipt_path,
    )
    assert second.status == "blocked"
    assert second.transport_category == "compatibility_attempt_claim_exists"


def test_missing_api_key_does_not_call_transport(tmp_path: Path) -> None:
    environment = dict(ENVIRONMENT)
    del environment["EVICON_LLM_API_KEY"]
    transport = FakeTransport([TransportResponse(200, _provider_body())])
    result = execute_compatibility_provider_subset_v2(
        allow_network=True,
        environment=environment,
        transport=transport,
        approval_path=_accepted_approval(tmp_path),
        receipt_path=tmp_path / "missing-key.json",
    )
    assert result.status == "provider_error"
    assert result.transport_category == "missing_api_key"
    assert result.attempt_count == 0
    assert not transport.calls


def test_provider_subset_is_distinct_from_canonical_schema() -> None:
    assert CANONICAL_SCHEMA_SHA256 != PROVIDER_RESPONSE_SCHEMA_SHA256
    assert CANONICAL_SCHEMA == SourceRelationComprehensionResponse.model_json_schema(
        mode="validation"
    )
    canonical_text = json.dumps(CANONICAL_SCHEMA, sort_keys=True)
    provider_text = json.dumps(PROVIDER_RESPONSE_SCHEMA, sort_keys=True)
    assert "prefixItems" in canonical_text
    assert "const" in canonical_text
    for keyword in REMOVED_PROVIDER_KEYWORDS:
        assert keyword not in provider_text


def test_minimal_context_is_fresh_public_material() -> None:
    context = minimal_public_context_v2()
    assert context.case_id == "source-relation-v1-c99"
    assert context.scenario_id == "source-relation-v1-s01"
    text = " ".join(item.public_summary for item in context.visible_material).lower()
    for forbidden in ("pilot", "wvs", "ground truth", "evaluator_private"):
        assert forbidden not in text


def test_default_paths_are_scoped_to_study_locks() -> None:
    assert DEFAULT_CONFIG.startswith("configs/provenance_cascade/identification/")
    assert DEFAULT_RECEIPT == (
        "outputs/study-locks/"
        "conformity_source_relation_comprehension_compatibility_provider_subset.v2.json"
    )


def test_receipt_contract_rejects_unknown_fields() -> None:
    fields = CompatibilityReceiptV2.model_fields
    assert "raw_response" not in fields
    assert "prompt" not in fields
    assert "headers" not in fields
