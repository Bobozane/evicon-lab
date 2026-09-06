from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

import pytest

from evicon.conformity_identification import sha256_file
from evicon.conformity_source_relation_comprehension_compatibility_approval_v1 import (
    DEFAULT_COMPATIBILITY_APPROVAL,
    DEFAULT_COMPATIBILITY_CONFIG,
)
from evicon.conformity_source_relation_comprehension_calibration_v1 import (
    DEFAULT_CONFIG,
    PROTOCOL_VERSION,
    RESPONSE_SCHEMA,
    RESPONSE_SCHEMA_SHA256,
    SCHEMA_NAME,
    TEMPLATE_VERSION,
    SourceRelationComprehensionResponse,
)
import evicon.conformity_source_relation_comprehension_compatibility_v1 as compatibility_module
from evicon.conformity_source_relation_comprehension_compatibility_v1 import (
    COMPATIBILITY_MODULE_VERSION,
    DEFAULT_ATTEMPT_CLAIM,
    DEFAULT_MODULE,
    DEFAULT_RECEIPT,
    MAX_RETRIES,
    MAX_TOKENS,
    RECEIPT_VERSION,
    SCHEMA_VERSION,
    SEED,
    TEMPERATURE,
    TIMEOUT_SECONDS,
    CompatibilityReceiptV1,
    execute_compatibility_v1,
    load_receipt_v1,
    minimal_public_context_v1,
)
from evicon.openai_provider import TransportResponse


ENVIRONMENT = {
    "EVICON_LLM_BASE_URL": "https://provider.invalid/v1",
    "EVICON_LLM_MODEL": "source-relation-compatibility-test-model",
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
    raw = (
        Path(__file__).resolve().parents[1] / DEFAULT_COMPATIBILITY_APPROVAL
    ).read_text(
        encoding="utf-8"
    )
    raw = raw.replace('acceptance_status = "pending"', 'acceptance_status = "accepted"')
    raw = raw.replace('accepted_by = ""', 'accepted_by = "test-reviewer"')
    raw = raw.replace('accepted_on = ""', 'accepted_on = "2026-08-29"')
    path = tmp_path / "compatibility-approval.toml"
    path.write_text(raw, encoding="utf-8")
    return path


def _pending_approval(tmp_path: Path) -> Path:
    raw = (
        Path(__file__).resolve().parents[1] / DEFAULT_COMPATIBILITY_APPROVAL
    ).read_text(encoding="utf-8")
    raw = raw.replace('acceptance_status = "accepted"', 'acceptance_status = "pending"')
    raw = raw.replace('accepted_by = "researcher_user"', 'accepted_by = ""')
    raw = raw.replace('accepted_on = "2026-08-29"', 'accepted_on = ""')
    path = tmp_path / "pending-compatibility-approval.toml"
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


def _provider_body_without_usage() -> str:
    return json.dumps(
        {
            "model": ENVIRONMENT["EVICON_LLM_MODEL"],
            "choices": [
                {
                    "message": {
                        "content": json.dumps(_valid_content(), sort_keys=True)
                    },
                    "finish_reason": "stop",
                }
            ],
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def _run(
    transport: FakeTransport,
    tmp_path: Path,
    *,
    approval_path: Path | None = None,
    receipt_name: str = "compatibility-receipt.json",
):
    return execute_compatibility_v1(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=transport,
        approval_path=approval_path or DEFAULT_COMPATIBILITY_APPROVAL,
        receipt_path=tmp_path / receipt_name,
    )


def _claim_path(receipt_path: Path) -> Path:
    return receipt_path.with_name(f".{receipt_path.name}.attempt-claim")


def test_default_path_is_offline_and_does_not_read_environment_or_transport(
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

    result = execute_compatibility_v1(
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
    assert result.safety.comprehension_calibration_executed is False
    assert result.safety.behavior_protocol_executed is False
    assert not list(tmp_path.iterdir())
    assert set(result.model_dump(by_alias=True)) == {
        "status",
        "model",
        "finish_reason",
        "parser_valid",
        "http_status_class",
        "transport_category",
        "token_usage",
        "latency_ms",
        "schema",
        "attempt_count",
        "safety",
    }


def test_pending_approval_blocks_before_environment_or_transport_access(
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

    result = execute_compatibility_v1(
        allow_network=True,
        environment=ExplodingEnvironment(),
        transport=FailTransport(),
        approval_path=_pending_approval(tmp_path),
        receipt_path=tmp_path / "must-not-exist.json",
    )
    assert result.status == "blocked"
    assert result.transport_category == "compatibility_approval_not_accepted"
    assert result.attempt_count == 0
    assert result.safety.api_key_read is False
    assert result.safety.provider_constructed is False


def test_valid_2xx_uses_exact_schema_parameters_and_one_call(tmp_path: Path) -> None:
    approval = _accepted_approval(tmp_path)
    transport = FakeTransport([TransportResponse(200, _provider_body())])
    result = _run(transport, tmp_path, approval_path=approval)
    assert result.status == "completed"
    assert result.parser_valid is True
    assert result.finish_reason == "stop"
    assert result.http_status_class == "2xx"
    assert result.token_usage.model_dump() == {
        "prompt_tokens": 90,
        "completion_tokens": 32,
        "total_tokens": 122,
    }
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
            "name": SCHEMA_NAME,
            "strict": True,
            "schema": RESPONSE_SCHEMA,
        },
    }
    assert RESPONSE_SCHEMA == SourceRelationComprehensionResponse.model_json_schema(
        mode="validation"
    )
    public_request = payload["messages"][1]["content"].lower()
    for forbidden in (
        "source-relation-v1-c99",
        "source-relation-v1-s01",
        "20261401",
        "ground_truth",
        "evaluator_private",
    ):
        assert forbidden not in public_request


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
        json.dumps(_valid_content() | {"source_relation": "same_shown_root"}),
        json.dumps(
            _valid_content() | {"source_structure_alone_decides_truth": True}
        ),
    ],
    ids=[
        "malformed-inner-json",
        "unknown-field",
        "unexposed-content-id",
        "wrong-relation",
        "truth-inference",
    ],
)
def test_invalid_structural_outputs_stop_after_one_call(
    content: str,
    tmp_path: Path,
) -> None:
    transport = FakeTransport(
        [
            TransportResponse(200, _provider_body(content=content)),
            TransportResponse(200, _provider_body()),
        ]
    )
    result = _run(transport, tmp_path, approval_path=_accepted_approval(tmp_path))
    assert result.status == "invalid_response"
    assert result.parser_valid is False
    assert result.transport_category == "invalid_provider_response"
    assert result.attempt_count == 1
    assert len(transport.calls) == 1


def test_finish_reason_length_is_invalid_without_retry(tmp_path: Path) -> None:
    transport = FakeTransport(
        [
            TransportResponse(
                200,
                _provider_body(finish_reason="length", completion_tokens=MAX_TOKENS),
            ),
            TransportResponse(200, _provider_body()),
        ]
    )
    result = _run(transport, tmp_path, approval_path=_accepted_approval(tmp_path))
    assert result.status == "invalid_response"
    assert result.parser_valid is True
    assert result.finish_reason == "length"
    assert result.transport_category == "invalid_provider_response"
    assert result.attempt_count == 1
    assert len(transport.calls) == 1


def test_missing_token_usage_is_invalid_without_retry(tmp_path: Path) -> None:
    transport = FakeTransport(
        [
            TransportResponse(200, _provider_body_without_usage()),
            TransportResponse(200, _provider_body()),
        ]
    )
    result = _run(transport, tmp_path, approval_path=_accepted_approval(tmp_path))
    assert result.status == "invalid_response"
    assert result.parser_valid is True
    assert result.transport_category == "invalid_token_usage"
    assert result.attempt_count == 1
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    ("outcome", "category", "status_class"),
    [
        (TimeoutError("sensitive-timeout"), "timeout", None),
        (ConnectionError("sensitive-connection"), "connection_failure", None),
        (
            TransportResponse(401, '{"error":{"message":"sensitive-auth"}}'),
            "authentication_failed",
            "4xx",
        ),
        (
            TransportResponse(429, '{"error":{"message":"sensitive-rate"}}'),
            "rate_limited",
            "4xx",
        ),
        (
            TransportResponse(500, '{"error":{"message":"sensitive-server"}}'),
            "http_server_error",
            "5xx",
        ),
        (
            TransportResponse(
                400,
                '{"error":{"message":"response_format json_schema unsupported secret"}}',
            ),
            "response_format_unsupported",
            "4xx",
        ),
    ],
)
def test_transport_failures_are_redacted_and_never_retried(
    outcome: TransportResponse | BaseException,
    category: str,
    status_class: str | None,
    tmp_path: Path,
) -> None:
    transport = FakeTransport([outcome, TransportResponse(200, _provider_body())])
    result = _run(transport, tmp_path, approval_path=_accepted_approval(tmp_path))
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
    transport = FakeTransport([TransportResponse(200, "malformed-provider-envelope")])
    result = _run(transport, tmp_path, approval_path=_accepted_approval(tmp_path))
    assert result.status == "provider_error"
    assert result.transport_category == "malformed_provider_response"
    assert result.http_status_class == "2xx"
    assert result.attempt_count == 1


def test_provider_model_reflection_never_enters_result_or_receipt(tmp_path: Path) -> None:
    approval = _accepted_approval(tmp_path)
    receipt_path = tmp_path / "compatibility-receipt.json"
    reflected_model = "provider-returned-untrusted-name"
    result = execute_compatibility_v1(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=FakeTransport(
            [TransportResponse(200, _provider_body(model_name=reflected_model))]
        ),
        approval_path=approval,
        receipt_path=receipt_path,
    )
    assert result.status == "completed"
    assert result.model == ENVIRONMENT["EVICON_LLM_MODEL"]
    assert reflected_model not in result.model_dump_json(by_alias=True)
    assert reflected_model not in receipt_path.read_text(encoding="utf-8")


def test_success_writes_only_a_bound_redacted_receipt(tmp_path: Path) -> None:
    approval = _accepted_approval(tmp_path)
    receipt_path = tmp_path / "compatibility-receipt.json"
    transport = FakeTransport([TransportResponse(200, _provider_body())])
    result = execute_compatibility_v1(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=transport,
        approval_path=approval,
        receipt_path=receipt_path,
    )
    assert result.status == "completed"
    assert result.safety.compatibility_receipt_written is True
    assert {item.name for item in tmp_path.iterdir() if item.is_file()} == {
        "compatibility-approval.toml",
        "compatibility-receipt.json",
    }
    receipt = load_receipt_v1(receipt_path, approval_path=approval)
    assert receipt.receipt_version == RECEIPT_VERSION
    assert receipt.compatibility_module_version == COMPATIBILITY_MODULE_VERSION
    assert receipt.compatibility_config_sha256 == sha256_file(
        DEFAULT_COMPATIBILITY_CONFIG
    )
    assert receipt.calibration_config_sha256 == sha256_file(DEFAULT_CONFIG)
    assert receipt.calibration_protocol_sha256
    assert receipt.compatibility_approval_sha256 == sha256_file(approval)
    assert receipt.compatibility_module_sha256 == sha256_file(DEFAULT_MODULE)
    assert receipt.protocol_version == PROTOCOL_VERSION
    assert receipt.template_version == TEMPLATE_VERSION
    assert receipt.schema_contract.schema_name == SCHEMA_NAME
    assert receipt.schema_contract.schema_version == SCHEMA_VERSION
    assert receipt.schema_contract.schema_sha256 == RESPONSE_SCHEMA_SHA256
    assert receipt.generation_parameters.model_dump() == {
        "max_tokens": 128,
        "temperature": 0.0,
        "seed": 20261401,
        "max_retries": 0,
        "timeout_seconds": 5.0,
        "reasoning_effort": "none",
    }
    serialized = receipt_path.read_text(encoding="utf-8").lower()
    for forbidden in (
        "test-only-secret",
        "authorization",
        "bearer",
        "system_prompt",
        "user_prompt",
        '"provider_metadata":',
        "maple transit",
        "message-a",
        "lineage-k7",
        "evidence-card-01",
    ):
        assert forbidden not in serialized


def test_existing_receipt_blocks_before_environment_or_transport_access(
    tmp_path: Path,
) -> None:
    approval = _accepted_approval(tmp_path)
    receipt_path = tmp_path / "existing.json"
    receipt_path.write_text("do-not-overwrite", encoding="utf-8")

    class ExplodingEnvironment(dict[str, str]):
        def get(self, key: str, default=None):
            raise AssertionError("environment must not be read")

    class FailTransport:
        def post(self, *args, **kwargs):
            raise AssertionError("transport must not be called")

    result = execute_compatibility_v1(
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


def test_retained_claim_blocks_a_second_attempt_before_environment_access(
    tmp_path: Path,
) -> None:
    approval = _accepted_approval(tmp_path)
    receipt_path = tmp_path / "claimed.json"
    first_transport = FakeTransport(
        [TransportResponse(200, _provider_body(content="not-json"))]
    )
    first = execute_compatibility_v1(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=first_transport,
        approval_path=approval,
        receipt_path=receipt_path,
    )
    assert first.status == "invalid_response"
    assert first.attempt_count == 1
    assert len(first_transport.calls) == 1
    assert _claim_path(receipt_path).exists()

    class ExplodingEnvironment(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            raise AssertionError("environment must not be read after a claim")

        def __iter__(self):
            raise AssertionError("environment must not be read after a claim")

        def __len__(self):
            raise AssertionError("environment must not be read after a claim")

        def get(self, key: str, default=None):
            raise AssertionError("environment must not be read after a claim")

    class FailTransport:
        def post(self, *args, **kwargs):
            raise AssertionError("transport must not be called after a claim")

    second = execute_compatibility_v1(
        allow_network=True,
        environment=ExplodingEnvironment(),
        transport=FailTransport(),
        approval_path=approval,
        receipt_path=receipt_path,
    )
    assert second.status == "blocked"
    assert second.transport_category == "compatibility_attempt_claim_exists"
    assert second.attempt_count == 0
    assert second.safety.api_key_read is False
    assert second.safety.provider_constructed is False


def test_binding_change_during_request_blocks_receipt_and_retains_claim(
    tmp_path: Path,
) -> None:
    approval = _accepted_approval(tmp_path)
    receipt_path = tmp_path / "binding-change.json"

    class ApprovalMutatingTransport(FakeTransport):
        def post(self, url, headers, payload, timeout_seconds):
            outcome = super().post(url, headers, payload, timeout_seconds)
            approval.write_text(
                approval.read_text(encoding="utf-8") + "\n# changed during request\n",
                encoding="utf-8",
            )
            return outcome

    transport = ApprovalMutatingTransport([TransportResponse(200, _provider_body())])
    result = execute_compatibility_v1(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=transport,
        approval_path=approval,
        receipt_path=receipt_path,
    )
    assert result.status == "blocked"
    assert result.transport_category == "configuration_changed"
    assert result.attempt_count == 1
    assert len(transport.calls) == 1
    assert not receipt_path.exists()
    assert _claim_path(receipt_path).exists()


def test_atomic_receipt_publication_never_overwrites_a_racing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approval = _accepted_approval(tmp_path)
    receipt_path = tmp_path / "racing-receipt.json"

    def competing_link(source: Path, target: Path) -> None:
        Path(target).write_text("interloper-record\n", encoding="utf-8")
        raise FileExistsError

    monkeypatch.setattr(compatibility_module.os, "link", competing_link)
    result = execute_compatibility_v1(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=FakeTransport([TransportResponse(200, _provider_body())]),
        approval_path=approval,
        receipt_path=receipt_path,
    )
    assert result.status == "blocked"
    assert result.transport_category == "compatibility_receipt_exists"
    assert receipt_path.read_text(encoding="utf-8") == "interloper-record\n"
    assert _claim_path(receipt_path).exists()
    assert not list(tmp_path.glob(".racing-receipt.json.*.tmp"))


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
    receipt_path = tmp_path / f"missing-{missing}.json"
    transport = FakeTransport([TransportResponse(200, _provider_body())])
    result = execute_compatibility_v1(
        allow_network=True,
        environment=environment,
        transport=transport,
        approval_path=_accepted_approval(tmp_path),
        receipt_path=receipt_path,
    )
    assert result.status == "provider_error"
    assert result.transport_category == category
    assert result.attempt_count == 0
    assert len(transport.calls) == 0
    assert not receipt_path.exists()
    assert not _claim_path(receipt_path).exists()


@pytest.mark.parametrize(
    "missing",
    ["EVICON_LLM_BASE_URL", "EVICON_LLM_MODEL"],
)
def test_missing_nonsecret_provider_settings_do_not_read_api_key(
    missing: str,
    tmp_path: Path,
) -> None:
    class ReadTrackingEnvironment(dict[str, str]):
        def __init__(self, values: dict[str, str]) -> None:
            super().__init__(values)
            self.requested_keys: list[str] = []

        def get(self, key: str, default=None):
            self.requested_keys.append(key)
            if key == "EVICON_LLM_API_KEY":
                raise AssertionError("API key must not be read")
            return super().get(key, default)

    environment = ReadTrackingEnvironment(
        {key: value for key, value in ENVIRONMENT.items() if key != missing}
    )
    transport = FakeTransport([TransportResponse(200, _provider_body())])
    result = execute_compatibility_v1(
        allow_network=True,
        environment=environment,
        transport=transport,
        approval_path=_accepted_approval(tmp_path),
        receipt_path=tmp_path / f"missing-nonsecret-{missing}.json",
    )
    assert result.status == "provider_error"
    assert result.transport_category == (
        "missing_base_url" if missing.endswith("BASE_URL") else "missing_model"
    )
    assert "EVICON_LLM_API_KEY" not in environment.requested_keys
    assert result.safety.api_key_read is False
    assert result.safety.provider_constructed is False
    assert not transport.calls


def test_receipt_schema_rejects_unknown_fields_and_duplicate_fields(tmp_path: Path) -> None:
    approval = _accepted_approval(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    result = execute_compatibility_v1(
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
        CompatibilityReceiptV1.model_validate(payload)

    serialized = receipt_path.read_text(encoding="utf-8")
    receipt_path.write_text(
        serialized.replace(
            '"status": "completed",',
            '"status": "completed",\n  "status": "completed",',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="source_relation_compatibility_receipt_invalid",
    ):
        load_receipt_v1(receipt_path, approval_path=approval)


def test_minimal_context_is_fresh_and_not_a_calibration_case() -> None:
    context = minimal_public_context_v1()
    assert context.case_id == "source-relation-v1-c99"
    assert context.scenario_id == "source-relation-v1-s01"
    assert len(context.visible_material) == 3
    assert tuple(item.content_id for item in context.public_root_assignments) == (
        "message-a",
        "message-b",
    )
    text = " ".join(item.public_summary for item in context.visible_material).lower()
    for forbidden in ("harborlink", "cedarworks", "wvs", "pilot", "ground truth"):
        assert forbidden not in text


def test_default_receipt_location_is_the_only_repository_target() -> None:
    assert DEFAULT_RECEIPT.startswith("outputs/study-locks/")
    assert DEFAULT_ATTEMPT_CLAIM.startswith("outputs/study-locks/")
    assert re.fullmatch(
        r"outputs/study-locks/conformity_source_relation_comprehension_compatibility\.v1\.json",
        DEFAULT_RECEIPT,
    )
