from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from evicon.conformity_identification import sha256_file
from evicon.conformity_source_relation_comprehension_calibration_v1 import (
    DEFAULT_CONFIG as CALIBRATION_CONFIG,
    RESPONSE_SCHEMA_SHA256 as CANONICAL_SCHEMA_SHA256,
    load_config as load_calibration_config,
)
from evicon.conformity_source_relation_comprehension_compatibility_provider_subset_short_name_v4 import (
    DEFAULT_CONFIG as V4_CONFIG,
    DEFAULT_MODULE as V4_MODULE,
    PROVIDER_RESPONSE_SCHEMA,
    PROVIDER_RESPONSE_SCHEMA_SHA256,
    SHORT_PROVIDER_SCHEMA_NAME,
    execute_compatibility_provider_subset_short_name_v4,
)
from evicon.conformity_source_relation_comprehension_provider_subset_short_name_v4_approval import (
    DEFAULT_APPROVAL as V4_APPROVAL,
    PROVIDER_SCHEMA_MODULE,
)
from evicon.conformity_source_relation_comprehension_provider_subset_v2 import (
    PROVIDER_SCHEMA_VERSION,
)
from evicon.conformity_source_relation_comprehension_qualification_provider_subset_short_name_v2 import (
    COMPLETION_RESERVATION_CAP,
    DEFAULT_RUNNER,
    MAX_RETRIES,
    MAX_TOKENS,
    REQUEST_CAP,
    SEED,
    TEMPERATURE,
    TIMEOUT_SECONDS,
    _execute_qualification_v2,
    execute_qualification_v2,
    load_qualification_receipt,
    safe_preflight,
)
from evicon.openai_provider import TransportResponse


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = {
    "EVICON_LLM_BASE_URL": "https://provider.invalid/v1",
    "EVICON_LLM_MODEL": "source-relation-qualification-v4-test-model",
    "EVICON_LLM_API_KEY": "test-only-secret",
}


class RecordingTransport:
    def __init__(
        self,
        responder: Callable[[int, dict[str, object]], TransportResponse | BaseException],
    ) -> None:
        self._responder = responder
        self.calls: list[dict[str, object]] = []

    def post(self, url, headers, payload, timeout_seconds):  # type: ignore[no-untyped-def]
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        outcome = self._responder(len(self.calls), payload)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _provider_body(
    content: str,
    *,
    finish_reason: str = "stop",
    completion_tokens: int = 30,
) -> TransportResponse:
    return TransportResponse(
        200,
        json.dumps(
            {
                "model": "untrusted-provider-model-name",
                "choices": [
                    {"message": {"content": content}, "finish_reason": finish_reason}
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": completion_tokens,
                    "total_tokens": 100 + completion_tokens,
                },
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
    )


def _compatibility_content() -> str:
    return json.dumps(
        {
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
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def _qualification_content(payload: dict[str, object]) -> str:
    messages = payload["messages"]
    assert isinstance(messages, list)
    user_message = messages[1]
    assert isinstance(user_message, dict)
    public = json.loads(user_message["content"])
    assert isinstance(public, dict)
    material = public["visible_material"]
    assignments = public["public_root_relations"]
    assert isinstance(material, list)
    assert isinstance(assignments, list)
    roots = [item["source_root_id"] for item in assignments]
    if all(root is None for root in roots):
        relation = "not_shown"
    elif len(set(roots)) == 1:
        relation = "same_shown_root"
    else:
        relation = "distinct_shown_roots"
    evidence_ids = [
        item["content_id"]
        for item in material
        if item["material_type"] == "direct_evidence_card"
    ]
    return json.dumps(
        {
            "visible_social_message_count": 2,
            "source_relation": relation,
            "visible_social_root_count": len(
                {root for root in roots if root is not None}
            ),
            "social_root_assignments": assignments,
            "direct_evidence_card_ids": evidence_ids,
            "support_kind": (
                "direct_evidence_present" if evidence_ids else "social_reports_only"
            ),
            "source_structure_alone_decides_truth": False,
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def _copy_v4_approval(tmp_path: Path) -> Path:
    target = tmp_path / "v4-compatibility-approval.toml"
    target.write_text((ROOT / V4_APPROVAL).read_text(encoding="utf-8"), encoding="utf-8")
    return target


def _write_toml_pair(
    tmp_path: Path,
    *,
    v4_receipt: Path,
    v4_approval: Path,
    accepted: bool,
) -> tuple[Path, Path]:
    calibration = load_calibration_config(CALIBRATION_CONFIG)
    config_path = tmp_path / "qualification-config.toml"
    quote = json.dumps
    config_text = "\n".join(
        (
            'qualification_id = "evicon-source-relation-comprehension-qualification-provider-subset-short-name-v2"',
            'qualification_version = "conformity_source_relation_comprehension_qualification_provider_subset_short_name.v2"',
            'runner_version = "conformity_source_relation_comprehension_qualification_provider_subset_short_name_runner.v2"',
            'status = "awaiting_explicit_authorization"',
            "development_only = true",
            "not_paper_result = true",
            "no_causal_conclusion = true",
            "network_enabled_by_default = false",
            "explicit_allow_network_required = true",
            "explicit_run_confirmation_required = true",
            "real_provider_permitted_after_authorization = true",
            "api_key_read_permitted_after_authorization = true",
            "safe_artifact_write_permitted_after_authorization = true",
            "evaluator_private_truth_available = false",
            "historical_artifacts_excluded = true",
            "public_structure_task_only = true",
            "compatibility_receipt_required = true",
            f"v4_compatibility_config_path = {quote(V4_CONFIG)}",
            f"v4_compatibility_config_sha256 = {quote(sha256_file(V4_CONFIG))}",
            f"v4_compatibility_approval_path = {quote(str(v4_approval))}",
            f"v4_compatibility_approval_sha256 = {quote(sha256_file(v4_approval))}",
            f"v4_compatibility_module_path = {quote(V4_MODULE)}",
            f"v4_compatibility_module_sha256 = {quote(sha256_file(V4_MODULE))}",
            f"compatibility_receipt_path = {quote(str(v4_receipt))}",
            'compatibility_receipt_version = "conformity_source_relation_comprehension_compatibility_provider_subset_short_name_receipt.v4"',
            f"compatibility_receipt_sha256 = {quote(sha256_file(v4_receipt))}",
            f"calibration_config_path = {quote(CALIBRATION_CONFIG)}",
            f"calibration_config_sha256 = {quote(sha256_file(CALIBRATION_CONFIG))}",
            f"calibration_protocol_sha256 = {quote(calibration.protocol_sha256)}",
            'calibration_protocol_version = "conformity_source_relation_comprehension_calibration.v1"',
            'template_version = "conformity_source_relation_comprehension_turn.v1"',
            'canonical_schema_name = "conformity_source_relation_comprehension_response_v1"',
            f"canonical_schema_sha256 = {quote(CANONICAL_SCHEMA_SHA256)}",
            f"provider_schema_name = {quote(SHORT_PROVIDER_SCHEMA_NAME)}",
            f"provider_schema_version = {quote(PROVIDER_SCHEMA_VERSION)}",
            f"provider_schema_sha256 = {quote(PROVIDER_RESPONSE_SCHEMA_SHA256)}",
            f"provider_schema_module_path = {quote(PROVIDER_SCHEMA_MODULE)}",
            f"provider_schema_module_sha256 = {quote(sha256_file(PROVIDER_SCHEMA_MODULE))}",
            f"runner_path = {quote(DEFAULT_RUNNER)}",
            f"runner_sha256 = {quote(sha256_file(DEFAULT_RUNNER))}",
            f"required_model = {quote(ENVIRONMENT['EVICON_LLM_MODEL'])}",
            "case_count = 24",
            "logical_request_cap = 24",
            "completion_reservation_cap = 3072",
            "max_tokens = 128",
            "temperature = 0.0",
            "seed = 20261401",
            "max_retries = 0",
            "timeout_seconds = 15.0",
            'reasoning_effort = "none"',
            'output_root = "outputs/conformity-source-relation-comprehension-qualification-provider-subset-short-name-v2"',
            'attempt_claim_path = "outputs/study-locks/conformity_source_relation_comprehension_qualification_provider_subset_short_name.v2.claim"',
            "",
        )
    )
    config_path.write_text(config_text, encoding="utf-8")
    approval_path = tmp_path / "qualification-approval.toml"
    approval_text = "\n".join(
        (
            'approval_id = "evicon-source-relation-comprehension-qualification-provider-subset-short-name-v2"',
            'approval_version = "conformity_source_relation_comprehension_qualification_provider_subset_short_name_approval.v2"',
            f'acceptance_status = {quote("accepted" if accepted else "pending")}',
            f'accepted_by = {quote("test-reviewer" if accepted else "")}',
            f'accepted_on = {quote("2026-08-30" if accepted else "")}',
            f"qualification_config_sha256 = {quote(sha256_file(config_path))}",
            f"calibration_config_sha256 = {quote(sha256_file(CALIBRATION_CONFIG))}",
            f"calibration_protocol_sha256 = {quote(calibration.protocol_sha256)}",
            f"canonical_schema_sha256 = {quote(CANONICAL_SCHEMA_SHA256)}",
            f"provider_schema_sha256 = {quote(PROVIDER_RESPONSE_SCHEMA_SHA256)}",
            f"provider_schema_module_sha256 = {quote(sha256_file(PROVIDER_SCHEMA_MODULE))}",
            f"v4_compatibility_config_sha256 = {quote(sha256_file(V4_CONFIG))}",
            f"v4_compatibility_approval_sha256 = {quote(sha256_file(v4_approval))}",
            f"v4_compatibility_module_sha256 = {quote(sha256_file(V4_MODULE))}",
            f"compatibility_receipt_sha256 = {quote(sha256_file(v4_receipt))}",
            f"runner_sha256 = {quote(sha256_file(DEFAULT_RUNNER))}",
            f"required_model = {quote(ENVIRONMENT['EVICON_LLM_MODEL'])}",
            "confirm_public_structure_task_only = true",
            "confirm_compatibility_receipt_required = true",
            "confirm_full_24_case_coverage_required = true",
            "confirm_no_retry_and_terminal_failure = true",
            "confirm_no_private_or_historical_material = true",
            "confirm_not_behavior_or_causal_evidence = true",
            "persistent_network_authorization_granted = false",
            "explicit_command_authorization_required = true",
            "",
        )
    )
    approval_path.write_text(approval_text, encoding="utf-8")
    return config_path, approval_path


def _fixture(
    tmp_path: Path, *, accepted: bool = True
) -> tuple[Path, Path, Path, Path]:
    v4_approval = _copy_v4_approval(tmp_path)
    v4_receipt = tmp_path / "v4-compatibility-receipt.json"
    compatibility_transport = RecordingTransport(
        lambda _index, _payload: _provider_body(_compatibility_content())
    )
    compatibility_result = execute_compatibility_provider_subset_short_name_v4(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=compatibility_transport,
        approval_path=v4_approval,
        receipt_path=v4_receipt,
    )
    assert compatibility_result.status == "completed"
    assert len(compatibility_transport.calls) == 1
    config, approval = _write_toml_pair(
        tmp_path,
        v4_receipt=v4_receipt,
        v4_approval=v4_approval,
        accepted=accepted,
    )
    return config, approval, v4_receipt, v4_approval


def _run(
    tmp_path: Path,
    transport: RecordingTransport,
    *,
    accepted: bool = True,
    environment: Mapping[str, str] | None = ENVIRONMENT,
) -> tuple[Any, RecordingTransport, tuple[Path, Path, Path, Path]]:
    bindings = _fixture(tmp_path, accepted=accepted)
    config, approval, v4_receipt, v4_approval = bindings
    result = _execute_qualification_v2(
        allow_network=True,
        confirm_run=True,
        confirm_request_cap=REQUEST_CAP,
        confirm_completion_reservation_cap=COMPLETION_RESERVATION_CAP,
        environment=environment,
        transport=transport,
        config_path=config,
        approval_path=approval,
        compatibility_receipt_path=v4_receipt,
        compatibility_approval_path=v4_approval,
        output_root=tmp_path / "qualification-output",
        attempt_claim_path=tmp_path / "qualification.claim",
    )
    return result, transport, bindings


def _all_created_text(tmp_path: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in tmp_path.rglob("*")
        if path.is_file()
    )


def test_default_execution_is_inert_and_does_not_read_environment_or_transport(
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

    result = execute_qualification_v2(
        allow_network=False,
        environment=ExplodingEnvironment(),
        transport=FailTransport(),
    )
    assert result.status == "network_disabled"
    assert result.qualification_status == "not_run"
    assert result.logical_request_count == 0
    assert result.transport_attempt_count == 0
    assert result.safety.api_key_read is False
    assert result.safety.provider_constructed is False
    assert not list(tmp_path.iterdir())


def test_pending_approval_blocks_before_environment_or_transport_access(
    tmp_path: Path,
) -> None:
    class ExplodingEnvironment(dict[str, str]):
        def get(self, key: str, default=None):
            raise AssertionError("environment must not be read")

    transport = RecordingTransport(lambda _index, _payload: AssertionError("called"))
    result, _, _ = _run(
        tmp_path,
        transport,
        accepted=False,
        environment=ExplodingEnvironment(),
    )
    assert result.status == "blocked"
    assert result.error_code == "source_relation_qualification_approval_not_accepted"
    assert not transport.calls
    assert not (tmp_path / "qualification-output").exists()
    assert not (tmp_path / "qualification.claim").exists()


def test_24_case_v4_wire_schema_and_canonical_parser_success(tmp_path: Path) -> None:
    transport = RecordingTransport(
        lambda _index, payload: _provider_body(_qualification_content(payload))
    )
    result, _, bindings = _run(tmp_path, transport)
    config, approval, v4_receipt, v4_approval = bindings
    assert result.status == "completed"
    assert result.qualification_status == "source_structure_qualified"
    assert result.logical_request_count == REQUEST_CAP == 24
    assert result.transport_attempt_count == REQUEST_CAP
    assert result.completed_case_count == REQUEST_CAP
    assert result.parser_invalid_count == 0
    assert len(transport.calls) == REQUEST_CAP
    assert result.safety.compatibility_receipt_loaded is True
    assert result.safety.qualification_executed is True
    assert result.safety.request_ledger_written is False
    assert result.safety.full_response_saved is False
    for call in transport.calls:
        payload = call["payload"]
        assert call["timeout_seconds"] == TIMEOUT_SECONDS == 15.0
        assert payload["max_tokens"] == MAX_TOKENS == 128
        assert payload["temperature"] == TEMPERATURE == 0.0
        assert payload["seed"] == SEED == 20261401
        assert payload["reasoning_effort"] == "none"
        assert MAX_RETRIES == 0
        schema = payload["response_format"]["json_schema"]
        assert schema["name"] == SHORT_PROVIDER_SCHEMA_NAME
        assert schema["strict"] is True
        assert schema["schema"] == PROVIDER_RESPONSE_SCHEMA

    receipt = load_qualification_receipt(
        Path(result.receipt_path or ""),
        config_path=config,
        approval_path=approval,
        compatibility_receipt_path=v4_receipt,
        compatibility_approval_path=v4_approval,
    )
    assert receipt.status == "source_structure_qualified"
    assert receipt.schema_contract.provider_schema_sha256 == PROVIDER_RESPONSE_SCHEMA_SHA256
    assert receipt.schema_contract.canonical_schema_sha256 == CANONICAL_SCHEMA_SHA256
    audits = (tmp_path / "qualification-output" / "safe_case_audits.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(audits) == REQUEST_CAP
    persisted = _all_created_text(tmp_path)
    for forbidden in (
        ENVIRONMENT["EVICON_LLM_API_KEY"],
        "Authorization",
        "Bearer",
        "message-a",
        "lineage-k7",
        "evidence-card-01",
        "public_summary",
    ):
        assert forbidden not in persisted


@pytest.mark.parametrize(
    "kind",
    ["malformed", "unknown_field", "unexposed_content", "length"],
)
def test_invalid_structural_response_is_terminal_and_claim_is_retained(
    tmp_path: Path, kind: str
) -> None:
    def responder(_index: int, payload: dict[str, object]) -> TransportResponse:
        if kind == "malformed":
            return _provider_body("not-json")
        if kind == "unknown_field":
            return _provider_body(_qualification_content(payload)[:-1] + ',"unknown":true}')
        if kind == "unexposed_content":
            content = json.loads(_qualification_content(payload))
            content["social_root_assignments"][1]["content_id"] = "message-x"
            return _provider_body(json.dumps(content, sort_keys=True))
        return _provider_body(
            _qualification_content(payload), finish_reason="length", completion_tokens=128
        )

    transport = RecordingTransport(responder)
    result, _, bindings = _run(tmp_path, transport)
    config, approval, v4_receipt, v4_approval = bindings
    assert result.status == "failed"
    assert result.logical_request_count == 1
    assert result.transport_attempt_count == 1
    assert len(transport.calls) == 1
    assert (tmp_path / "qualification.claim").exists()
    assert Path(result.receipt_path or "").is_file()
    persisted = _all_created_text(tmp_path)
    assert ENVIRONMENT["EVICON_LLM_API_KEY"] not in persisted

    second = _execute_qualification_v2(
        allow_network=True,
        confirm_run=True,
        confirm_request_cap=REQUEST_CAP,
        confirm_completion_reservation_cap=COMPLETION_RESERVATION_CAP,
        environment=ENVIRONMENT,
        transport=RecordingTransport(lambda _index, payload: _provider_body(_qualification_content(payload))),
        config_path=config,
        approval_path=approval,
        compatibility_receipt_path=v4_receipt,
        compatibility_approval_path=v4_approval,
        output_root=tmp_path / "qualification-output",
        attempt_claim_path=tmp_path / "qualification.claim",
    )
    assert second.status == "blocked"
    assert second.error_code == "source_relation_qualification_output_exists"


@pytest.mark.parametrize(
    ("outcome", "category"),
    [
        (TimeoutError("sensitive-timeout"), "timeout"),
        (ConnectionError("sensitive-connection"), "connection_failure"),
        (TransportResponse(401, '{"error":{"message":"sensitive-auth"}}'), "authentication_failed"),
        (TransportResponse(429, '{"error":{"message":"sensitive-rate"}}'), "rate_limited"),
        (TransportResponse(500, '{"error":{"message":"sensitive-server"}}'), "http_server_error"),
        (
            TransportResponse(400, '{"error":{"message":"response_format json_schema unsupported sensitive"}}'),
            "response_format_unsupported",
        ),
    ],
)
def test_transport_failures_are_one_shot_and_redacted(
    tmp_path: Path, outcome: TransportResponse | BaseException, category: str
) -> None:
    transport = RecordingTransport(lambda _index, _payload: outcome)
    result, _, _ = _run(tmp_path, transport)
    assert result.status == "failed"
    assert result.error_code == category
    assert result.transport_attempt_count == 1
    assert len(transport.calls) == 1
    assert result.safety.full_response_saved is False
    persisted = _all_created_text(tmp_path)
    for forbidden in ("sensitive-timeout", "sensitive-connection", "sensitive-auth", "sensitive-rate", "sensitive-server"):
        assert forbidden not in persisted


def test_binding_change_after_first_call_prevents_a_second_transport_call(
    tmp_path: Path,
) -> None:
    bindings_holder: dict[str, Path] = {}

    def responder(index: int, payload: dict[str, object]) -> TransportResponse:
        if index == 1:
            receipt = bindings_holder["receipt"]
            receipt.write_text(receipt.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return _provider_body(_qualification_content(payload))

    transport = RecordingTransport(responder)
    v4_approval = _copy_v4_approval(tmp_path)
    v4_receipt = tmp_path / "v4-compatibility-receipt.json"
    compatibility_transport = RecordingTransport(
        lambda _index, _payload: _provider_body(_compatibility_content())
    )
    compatibility_result = execute_compatibility_provider_subset_short_name_v4(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=compatibility_transport,
        approval_path=v4_approval,
        receipt_path=v4_receipt,
    )
    assert compatibility_result.status == "completed"
    config, approval = _write_toml_pair(
        tmp_path,
        v4_receipt=v4_receipt,
        v4_approval=v4_approval,
        accepted=True,
    )
    bindings_holder["receipt"] = v4_receipt
    result = _execute_qualification_v2(
        allow_network=True,
        confirm_run=True,
        confirm_request_cap=REQUEST_CAP,
        confirm_completion_reservation_cap=COMPLETION_RESERVATION_CAP,
        environment=ENVIRONMENT,
        transport=transport,
        config_path=config,
        approval_path=approval,
        compatibility_receipt_path=v4_receipt,
        compatibility_approval_path=v4_approval,
        output_root=tmp_path / "qualification-output",
        attempt_claim_path=tmp_path / "qualification.claim",
    )
    assert result.status == "failed"
    assert result.error_code == "source_relation_qualification_binding_changed_before_request"
    assert len(transport.calls) == 1
    assert (tmp_path / "qualification.claim").exists()


def test_safe_preflight_stays_offline_and_reports_pending_human_review(tmp_path: Path) -> None:
    config, approval, v4_receipt, v4_approval = _fixture(tmp_path, accepted=False)
    preflight = safe_preflight(
        config_path=config,
        approval_path=approval,
        compatibility_receipt_path=v4_receipt,
        compatibility_approval_path=v4_approval,
    )
    assert preflight["status"] == "source_relation_qualification_preflight_ready"
    assert preflight["network"] == "disabled"
    assert preflight["api_key_read"] is False
    assert preflight["provider_constructed"] is False
    assert preflight["results_written"] is False
    assert "human_approval_required" in preflight["blocking_reasons"]
