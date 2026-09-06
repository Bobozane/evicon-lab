from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

import pytest

from evicon.conformity_source_behavior_qualification_runner_v3 import (
    COMPLETION_RESERVATION_CAP,
    DEFAULT_APPROVAL,
    DEFAULT_CONFIG,
    DEFAULT_RUNNER,
    MAX_TOKENS,
    RECEIPT_NAME,
    REQUEST_CAP,
    RESPONSE_SCHEMA,
    SCHEMA_NAME,
    SAFE_AUDIT_NAME,
    SEEDS,
    TEMPLATE_VERSION,
    V3RunReceipt,
    execute_qualification_v3,
    load_execution_receipt,
    safe_preflight,
)
from evicon.conformity_source_behavior_qualification_v3_smoke import build_cases
from evicon.conformity_source_behavior_qualification_v3 import (
    PROTOCOL_VERSION,
    render_behavior_request_v3,
)
from evicon.conformity_identification import sha256_file
from evicon.openai_provider import TransportResponse


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = {
    "EVICON_LLM_BASE_URL": "https://provider.invalid/v1",
    "EVICON_LLM_MODEL": "gpt-5.6-luna",
    "EVICON_LLM_API_KEY": "test-only-secret",
}


class FakeTransport:
    """A deterministic transport that records wire calls but never opens a socket."""

    def __init__(
        self,
        outcome: TransportResponse | BaseException | Callable[[int], TransportResponse | BaseException],
    ) -> None:
        self.outcome = outcome
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
        outcome = self.outcome(len(self.calls)) if callable(self.outcome) else self.outcome
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _valid_response() -> dict[str, object]:
    return {
        "ordinal_judgment": "uncertain",
        "public_action": "withhold",
        "share_decision": "do_not_share",
        "confidence": 3,
        "content_ids_used": ["content-01", "content-03"],
        "share_content_id": None,
    }


def _provider_body(
    *,
    content: dict[str, object] | str | None = None,
    finish_reason: str = "stop",
    model: str = "gpt-5.6-luna",
    usage: dict[str, int] | None = None,
) -> str:
    if content is None:
        content = _valid_response()
    if isinstance(content, dict):
        content = json.dumps(content, ensure_ascii=True, sort_keys=True)
    if usage is None:
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    return json.dumps(
        {
            "model": model,
            "choices": [
                {"message": {"content": content}, "finish_reason": finish_reason}
            ],
            "usage": usage,
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def _replace_toml_value(raw: str, key: str, value: str) -> str:
    pattern = rf"(?m)^{re.escape(key)}\s*=\s*.*$"
    replacement = f'{key} = "{value}"'
    updated, count = re.subn(pattern, replacement, raw)
    assert count == 1, key
    return updated


def _approval(tmp_path: Path, *, accepted: bool) -> Path:
    """Create a test-only approval bound to the current production config."""

    raw = (ROOT / DEFAULT_APPROVAL).read_text(encoding="utf-8")
    raw = _replace_toml_value(raw, "acceptance_status", "accepted" if accepted else "pending")
    raw = _replace_toml_value(raw, "accepted_by", "test-reviewer" if accepted else "")
    raw = _replace_toml_value(raw, "accepted_on", "2026-08-30" if accepted else "")
    raw = _replace_toml_value(raw, "config_sha256", sha256_file(DEFAULT_CONFIG))
    raw = _replace_toml_value(raw, "runner_sha256", sha256_file(DEFAULT_RUNNER))
    path = tmp_path / ("accepted-runner-v3.toml" if accepted else "pending-runner-v3.toml")
    path.write_text(raw, encoding="utf-8")
    return path


def _run(
    tmp_path: Path,
    transport: FakeTransport,
    *,
    approval: Path | None = None,
    environment: Mapping[str, str] | None = ENVIRONMENT,
    output_name: str = "qualification",
    claim_name: str = "attempt.claim",
):
    return execute_qualification_v3(
        allow_network=True,
        confirm_run=True,
        confirm_request_cap=REQUEST_CAP,
        confirm_completion_reservation_cap=COMPLETION_RESERVATION_CAP,
        environment=environment,
        transport=transport,
        approval_path=approval or _approval(tmp_path, accepted=True),
        output_root=tmp_path / output_name,
        attempt_claim_path=tmp_path / claim_name,
    )


def test_default_path_is_offline_without_environment_transport_or_files(tmp_path: Path) -> None:
    class ExplodingEnvironment(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            raise AssertionError("environment must not be read")

        def __iter__(self):
            raise AssertionError("environment must not be read")

        def __len__(self):
            raise AssertionError("environment must not be read")

        def get(self, key: str, default=None):
            raise AssertionError("environment must not be read")

    class ExplodingTransport:
        def post(self, *args, **kwargs):
            raise AssertionError("transport must not be called")

    result = execute_qualification_v3(
        environment=ExplodingEnvironment(),
        transport=ExplodingTransport(),
        output_root=tmp_path / "must-not-exist",
        attempt_claim_path=tmp_path / "must-not-claim",
    )
    assert result.status == "network_disabled"
    assert result.qualification_status == "not_run"
    assert result.error_code == "network_disabled"
    assert result.safety.network_enabled is False
    assert result.safety.api_key_read is False
    assert result.safety.provider_constructed is False
    assert not list(tmp_path.iterdir())


def test_safe_preflight_is_read_only_and_pending_by_default() -> None:
    report = safe_preflight()
    assert report["status"] == "source_behavior_v3_runner_preflight_ready"
    assert report["approval_status"] == "pending"
    assert report["network"] == "disabled"
    assert report["api_key_read"] is False
    assert report["provider_constructed"] is False
    assert report["results_written"] is False
    assert report["blocking_reasons"] == ["runner_execution_approval_required"]
    assert report["case_count"] == REQUEST_CAP == 36


def test_pending_approval_blocks_before_environment_claim_or_transport(tmp_path: Path) -> None:
    class ExplodingEnvironment(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            raise AssertionError("environment must not be read")

        def __iter__(self):
            raise AssertionError("environment must not be read")

        def __len__(self):
            raise AssertionError("environment must not be read")

        def get(self, key: str, default=None):
            raise AssertionError("environment must not be read")

    transport = FakeTransport(TransportResponse(200, _provider_body()))
    result = _run(
        tmp_path,
        transport,
        approval=_approval(tmp_path, accepted=False),
        environment=ExplodingEnvironment(),
    )
    assert result.status == "blocked"
    assert result.qualification_status == "execution_blocked"
    assert result.error_code == "source_behavior_v3_runner_approval_not_accepted"
    assert result.claim_cleanup_status == "not_applicable"
    assert not transport.calls
    assert not (tmp_path / "qualification").exists()
    assert not (tmp_path / "attempt.claim").exists()


def test_confirmation_caps_block_before_environment_or_transport(tmp_path: Path) -> None:
    class ExplodingEnvironment(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            raise AssertionError("environment must not be read")

        def __iter__(self):
            raise AssertionError("environment must not be read")

        def __len__(self):
            raise AssertionError("environment must not be read")

        def get(self, key: str, default=None):
            raise AssertionError("environment must not be read")

    transport = FakeTransport(TransportResponse(200, _provider_body()))
    result = execute_qualification_v3(
        allow_network=True,
        confirm_run=True,
        confirm_request_cap=REQUEST_CAP - 1,
        confirm_completion_reservation_cap=COMPLETION_RESERVATION_CAP,
        environment=ExplodingEnvironment(),
        transport=transport,
        approval_path=_approval(tmp_path, accepted=True),
        output_root=tmp_path / "qualification",
        attempt_claim_path=tmp_path / "attempt.claim",
    )
    assert result.status == "blocked"
    assert result.error_code == "source_behavior_v3_runner_request_cap_confirmation_mismatch"
    assert not transport.calls
    assert not (tmp_path / "qualification").exists()
    assert not (tmp_path / "attempt.claim").exists()


def test_success_runs_exact_36_cases_with_fixed_wire_contract_and_redacted_outputs(
    tmp_path: Path,
) -> None:
    approval = _approval(tmp_path, accepted=True)
    transport = FakeTransport(TransportResponse(200, _provider_body()))
    result = _run(tmp_path, transport, approval=approval)

    assert result.status == "completed"
    assert result.qualification_status == "source_behavior_qualified"
    assert result.logical_request_count == REQUEST_CAP
    assert result.transport_attempt_count == REQUEST_CAP
    assert result.completed_case_count == REQUEST_CAP
    assert result.parser_invalid_count == 0
    assert result.claim_cleanup_status == "released"
    assert result.safety.network_enabled is True
    assert result.safety.provider_constructed is True
    assert result.safety.api_key_read is True
    assert result.safety.qualification_executed is True
    assert result.safety.safe_case_audits_written is True
    assert result.safety.receipt_written is True
    assert result.safety.full_response_saved is False
    assert result.safety.prompt_saved is False
    assert result.safety.headers_saved is False
    assert result.safety.provider_metadata_saved is False
    assert result.safety.request_ledger_written is False
    assert len(transport.calls) == REQUEST_CAP

    assert {call["timeout_seconds"] for call in transport.calls} == {5.0}
    assert {call["payload"]["max_tokens"] for call in transport.calls} == {MAX_TOKENS}
    assert {call["payload"]["temperature"] for call in transport.calls} == {0.2}
    assert {call["payload"]["reasoning_effort"] for call in transport.calls} == {"none"}
    assert {call["payload"]["seed"] for call in transport.calls} == set(SEEDS)
    assert {call["payload"]["response_format"]["type"] for call in transport.calls} == {
        "json_schema"
    }
    for call in transport.calls:
        assert call["url"] == "https://provider.invalid/v1/chat/completions"
        assert call["headers"]["Content-Type"] == "application/json"
        schema = call["payload"]["response_format"]["json_schema"]
        assert schema["name"] == SCHEMA_NAME
        assert schema["strict"] is True
        assert schema["schema"] == RESPONSE_SCHEMA
        # The wire request can contain public synthetic material, but never a
        # private condition label or credential.
        wire = json.dumps(call["payload"], ensure_ascii=True).lower()
        for forbidden in ("evaluator_private", "ground_truth", "test-only-secret", "api_key"):
            assert forbidden not in wire

    output = tmp_path / "qualification"
    receipt_path = output / RECEIPT_NAME
    audit_path = output / SAFE_AUDIT_NAME
    receipt = load_execution_receipt(receipt_path, approval_path=approval)
    assert isinstance(receipt, V3RunReceipt)
    assert receipt.status == "source_behavior_qualified"
    assert receipt.logical_request_count == REQUEST_CAP
    assert receipt.completed_case_count == REQUEST_CAP
    assert receipt.transport_attempt_count == REQUEST_CAP
    assert receipt.parser_invalid_count == 0
    assert receipt.schema_contract.schema_name == SCHEMA_NAME
    assert receipt.schema_contract.template_version == TEMPLATE_VERSION
    assert receipt.generation_parameters.max_tokens == MAX_TOKENS
    assert receipt.generation_parameters.temperature == 0.2
    assert receipt.generation_parameters.max_retries == 0
    assert receipt.generation_parameters.timeout_seconds == 5.0
    assert receipt.generation_parameters.logical_request_cap == REQUEST_CAP
    assert receipt.generation_parameters.completion_reservation_cap == COMPLETION_RESERVATION_CAP
    assert receipt.safe_case_audit_sha256 is not None
    assert not (tmp_path / "attempt.claim").exists()

    lines = [line for line in audit_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == REQUEST_CAP
    audits = [json.loads(line) for line in lines]
    assert len({item["case_id"] for item in audits}) == REQUEST_CAP
    assert {item["projection"] for item in audits} == {
        "source_free",
        "same_root",
        "independent_roots",
    }
    assert {item["seed"] for item in audits} == set(SEEDS)
    for item in audits:
        assert "content_ids_used" not in item
        assert "public_summary" not in item
        assert "target_claim" not in item
        assert "source_root_id" not in json.dumps(item)

    persisted = (receipt_path.read_text(encoding="utf-8") + audit_path.read_text(encoding="utf-8")).lower()
    for forbidden in (
        "test-only-secret",
        "authorization",
        "bearer",
        "system_prompt",
        "user_prompt",
        '"provider_metadata":',
        "content_ids_used",
        "public_summary",
        "target_claim",
        "source_root_id",
    ):
        assert forbidden not in persisted


@pytest.mark.parametrize(
    ("outcome", "error_code", "stage", "parser_invalid"),
    [
        (
            TransportResponse(200, _provider_body(content="not-json")),
            "malformed_json",
            "response",
            1,
        ),
        (
            TransportResponse(
                200,
                _provider_body(content=_valid_response() | {"unknown_field": "forbidden"}),
            ),
            "invalid_schema",
            "response",
            1,
        ),
        (
            TransportResponse(
                200,
                _provider_body(content=_valid_response() | {"content_ids_used": ["content-99"]}),
            ),
            "unavailable_content_id",
            "response",
            1,
        ),
        (
            TransportResponse(200, _provider_body(finish_reason="length")),
            "finish_reason_invalid",
            "response",
            0,
        ),
        (TimeoutError("sensitive-timeout"), "timeout", "transport", 0),
        (ConnectionError("sensitive-connection"), "connection_failure", "transport", 0),
        (
            TransportResponse(401, '{"error":{"message":"sensitive-auth"}}'),
            "authentication_failed",
            "transport",
            0,
        ),
        (
            TransportResponse(429, '{"error":{"message":"sensitive-rate"}}'),
            "rate_limited",
            "transport",
            0,
        ),
        (
            TransportResponse(500, '{"error":{"message":"sensitive-server"}}'),
            "http_server_error",
            "transport",
            0,
        ),
        (
            TransportResponse(
                400,
                '{"error":{"message":"response_format json_schema unsupported secret"}}',
            ),
            "response_format_unsupported",
            "transport",
            0,
        ),
        (TransportResponse(200, "not-provider-json"), "malformed_provider_response", "transport", 0),
    ],
    ids=[
        "malformed-json",
        "unknown-field",
        "unavailable-content-id",
        "finish-reason-length",
        "timeout",
        "connection",
        "401",
        "429",
        "500",
        "response-format-unsupported",
        "malformed-provider-envelope",
    ],
)
def test_invalid_or_transport_outcomes_stop_after_one_call_and_are_redacted(
    tmp_path: Path,
    outcome: TransportResponse | BaseException,
    error_code: str,
    stage: str,
    parser_invalid: int,
) -> None:
    transport = FakeTransport(outcome)
    result = _run(tmp_path, transport)

    assert result.status == "failed"
    assert result.qualification_status == "execution_failed"
    assert result.error_code == error_code
    assert result.transport_attempt_count == 1
    assert result.completed_case_count == 0
    assert result.parser_invalid_count == parser_invalid
    assert len(transport.calls) == 1
    assert result.claim_cleanup_status == "retained_after_attempt"
    assert result.receipt_path is not None
    if result.safe_case_audit_path is not None:
        assert not (tmp_path / "qualification" / SAFE_AUDIT_NAME).exists()

    receipt_path = tmp_path / "qualification" / RECEIPT_NAME
    receipt = load_execution_receipt(receipt_path, approval_path=_approval(tmp_path, accepted=True))
    assert receipt.status == "execution_failed"
    assert receipt.failure_audit is not None
    assert receipt.failure_audit.stage == stage
    assert receipt.failure_audit.error_category == error_code
    if error_code in {"timeout", "connection_failure"}:
        assert receipt.failure_audit.http_status_class is None
    elif error_code == "malformed_provider_response":
        assert receipt.failure_audit.http_status_class == "2xx"
    serialized = receipt_path.read_text(encoding="utf-8").lower()
    for forbidden in (
        "sensitive-timeout",
        "sensitive-connection",
        "sensitive-auth",
        "sensitive-rate",
        "sensitive-server",
        "unsupported secret",
        "test-only-secret",
        "authorization",
        "bearer",
        "system_prompt",
        "user_prompt",
        '"provider_metadata":',
    ):
        assert forbidden not in serialized


def test_missing_api_key_is_pre_request_failure_and_releases_claim(tmp_path: Path) -> None:
    transport = FakeTransport(TransportResponse(200, _provider_body()))
    missing_key = {
        "EVICON_LLM_BASE_URL": ENVIRONMENT["EVICON_LLM_BASE_URL"],
        "EVICON_LLM_MODEL": ENVIRONMENT["EVICON_LLM_MODEL"],
    }
    result = _run(tmp_path, transport, environment=missing_key)
    assert result.status == "failed"
    assert result.error_code == "missing_api_key"
    assert result.transport_attempt_count == 0
    assert result.completed_case_count == 0
    assert result.claim_cleanup_status == "released"
    assert result.safety.provider_constructed is False
    assert result.safety.api_key_read is True
    assert not transport.calls
    assert not (tmp_path / "qualification").exists()
    assert not (tmp_path / "attempt.claim").exists()


def test_first_attempt_claim_blocks_a_second_run(tmp_path: Path) -> None:
    approval = _approval(tmp_path, accepted=True)
    first_transport = FakeTransport(TimeoutError("sensitive-timeout"))
    first = _run(tmp_path, first_transport, approval=approval)
    assert first.status == "failed"
    assert first.claim_cleanup_status == "retained_after_attempt"
    assert (tmp_path / "attempt.claim").exists()

    second_transport = FakeTransport(TransportResponse(200, _provider_body()))
    second = _run(
        tmp_path,
        second_transport,
        approval=approval,
        output_name="qualification-second",
    )
    assert second.status == "blocked"
    assert second.error_code == "source_behavior_v3_runner_attempt_claim_exists"
    assert not second_transport.calls


def test_binding_change_after_first_call_writes_failed_receipt_and_keeps_claim(
    tmp_path: Path,
) -> None:
    approval = _approval(tmp_path, accepted=True)

    def mutate_approval(call_count: int) -> TransportResponse:
        if call_count == 1:
            approval.write_text(approval.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return TransportResponse(200, _provider_body())

    transport = FakeTransport(mutate_approval)
    result = _run(tmp_path, transport, approval=approval)
    assert result.status == "failed"
    assert result.error_code == "source_behavior_v3_runner_binding_changed_before_request"
    assert result.completed_case_count == 1
    assert result.transport_attempt_count == 1
    assert result.claim_cleanup_status == "retained_after_attempt"
    assert len(transport.calls) == 1
    assert (tmp_path / "attempt.claim").exists()
    receipt = json.loads((tmp_path / "qualification" / RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["status"] == "execution_failed"
    assert receipt["failure_audit"]["stage"] == "binding"
    assert receipt["bindings_unchanged_before_receipt"] is False


def test_receipt_and_audit_tampering_is_rejected(tmp_path: Path) -> None:
    approval = _approval(tmp_path, accepted=True)
    transport = FakeTransport(TransportResponse(200, _provider_body()))
    result = _run(tmp_path, transport, approval=approval)
    assert result.status == "completed"
    receipt_path = tmp_path / "qualification" / RECEIPT_NAME
    audit_path = tmp_path / "qualification" / SAFE_AUDIT_NAME

    original_receipt = receipt_path.read_text(encoding="utf-8")
    receipt_payload = json.loads(original_receipt)
    receipt_payload["prompt"] = "must-not-be-accepted"
    receipt_path.write_text(json.dumps(receipt_payload), encoding="utf-8")
    with pytest.raises(Exception):
        load_execution_receipt(receipt_path, approval_path=approval)

    # Restore the receipt, then tamper with the append-only audit.
    receipt_path.write_text(original_receipt, encoding="utf-8")
    audit_path.write_text(audit_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(Exception):
        load_execution_receipt(receipt_path, approval_path=approval)


def test_case_builder_and_request_template_are_the_bound_v3_contract() -> None:
    cases = build_cases()
    assert len(cases) == REQUEST_CAP
    assert len({case.case_id for case in cases}) == REQUEST_CAP
    request = render_behavior_request_v3(cases[0])
    assert request.temperature == 0.2
    assert request.max_tokens == MAX_TOKENS
    assert request.seed in SEEDS
    assert request.metadata["protocol_version"] == PROTOCOL_VERSION
    assert request.metadata["template_version"] == TEMPLATE_VERSION
    payload = json.loads(request.user_prompt)
    assert tuple(item["content_id"] for item in payload["visible_content"]) == (
        "content-01",
        "content-02",
        "content-03",
        "content-04",
    )
