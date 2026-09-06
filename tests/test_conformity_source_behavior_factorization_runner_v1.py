from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from evicon.conformity_source_behavior_factorization_runner_v1 import (
    DEFAULT_CONFIG,
    FactorizationPilotReceipt,
    build_pilot_cases,
    execute_factorization_pilot,
    load_pilot_receipt,
    pilot_preflight,
)
from evicon.openai_provider import TransportResponse


ENVIRONMENT = {
    "EVICON_LLM_BASE_URL": "https://provider.invalid/v1",
    "EVICON_LLM_MODEL": "gpt-5.6-luna",
    "EVICON_LLM_API_KEY": "test-only-secret",
}


class FakeTransport:
    def __init__(self, outcome: TransportResponse | BaseException) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, object]] = []

    def post(self, url, headers, payload, timeout_seconds):
        self.calls.append({
            "url": url,
            "headers": dict(headers),
            "payload": payload,
            "timeout_seconds": timeout_seconds,
        })
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _valid_content() -> dict[str, object]:
    return {
        "ordinal_judgment": "uncertain",
        "public_action": "withhold",
        "share_decision": "do_not_share",
        "confidence": 3,
        "content_ids_used": ["content-initial"],
        "share_content_id": None,
    }


def _body(
    *,
    content: dict[str, object] | str | None = None,
    finish_reason: str = "stop",
    model: str = "gpt-5.6-luna",
) -> str:
    if content is None:
        content = _valid_content()
    if isinstance(content, dict):
        content = json.dumps(content, ensure_ascii=True, sort_keys=True)
    return json.dumps({
        "model": model,
        "choices": [{
            "message": {"content": content},
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }, ensure_ascii=True, sort_keys=True)


def _run(tmp_path: Path, transport: FakeTransport):
    return execute_factorization_pilot(
        allow_network=True,
        confirm_run=True,
        confirm_request_cap=32,
        confirm_completion_reservation_cap=8192,
        environment=ENVIRONMENT,
        transport=transport,
        output_root=tmp_path / "pilot",
    )


def test_default_preflight_is_offline_and_does_not_read_environment(tmp_path: Path) -> None:
    class ExplodingEnvironment(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            raise AssertionError("environment must not be read")

        def __iter__(self):
            raise AssertionError("environment must not be read")

        def __len__(self):
            raise AssertionError("environment must not be read")

        def get(self, key: str, default=None):
            raise AssertionError("environment must not be read")

    report = execute_factorization_pilot(
        allow_network=False,
        environment=ExplodingEnvironment(),
        transport=FakeTransport(TransportResponse(200, _body())),
        output_root=tmp_path / "must-not-exist",
    )
    assert report["status"] == "factorization_pilot_preflight_ready"
    assert report["network"] == "disabled"
    assert report["api_key_read"] is False
    assert report["provider_constructed"] is False
    assert report["results_written"] is False
    assert not list(tmp_path.iterdir())


def test_preflight_and_matrix_are_exact() -> None:
    report = pilot_preflight(DEFAULT_CONFIG)
    assert report["case_count"] == 32
    assert report["scenario_count"] == 2
    assert report["seed_count"] == 2
    assert report["condition_count"] == 8
    cases = build_pilot_cases(DEFAULT_CONFIG)
    assert len(cases) == 32
    assert len({case.group_id for case in cases}) == 4
    assert all(case.scenario_id in {"source-factor-v1-s01", "source-factor-v1-s02"} for case in cases)
    assert all(case.seed in {20261301, 20261302} for case in cases)


def test_network_flag_alone_does_not_read_environment_or_construct_provider(tmp_path: Path) -> None:
    class ExplodingEnvironment(dict[str, str]):
        def get(self, key: str, default=None):
            raise AssertionError("environment must not be read before exact confirmations")

    transport = FakeTransport(TransportResponse(200, _body()))
    result = execute_factorization_pilot(
        allow_network=True,
        confirm_run=True,
        confirm_request_cap=31,
        confirm_completion_reservation_cap=8192,
        environment=ExplodingEnvironment(),
        transport=transport,
        output_root=tmp_path / "pilot",
    )
    assert result.status == "blocked"
    assert result.error_code == "factorization_pilot_explicit_cap_confirmation_required"
    assert not transport.calls
    assert not list(tmp_path.iterdir())


def test_valid_transport_runs_exactly_32_cases_and_writes_redacted_receipt(tmp_path: Path) -> None:
    transport = FakeTransport(TransportResponse(200, _body()))
    result = _run(tmp_path, transport)
    assert result.status == "completed"
    assert result.logical_request_count == 32
    assert result.transport_attempt_count == 32
    assert result.completed_case_count == 32
    assert result.parser_invalid_count == 0
    assert result.safety.api_key_read is True
    assert result.safety.results_written is True
    assert len(transport.calls) == 32
    assert {call["timeout_seconds"] for call in transport.calls} == {5.0}
    assert {call["payload"]["max_tokens"] for call in transport.calls} == {256}
    assert {call["payload"]["temperature"] for call in transport.calls} == {0.0}
    assert {call["payload"]["response_format"]["type"] for call in transport.calls} == {"json_schema"}
    assert {call["payload"]["seed"] for call in transport.calls} == {20261301, 20261302}
    for call in transport.calls:
        prompt = json.dumps(call["payload"]["messages"], ensure_ascii=True)
        for forbidden in ("source-factor-v1-g", "latent_focal_root_ids", "evaluator_private", "test-only-secret"):
            assert forbidden not in prompt

    output = tmp_path / "pilot"
    receipt_path = output / "pilot_receipt.json"
    audit_path = output / "safe_case_audits.jsonl"
    receipt = load_pilot_receipt(receipt_path)
    assert isinstance(receipt, FactorizationPilotReceipt)
    assert receipt.logical_request_count == 32
    assert receipt.condition_case_counts["same_root_hidden"] == 4
    assert receipt.condition_case_counts["independent_roots_shown"] == 4
    assert receipt.schema_contract.schema_name == "conformity_source_behavior_factorized_response_v1"
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in (receipt_path, audit_path)).lower()
    for forbidden in (
        "test-only-secret",
        "authorization",
        "bearer",
        "system_prompt",
        "user_prompt",
        "target_claim",
        "public_summary",
        "content-initial",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    ("outcome", "category", "status_class", "parser_invalid", "parser_error_category"),
    [
        (TransportResponse(200, "malformed-provider-json"), "malformed_response", "2xx", 0, None),
        (TransportResponse(200, _body(content=_valid_content() | {"unknown_field": "forbidden"})), "factorization_pilot_parser_invalid", "2xx", 1, "invalid_schema"),
        (TransportResponse(200, _body(content=_valid_content() | {"content_ids_used": ["not-visible"]})), "factorization_pilot_parser_invalid", "2xx", 1, "unavailable_content_id"),
        (TransportResponse(200, _body(finish_reason="length")), "factorization_pilot_finish_reason_invalid", "2xx", 0, None),
        (TimeoutError("sensitive-timeout"), "timeout", None, 0, None),
        (ConnectionError("sensitive-connection"), "connection_failure", None, 0, None),
        (TransportResponse(401, '{"error":{"message":"sensitive-auth"}}'), "authentication_failed", "4xx", 0, None),
        (TransportResponse(429, '{"error":{"message":"sensitive-rate"}}'), "rate_limited", "4xx", 0, None),
        (TransportResponse(500, '{"error":{"message":"sensitive-server"}}'), "http_server_error", "5xx", 0, None),
        (TransportResponse(400, '{"error":{"message":"response_format json_schema unsupported secret"}}'), "response_format_unsupported", "4xx", 0, None),
    ],
    ids=["malformed-json", "unknown-field", "unavailable-content", "finish-length", "timeout", "connection", "401", "429", "500", "response-format-unsupported"],
)
def test_failures_stop_after_one_transport_call_and_redact_sensitive_material(
    tmp_path: Path,
    outcome: TransportResponse | BaseException,
    category: str,
    status_class: str | None,
    parser_invalid: int,
    parser_error_category: str | None,
) -> None:
    transport = FakeTransport(outcome)
    result = _run(tmp_path, transport)
    assert result.status == "failed"
    assert result.transport_category == category
    assert result.http_status_class == status_class
    assert result.parser_invalid_count == parser_invalid
    assert result.parser_error_category == parser_error_category
    assert len(transport.calls) == 1
    serialized = result.model_dump_json().lower()
    for forbidden in ("sensitive-", "unsupported secret", "test-only-secret", "authorization", '"provider_metadata":', "system_prompt", "user_prompt"):
        assert forbidden not in serialized
    output = tmp_path / "pilot"
    if output.exists():
        for path in output.iterdir():
            assert "test-only-secret" not in path.read_text(encoding="utf-8")


def test_existing_output_blocks_before_environment_or_transport(tmp_path: Path) -> None:
    output = tmp_path / "pilot"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("do-not-overwrite", encoding="utf-8")

    class ExplodingEnvironment(dict[str, str]):
        def get(self, key: str, default=None):
            raise AssertionError("environment must not be read")

    transport = FakeTransport(TransportResponse(200, _body()))
    result = execute_factorization_pilot(
        allow_network=True,
        confirm_run=True,
        confirm_request_cap=32,
        confirm_completion_reservation_cap=8192,
        environment=ExplodingEnvironment(),
        transport=transport,
        output_root=output,
    )
    assert result.status == "blocked"
    assert result.error_code == "factorization_pilot_output_exists"
    assert not transport.calls
    assert marker.read_text(encoding="utf-8") == "do-not-overwrite"


def test_receipt_rejects_unknown_fields_and_duplicate_json_keys(tmp_path: Path) -> None:
    transport = FakeTransport(TransportResponse(200, _body()))
    _run(tmp_path, transport)
    receipt_path = tmp_path / "pilot" / "pilot_receipt.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["raw_response"] = "forbidden"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception):
        load_pilot_receipt(receipt_path)
