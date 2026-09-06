from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from evicon.cascade_hg_compatibility import (
    COMPATIBILITY_MAX_RETRIES,
    COMPATIBILITY_MAX_TOKENS,
    COMPATIBILITY_SEED,
    COMPATIBILITY_TEMPERATURE,
    COMPATIBILITY_TIMEOUT_SECONDS,
    HGCompatibilityResult,
    run_compatibility_check,
)
from evicon.cascade_hg_compatibility_receipt import (
    HGCompatibilityReceiptError,
    register_receipt,
    sha256_file,
    validate_receipt,
)
from evicon.openai_provider import TransportResponse
from evicon.provenance_cascade_identifiability import hg_preflight

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg.v1.toml"
PROTOCOL = ROOT / "src/evicon/cascade_agent_protocol_hg.py"
APPROVAL = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg_approval_template.toml"
AMENDMENT_SHA = "f840925cd46a27741c2245e011acf68725b4654c85a2a9ce899d69cb4b34e49a"
ENVIRONMENT = {
    "EVICON_LLM_BASE_URL": "https://provider.invalid/v1",
    "EVICON_LLM_MODEL": "compat-model",
    "EVICON_LLM_API_KEY": "test-secret",
}


class FakeTransport:
    def __init__(self, outcomes: list[object]) -> None:
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


def _provider_response(
    *,
    content: str | None = None,
    finish_reason: str = "stop",
    completion_tokens: int = 20,
) -> str:
    content = content or json.dumps({
        "stance": "uncertain",
        "content_ids_used": ["hg-compatibility-content-01"],
        "evidence_ids_used": ["hg-compatibility-evidence-01"],
        "share_content_id": "hg-compatibility-content-01",
    })
    return json.dumps({
        "model": "compat-model",
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": completion_tokens,
            "total_tokens": 10 + completion_tokens,
        },
    })


def test_valid_strict_json_uses_hg_contract_and_one_transport_call(tmp_path: Path) -> None:
    transport = FakeTransport([TransportResponse(200, _provider_response())])
    result = run_compatibility_check(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=transport,
    )
    assert result.status == "completed"
    assert result.parser_valid is True
    assert result.finish_reason == "stop"
    assert result.http_status_class == "2xx"
    assert result.attempt_count == 1
    assert len(transport.calls) == 1
    call = transport.calls[0]
    payload = call["payload"]
    assert payload["max_tokens"] == COMPATIBILITY_MAX_TOKENS == 512
    assert payload["temperature"] == COMPATIBILITY_TEMPERATURE == 0.2
    assert payload["seed"] == COMPATIBILITY_SEED == 20260911
    assert call["timeout_seconds"] == COMPATIBILITY_TIMEOUT_SECONDS == 5.0
    assert COMPATIBILITY_MAX_RETRIES == 0
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "cascade_agent_response_v2_1"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False
    assert set(response_format["json_schema"]["schema"]["properties"]) == {
        "stance", "content_ids_used", "evidence_ids_used", "share_content_id",
    }
    serialized_messages = json.dumps(payload["messages"])
    assert "synthetic notice" in serialized_messages.lower()
    assert "cascade-hg-false-majority" not in serialized_messages
    assert not list(tmp_path.iterdir())
    dumped = result.model_dump_json().lower()
    for forbidden in ("test-secret", "authorization", "system_prompt", "user_prompt", "provider_metadata", "ground_truth_label"):
        assert forbidden not in dumped


def test_finish_reason_length_is_parser_invalid_without_retry() -> None:
    transport = FakeTransport([
        TransportResponse(200, _provider_response(
            content='{"stance":"uncertain"',
            finish_reason="length",
            completion_tokens=512,
        )),
        TransportResponse(200, _provider_response()),
    ])
    result = run_compatibility_check(allow_network=True, environment=ENVIRONMENT, transport=transport)
    assert result.status == "invalid_response"
    assert result.finish_reason == "length"
    assert result.parser_valid is False
    assert result.attempt_count == 1
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    ("outcome", "category", "status_class"),
    [
        (TimeoutError("sensitive-timeout"), "timeout", None),
        (ConnectionError("sensitive-connection"), "connection_failure", None),
        (socket.gaierror("sensitive-dns"), "dns_or_url_failure", None),
        (TransportResponse(400, '{"error":{"message":"invalid request"}}'), "http_client_error", "4xx"),
        (TransportResponse(400, '{"error":{"message":"response_format json_schema unsupported"}}'), "response_format_unsupported", "4xx"),
        (TransportResponse(401, '{"error":{"message":"secret"}}'), "authentication_failed", "4xx"),
        (TransportResponse(429, '{"error":{"message":"secret"}}'), "rate_limited", "4xx"),
        (TransportResponse(500, '{"error":{"message":"secret"}}'), "http_server_error", "5xx"),
    ],
)
def test_transport_failures_are_stable_redacted_and_single_attempt(outcome, category, status_class) -> None:
    transport = FakeTransport([outcome, TransportResponse(200, _provider_response())])
    result = run_compatibility_check(allow_network=True, environment=ENVIRONMENT, transport=transport)
    assert result.status == "provider_error"
    assert result.provider_error_code == category
    assert result.transport_category == category
    assert result.http_status_class == status_class
    assert result.attempt_count == 1
    assert len(transport.calls) == 1
    dumped = result.model_dump_json().lower()
    for forbidden in ("sensitive-", "invalid request", "response_format json_schema unsupported", "test-secret", "authorization", "provider_metadata"):
        assert forbidden not in dumped


def test_default_path_does_not_construct_provider_read_env_or_call_transport(monkeypatch, tmp_path: Path) -> None:
    class FailTransport:
        def post(self, *args, **kwargs):
            raise AssertionError("transport must not be called")

    def fail_from_env(*args, **kwargs):
        raise AssertionError("environment must not be read")

    monkeypatch.setattr("evicon.cascade_hg_compatibility.ProviderConfig.from_env", fail_from_env)
    result = run_compatibility_check(allow_network=False, transport=FailTransport())
    assert result.status == "network_disabled"
    assert result.attempt_count == 0
    assert result.network == "disabled"
    assert not list(tmp_path.iterdir())


def test_compatibility_path_does_not_load_evaluator_private_truth(monkeypatch) -> None:
    monkeypatch.setattr(
        "evicon.provenance_cascade.EvaluatorTruthLoader.load",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("private truth must not be read")),
    )
    transport = FakeTransport([TransportResponse(200, _provider_response())])
    result = run_compatibility_check(allow_network=True, environment=ENVIRONMENT, transport=transport)
    assert result.status == "completed"


def _successful_result() -> HGCompatibilityResult:
    return HGCompatibilityResult(
        status="completed",
        model="compat-model",
        finish_reason="stop",
        parser_valid=True,
        http_status_class="2xx",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        latency_ms=1.0,
        attempt_count=1,
        network="enabled",
    )


def test_safe_receipt_registration_binds_approval_before_overwrite_gate(tmp_path: Path) -> None:
    result_path = tmp_path / "safe-result.json"
    result_path.write_text(_successful_result().model_dump_json(), encoding="utf-8")
    approval_path = tmp_path / "approval.toml"
    approval_text = "\n".join(
        'compatibility_receipt_sha256 = ""'
        if line.startswith("compatibility_receipt_sha256 =") else line
        for line in APPROVAL.read_text(encoding="utf-8").splitlines()
    ) + "\n"
    approval_path.write_text(approval_text, encoding="utf-8")
    receipt_path = ROOT / "outputs/study-locks/_test_hg_compatibility_receipt.json"
    if receipt_path.exists():
        receipt_path.unlink()
    try:
        _, digest, _ = register_receipt(
            result_path=result_path,
            output_path=receipt_path,
            approval_path=approval_path,
            config_path=CONFIG,
            protocol_path=PROTOCOL,
            amendment_sha256=AMENDMENT_SHA,
        )
        validate_receipt(
            receipt_path,
            expected_hash=digest,
            config_path=CONFIG,
            protocol_path=PROTOCOL,
            amendment_sha256=AMENDMENT_SHA,
        )
        assert f'compatibility_receipt_sha256 = "{digest}"' in approval_path.read_text(encoding="utf-8")
        report = hg_preflight(
            CONFIG,
            approval_path_override=approval_path,
            compatibility_receipt_path_override=receipt_path,
        )
        assert report["status"] == "blocked"
        assert report["ready_for_real_pilot"] is False
        assert report["compatibility_receipt_sha256"] == digest
        assert report["blocking_reasons"] == ["output_root_exists"]
        assert report["network"] == "disabled"
        raw = receipt_path.read_text(encoding="utf-8").lower()
        for forbidden in ("system_prompt", "user_prompt", "request_id", "api_key", "headers", "provider_metadata", "ground_truth_label"):
            assert forbidden not in raw
    finally:
        if receipt_path.exists():
            receipt_path.unlink()


def test_receipt_registration_refuses_invalid_or_existing_receipt(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(HGCompatibilityResult(status="invalid_response", network="enabled").model_dump_json(), encoding="utf-8")
    approval_path = tmp_path / "approval.toml"
    approval_path.write_text(APPROVAL.read_text(encoding="utf-8"), encoding="utf-8")
    receipt_path = ROOT / "outputs/study-locks/_test_hg_compatibility_receipt_invalid.json"
    if receipt_path.exists():
        receipt_path.unlink()
    try:
        with pytest.raises(HGCompatibilityReceiptError) as error:
            register_receipt(
                result_path=invalid_path,
                output_path=receipt_path,
                approval_path=approval_path,
                config_path=CONFIG,
                protocol_path=PROTOCOL,
                amendment_sha256=AMENDMENT_SHA,
            )
        assert error.value.code == "compatibility_result_not_completed"
        receipt_path.write_text("{}", encoding="utf-8")
        valid_path = tmp_path / "valid.json"
        valid_path.write_text(_successful_result().model_dump_json(), encoding="utf-8")
        with pytest.raises(HGCompatibilityReceiptError) as error:
            register_receipt(
                result_path=valid_path,
                output_path=receipt_path,
                approval_path=approval_path,
                config_path=CONFIG,
                protocol_path=PROTOCOL,
                amendment_sha256=AMENDMENT_SHA,
            )
        assert error.value.code == "receipt_exists"
    finally:
        if receipt_path.exists():
            receipt_path.unlink()


def test_official_preflight_is_accepted_and_blocks_overwrite_after_start() -> None:
    report = hg_preflight(CONFIG)
    assert report["approval_status"] == "accepted"
    assert report["compatibility_receipt_sha256"] is not None
    assert report["blocking_reasons"] == ["output_root_exists"]
    assert report["ready_for_real_pilot"] is False
    assert report["provider_constructed"] is False
    assert report["api_key_read"] is False
    resume_report = hg_preflight(CONFIG, allow_existing_output=True)
    assert resume_report["status"] == "ready_for_real_pilot"
    assert resume_report["ready_for_real_pilot"] is True
    assert resume_report["blocking_reasons"] == []
    assert resume_report["resume_mode"] is True
