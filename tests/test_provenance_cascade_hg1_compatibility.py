from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from evicon.openai_provider import TransportResponse
from evicon.provenance_cascade_hg1 import hg1_preflight
from evicon.provenance_cascade_hg1_compatibility import (
    COMPATIBILITY_MAX_RETRIES,
    COMPATIBILITY_MAX_TOKENS,
    COMPATIBILITY_SEED,
    COMPATIBILITY_TEMPERATURE,
    COMPATIBILITY_TIMEOUT_SECONDS,
    HG1CompatibilityResult,
    run_compatibility_check,
)
from evicon.provenance_cascade_hg1_compatibility_receipt import (
    HG1CompatibilityReceiptError,
    register_receipt,
    validate_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg1.v1.toml"
PROTOCOL = ROOT / "src/evicon/cascade_agent_protocol_hg1.py"
APPROVAL = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg1_approval_template.toml"
AMENDMENT_SHA = "31b85944a7886bca6c0bc98601e438c5083bbb25de5f87a4585c9fd56f60d2bf"
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


def _provider_response(*, content: str | None = None, finish_reason: str = "stop", completion_tokens: int = 20) -> str:
    content = content or json.dumps({
        "stance": "uncertain",
        "content_ids_used": ["hg1-compatibility-content-01"],
        "evidence_ids_used": ["hg1-compatibility-evidence-01"],
        "share_content_id": "hg1-compatibility-content-01",
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


def _successful_result() -> HG1CompatibilityResult:
    return HG1CompatibilityResult(
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


def _accepted_approval_text() -> str:
    text = APPROVAL.read_text(encoding="utf-8")
    return __import__("re").sub(
        r'^compatibility_receipt_sha256 = "[0-9a-f]*"$',
        'compatibility_receipt_sha256 = ""', text, flags=__import__("re").MULTILINE,
    )


def _pending_approval_text() -> str:
    text = _accepted_approval_text()
    text = text.replace('acceptance_status = "accepted"', 'acceptance_status = "pending"')
    text = __import__("re").sub(r'^accepted_by = ".*"$', 'accepted_by = ""', text, flags=__import__("re").MULTILINE)
    text = __import__("re").sub(r'^accepted_on = ".*"$', 'accepted_on = ""', text, flags=__import__("re").MULTILINE)
    for field in (
        "confirm_round_zero_preload", "confirm_public_view_only_controller",
        "confirm_paired_root_design", "confirm_eligibility_contract",
        "confirm_parent_results_excluded", "confirm_no_overwrite",
        "confirm_append_only_ledger", "confirm_resume_rules",
    ):
        text = text.replace(f"{field} = true", f"{field} = false")
    return text


def test_valid_strict_json_uses_hg1_contract_and_one_transport_call(tmp_path: Path) -> None:
    transport = FakeTransport([TransportResponse(200, _provider_response())])
    result = run_compatibility_check(allow_network=True, environment=ENVIRONMENT, transport=transport)

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
    assert payload["seed"] == COMPATIBILITY_SEED == 20261001
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
    assert "cascade-hg1-false-majority" not in serialized_messages
    assert not list(tmp_path.iterdir())
    dumped = result.model_dump_json().lower()
    for forbidden in ("test-secret", "authorization", "system_prompt", "user_prompt", "provider_metadata", "ground_truth_label"):
        assert forbidden not in dumped


def test_finish_reason_length_is_parser_invalid_without_retry() -> None:
    transport = FakeTransport([
        TransportResponse(200, _provider_response(content='{"stance":"uncertain"', finish_reason="length", completion_tokens=512)),
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
def test_transport_failures_are_redacted_and_single_attempt(outcome, category, status_class) -> None:
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


def test_default_path_does_not_construct_provider_read_environment_or_call_transport(monkeypatch) -> None:
    def fail_from_env(*args, **kwargs):
        raise AssertionError("environment must not be read")

    monkeypatch.setattr("evicon.provenance_cascade_hg1_compatibility.ProviderConfig.from_env", fail_from_env)
    result = run_compatibility_check(allow_network=False)
    assert result.status == "network_disabled"
    assert result.attempt_count == 0
    assert result.network == "disabled"


def test_compatibility_path_does_not_load_evaluator_private_truth(monkeypatch) -> None:
    monkeypatch.setattr(
        "evicon.provenance_cascade.EvaluatorTruthLoader.load",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("private truth must not be read")),
    )
    transport = FakeTransport([TransportResponse(200, _provider_response())])
    result = run_compatibility_check(allow_network=True, environment=ENVIRONMENT, transport=transport)
    assert result.status == "completed"


def test_safe_receipt_registration_requires_accepted_approval_and_binds_hashes(tmp_path: Path) -> None:
    result_path = tmp_path / "safe-result.json"
    result_path.write_text(_successful_result().model_dump_json(), encoding="utf-8")
    pending_approval = tmp_path / "pending.toml"
    pending_approval.write_text(_pending_approval_text(), encoding="utf-8")
    accepted_approval = tmp_path / "accepted.toml"
    accepted_approval.write_text(_accepted_approval_text(), encoding="utf-8")
    receipt_path = ROOT / "outputs/study-locks/_test_hg1_compatibility_receipt.json"
    if receipt_path.exists():
        receipt_path.unlink()
    try:
        with pytest.raises(HG1CompatibilityReceiptError) as error:
            register_receipt(
                result_path=result_path,
                output_path=receipt_path,
                approval_path=pending_approval,
                config_path=CONFIG,
                protocol_path=PROTOCOL,
                amendment_sha256=AMENDMENT_SHA,
            )
        assert error.value.code == "approval_not_accepted"

        _, digest, approval_digest = register_receipt(
            result_path=result_path,
            output_path=receipt_path,
            approval_path=accepted_approval,
            config_path=CONFIG,
            protocol_path=PROTOCOL,
            amendment_sha256=AMENDMENT_SHA,
        )
        receipt = validate_receipt(
            receipt_path,
            expected_hash=digest,
            config_path=CONFIG,
            protocol_path=PROTOCOL,
            amendment_sha256=AMENDMENT_SHA,
        )
        assert receipt.schema_name == "cascade_agent_response_v2_1"
        assert len(digest) == 64
        assert len(approval_digest) == 64
        assert f'compatibility_receipt_sha256 = "{digest}"' in accepted_approval.read_text(encoding="utf-8")
        raw = receipt_path.read_text(encoding="utf-8").lower()
        for forbidden in ("system_prompt", "user_prompt", "request_id", "api_key", "headers", "provider_metadata", "ground_truth_label"):
            assert forbidden not in raw
    finally:
        if receipt_path.exists():
            receipt_path.unlink()


def test_official_preflight_blocks_existing_historical_output_after_acceptance() -> None:
    report = hg1_preflight(CONFIG)
    assert report["status"] == "blocked"
    assert report["ready_for_real_pilot"] is False
    assert report["blocking_reasons"] == ["output_root_exists"]
    assert report["network"] == "disabled"
    assert report["provider_constructed"] is False
    assert report["api_key_read"] is False
