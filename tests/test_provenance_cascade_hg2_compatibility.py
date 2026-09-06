from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from evicon.openai_provider import TransportResponse
from evicon.provenance_cascade_hg2_compatibility import (
    COMPATIBILITY_MAX_RETRIES,
    COMPATIBILITY_MAX_TOKENS,
    COMPATIBILITY_SEED,
    COMPATIBILITY_TEMPERATURE,
    COMPATIBILITY_TIMEOUT_SECONDS,
    HG2CompatibilityResult,
    run_compatibility_check,
    run_fake_compatibility,
)
from evicon.provenance_cascade_hg2_compatibility_receipt import (
    register_compatibility_result,
    validate_receipt,
)
from evicon.provenance_cascade_hg2_governance import HG2Approval

ROOT = Path(__file__).resolve().parents[1]
APPROVAL = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg2_approval_template.toml"
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
        self.calls.append({"url": url, "headers": dict(headers), "payload": payload, "timeout_seconds": timeout_seconds})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _content() -> str:
    return json.dumps({
        "epistemic_stance": "uncertain",
        "behavioral_decision": "share_with_caveat",
        "content_ids_used": ["hg11-compatibility-content-01"],
        "evidence_ids_used": ["hg11-compatibility-evidence-01"],
        "share_content_id": "hg11-compatibility-content-01",
    })


def _provider_response(*, content: str | None = None, finish_reason: str = "stop") -> str:
    return json.dumps({
        "model": "compat-model",
        "choices": [{"message": {"content": content or _content()}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 25, "total_tokens": 45},
    })


def _completed_result() -> HG2CompatibilityResult:
    return HG2CompatibilityResult(
        status="completed", model="compat-model", finish_reason="stop", parser_valid=True,
        http_status_class="2xx", prompt_tokens=20, completion_tokens=25, total_tokens=45,
        latency_ms=2.0, attempt_count=1, network="enabled",
    )


def test_valid_strict_schema_payload_and_single_attempt() -> None:
    transport = FakeTransport([TransportResponse(200, _provider_response())])
    result = run_compatibility_check(allow_network=True, environment=ENVIRONMENT, transport=transport)
    assert result.status == "completed" and result.parser_valid is True
    assert result.attempt_count == 1 and len(transport.calls) == 1
    call = transport.calls[0]
    payload = call["payload"]
    assert payload["max_tokens"] == COMPATIBILITY_MAX_TOKENS == 1024
    assert payload["temperature"] == COMPATIBILITY_TEMPERATURE == 0.2
    assert payload["seed"] == COMPATIBILITY_SEED == 20261021
    assert call["timeout_seconds"] == COMPATIBILITY_TIMEOUT_SECONDS == 5.0
    assert COMPATIBILITY_MAX_RETRIES == 0
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "cascade_agent_epistemic_behavior_response_v1"
    assert response_format["json_schema"]["strict"] is True
    assert set(response_format["json_schema"]["schema"]["properties"]) == {
        "epistemic_stance", "behavioral_decision", "content_ids_used", "evidence_ids_used", "share_content_id",
    }
    serialized = json.dumps(payload["messages"]).lower()
    assert "synthetic" in serialized
    assert "cascade-hg1-false-majority" not in serialized
    dumped = result.model_dump_json().lower()
    for forbidden in ("test-secret", "authorization", "system_prompt", "user_prompt", "provider_metadata", "ground_truth_label"):
        assert forbidden not in dumped


def test_truncated_output_is_invalid_without_retry() -> None:
    transport = FakeTransport([
        TransportResponse(200, _provider_response(content='{"epistemic_stance":"uncertain"', finish_reason="length")),
        TransportResponse(200, _provider_response()),
    ])
    result = run_compatibility_check(allow_network=True, environment=ENVIRONMENT, transport=transport)
    assert result.status == "invalid_response"
    assert result.finish_reason == "length" and result.parser_valid is False
    assert result.attempt_count == 1 and len(transport.calls) == 1


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
def test_transport_errors_are_safe_and_never_retried(outcome, category, status_class) -> None:
    transport = FakeTransport([outcome, TransportResponse(200, _provider_response())])
    result = run_compatibility_check(allow_network=True, environment=ENVIRONMENT, transport=transport)
    assert result.status == "provider_error"
    assert result.provider_error_code == category
    assert result.transport_category == category
    assert result.http_status_class == status_class
    assert result.attempt_count == 1 and len(transport.calls) == 1
    dumped = result.model_dump_json().lower()
    for forbidden in ("sensitive-", "invalid request", "json_schema unsupported\"", "test-secret", "authorization", "provider_metadata"):
        assert forbidden not in dumped


def test_default_is_offline_and_does_not_read_environment(monkeypatch) -> None:
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg2_compatibility.ProviderConfig.from_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("environment read")),
    )
    result = run_compatibility_check()
    assert result.status == "network_disabled"
    assert result.attempt_count == 0 and result.network == "disabled"


def test_fake_smoke_uses_only_two_in_memory_calls() -> None:
    report = run_fake_compatibility()
    assert report["status"] == "passed"
    assert report["provider_call_count"] == 2
    assert report["network"] == "disabled"
    assert report["results_written"] is False


def test_safe_receipt_registration_binds_accepted_approval(tmp_path: Path) -> None:
    approval = tmp_path / "approval.toml"
    approval.write_text(APPROVAL.read_text(encoding="utf-8"), encoding="utf-8")
    target = ROOT / "outputs/study-locks/_test_hg2_compatibility_receipt.json"
    if target.exists():
        target.unlink()
    try:
        receipt_sha, _ = register_compatibility_result(_completed_result(), receipt_path=target, approval_path=approval)
        receipt = validate_receipt(target, expected_hash=receipt_sha)
        assert receipt.schema_name == "cascade_agent_epistemic_behavior_response_v1"
        assert receipt.max_tokens == 1024 and receipt.max_retries == 0
        parsed = __import__("tomllib").loads(approval.read_text(encoding="utf-8"))
        accepted = HG2Approval.model_validate(parsed)
        assert accepted.acceptance_status == "accepted"
        assert accepted.compatibility_receipt_sha256 == receipt_sha
        dumped = target.read_text(encoding="utf-8").lower()
        for forbidden in ("system_prompt", "user_prompt", "response_body", "request_id", "api_key", "authorization", "provider_metadata", "ground_truth_label"):
            assert forbidden not in dumped
    finally:
        if target.exists():
            target.unlink()
