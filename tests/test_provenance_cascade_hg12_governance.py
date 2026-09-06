from __future__ import annotations

import json
import re
import socket
from pathlib import Path

import pytest

from evicon.openai_provider import TransportResponse
from evicon.provenance_cascade_hg12_compatibility import (
    MAX_RETRIES, MAX_TOKENS, SEED, TEMPERATURE, TIMEOUT_SECONDS,
    run_compatibility_check, run_fake_smoke,
)
from evicon.provenance_cascade_hg12_governance import (
    DEFAULT_AMENDMENT_RECEIPT, DEFAULT_APPROVAL, DEFAULT_CONFIG,
    HG12GovernanceError, final_preflight, load_approval,
    register_compatibility_result, sha256_file, validate_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = {
    "EVICON_LLM_BASE_URL": "https://provider.invalid/v1",
    "EVICON_LLM_MODEL": "hg12-compat-model",
    "EVICON_LLM_API_KEY": "test-secret",
}


class FakeTransport:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def post(self, url, headers, payload, timeout_seconds):
        self.calls.append({"payload": payload, "timeout_seconds": timeout_seconds})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _body(*, content: str | None = None, finish_reason: str = "stop") -> str:
    content = content or json.dumps({
        "stance": "uncertain",
        "content_ids_used": ["hg11-compatibility-content-01"],
        "evidence_ids_used": ["hg11-compatibility-evidence-01"],
        "share_content_id": "hg11-compatibility-content-01",
    })
    return json.dumps({
        "model": "hg12-compat-model",
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 13, "total_tokens": 24},
    })


def test_accepted_approval_binds_locked_public_contract() -> None:
    approval = load_approval()
    assert approval.acceptance_status == "accepted"
    assert approval.config_sha256 == sha256_file(DEFAULT_CONFIG)
    assert approval.amendment_receipt_sha256 == sha256_file(DEFAULT_AMENDMENT_RECEIPT)
    assert approval.compatibility_receipt_sha256 == sha256_file(
        "outputs/study-locks/provenance_cascade_hg12_compatibility_receipt.json"
    )
    assert approval.logical_request_cap == 864
    assert approval.completion_reservation_cap == 884736


def test_preflight_is_offline_and_blocks_overwrite_after_completed_pilot() -> None:
    report = final_preflight()
    assert report["status"] == "blocked"
    assert report["blocking_reasons"] == ["output_root_exists"]
    assert report["ready_for_real_pilot"] is False
    assert report["network"] == "disabled"
    assert report["provider_constructed"] is False
    assert report["api_key_read"] is False
    assert report["results_written"] is False


def test_default_compatibility_does_not_read_environment_or_call_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg12_compatibility.ProviderConfig.from_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("environment must not be read")),
    )
    result = run_compatibility_check()
    assert result.status == "network_disabled"
    assert result.attempt_count == 0
    assert result.network == "disabled"


def test_one_shot_payload_uses_locked_schema_and_generation_contract() -> None:
    transport = FakeTransport([TransportResponse(200, _body())])
    result = run_compatibility_check(allow_network=True, environment=ENVIRONMENT, transport=transport)
    assert result.status == "completed" and result.parser_valid is True
    assert result.attempt_count == 1 and len(transport.calls) == 1
    call = transport.calls[0]
    payload = call["payload"]
    assert payload["max_tokens"] == MAX_TOKENS == 1024
    assert payload["temperature"] == TEMPERATURE == 0.2
    assert payload["seed"] == SEED == 20261011
    assert call["timeout_seconds"] == TIMEOUT_SECONDS == 5.0
    assert MAX_RETRIES == 0
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "cascade_agent_response_v2_1"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False
    serialized = json.dumps(payload)
    assert "cascade-hg1-false-majority" not in serialized
    assert "test-secret" not in result.model_dump_json()


def test_finish_reason_length_is_invalid_without_retry() -> None:
    transport = FakeTransport([
        TransportResponse(200, _body(content='{"stance":"uncertain"', finish_reason="length")),
        TransportResponse(200, _body()),
    ])
    result = run_compatibility_check(allow_network=True, environment=ENVIRONMENT, transport=transport)
    assert result.status == "invalid_response"
    assert result.finish_reason == "length"
    assert result.parser_valid is False
    assert result.attempt_count == len(transport.calls) == 1


@pytest.mark.parametrize(
    ("outcome", "category"),
    [
        (TimeoutError("sensitive-error-detail"), "timeout"),
        (ConnectionError("sensitive-error-detail"), "connection_failure"),
        (socket.gaierror("sensitive-error-detail"), "dns_or_url_failure"),
        (TransportResponse(400, '{"error":{"message":"response_format unsupported"}}'), "response_format_unsupported"),
        (TransportResponse(401, "sensitive-error-detail"), "authentication_failed"),
        (TransportResponse(429, "sensitive-error-detail"), "rate_limited"),
        (TransportResponse(500, "sensitive-error-detail"), "http_server_error"),
    ],
)
def test_transport_errors_are_safe_and_never_retry(outcome, category) -> None:
    transport = FakeTransport([outcome, TransportResponse(200, _body())])
    result = run_compatibility_check(allow_network=True, environment=ENVIRONMENT, transport=transport)
    assert result.status == "provider_error"
    assert result.provider_error_code == category
    assert len(transport.calls) == result.attempt_count == 1
    dumped = result.model_dump_json().lower()
    assert "sensitive-error-detail" not in dumped and "authorization" not in dumped


def test_safe_receipt_registration_updates_only_temp_approval(monkeypatch, tmp_path: Path) -> None:
    approval = tmp_path / "approval.toml"
    approval.write_text(_approval_without_receipt(), encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr("evicon.provenance_cascade_hg12_governance.DEFAULT_APPROVAL", str(approval))
    monkeypatch.setattr("evicon.provenance_cascade_hg12_governance.DEFAULT_COMPATIBILITY_RECEIPT", str(receipt))
    transport = FakeTransport([TransportResponse(200, _body())])
    result = run_compatibility_check(allow_network=True, environment=ENVIRONMENT, transport=transport)
    digest, approval_digest = register_compatibility_result(result)
    registered = validate_receipt(receipt, expected_hash=digest)
    assert registered.max_tokens == 1024 and registered.attempt_count == 1
    assert digest == sha256_file(receipt)
    assert approval_digest == sha256_file(approval)
    raw = receipt.read_text(encoding="utf-8").lower()
    for forbidden in ("system_prompt", "user_prompt", "test-secret", "authorization", "provider_metadata", "ground_truth_label"):
        assert forbidden not in raw
    ready = final_preflight()
    assert ready["status"] == "blocked"
    assert ready["blocking_reasons"] == ["output_root_exists"]
    assert ready["ready_for_real_pilot"] is False


def test_receipt_is_no_overwrite_and_fake_smoke_is_offline(monkeypatch, tmp_path: Path) -> None:
    approval = tmp_path / "approval.toml"
    approval.write_text(_approval_without_receipt(), encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr("evicon.provenance_cascade_hg12_governance.DEFAULT_APPROVAL", str(approval))
    monkeypatch.setattr("evicon.provenance_cascade_hg12_governance.DEFAULT_COMPATIBILITY_RECEIPT", str(receipt))
    result = run_compatibility_check(
        allow_network=True, environment=ENVIRONMENT,
        transport=FakeTransport([TransportResponse(200, _body())]),
    )
    register_compatibility_result(result)
    with pytest.raises(HG12GovernanceError, match="compatibility_receipt_exists"):
        register_compatibility_result(result)
    smoke = run_fake_smoke()
    assert smoke["status"] == "passed"
    assert smoke["network"] == "disabled"
    assert smoke["max_tokens"] == 1024


def _approval_without_receipt() -> str:
    return re.sub(
        r'^compatibility_receipt_sha256 = "[0-9a-f]*"$',
        'compatibility_receipt_sha256 = ""',
        (ROOT / DEFAULT_APPROVAL).read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
