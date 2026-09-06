from __future__ import annotations

import json
import socket
import re
from pathlib import Path

import pytest

from evicon.openai_provider import TransportResponse
from evicon.provenance_cascade_hg232_compatibility import (
    COMPATIBILITY_MAX_RETRIES, COMPATIBILITY_MAX_TOKENS,
    COMPATIBILITY_SEED, COMPATIBILITY_TEMPERATURE,
    COMPATIBILITY_TIMEOUT_SECONDS, run_compatibility_check, run_fake_smoke,
)
from evicon.provenance_cascade_hg232_compatibility_receipt import (
    register_compatibility_result, validate_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
APPROVAL = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg232_approval_template.toml"
ENVIRONMENT = {
    "EVICON_LLM_BASE_URL": "https://provider.invalid/v1",
    "EVICON_LLM_MODEL": "gpt-5.6-luna",
    "EVICON_LLM_API_KEY": "test-secret",
}


class FakeTransport:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def post(self, url, headers, payload, timeout_seconds):
        self.calls.append({"url": url, "headers": dict(headers), "payload": payload, "timeout": timeout_seconds})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _content() -> str:
    return json.dumps({
        "epistemic_stance": "uncertain",
        "adoption_decision": "adopt_claim",
        "sharing_decision": "share_with_caveat",
        "content_ids_used": ["hg11-compatibility-content-01"],
        "evidence_ids_used": ["hg11-compatibility-evidence-01"],
        "share_content_id": "hg11-compatibility-content-01",
    })


def _response(*, content: str | None = None, finish_reason: str = "stop") -> str:
    return json.dumps({
        "model": "gpt-5.6-luna",
        "choices": [{"message": {"content": content or _content()}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 31, "total_tokens": 61},
    })


def _accepted_approval(tmp_path: Path) -> Path:
    raw = APPROVAL.read_text(encoding="utf-8")
    raw = raw.replace('acceptance_status = "pending"', 'acceptance_status = "accepted"')
    raw = raw.replace('accepted_by = ""', 'accepted_by = "synthetic-reviewer"')
    raw = raw.replace('accepted_on = ""', 'accepted_on = "2026-08-26"')
    raw = re.sub(r"^compatibility_receipt_sha256 = .*?$", 'compatibility_receipt_sha256 = ""', raw, flags=re.MULTILINE)
    raw = re.sub(r"^(confirm_[a-z0-9_]+) = false$", r"\1 = true", raw, flags=re.MULTILINE)
    target = tmp_path / "accepted.toml"
    target.write_text(raw, encoding="utf-8")
    return target

def test_valid_schema_payload_is_one_shot_and_uses_locked_parameters(tmp_path: Path) -> None:
    transport = FakeTransport([TransportResponse(200, _response())])
    result = run_compatibility_check(allow_network=True, environment=ENVIRONMENT, transport=transport, approval_path=_accepted_approval(tmp_path))
    assert result.status == "completed" and result.parser_valid is True
    assert result.attempt_count == len(transport.calls) == 1
    call = transport.calls[0]
    payload = call["payload"]
    assert payload["max_tokens"] == COMPATIBILITY_MAX_TOKENS == 2048
    assert payload["temperature"] == COMPATIBILITY_TEMPERATURE == 0.2
    assert payload["seed"] == COMPATIBILITY_SEED == 20261031
    assert call["timeout"] == COMPATIBILITY_TIMEOUT_SECONDS == 5.0
    assert COMPATIBILITY_MAX_RETRIES == 0
    schema = payload["response_format"]["json_schema"]
    assert schema["name"] == "cascade_agent_epistemic_adoption_sharing_v1_provider_subset_2048"
    assert schema["strict"] is True
    assert set(schema["schema"]["properties"]) == {
        "epistemic_stance", "adoption_decision", "sharing_decision",
        "content_ids_used", "evidence_ids_used", "share_content_id",
    }
    serialized = json.dumps(payload["messages"]).lower()
    assert "synthetic" in serialized
    assert "cascade-hg1-false-majority" not in serialized
    assert "ground_truth_label" not in serialized


def test_length_or_malformed_output_is_invalid_without_retry(tmp_path: Path) -> None:
    transport = FakeTransport([
        TransportResponse(200, _response(content='{"epistemic_stance":"uncertain"', finish_reason="length")),
        TransportResponse(200, _response()),
    ])
    result = run_compatibility_check(allow_network=True, environment=ENVIRONMENT, transport=transport, approval_path=_accepted_approval(tmp_path))
    assert result.status == "invalid_response" and result.parser_valid is False
    assert result.finish_reason == "length"
    assert result.attempt_count == len(transport.calls) == 1


@pytest.mark.parametrize(
    ("outcome", "category", "status_class"),
    [
        (TimeoutError("sensitive-timeout"), "timeout", None),
        (ConnectionError("sensitive-connection"), "connection_failure", None),
        (socket.gaierror("sensitive-dns"), "dns_or_url_failure", None),
        (TransportResponse(400, '{"error":{"message":"ordinary bad request"}}'), "http_client_error", "4xx"),
        (TransportResponse(400, '{"error":{"message":"response_format json_schema unsupported"}}'), "response_format_unsupported", "4xx"),
        (TransportResponse(401, '{"error":{"message":"secret"}}'), "authentication_failed", "4xx"),
        (TransportResponse(429, '{"error":{"message":"secret"}}'), "rate_limited", "4xx"),
        (TransportResponse(500, '{"error":{"message":"secret"}}'), "http_server_error", "5xx"),
    ],
)
def test_transport_failures_are_safe_and_never_retried(outcome, category, status_class, tmp_path: Path) -> None:
    transport = FakeTransport([outcome, TransportResponse(200, _response())])
    result = run_compatibility_check(allow_network=True, environment=ENVIRONMENT, transport=transport, approval_path=_accepted_approval(tmp_path))
    assert result.status == "provider_error"
    assert result.provider_error_code == result.transport_category == category
    assert result.http_status_class == status_class
    assert result.attempt_count == len(transport.calls) == 1
    safe = result.model_dump_json().lower()
    for forbidden in ("sensitive-", "ordinary bad request", "test-secret", "authorization", "provider_metadata"):
        assert forbidden not in safe


def test_default_is_offline_and_does_not_read_environment(monkeypatch) -> None:
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg232_compatibility.ProviderConfig.from_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("environment read")),
    )
    result = run_compatibility_check()
    assert result.status == "network_disabled"
    assert result.attempt_count == 0 and result.network == "disabled"


def test_fake_smoke_and_safe_receipt_registration(tmp_path: Path, monkeypatch) -> None:
    from evicon import provenance_cascade_hg232_compatibility_receipt as receipt_module

    assert run_fake_smoke()["status"] == "passed"
    approval = tmp_path / "approval.toml"
    approval.write_text(_accepted_approval(tmp_path).read_text(encoding="utf-8"), encoding="utf-8")
    original_path = receipt_module._path
    monkeypatch.setattr(receipt_module, "_ROOT", tmp_path)
    monkeypatch.setattr(
        receipt_module,
        "_path",
        lambda value: Path(value).resolve() if Path(value).is_absolute() else original_path(value),
    )
    target = tmp_path / "outputs/study-locks/compatibility_receipt.json"
    transport = FakeTransport([TransportResponse(200, _response())])
    result = run_compatibility_check(allow_network=True, environment=ENVIRONMENT, transport=transport, approval_path=_accepted_approval(tmp_path))
    receipt_sha, _ = register_compatibility_result(result, receipt_path=target, approval_path=approval)
    receipt = validate_receipt(target, expected_hash=receipt_sha)
    assert receipt.schema_name == "cascade_agent_epistemic_adoption_sharing_v1_provider_subset_2048"
    assert receipt.max_tokens == 2048 and receipt.seed == 20261031
    safe = target.read_text(encoding="utf-8").lower()
    for forbidden in ("system_prompt", "user_prompt", "request_id", "api_key", "authorization", "provider_metadata", "ground_truth_label"):
        assert forbidden not in safe
