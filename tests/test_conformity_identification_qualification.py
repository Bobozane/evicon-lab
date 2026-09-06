from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from evicon.conformity_identification_compatibility import (
    RESPONSE_SCHEMA,
    check_provider,
    run_compatibility_check,
    run_fake_smoke as run_compatibility_fake,
)
from evicon.conformity_identification_network_gates import (
    run_manipulation_gate,
    run_stability_gate,
)
from evicon.conformity_identification_qualification import (
    DEFAULT_QUALIFICATION_APPROVAL,
    load_qualification_approval,
    safe_qualification_preflight,
)
from evicon.conformity_identification_stability import run_fake_smoke as run_stability_fake
from evicon.conformity_source_manipulation_smoke import (
    ManipulationFakeProvider,
    run_fake_smoke as run_manipulation_fake,
)
from evicon.llm_contract import LLMResponse
from evicon.openai_provider import TransportResponse

ROOT = Path(__file__).resolve().parents[1]
APPROVAL = ROOT / DEFAULT_QUALIFICATION_APPROVAL
ENV = {
    "EVICON_LLM_BASE_URL": "https://provider.invalid/v1",
    "EVICON_LLM_MODEL": "qualification-model",
    "EVICON_LLM_API_KEY": "test-only-secret",
}


def _accepted_approval(tmp_path: Path) -> Path:
    raw = APPROVAL.read_text(encoding="utf-8")
    raw = raw.replace('acceptance_status = "pending"', 'acceptance_status = "accepted"')
    raw = raw.replace('accepted_by = ""', 'accepted_by = "test-reviewer"')
    raw = raw.replace('accepted_on = ""', 'accepted_on = "2026-08-27"')
    raw = re.sub(r"(confirm_[a-z0-9_]+) = false", r"\1 = true", raw)
    path = tmp_path / "approval.toml"
    path.write_text(raw, encoding="utf-8")
    return path


class FakeTransport:
    def __init__(self, response: TransportResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def post(self, url, headers, payload, timeout_seconds):
        self.calls.append({"payload": payload, "timeout": timeout_seconds})
        return self.response


def _transport_response() -> TransportResponse:
    content = json.dumps({
        "ordinal_judgment": "uncertain", "public_action": "withhold",
        "share_decision": "do_not_share", "confidence": 4,
        "content_ids_used": ["compatibility-content-01"], "evidence_ids_used": [],
    })
    return TransportResponse(200, json.dumps({
        "model": "qualification-model",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 24, "total_tokens": 44},
    }))


def test_accepted_approval_and_offline_preflight_are_safe(monkeypatch) -> None:
    monkeypatch.setenv("EVICON_LLM_API_KEY", "must-not-be-read")
    approval = load_qualification_approval()
    result = safe_qualification_preflight()
    assert approval.acceptance_status == "accepted"
    assert result["status"] == "qualification_gates_offline_ready"
    assert "exact_hash_human_approval_required" not in result["blocking_reasons"]
    assert result["blocking_reasons"] == [
        "real_source_manipulation_check_required",
        "protocol_stability_probe_required",
    ]
    assert result["gate_receipts"]["provider_compatibility"] is True
    assert result["network"] == "disabled"
    assert result["provider_constructed"] is False
    assert result["api_key_read"] is False


def test_approval_rejects_unknown_field(tmp_path: Path) -> None:
    path = tmp_path / "approval.toml"
    path.write_text(APPROVAL.read_text(encoding="utf-8") + "\nquestion_text = \"forbidden\"\n", encoding="utf-8")
    with pytest.raises(Exception):
        load_qualification_approval(path)


def test_accepted_approval_requires_all_confirmations(tmp_path: Path) -> None:
    raw = APPROVAL.read_text(encoding="utf-8")
    raw = raw.replace("confirm_shared_t0_design = true", "confirm_shared_t0_design = false")
    path = tmp_path / "incomplete.toml"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(Exception):
        load_qualification_approval(path)


def test_compatibility_default_and_fake_are_network_free() -> None:
    default = run_compatibility_check()
    fake = run_compatibility_fake()
    assert default.status == "network_disabled" and default.attempt_count == 0
    assert fake["status"] == "compatibility_fake_smoke_passed"
    assert fake["provider_call_count"] == 1
    assert fake["network"] == "disabled"


def test_compatibility_payload_uses_strict_schema_once(tmp_path: Path) -> None:
    transport = FakeTransport(_transport_response())
    result = run_compatibility_check(
        allow_network=True, environment=ENV, transport=transport,
        approval_path=_accepted_approval(tmp_path),
    )
    assert result.status == "completed" and result.parser_valid is True
    assert result.attempt_count == 1 and len(transport.calls) == 1
    response_format = transport.calls[0]["payload"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == RESPONSE_SCHEMA


def test_compatibility_rejects_unavailable_id() -> None:
    class Provider:
        def complete(self, request):
            return LLMResponse(
                request_id=request.request_id, model_name=request.model_name,
                content=json.dumps({
                    "ordinal_judgment": "uncertain", "public_action": "withhold",
                    "share_decision": "do_not_share", "confidence": 4,
                    "content_ids_used": ["not-visible"], "evidence_ids_used": [],
                }),
                finish_reason="stop", latency_ms=0.0,
            )

    assert check_provider(Provider()).status == "invalid_response"


def test_fake_gate_smokes_remain_complete() -> None:
    manipulation = run_manipulation_fake()
    stability = run_stability_fake()
    assert manipulation["passed_case_count"] == 12
    assert manipulation["root_count_totals"] == {
        "independent_roots": 8, "same_root": 4, "source_free": 0,
    }
    assert stability["logical_request_count"] == 30
    assert stability["parser_invalid_count"] == 0


def test_manipulation_gate_writes_safe_append_only_receipt(tmp_path: Path) -> None:
    output = tmp_path / "manipulation"
    result = run_manipulation_gate(
        ManipulationFakeProvider(), output_root=output,
        approval_path=_accepted_approval(tmp_path),
    )
    assert result.status == "completed"
    assert result.logical_request_count == result.transport_attempt_count == 12
    assert result.completed_count == 12
    ledger_lines = (output / "request_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ledger_lines) == 24
    safe = "\n".join(path.read_text(encoding="utf-8") for path in output.iterdir()).lower()
    for forbidden in ("system_prompt", "user_prompt", "ground_truth_label", "source_independence_label", "authorization", "test-only-secret"):
        assert forbidden not in safe


def test_stability_gate_completes_30_requests(tmp_path: Path) -> None:
    output = tmp_path / "stability"
    result = run_stability_gate(
        __import__("evicon.conformity_identification_smoke", fromlist=["QualificationFakeProvider"]).QualificationFakeProvider(),
        output_root=output, approval_path=_accepted_approval(tmp_path),
    )
    assert result.status == "completed"
    assert result.logical_request_count == result.transport_attempt_count == 30
    assert result.completed_count == 30 and result.failed_count == 0
    receipt = json.loads((output / "protocol_stability_receipt.json").read_text(encoding="utf-8"))
    assert receipt["stage_counts"] == {
        "correction_evidence": 6, "final_judgment": 6, "initial_private": 6,
        "self_reflection": 6, "social_exposure": 6,
    }


def test_gate_no_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    result = run_manipulation_gate(
        ManipulationFakeProvider(), output_root=output,
        approval_path=_accepted_approval(tmp_path),
    )
    assert result.status == "failed"
    assert result.error_code == "qualification_output_exists"


def test_public_files_do_not_contain_private_truth() -> None:
    paths = [
        ROOT / "src/evicon/conformity_identification_compatibility.py",
        ROOT / "src/evicon/conformity_identification_network_gates.py",
        APPROVAL,
    ]
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in paths).lower()
    assert "evaluatortruthrecord" not in serialized
    assert "api_key_value" not in serialized
