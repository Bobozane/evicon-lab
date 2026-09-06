from __future__ import annotations

import json
import re
from pathlib import Path

from evicon.cascade_agent_protocol_hg2 import parse_hg2_response, render_hg2_turn
from evicon.cascade_agent_protocol_hg21 import HG21_RESPONSE_JSON_SCHEMA, render_hg21_turn
from evicon.openai_provider import TransportResponse
from evicon.provenance_cascade_hg2_compatibility import build_minimal_hg2_context
from evicon.provenance_cascade_hg21 import preflight, run_fake_smoke
from evicon.provenance_cascade_hg21_compatibility import run_compatibility_check
from evicon.provenance_cascade_hg21_compatibility_receipt import register_compatibility_result, validate_receipt

ROOT = Path(__file__).resolve().parents[1]
APPROVAL = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg21_approval_template.toml"
ENV = {"EVICON_LLM_BASE_URL": "https://provider.invalid/v1", "EVICON_LLM_MODEL": "compat-model", "EVICON_LLM_API_KEY": "secret"}


class FakeTransport:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def post(self, url, headers, payload, timeout_seconds):
        self.calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout_seconds})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _accepted_approval(tmp_path: Path) -> Path:
    raw = APPROVAL.read_text(encoding="utf-8")
    raw = raw.replace('acceptance_status = "pending"', 'acceptance_status = "accepted"')
    raw = raw.replace('accepted_by = ""', 'accepted_by = "synthetic-reviewer"')
    raw = re.sub(r'^(confirm_[a-z0-9_]+) = false$', r'\1 = true', raw, flags=re.MULTILINE)
    raw = re.sub(r'^compatibility_receipt_sha256 = "[0-9a-f]*"$', 'compatibility_receipt_sha256 = ""', raw, flags=re.MULTILINE)
    path = tmp_path / "accepted.toml"
    path.write_text(raw, encoding="utf-8")
    return path


def _pending_approval(tmp_path: Path) -> Path:
    raw = APPROVAL.read_text(encoding="utf-8")
    raw = raw.replace('acceptance_status = "accepted"', 'acceptance_status = "pending"')
    raw = re.sub(r'^accepted_by = ".*"$', 'accepted_by = ""', raw, flags=re.MULTILINE)
    raw = re.sub(r'^(confirm_[a-z0-9_]+) = true$', r'\1 = false', raw, flags=re.MULTILINE)
    path = tmp_path / "pending.toml"
    path.write_text(raw, encoding="utf-8")
    return path


def _content(*, duplicate: bool = False) -> str:
    ids = ["hg11-compatibility-content-01"] * (2 if duplicate else 1)
    return json.dumps({
        "epistemic_stance": "uncertain", "behavioral_decision": "share_with_caveat",
        "content_ids_used": ids, "evidence_ids_used": ["hg11-compatibility-evidence-01"],
        "share_content_id": "hg11-compatibility-content-01",
    })


def _response(content: str | None = None) -> str:
    return json.dumps({
        "model": "compat-model", "choices": [{"message": {"content": content or _content()}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 25, "total_tokens": 45},
    })


def test_amendment_changes_only_transport_schema_and_request_identity() -> None:
    context = build_minimal_hg2_context()
    old, new = render_hg2_turn(context), render_hg21_turn(context)
    assert old.system_prompt == new.system_prompt
    assert old.user_prompt == new.user_prompt
    assert old.request_id != new.request_id
    assert "uniqueItems" not in json.dumps(HG21_RESPONSE_JSON_SCHEMA)
    assert set(HG21_RESPONSE_JSON_SCHEMA["properties"]) == {
        "epistemic_stance", "behavioral_decision", "content_ids_used", "evidence_ids_used", "share_content_id",
    }


def test_local_parser_still_rejects_duplicate_ids() -> None:
    parsed, error = parse_hg2_response(_content(duplicate=True), build_minimal_hg2_context())
    assert parsed is None and error is not None


def test_pending_approval_blocks_before_provider_construction(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg21_compatibility.ProviderConfig.from_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("provider must not be constructed")),
    )
    result = run_compatibility_check(allow_network=True, approval_path=_pending_approval(tmp_path))
    assert result.status == "blocked"
    assert result.provider_error_code == "exact_hash_human_approval_required"
    assert result.attempt_count == 0


def test_provider_subset_payload_succeeds_with_one_fake_transport_call(tmp_path: Path) -> None:
    approval = _accepted_approval(tmp_path)
    transport = FakeTransport([TransportResponse(200, _response())])
    result = run_compatibility_check(allow_network=True, environment=ENV, transport=transport, approval_path=approval)
    assert result.status == "completed" and result.parser_valid is True
    assert result.attempt_count == 1 and len(transport.calls) == 1
    payload = transport.calls[0]["payload"]
    schema = payload["response_format"]["json_schema"]
    assert schema["name"] == "cascade_agent_epistemic_behavior_response_v1_provider_subset"
    assert "uniqueItems" not in json.dumps(schema["schema"])
    assert payload["max_tokens"] == 1024 and payload["temperature"] == 0.2 and payload["seed"] == 20261021
    assert transport.calls[0]["timeout"] == 5.0


def test_response_format_rejection_is_safe_and_not_retried(tmp_path: Path) -> None:
    approval = _accepted_approval(tmp_path)
    transport = FakeTransport([
        TransportResponse(400, '{"error":{"message":"response_format json_schema unsupported"}}'),
        TransportResponse(200, _response()),
    ])
    result = run_compatibility_check(allow_network=True, environment=ENV, transport=transport, approval_path=approval)
    assert result.status == "provider_error"
    assert result.provider_error_code == "response_format_unsupported"
    assert result.attempt_count == 1 and len(transport.calls) == 1
    assert "test-secret" not in result.model_dump_json().lower()


def test_receipt_registration_is_safe_and_hash_bound(tmp_path: Path) -> None:
    approval = _accepted_approval(tmp_path)
    target = ROOT / "outputs/study-locks/_test_hg21_compatibility_receipt.json"
    if target.exists():
        target.unlink()
    result = run_compatibility_check(
        allow_network=True, environment=ENV,
        transport=FakeTransport([TransportResponse(200, _response())]), approval_path=approval,
    )
    try:
        digest, _ = register_compatibility_result(result, receipt_path=target, approval_path=approval)
        receipt = validate_receipt(target, expected_hash=digest)
        assert receipt.schema_name == "cascade_agent_epistemic_behavior_response_v1_provider_subset"
        dumped = target.read_text(encoding="utf-8").lower()
        for forbidden in ("system_prompt", "user_prompt", "request_id", "api_key", "authorization", "provider_metadata", "ground_truth_label"):
            assert forbidden not in dumped
    finally:
        if target.exists():
            target.unlink()


def test_fake_smoke_and_preflight_are_offline_and_do_not_create_results() -> None:
    output = ROOT / "results/provenance-cascade-hg21-eligibility-calibration-v1"
    before = output.exists()
    smoke = run_fake_smoke()
    report = preflight()
    assert smoke["status"] == "offline_fake_smoke_passed"
    assert smoke["fake_replay_passed_count"] == 16
    assert report["blocking_reasons"] == (["one_seed_real_eligibility_calibration_required", "output_root_exists"] if before else ["one_seed_real_eligibility_calibration_required"])
    assert report["network"] == "disabled" and report["provider_constructed"] is False
    assert output.exists() is before


def test_parent_hg2_hashes_remain_locked() -> None:
    from evicon.provenance_cascade_hg2 import sha256_file
    assert sha256_file(ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg2_calibration.v1.toml") == "ce0ca6c87fa23ed818bec55a5c061f45fb18b24acab311e85d793762f3548bc0"
    assert sha256_file(ROOT / "src/evicon/cascade_agent_protocol_hg2.py") == "dfe7ca194b61c3b5f0b3c44fd569cbdc7e0247dea785399869b5ea26d7c6f172"
