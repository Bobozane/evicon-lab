from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from evicon.cascade_agent_protocol_hg2 import BehavioralDecision
from evicon.cascade_agent_protocol_hg22 import (
    HG22ParserError,
    HG22_RESPONSE_JSON_SCHEMA,
    parse_hg22_response,
    render_hg22_turn,
)
from evicon.openai_provider import TransportResponse
from evicon.provenance_cascade_hg2 import sha256_file
from evicon.provenance_cascade_hg2_compatibility import build_minimal_hg2_context
from evicon.provenance_cascade_hg22 import (
    DEFAULT_AMENDMENT_RECEIPT,
    DEFAULT_APPROVAL,
    DEFAULT_CONFIG,
    load_config,
    preflight,
    run_fake_smoke,
)
from evicon.provenance_cascade_hg22_compatibility import run_compatibility_check

ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT / "results/provenance-cascade-hg21-eligibility-calibration-v2"
NEW_ROOT = ROOT / "results/provenance-cascade-hg22-eligibility-calibration-v1"
ENV = {
    "EVICON_LLM_BASE_URL": "https://provider.invalid/v1",
    "EVICON_LLM_MODEL": "gpt-5.6-luna",
    "EVICON_LLM_API_KEY": "not-logged",
}


class FakeTransport:
    def __init__(self, response: TransportResponse) -> None:
        self.response = response
        self.calls = []

    def post(self, url, headers, payload, timeout_seconds):
        self.calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout_seconds})
        return self.response


def _payload(**updates):
    context = build_minimal_hg2_context()
    content_id = context.public_context.contents[0].content_id
    payload = {
        "epistemic_stance": "uncertain",
        "behavioral_decision": "share_with_caveat",
        "content_ids_used": [content_id],
        "evidence_ids_used": [],
        "share_content_id": content_id,
    }
    payload.update(updates)
    return context, payload


def _accepted_approval(tmp_path: Path) -> Path:
    raw = (ROOT / DEFAULT_APPROVAL).read_text(encoding="utf-8")
    raw = raw.replace('acceptance_status = "pending"', 'acceptance_status = "accepted"')
    raw = raw.replace('accepted_by = ""', 'accepted_by = "synthetic-reviewer"')
    raw = re.sub(r'^(confirm_[a-z0-9_]+) = false$', r'\1 = true', raw, flags=re.MULTILINE)
    path = tmp_path / "accepted.toml"
    path.write_text(raw, encoding="utf-8")
    return path


def _provider_body(payload: dict[str, object]) -> str:
    return json.dumps({
        "model": "gpt-5.6-luna",
        "choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 25, "total_tokens": 45},
    })


def test_diagnostic_lock_is_safe_and_preserves_old_failure() -> None:
    config, path = load_config()
    assert sha256_file(path) == "2d93fe9db87f36f960598f84ed66b3e9e01f1d7ab8a63886f702f06204eeafb9"
    assert sha256_file(DEFAULT_AMENDMENT_RECEIPT) == "140da82706429df74ba8d007aacdb0be7760f1737ee91918bd7639312f904ccc"
    assert sha256_file(config.failed_hg21v2_batch_record_path) == config.failed_hg21v2_batch_record_sha256
    assert sha256_file(config.failed_hg21v2_request_ledger_path) == config.failed_hg21v2_request_ledger_sha256
    assert config.diagnostic_missing_field_ruled_out is True
    assert config.diagnostic_share_type_vs_semantic_not_losslessly_distinguishable is True
    assert config.diagnostic_most_consistent_with == "share_behavior_inconsistent"


@pytest.mark.parametrize(
    ("updates", "expected_behavior"),
    [
        ({}, BehavioralDecision.SHARE_WITH_CAVEAT),
        ({"behavioral_decision": "share",}, BehavioralDecision.SHARE),
        ({"behavioral_decision": "do_not_share", "share_content_id": None}, BehavioralDecision.DO_NOT_SHARE),
        ({"behavioral_decision": "defer_action", "share_content_id": None}, BehavioralDecision.DEFER_ACTION),
    ],
)
def test_valid_share_and_nonshare_contracts(updates, expected_behavior) -> None:
    context, payload = _payload(**updates)
    parsed, error = parse_hg22_response(json.dumps(payload), context)
    assert error is None and parsed is not None
    assert parsed.behavioral_decision is expected_behavior


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing", HG22ParserError.MISSING_FIELD),
        ("type", HG22ParserError.SHARE_FIELD_TYPE),
        ("sharing_null", HG22ParserError.SHARE_BEHAVIOR_INCONSISTENT),
        ("nonsharing_id", HG22ParserError.SHARE_BEHAVIOR_INCONSISTENT),
    ],
)
def test_missing_type_and_cross_field_failures_are_distinct(mutation, expected) -> None:
    context, payload = _payload()
    if mutation == "missing":
        payload.pop("share_content_id")
    elif mutation == "type":
        payload["share_content_id"] = 9
    elif mutation == "sharing_null":
        payload["share_content_id"] = None
    else:
        payload["behavioral_decision"] = "do_not_share"
    parsed, error = parse_hg22_response(json.dumps(payload), context)
    assert parsed is None and error is expected


def test_schema_is_structural_and_local_validator_is_semantic() -> None:
    serialized = json.dumps(HG22_RESPONSE_JSON_SCHEMA)
    assert '"if"' not in serialized and '"then"' not in serialized
    assert set(HG22_RESPONSE_JSON_SCHEMA["required"]) == {
        "epistemic_stance", "behavioral_decision", "content_ids_used", "evidence_ids_used", "share_content_id",
    }
    context, _ = _payload()
    request = render_hg22_turn(context)
    assert request.request_id.startswith("hg22-")
    assert "share_content_id MUST be null" in request.system_prompt
    assert "ground_truth_label" not in request.user_prompt
    assert "source_independence_label" not in request.user_prompt


def test_offline_smoke_and_preflight_do_not_construct_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg22_compatibility.ProviderConfig.from_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("provider must not be constructed")),
    )
    before_exists = NEW_ROOT.exists()
    before_batch = sha256_file(NEW_ROOT / "calibration_batch_record.json") if before_exists else None
    smoke = run_fake_smoke()
    report = preflight()
    disabled = run_compatibility_check()
    assert smoke["case_count"] == smoke["case_passed_count"] == 7
    assert "real_eligibility_calibration_not_authorized" in report["blocking_reasons"]
    assert set(report["blocking_reasons"]).issubset({
        "exact_hash_human_approval_required",
        "provider_compatibility_check_required",
        "provider_compatibility_receipt_hash_mismatch",
        "real_eligibility_calibration_not_authorized",
        "output_root_exists",
    })
    assert disabled.status == "network_disabled"
    assert report["network"] == "disabled" and report["provider_constructed"] is False
    assert NEW_ROOT.exists() is before_exists
    if before_batch is not None:
        assert sha256_file(NEW_ROOT / "calibration_batch_record.json") == before_batch


def test_fake_transport_uses_new_schema_once(tmp_path: Path) -> None:
    context, payload = _payload()
    transport = FakeTransport(TransportResponse(200, _provider_body(payload)))
    result = run_compatibility_check(
        allow_network=True,
        environment=ENV,
        transport=transport,
        approval_path=_accepted_approval(tmp_path),
    )
    assert result.status == "completed" and result.parser_valid is True
    assert result.attempt_count == len(transport.calls) == 1
    sent = transport.calls[0]["payload"]
    assert sent["response_format"]["json_schema"]["name"] == "cascade_agent_epistemic_behavior_share_contract_v2"
    assert sent["max_tokens"] == 1024 and sent["temperature"] == 0.2 and sent["seed"] == 20261021
    assert transport.calls[0]["timeout"] == 5.0


def test_config_rejects_unknown_fields(tmp_path: Path) -> None:
    raw = (ROOT / DEFAULT_CONFIG).read_text(encoding="utf-8")
    path = tmp_path / "invalid.toml"
    path.write_text(raw + '\nprompt = "forbidden"\n', encoding="utf-8")
    with pytest.raises(Exception, match="hg22_config_invalid"):
        load_config(path)


def test_old_failure_remains_incomplete_and_unmodified() -> None:
    batch = json.loads((OLD_ROOT / "calibration_batch_record.json").read_text(encoding="utf-8"))
    assert batch["status"] == "failed"
    assert batch["failure_code"] == "share_field"
    assert not (OLD_ROOT / "calibration_receipt.json").exists()
