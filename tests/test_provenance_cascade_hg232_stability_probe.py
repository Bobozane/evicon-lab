from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.cascade_real_agent_runner import CascadeRealAgentRunError
from evicon.llm_contract import LLMResponse
from evicon.provenance_cascade_hg2 import sha256_file
from evicon.provenance_cascade_hg23 import HG23FakeProvider
from evicon.provenance_cascade_hg232_calibration import final_preflight as calibration_preflight
from evicon.provenance_cascade_hg232_stability_contract import (
    HG232ProbeError,
    load_probe_config,
    validate_probe_receipt,
)
from evicon.provenance_cascade_hg232_stability_probe import (
    execute_real,
    preflight,
    run_fake_smoke,
    run_probe,
)

ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = ROOT / "results/provenance-cascade-hg232-protocol-stability-probe-v1"


class LengthProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        response = HG23FakeProvider().complete(request)
        return response.model_copy(update={"finish_reason": "length", "completion_tokens": 100, "total_tokens": 200})


class AtLimitProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        response = HG23FakeProvider().complete(request)
        return response.model_copy(update={"finish_reason": "stop", "completion_tokens": 2048, "total_tokens": 2148})


class MalformedProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content="{",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=3,
            total_tokens=13,
            latency_ms=1.0,
        )


def test_probe_config_is_fixed_and_separate() -> None:
    config, _ = load_probe_config()
    assert config.scenario_id == "cascade-hg1-true-minority-correction"
    assert config.condition == "generic_dissent"
    assert config.logical_request_cap == 18
    assert config.completion_reservation_cap == 36864
    assert config.max_tokens == 2048
    assert config.output_root != "results/provenance-cascade-hg232-adoption-identifiability-v1"
    assert config.run_namespace == "hg232-stability"


def test_preflight_is_offline_and_ready_after_authorization(monkeypatch) -> None:
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg232_stability_probe.ProviderConfig.from_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("provider constructed")),
    )
    report = preflight()
    assert report["blocking_reasons"] == (["protocol_stability_probe_output_exists"] if FORMAL_ROOT.exists() else [])
    assert report["ready_for_protocol_stability_probe"] is (not FORMAL_ROOT.exists())
    assert report["network"] == "disabled"
    assert report["provider_constructed"] is False
    assert report["api_key_read"] is False


def test_full_fake_probe_passes_all_three_replays() -> None:
    report = run_fake_smoke()
    assert report["run_count"] == 1
    assert report["logical_request_count"] == report["provider_call_count"] == 18
    assert report["transport_attempt_count"] == 18
    assert report["cascade_replay_status"] == "passed"
    assert report["application_replay_status"] == "passed"
    assert report["outcome_replay_status"] == "passed"
    assert report["sensitive_scan_status"] == "passed"


def test_fake_receipt_cannot_unlock_real_gate(tmp_path: Path) -> None:
    root = tmp_path / "probe"
    run_probe(provider=HG23FakeProvider(), root=root, model_name="gpt-5.6-luna", network="disabled")
    with pytest.raises(HG232ProbeError, match="protocol_stability_probe_receipt_not_real"):
        validate_probe_receipt(root / "protocol_stability_probe_receipt.json")


@pytest.mark.parametrize(
    ("provider", "code"),
    [
        (LengthProvider(), "finish_reason_length"),
        (AtLimitProvider(), "completion_token_limit_reached"),
        (MalformedProvider(), "malformed_json"),
    ],
)
def test_invalid_or_truncated_response_stops_immediately(tmp_path: Path, provider, code: str) -> None:
    root = tmp_path / code
    with pytest.raises(CascadeRealAgentRunError, match=code):
        run_probe(provider=provider, root=root, model_name="gpt-5.6-luna", network="disabled")
    assert provider.calls == 1
    assert not (root / "protocol_stability_probe_receipt.json").exists()
    audit_path = next(root.rglob("response_audit.jsonl"))
    audit = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert set(("finish_reason", "http_status_class", "completion_tokens", "completion_token_limit_reached")) <= set(audit)
    raw = audit_path.read_text(encoding="utf-8").lower()
    for forbidden in ("system_prompt", "user_prompt", "raw_response", "api_key", "authorization", "ground_truth_label"):
        assert forbidden not in raw


def test_no_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "probe"
    run_probe(provider=HG23FakeProvider(), root=root, model_name="gpt-5.6-luna", network="disabled")
    with pytest.raises(HG232ProbeError, match="protocol_stability_probe_output_exists"):
        run_probe(provider=HG23FakeProvider(), root=root, model_name="gpt-5.6-luna", network="disabled")


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({}, "allow_network_required"),
        ({"allow_network": True}, "confirm_run_required"),
        ({"allow_network": True, "confirm_run": True, "confirm_request_cap": 17}, "confirm_request_cap_must_equal_18"),
        ({"allow_network": True, "confirm_run": True, "confirm_request_cap": 18, "confirm_completion_reservation_cap": 1}, "confirm_completion_reservation_cap_must_equal_36864"),
    ],
)
def test_real_gate_rejects_before_provider_construction(monkeypatch, kwargs, code: str) -> None:
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg232_stability_probe.ProviderConfig.from_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("provider constructed")),
    )
    result = execute_real(**kwargs)
    assert result["status"] == "blocked" and result["error_code"] == code
    assert result["provider_constructed"] is False


def test_full_calibration_retains_independent_probe_block() -> None:
    if FORMAL_ROOT.exists():
        pytest.skip("real probe output exists")
    report = calibration_preflight()
    assert "protocol_stability_probe_required" in report["blocking_reasons"]
    assert report["protocol_stability_probe_receipt_sha256"] is None


def test_formal_probe_output_is_absent_or_has_valid_real_receipt() -> None:
    if FORMAL_ROOT.exists():
        receipt = validate_probe_receipt(FORMAL_ROOT / "protocol_stability_probe_receipt.json")
        assert receipt.status == "passed" and receipt.network == "enabled"
