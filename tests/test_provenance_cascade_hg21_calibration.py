from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.cascade_real_agent_runner import CascadeRealAgentRunError
from evicon.llm_contract import LLMProviderError, LLMResponse, ProviderErrorCode
from evicon.provenance_cascade_hg2 import HG2FakeProvider
from evicon.provenance_cascade_hg21_calibration import (
    HG21EligibilityCalibrationRunner,
    execute_real,
    final_preflight,
    run_fake_smoke,
)

ROOT = Path(__file__).resolve().parents[1]


class FailingProvider:
    def complete(self, request):
        raise LLMProviderError(ProviderErrorCode.CONNECTION_FAILURE, "redacted")


class InvalidProvider:
    def complete(self, request):
        return LLMResponse(
            request_id=request.request_id, model_name=request.model_name,
            content='{"epistemic_stance":', finish_reason="stop",
            prompt_tokens=10, completion_tokens=5, total_tokens=15, latency_ms=1.0,
        )


def test_fake_calibration_is_complete_resumable_and_safe() -> None:
    report = run_fake_smoke()
    assert report["run_count"] == report["resumed_run_count"] == 16
    assert report["matched_group_count"] == 4
    assert report["logical_request_count"] == report["provider_call_count"] == 288
    assert report["resume_additional_provider_call_count"] == 0
    assert report["replay_passed_count"] == 16
    assert report["behavior_observation_count"] == 288
    assert report["round0_non_defer_behavior_count"] > 0
    assert report["share_decision_count"] > 0
    assert report["effectiveness_claimed"] is False


def test_final_preflight_is_offline_and_only_waits_for_execution_authorization(monkeypatch) -> None:
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg21_calibration.ProviderConfig.from_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("provider must not be constructed")),
    )
    report = final_preflight()
    root_exists = (ROOT / "results/provenance-cascade-hg21-eligibility-calibration-v1").exists()
    assert report["blocking_reasons"] == (["output_root_exists"] if root_exists else [])
    assert report["network"] == "disabled"
    assert report["provider_constructed"] is False and report["api_key_read"] is False


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({}, "allow_network_required"),
        ({"allow_network": True}, "confirm_run_required"),
        ({"allow_network": True, "confirm_run": True, "confirm_request_cap": 287}, "confirm_request_cap_must_equal_288"),
        ({"allow_network": True, "confirm_run": True, "confirm_request_cap": 288, "confirm_completion_reservation_cap": 1}, "confirm_completion_reservation_cap_must_equal_294912"),
    ],
)
def test_real_gate_rejects_before_provider_construction(monkeypatch, kwargs, code) -> None:
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg21_calibration.ProviderConfig.from_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("provider must not be constructed")),
    )
    result = execute_real(**kwargs)
    assert result.status == "blocked" and result.error_code == code
    assert result.provider_constructed_count == 0


def test_no_overwrite_and_resume_binding_protection(tmp_path: Path) -> None:
    runner = HG21EligibilityCalibrationRunner()
    root = tmp_path / "calibration"
    runner.run_all(provider_factory=lambda _spec: HG2FakeProvider(), root=root, model_name="fake", write_receipt=False, network="disabled")
    with pytest.raises(CascadeRealAgentRunError, match="output_root_exists"):
        runner.run_all(provider_factory=lambda _spec: HG2FakeProvider(), root=root, model_name="fake", write_receipt=False, network="disabled")
    with pytest.raises(CascadeRealAgentRunError, match="resume_binding_mismatch"):
        runner.run_all(provider_factory=lambda _spec: HG2FakeProvider(), root=root, model_name="changed", resume=True, write_receipt=False, network="disabled")


def test_failed_transport_can_resume_without_replaying_completed_runs(tmp_path: Path) -> None:
    runner = HG21EligibilityCalibrationRunner()
    root = tmp_path / "calibration"
    first = True
    def failing_factory(_spec):
        nonlocal first
        if first:
            first = False
            return FailingProvider()
        return HG2FakeProvider()
    with pytest.raises(CascadeRealAgentRunError, match="connection_failure"):
        runner.run_all(provider_factory=failing_factory, root=root, model_name="fake", write_receipt=False, network="disabled")
    records, receipt = runner.run_all(
        provider_factory=lambda _spec: HG2FakeProvider(), root=root,
        model_name="fake", resume=True, write_receipt=False, network="disabled",
    )
    assert len(records) == 16 and receipt.logical_request_count == 288
    assert receipt.transport_attempt_count == 289
    assert receipt.replay_passed_count == 16


def test_parser_invalid_stops_without_receipt_or_recovery(tmp_path: Path) -> None:
    runner = HG21EligibilityCalibrationRunner()
    root = tmp_path / "calibration"
    with pytest.raises(CascadeRealAgentRunError, match="malformed_json"):
        runner.run_all(provider_factory=lambda _spec: InvalidProvider(), root=root, model_name="fake", write_receipt=True, network="disabled")
    assert not (root / "calibration_receipt.json").exists()
    batch = json.loads((root / "calibration_batch_record.json").read_text(encoding="utf-8"))
    assert batch["status"] == "failed"
    assert batch["failure_code"] == "malformed_json"


def test_formal_v1_failure_is_preserved_without_complete_receipt() -> None:
    root = ROOT / "results/provenance-cascade-hg21-eligibility-calibration-v1"
    if not root.exists():
        return
    batch = json.loads((root / "calibration_batch_record.json").read_text(encoding="utf-8"))
    assert batch["status"] == "failed"
    assert not (root / "calibration_receipt.json").exists()
