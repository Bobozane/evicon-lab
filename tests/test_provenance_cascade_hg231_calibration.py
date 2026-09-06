from __future__ import annotations

from pathlib import Path

import pytest

from evicon.cascade_real_agent_runner import CascadeRealAgentRunError
from evicon.llm_contract import LLMResponse
from evicon.provenance_cascade_hg2 import sha256_file
from evicon.provenance_cascade_hg23 import HG23FakeProvider
from evicon.provenance_cascade_hg231_calibration import (
    HG231CalibrationRunner,
    execute_real,
    final_preflight,
    run_fake_smoke,
)

ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = ROOT / "results/provenance-cascade-hg231-adoption-identifiability-v1"
OLD_BATCH = ROOT / "results/provenance-cascade-hg21-eligibility-calibration-v2/calibration_batch_record.json"
OLD_LEDGER = ROOT / "results/provenance-cascade-hg21-eligibility-calibration-v2/hg21v2-cascade-hg1-false-majority-20261021-no_intervention/request_ledger.jsonl"


class ShareConflictProvider:
    def complete(self, request):
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content=(
                '{"epistemic_stance":"uncertain","adoption_decision":"adopt_claim","sharing_decision":"share_with_caveat",'
                '"content_ids_used":[],"evidence_ids_used":[],"share_content_id":null}'
            ),
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            latency_ms=1.0,
        )


def test_full_fake_calibration_and_resume_are_complete() -> None:
    report = run_fake_smoke()
    assert report["run_count"] == 16
    assert report["matched_group_count"] == 4
    assert report["logical_request_count"] == report["provider_call_count"] == 288
    assert report["transport_attempt_count"] == 288
    assert report["replay_passed_count"] == 16
    assert report["behavior_observation_count"] == 288
    assert report["resume_additional_provider_call_count"] == 0
    assert report["directive_applied_count"] > 0
    assert report["share_decision_count"] > 0
    assert report["effectiveness_claimed"] is False


def test_final_preflight_is_offline_and_waits_for_execution_authorization(monkeypatch) -> None:
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg231_calibration.ProviderConfig.from_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("provider must not be constructed")),
    )
    report = final_preflight()
    expected = ["output_root_exists"] if FORMAL_ROOT.exists() else ["calibration_network_authorization_required"]
    assert report["blocking_reasons"] == expected
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
        "evicon.provenance_cascade_hg231_calibration.ProviderConfig.from_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("provider must not be constructed")),
    )
    result = execute_real(**kwargs)
    assert result.status == "blocked" and result.error_code == code
    assert result.provider_constructed_count == 0


def test_model_mismatch_rejected_before_provider_construction(monkeypatch) -> None:
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg231_calibration.final_preflight",
        lambda: {"ready_for_network_authorization": True, "blocking_reasons": []},
    )
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg231_calibration.ProviderConfig.from_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("provider must not be constructed")),
    )
    result = execute_real(
        allow_network=True,
        confirm_run=True,
        confirm_request_cap=288,
        confirm_completion_reservation_cap=294912,
        environment={
            "EVICON_LLM_BASE_URL": "https://provider.invalid/v1",
            "EVICON_LLM_MODEL": "wrong-model",
            "EVICON_LLM_API_KEY": "not-logged",
        },
    )
    assert result.status == "blocked" and result.error_code == "provider_model_mismatch"
    assert result.provider_constructed_count == 0


def test_no_overwrite_resume_and_binding_protection(tmp_path: Path) -> None:
    runner = HG231CalibrationRunner()
    root = tmp_path / "hg231"
    records, receipt = runner.run_all(
        provider_factory=lambda _spec: HG23FakeProvider(),
        root=root,
        model_name="gpt-5.6-luna",
        write_receipt=False,
        network="disabled",
    )
    assert len(records) == 16 and receipt.replay_passed_count == 16
    with pytest.raises(CascadeRealAgentRunError, match="output_root_exists"):
        runner.run_all(
            provider_factory=lambda _spec: HG23FakeProvider(),
            root=root,
            model_name="gpt-5.6-luna",
            write_receipt=False,
            network="disabled",
        )
    with pytest.raises(CascadeRealAgentRunError, match="resume_binding_mismatch"):
        runner.run_all(
            provider_factory=lambda _spec: HG23FakeProvider(),
            root=root,
            model_name="different-model",
            resume=True,
            write_receipt=False,
            network="disabled",
        )


def test_share_semantic_parser_invalid_stops_without_recovery(tmp_path: Path) -> None:
    runner = HG231CalibrationRunner()
    root = tmp_path / "invalid"
    with pytest.raises(CascadeRealAgentRunError, match="share_behavior_inconsistent"):
        runner.run_all(
            provider_factory=lambda _spec: ShareConflictProvider(),
            root=root,
            model_name="gpt-5.6-luna",
            write_receipt=True,
            network="disabled",
        )
    assert not (root / "calibration_receipt.json").exists()
    batch = (root / "calibration_batch_record.json").read_text(encoding="utf-8")
    assert '"failure_code": "share_behavior_inconsistent"' in batch


def test_old_failure_hashes_remain_unchanged_when_new_root_exists() -> None:
    assert sha256_file(OLD_BATCH) == "a4a01686c813ab955876607f402e0e4d2e96ba7b0635f0ef0a2087ebb919cba8"
    assert sha256_file(OLD_LEDGER) == "c8724b88c2539c8052f3837c23e5af9e5888dcf2a798758c599fec66eb8987f2"
    if FORMAL_ROOT.exists():
        assert (FORMAL_ROOT / "calibration_batch_record.json").is_file()
