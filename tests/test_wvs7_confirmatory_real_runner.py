from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.wvs7_baseline_pilot import WVS7BaselineFakeLLMProvider
from evicon.wvs7_confirmatory_real_runner import (
    WVS7ConfirmatoryRealError,
    WVS7ConfirmatoryRealRunner,
    main,
    prepare_wvs7_confirmatory_real_experiment,
    preflight_wvs7_confirmatory_real_experiment,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/studies/wvs7_english_protocol_blind_v2_confirmatory_plan.toml"
LOCK = ROOT / "outputs/study-locks/wvs7_protocol_blind_v2_confirmatory_plan_lock.json"


def test_default_runner_is_network_disabled_and_does_not_create_results(capsys: pytest.CaptureFixture[str]) -> None:
    before = (ROOT / "results/batches/wvs7-protocol-blind-v2-confirmatory-execution").exists()
    assert main(["--config", str(CONFIG)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"provider_call_count": 0, "status": "network_disabled"}
    assert (ROOT / "results/batches/wvs7-protocol-blind-v2-confirmatory-execution").exists() is before


def test_preflight_is_ready_offline_and_reports_fixed_caps(tmp_path: Path) -> None:
    prepared, report = preflight_wvs7_confirmatory_real_experiment(CONFIG, output_root=tmp_path)
    assert report.status == "ready"
    assert report.run_count == 12
    assert report.request_cap == 1152
    assert report.completion_reservation_cap == 153600
    assert report.estimated_total_token_reference == 571806
    assert prepared.plan_lock.status == "locked"


def test_tampered_plan_lock_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    payload["input_file_sha256"]["confirmatory_manifest"] = "0" * 64
    tampered = tmp_path / "tampered-lock.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WVS7ConfirmatoryRealError, match="confirmatory_plan_lock_mismatch"):
        prepare_wvs7_confirmatory_real_experiment(CONFIG, plan_lock_path=tampered, output_root=tmp_path)


def test_confirmation_parameters_are_required_without_provider_calls(capsys: pytest.CaptureFixture[str]) -> None:
    status = main([
        "--config", str(CONFIG), "--allow-network", "--confirm-run",
        "--confirm-request-cap", "1", "--confirm-completion-reservation-cap", "2",
        "--confirm-estimated-total-token-reference", "3",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert status == 1
    assert payload["provider_call_count"] == 0
    assert set(payload["blocking_reasons"]) == {
        "confirm_request_cap_must_equal_1152",
        "confirm_completion_reservation_cap_must_equal_153600",
        "confirm_estimated_total_token_reference_must_equal_571806",
    }


def test_fake_four_condition_execution_writes_safe_receipt_and_report(tmp_path: Path) -> None:
    prepared = prepare_wvs7_confirmatory_real_experiment(CONFIG, output_root=tmp_path)
    provider = WVS7BaselineFakeLLMProvider()
    summary = WVS7ConfirmatoryRealRunner(
        prepared, agent_provider=provider, probe_provider=provider,
    ).run()
    assert summary.status == "confirmatory"
    assert summary.completed_run_count == 12
    assert provider.request_count == 1152
    batch_dir = Path(summary.receipt_path).parent
    receipt = json.loads((batch_dir / "confirmatory_batch_receipt.json").read_text(encoding="utf-8"))
    assert receipt["confirmatory"] is True
    assert receipt["not_causal_conclusion"] is True
    assert receipt["statistical_analysis_deferred"] is True
    assert receipt["request_ledger"]["provider_call_count"] == 1152
    assert receipt["request_ledger"]["completion_reserved_token_count"] == 153600
    report = json.loads((batch_dir / "offline_evaluation_report_initial_final_v2.json").read_text(encoding="utf-8"))
    assert sum(metric["metric_name"] == "profile_drift" for metric in report["metrics"]) == 12
    assert not any(key in receipt for key in ("prompt", "response", "api_key", "provider_metadata"))


def test_completed_runs_are_not_repeated_on_resume(tmp_path: Path) -> None:
    prepared = prepare_wvs7_confirmatory_real_experiment(CONFIG, output_root=tmp_path)
    first_provider = WVS7BaselineFakeLLMProvider()
    first = WVS7ConfirmatoryRealRunner(prepared, agent_provider=first_provider, probe_provider=first_provider).run()
    second_provider = WVS7BaselineFakeLLMProvider()
    second = WVS7ConfirmatoryRealRunner(prepared, agent_provider=second_provider, probe_provider=second_provider).run(resume=True)
    assert first.status == second.status == "confirmatory"
    assert second_provider.request_count == 0


def test_failed_condition_blocks_reports_until_explicit_resume(tmp_path: Path) -> None:
    prepared = prepare_wvs7_confirmatory_real_experiment(CONFIG, output_root=tmp_path)

    class FailingProvider(WVS7BaselineFakeLLMProvider):
        def complete(self, request):
            raise RuntimeError("local synthetic failure")

    failed_provider = FailingProvider()
    failed = WVS7ConfirmatoryRealRunner(prepared, agent_provider=failed_provider, probe_provider=failed_provider).run()
    assert failed.status == "failed"
    batch_dir = Path(prepared.batch_manifest.batch_output_dir) / prepared.batch_manifest.batch_id
    assert not (batch_dir / "confirmatory_batch_receipt.json").exists()
    assert not (batch_dir / "offline_evaluation_report_initial_final_v2.json").exists()


def test_resume_manifest_binding_rejects_fingerprint_change(tmp_path: Path) -> None:
    prepared = prepare_wvs7_confirmatory_real_experiment(CONFIG, output_root=tmp_path)
    provider = WVS7BaselineFakeLLMProvider()
    WVS7ConfirmatoryRealRunner(prepared, agent_provider=provider, probe_provider=provider).run()
    binding = Path(prepared.batch_manifest.batch_output_dir) / prepared.batch_manifest.batch_id / "confirmatory_execution_binding.json"
    payload = json.loads(binding.read_text(encoding="utf-8"))
    payload["plan_sha256"] = "f" * 64
    binding.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WVS7ConfirmatoryRealError, match="confirmatory_resume_fingerprint_mismatch"):
        WVS7ConfirmatoryRealRunner(prepared, agent_provider=provider, probe_provider=provider).run(resume=True)
