from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.wvs7_baseline_pilot import WVS7BaselineFakeLLMProvider
from evicon.wvs7_protocol_blind_v2_calibration import (
    WVS7ProtocolBlindV2CalibrationRunner,
    WVS7ProtocolBlindV2CalibrationError,
    main,
    prepare_protocol_blind_v2_calibration,
    preflight_protocol_blind_v2_calibration,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/studies/wvs7_english_protocol_blind_v2_calibration.toml"


def test_default_cli_is_network_disabled_and_does_not_create_results(capsys: pytest.CaptureFixture[str]) -> None:
    before = (ROOT / "results").exists()
    status = main(["--config", str(CONFIG)])
    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert payload["status"] == "network_disabled"
    assert payload["provider_call_count"] == 0
    assert (ROOT / "results").exists() is before


def test_preflight_uses_only_future_003_and_rejects_existing_output(tmp_path: Path) -> None:
    prepared = prepare_protocol_blind_v2_calibration(CONFIG, test_output_root=tmp_path)
    assert prepared.preregistration.planned_calibration_run_id == "wvs7-real-baseline-pilot-seed-003"
    prepared, report = preflight_protocol_blind_v2_calibration(CONFIG)
    assert report.request_cap == 384
    assert report.completion_reservation_cap == 51_200
    assert report.estimated_total_token_reference == 190_602


def test_fake_calibration_completes_four_conditions_and_resume_is_idempotent(tmp_path: Path) -> None:
    protected_002 = ROOT / "results/batches/wvs7-real-baseline-pilot-002"
    before_002 = {
        path.relative_to(protected_002): path.read_bytes()
        for path in protected_002.rglob("*") if path.is_file()
    } if protected_002.is_dir() else {}
    prepared = prepare_protocol_blind_v2_calibration(CONFIG, test_output_root=tmp_path)
    provider = WVS7BaselineFakeLLMProvider()
    runner = WVS7ProtocolBlindV2CalibrationRunner(
        prepared,
        agent_provider=provider,
        probe_provider=provider,
    )
    result = runner.run()
    assert result.status == "calibration"
    assert result.completed_run_count == 4
    assert result.matched_groups_complete is True
    first_calls = provider.request_count
    assert first_calls == 384

    ledger_lines = (runner.batch_directory / "request_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in ledger_lines]
    assert {entry["template_version"] for entry in entries if entry["phase"] == "probe"} == {"contextual_value_probe.v2"}
    assert len({entry["fingerprint"] for entry in entries if entry["status"] == "started"}) == 384

    receipt = json.loads((runner.batch_directory / "protocol_blind_v2_calibration_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "calibration"
    assert receipt["development_only"] is True
    assert receipt["not_paper_result"] is True
    assert receipt["no_causal_conclusion"] is True
    assert receipt["pilot_version"] == "protocol_blind_v2"
    assert receipt["v1_002_excluded_from_v2_main_analysis"] is True
    assert receipt["request_ledger"]["completion_reservation_cap"] == 51_200
    assert receipt["request_ledger"]["actual_total_token_count"] is not None
    assert receipt["request_ledger"]["completion_reserved_token_count"] == 51_200
    assert receipt["profile_drift_condition_count"] == 4

    resumed = WVS7ProtocolBlindV2CalibrationRunner(
        prepared,
        agent_provider=provider,
        probe_provider=provider,
    ).run(resume=True)
    assert resumed.status == "calibration"
    assert provider.request_count == first_calls
    if protected_002.is_dir():
        after_002 = {
            path.relative_to(protected_002): path.read_bytes()
            for path in protected_002.rglob("*") if path.is_file()
        }
        assert after_002 == before_002


def test_confirmation_flags_are_required_before_any_provider_or_output(capsys: pytest.CaptureFixture[str]) -> None:
    status = main([
        "--config", str(CONFIG),
        "--allow-network", "--confirm-run",
        "--confirm-request-cap", "1",
        "--confirm-completion-reservation-cap", "2",
        "--confirm-estimated-total-token-reference", "3",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert status == 1
    assert payload["provider_call_count"] == 0
    assert set(payload["blocking_reasons"]) == {
        "confirm_request_cap_must_equal_384",
        "confirm_completion_reservation_cap_must_equal_51200",
        "confirm_estimated_total_token_reference_must_equal_190602",
    }


def test_preparation_rejects_preregistration_parameter_drift(tmp_path: Path) -> None:
    source = CONFIG.read_text(encoding="utf-8")
    drifted = tmp_path / "drifted.toml"
    drifted.write_text(source.replace('reasoning_effort = "none"', 'reasoning_effort = "low"'), encoding="utf-8")
    with pytest.raises(WVS7ProtocolBlindV2CalibrationError):
        prepare_protocol_blind_v2_calibration(drifted, test_output_root=tmp_path)
