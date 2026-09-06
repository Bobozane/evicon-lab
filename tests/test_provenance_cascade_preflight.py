from __future__ import annotations

import json
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

import evicon.provenance_cascade_preflight as hb
from evicon.provenance_cascade_preflight import (
    HBPilotError,
    ProvenanceCascadeHBConfig,
    _validate_runs,
    load_hb_config,
    preflight_hb,
)
from evicon.validate_provenance_cascade_preflight import main

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hb.v1.toml"


def payload() -> dict[str, object]:
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))


def test_valid_hb_config_is_explicit_and_offline() -> None:
    config = load_hb_config(CONFIG)
    assert len(config.scenario_materials) == 4
    assert len(config.matched_groups) == 12
    assert len(config.runs) == 48
    assert config.budget.request_cap == 864
    assert config.budget.completion_reservation_cap == 221184
    report = preflight_hb(CONFIG, environment={"EVICON_LLM_API_KEY": "must-not-be-read"})
    assert report.status == "ready_for_offline_preflight"
    assert report.credential_value_read is False
    assert report.evaluator_truth_loaded is False
    assert report.output_paths_available is True


@pytest.mark.parametrize("field", ["unexpected", "ground_truth_label", "results_path"])
def test_unknown_or_private_fields_are_rejected(field: str) -> None:
    data = payload()
    data[field] = "forbidden"
    with pytest.raises(Exception):
        ProvenanceCascadeHBConfig.model_validate(data)


def test_wrong_budget_is_rejected() -> None:
    data = payload()
    data["budget"]["request_cap"] = 863  # type: ignore[index]
    with pytest.raises(Exception):
        ProvenanceCascadeHBConfig.model_validate(data)


def test_missing_condition_or_seed_coordinate_is_rejected() -> None:
    data = payload()
    data["conditions"] = ["no_intervention", "generic_dissent", "source_blind_controller"]
    with pytest.raises(Exception):
        ProvenanceCascadeHBConfig.model_validate(data)

    config = load_hb_config(CONFIG)
    mutated = config.runs[0].model_copy(update={"condition": "generic_dissent"})
    replacement = config.model_copy(update={"runs": (mutated,) + config.runs[1:]})
    with pytest.raises(HBPilotError) as error:
        _validate_runs(replacement)
    assert error.value.code == "run_coordinate_set_mismatch"


def test_hash_tampering_is_blocked_without_reading_private_material() -> None:
    original = hb._sha256

    def tampered(path: Path) -> str:
        if path.name == "false_majority.toml":
            return "0" * 64
        return original(path)

    with patch.object(hb, "_sha256", side_effect=tampered):
        with pytest.raises(HBPilotError) as error:
            load_hb_config(CONFIG)
    assert error.value.code == "scenario_config_hash_mismatch"


def test_existing_output_root_is_blocked_without_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    existing = tmp_path / "already-there"
    existing.mkdir()
    monkeypatch.setattr(hb, "_output_path", lambda config: existing)
    report = preflight_hb(CONFIG)
    assert report.status == "blocked"
    assert report.output_paths_available is False
    assert "output_root_exists" in report.blocking_reasons


def test_network_confirmation_is_explicit_but_never_executes() -> None:
    blocked = preflight_hb(CONFIG, allow_network=True)
    assert blocked.status == "blocked"
    assert "confirm_run_required" in blocked.blocking_reasons
    assert "confirm_request_cap_mismatch" in blocked.blocking_reasons
    assert "confirm_token_cap_mismatch" in blocked.blocking_reasons

    gated = preflight_hb(
        CONFIG, allow_network=True, confirm_run=True,
        confirm_request_cap=864, confirm_token_cap=221184,
    )
    assert gated.status == "network_gate_validated_no_execution"
    assert "network_execution_deferred_until_hc" in gated.blocking_reasons
    assert gated.credential_value_read is False


def test_cli_safe_summary_and_no_provider(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(CONFIG)]) == 0
    output = capsys.readouterr().out
    parsed = json.loads(output)
    assert parsed["status"] == "ready_for_offline_preflight"
    assert parsed["run_count"] == 48
    lowered = output.lower()
    assert "ground_truth_label" not in lowered
    assert "source_independence_label" not in lowered
    assert "api_key" not in lowered
    assert "prompt" not in lowered


def test_cli_network_gate_never_runs(capsys: pytest.CaptureFixture[str]) -> None:
    code = main([
        "--config", str(CONFIG), "--allow-network", "--confirm-run",
        "--confirm-request-cap", "864", "--confirm-token-cap", "221184",
    ])
    assert code == 1
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["status"] == "network_gate_validated_no_execution"
    assert parsed["credential_value_read"] is False
