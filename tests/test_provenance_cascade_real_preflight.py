from __future__ import annotations

import json
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

import evicon.provenance_cascade_real_preflight as final_gate
from evicon.provenance_cascade_real_preflight import (
    HDFinalPreflightError,
    HDResearcherApproval,
    final_preflight_hd,
    load_hd_approval,
)
from evicon.validate_provenance_cascade_real_preflight import main

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd.v2.toml"
APPROVAL = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd_approval_2026-08-21.toml"
OUTPUT_ROOT = ROOT / "results/provenance-cascade-pilot-hd-v2"


def _payload() -> dict[str, object]:
    return tomllib.loads(APPROVAL.read_text(encoding="utf-8"))


def test_researcher_approval_binds_all_materials_and_keeps_network_separate() -> None:
    approval, config, amendment, receipt = load_hd_approval(APPROVAL)
    assert approval.status == "approved"
    assert approval.network_execution_authorized is False
    assert approval.network_authorization_required_separately is True
    assert len(approval.approved_materials) == 4
    assert config.budget.request_cap == approval.approved_scope.logical_request_cap == 864
    assert config.budget.completion_reservation_cap == approval.approved_scope.completion_reservation_cap == 221184
    assert amendment.amendment_version == approval.amendment_version
    assert receipt.smoke_replay_status == "passed"


def test_final_preflight_passes_but_does_not_authorize_execution_or_create_results() -> None:
    original = Path.exists

    def available(path: Path) -> bool:
        if path.resolve() == OUTPUT_ROOT.resolve():
            return False
        return original(path)

    with patch.object(Path, "exists", available):
        report = final_preflight_hd(CONFIG, APPROVAL)
    assert report.status == "passed_waiting_for_network_authorization"
    assert report.design_approved is True
    assert report.ready_for_explicit_network_authorization is True
    assert report.ready_for_execution is False
    assert report.network_execution_authorized is False
    assert report.network == "disabled"
    assert report.provider_constructed is False
    assert report.environment_read is False
    assert report.credential_value_read is False
    assert report.result_directory_created is False


@pytest.mark.parametrize("field", ["unexpected", "ground_truth_label", "provider_metadata", "prompt"])
def test_unknown_or_private_approval_fields_are_rejected(field: str) -> None:
    payload = _payload()
    payload[field] = "forbidden"
    with pytest.raises(Exception):
        HDResearcherApproval.model_validate(payload)


def test_missing_confirmation_or_network_authorization_in_approval_is_rejected() -> None:
    payload = _payload()
    payload["confirm_resume_contract"] = False
    with pytest.raises(Exception):
        HDResearcherApproval.model_validate(payload)
    payload = _payload()
    payload["network_execution_authorized"] = True
    with pytest.raises(Exception):
        HDResearcherApproval.model_validate(payload)


def test_approved_hash_tampering_blocks_preflight() -> None:
    original = final_gate._sha256

    def tampered(path: Path) -> str:
        if path.name == "provenance_cascade_pilot_hd_amendment_receipt.json":
            return "0" * 64
        return original(path)

    with patch.object(final_gate, "_sha256", side_effect=tampered):
        with pytest.raises(HDFinalPreflightError) as error:
            load_hd_approval(APPROVAL)
    assert error.value.code == "approved_receipt_hash_mismatch"


def test_existing_output_root_blocks_without_modifying_it() -> None:
    original = Path.exists

    def exists(path: Path) -> bool:
        if path.resolve() == OUTPUT_ROOT.resolve():
            return True
        return original(path)

    with patch.object(Path, "exists", exists):
        report = final_preflight_hd(CONFIG, APPROVAL)
    assert report.status == "blocked"
    assert report.blocking_reasons == ("output_root_exists",)
    assert report.result_directory_created is False


def test_cli_is_safe_offline_and_has_no_network_option(capsys: pytest.CaptureFixture[str]) -> None:
    original = Path.exists

    def available(path: Path) -> bool:
        if path.resolve() == OUTPUT_ROOT.resolve():
            return False
        return original(path)

    with patch.object(Path, "exists", available):
        assert main(["--config", str(CONFIG), "--approval", str(APPROVAL)]) == 0
    output = capsys.readouterr().out
    parsed = json.loads(output)
    assert parsed["status"] == "passed_waiting_for_network_authorization"
    assert parsed["network"] == "disabled"
    assert parsed["ready_for_execution"] is False
    lowered = output.lower()
    for token in ("ground_truth_label", "source_independence_label", "api_key", "prompt", "provider_metadata"):
        assert token not in lowered
    with pytest.raises(SystemExit):
        main(["--config", str(CONFIG), "--approval", str(APPROVAL), "--allow-network"])
