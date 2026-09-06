from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.cascade_agent_protocol_v2_receipt import (
    COMPATIBILITY_MODULE_VERSION,
    CompatibilityReceiptError,
    DEFAULT_RECEIPT_RELATIVE,
    register_receipt,
    sha256_file,
    validate_receipt,
)
from evicon.cascade_agent_protocol_v2_pilot import final_preflight

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd2.v1.toml"
PROTOCOL = ROOT / "src/evicon/cascade_agent_protocol_v2.py"
RECEIPT = ROOT / DEFAULT_RECEIPT_RELATIVE
APPROVAL = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd2_approval_template.toml"


def test_registered_receipt_has_only_safe_observed_fields() -> None:
    receipt = validate_receipt(RECEIPT, expected_hash=sha256_file(RECEIPT), config_path=CONFIG, protocol_path=PROTOCOL)
    assert receipt.compatibility_module_version == COMPATIBILITY_MODULE_VERSION
    assert receipt.prompt_tokens == 393
    assert receipt.completion_tokens == 158
    assert receipt.total_tokens == 551
    assert receipt.max_tokens == 256
    assert receipt.temperature == 0.2
    assert receipt.seed == 20260911
    assert receipt.max_retries == 0
    raw = RECEIPT.read_text(encoding="utf-8").lower()
    for forbidden in ("system_prompt", "user_prompt", "api_key", "authorization", "headers", "provider_metadata", "request_id", "ground_truth_label"):
        assert forbidden not in raw


def test_receipt_registration_never_overwrites(tmp_path) -> None:
    target = tmp_path / "receipt.json"
    # Registration is restricted to study-lock paths, so use a temporary lock-like path under the repository.
    target = ROOT / "outputs" / "study-locks" / "_test_h_e_receipt.json"
    if target.exists():
        target.unlink()
    try:
        _, digest = register_receipt(output_path=target, config_path=CONFIG, protocol_path=PROTOCOL)
        assert digest == sha256_file(target)
        with pytest.raises(CompatibilityReceiptError) as error:
            register_receipt(output_path=target, config_path=CONFIG, protocol_path=PROTOCOL)
        assert error.value.code == "receipt_exists"
    finally:
        if target.exists():
            target.unlink()


def test_tampered_receipt_hash_is_rejected(tmp_path) -> None:
    target = tmp_path / "receipt.json"
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    payload["config_sha256"] = "0" * 64
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CompatibilityReceiptError) as error:
        validate_receipt(target, config_path=CONFIG, protocol_path=PROTOCOL)
    assert error.value.code == "compatibility_config_hash_mismatch"


def test_h_e_preflight_accepts_registered_receipt_and_signed_design() -> None:
    result = final_preflight(CONFIG, APPROVAL, resume=True)
    assert result["blocking_reasons"] == []
    assert result["compatibility_receipt_sha256"] == sha256_file(RECEIPT)
    assert result["ready_for_network_authorization"] is True
    assert result["network"] == "disabled"
    assert result["provider_constructed"] is False
    assert result["transport_called"] is False
