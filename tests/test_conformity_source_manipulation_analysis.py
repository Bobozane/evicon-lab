from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import pytest

from evicon.conformity_identification import IdentificationError
from evicon.conformity_source_manipulation_analysis import (
    DEFAULT_ANALYSIS_CONFIG,
    analyze_source_manipulation_gate,
    safe_analysis_preflight,
    write_analysis_report,
)

ROOT = Path(__file__).resolve().parents[1]


def _config_for_tmp(tmp_path: Path) -> Path:
    inputs = {
        "gate": ROOT / "configs/provenance_cascade/identification/conformity_source_manipulation_gate.v1.toml",
        "amendment": ROOT / "configs/provenance_cascade/identification/conformity_source_manipulation_timeout_resume.v3.toml",
        "receipt": ROOT / "outputs/conformity-source-manipulation-v1/source_manipulation_resume_v3_receipt.json",
        "ledger": ROOT / "outputs/conformity-source-manipulation-v1/request_ledger.jsonl",
    }
    copied = {}
    for name, source in inputs.items():
        target = tmp_path / source.name
        shutil.copyfile(source, target)
        copied[name] = target
    raw = (ROOT / DEFAULT_ANALYSIS_CONFIG).read_text(encoding="utf-8")
    for name, target in copied.items():
        raw = re.sub(rf'{name if name != "gate" else "gate_config"}_path = "[^"]+"', f'{name if name != "gate" else "gate_config"}_path = "{target}"', raw)
        raw = re.sub(rf'{name if name != "gate" else "gate_config"}_sha256 = "[0-9a-f]{{64}}"', f'{name if name != "gate" else "gate_config"}_sha256 = "{hashlib.sha256(target.read_bytes()).hexdigest()}"', raw)
    target = tmp_path / "analysis.toml"
    target.write_text(raw, encoding="utf-8")
    return target


def test_completed_gate_is_audited_but_not_promoted_to_behavior_effect() -> None:
    result = analyze_source_manipulation_gate()
    assert result["status"] == "completed_with_semantic_audit_limit"
    assert result["logical_request_count"] == 12
    assert result["transport_attempt_count"] == 15
    assert result["behavior_effect_estimated"] is False
    assert result["structural_response_score"] == "not_observable_from_safe_artifacts"


def test_tampered_ledger_hash_is_rejected(tmp_path: Path) -> None:
    config = _config_for_tmp(tmp_path)
    ledger = tmp_path / "request_ledger.jsonl"
    ledger.write_text(ledger.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(IdentificationError) as error:
        analyze_source_manipulation_gate(config)
    assert error.value.code == "source_manipulation_analysis_ledger_hash_mismatch"


def test_report_is_safe_and_refuses_overwrite(tmp_path: Path) -> None:
    report = write_analysis_report(analyze_source_manipulation_gate(), tmp_path / "report.json")
    content = report.read_text(encoding="utf-8")
    for forbidden in ("system_prompt", "user_prompt", "raw_response", "api_key", "ground_truth_label", "source_independence_label"):
        assert forbidden not in content
    with pytest.raises(IdentificationError) as error:
        write_analysis_report(analyze_source_manipulation_gate(), report)
    assert error.value.code == "source_manipulation_analysis_output_exists"


def test_preflight_is_local_only() -> None:
    result = safe_analysis_preflight()
    assert result["network"] == "disabled"
    assert result["provider_constructed"] is False
    assert result["api_key_read"] is False
