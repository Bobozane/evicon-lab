from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.cascade_real_agent_runner import CascadeRealAgentRunError
from evicon.provenance_cascade_hg2 import HG2FakeProvider, sha256_file
from evicon.provenance_cascade_hg21_calibration_v2 import (
    DEFAULT_AMENDMENT_RECEIPT,
    DEFAULT_CONFIG,
    HG21ExecutionV2Runner,
    execute_real,
    final_preflight,
    load_config,
    run_fake_smoke,
)

ROOT = Path(__file__).resolve().parents[1]
OLD_BATCH = ROOT / "results/provenance-cascade-hg21-eligibility-calibration-v1/calibration_batch_record.json"
OLD_LEDGER = ROOT / "results/provenance-cascade-hg21-eligibility-calibration-v1/hg21-cascade-hg1-false-majority-20261021-no_intervention/request_ledger.jsonl"
NEW_ROOT = ROOT / "results/provenance-cascade-hg21-eligibility-calibration-v2"


def test_v2_lock_preserves_failed_v1_and_uses_new_identity() -> None:
    config, path = load_config()
    assert sha256_file(path) == "103f10ee9ec361f8acba33c2e99c84349059efca7b15fc88dcff49602e15268a"
    assert sha256_file(DEFAULT_AMENDMENT_RECEIPT) == "cc91993a89fa6b59669cf96fc7a745f186370d0c25ab93bb574671df62076b5a"
    assert sha256_file(OLD_BATCH) == config.failed_v1_batch_record_sha256
    assert sha256_file(OLD_LEDGER) == config.failed_v1_request_ledger_sha256
    runner = HG21ExecutionV2Runner()
    assert all(item.run_id.startswith("hg21v2-") for item in runner.runs)
    assert all(item.matched_group_id.startswith("hg21v2-") for item in runner.runs)
    assert runner.amendment.output_root == config.output_root
    assert config.required_model_name == "gpt-5.6-luna"


def test_preflight_is_offline_and_reflects_current_output_state(monkeypatch) -> None:
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg21_calibration_v2.ProviderConfig.from_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("provider must not be constructed")),
    )
    report = final_preflight()
    expected = ["output_root_exists"] if NEW_ROOT.exists() else []
    assert report["blocking_reasons"] == expected
    assert report["network"] == "disabled"
    assert report["provider_constructed"] is False
    assert report["failed_v1_preserved"] is True


def test_provider_model_mismatch_is_rejected_before_provider_construction(monkeypatch) -> None:
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg21_calibration_v2.final_preflight",
        lambda: {"ready_for_network_authorization": True, "blocking_reasons": []},
    )
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg21_calibration_v2.ProviderConfig.from_env",
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
    assert result.status == "blocked"
    assert result.error_code == "provider_model_mismatch"
    assert result.provider_constructed_count == 0


def test_v2_fake_smoke_completes_and_resumes_without_calls() -> None:
    result = run_fake_smoke()
    assert result["run_count"] == 16
    assert result["matched_group_count"] == 4
    assert result["logical_request_count"] == result["provider_call_count"] == 288
    assert result["replay_passed_count"] == 16
    assert result["resume_additional_provider_call_count"] == 0
    assert result["directive_applied_count"] > 0


def test_v2_no_overwrite_and_binding_protection(tmp_path: Path) -> None:
    runner = HG21ExecutionV2Runner()
    root = tmp_path / "v2"
    records, receipt = runner.run_all(
        provider_factory=lambda _spec: HG2FakeProvider(),
        root=root,
        model_name="gpt-5.6-luna",
        write_receipt=False,
        network="disabled",
    )
    assert len(records) == 16 and receipt.replay_passed_count == 16
    with pytest.raises(CascadeRealAgentRunError, match="output_root_exists"):
        runner.run_all(
            provider_factory=lambda _spec: HG2FakeProvider(),
            root=root,
            model_name="gpt-5.6-luna",
            write_receipt=False,
            network="disabled",
        )
    with pytest.raises(CascadeRealAgentRunError, match="resume_binding_mismatch"):
        runner.run_all(
            provider_factory=lambda _spec: HG2FakeProvider(),
            root=root,
            model_name="another-model",
            resume=True,
            write_receipt=False,
            network="disabled",
        )


def test_v2_config_rejects_unknown_fields(tmp_path: Path) -> None:
    source = (ROOT / DEFAULT_CONFIG).read_text(encoding="utf-8")
    path = tmp_path / "invalid.toml"
    path.write_text(source + '\nquestion_text = "forbidden"\n', encoding="utf-8")
    with pytest.raises(Exception, match="execution_v2_config_invalid"):
        load_config(path)


def test_formal_roots_preserve_incomplete_failures_without_receipts() -> None:
    if NEW_ROOT.exists():
        batch = json.loads((NEW_ROOT / "calibration_batch_record.json").read_text(encoding="utf-8"))
        assert batch["status"] == "failed"
        assert batch["model_name"] == "gpt-5.6-luna"
        assert not (NEW_ROOT / "calibration_receipt.json").exists()
    old = json.loads(OLD_BATCH.read_text(encoding="utf-8"))
    assert old["status"] == "failed"
    assert old["model_name"] == "gemini-3.6-flash"
