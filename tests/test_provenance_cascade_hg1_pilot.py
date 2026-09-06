from __future__ import annotations

from pathlib import Path

import pytest

from evicon.cascade_real_agent_runner import CascadeRealAgentRunError
from evicon.llm_contract import LLMProviderError, LLMResponse, ProviderErrorCode
from evicon.provenance_cascade_hg1 import HG1FakeProvider
from evicon.provenance_cascade_hg1_pilot import (
    HG1PilotRunner,
    execute_hg1_real_pilot,
    final_preflight_hg1,
    run_hg1_pilot_fake_smoke,
)
from evicon.request_ledger import RequestLedger

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg1.v1.toml"
APPROVAL = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg1_approval_template.toml"
OFFICIAL_OUTPUT = ROOT / "results/provenance-cascade-pilot-hg1-v1"


def test_hg1_final_preflight_is_offline_and_blocks_existing_output() -> None:
    report = final_preflight_hg1(CONFIG, APPROVAL)
    assert report["status"] == "blocked"
    assert report["ready_for_real_pilot"] is False
    assert report["blocking_reasons"] == ["output_root_exists"]
    assert report["run_count"] == 48
    assert report["matched_group_count"] == 12
    assert report["logical_request_cap"] == 864
    assert report["completion_reservation_cap"] == 442368
    assert report["network"] == "disabled"
    assert report["provider_constructed"] is False
    assert report["api_key_read"] is False
    assert report["results_written"] is False
    assert OFFICIAL_OUTPUT.name == "provenance-cascade-pilot-hg1-v1"


def test_hg1_pilot_full_fake_smoke_has_complete_replay_and_budget() -> None:
    result = run_hg1_pilot_fake_smoke(CONFIG)
    assert result["status"] == "fake_smoke_passed"
    assert result["run_count"] == 48
    assert result["matched_group_count"] == 12
    assert result["logical_request_count"] == 864
    assert result["provider_call_count"] == 864
    assert result["transport_attempt_count"] == 864
    assert result["replay_passed_count"] == 48
    assert result["directive_applied_count"] == 24
    assert result["completion_reservation_cap"] == 442368
    assert result["results_written"] is False
    assert OFFICIAL_OUTPUT.name == "provenance-cascade-pilot-hg1-v1"


def test_hg1_run_all_no_overwrite_completed_resume_and_binding(tmp_path: Path) -> None:
    runner = HG1PilotRunner(CONFIG)
    root = tmp_path / "pilot"
    records, receipt = runner.run_all(
        provider_factory=lambda _spec: HG1FakeProvider(),
        root=root,
        model_name="hg1-fake",
        approval_sha256="a" * 64,
        compatibility_receipt_sha256="b" * 64,
        amendment_sha256=runner.config.amendment_sha256,
        write_receipt=False,
        network="disabled",
    )
    assert len(records) == 48
    assert receipt.logical_request_count == 864
    assert receipt.replay_passed_count == 48
    assert receipt.directive_applied_count == 24

    with pytest.raises(CascadeRealAgentRunError) as error:
        runner.run_all(
            provider_factory=lambda _spec: HG1FakeProvider(), root=root,
            model_name="hg1-fake", approval_sha256="a" * 64,
            compatibility_receipt_sha256="b" * 64,
            amendment_sha256=runner.config.amendment_sha256,
            write_receipt=False, network="disabled",
        )
    assert error.value.code == "output_root_exists"

    calls = 0
    def fail_if_called(_spec):
        nonlocal calls
        calls += 1
        raise AssertionError("completed run must not construct a provider")

    resumed, resumed_receipt = runner.run_all(
        provider_factory=fail_if_called, root=root, model_name="hg1-fake",
        approval_sha256="a" * 64, compatibility_receipt_sha256="b" * 64,
        amendment_sha256=runner.config.amendment_sha256,
        resume=True, write_receipt=False, network="disabled",
    )
    assert calls == 0
    assert len(resumed) == 48
    assert resumed_receipt.logical_request_count == 864

    with pytest.raises(CascadeRealAgentRunError) as error:
        runner.run_all(
            provider_factory=fail_if_called, root=root, model_name="changed-model",
            approval_sha256="a" * 64, compatibility_receipt_sha256="b" * 64,
            amendment_sha256=runner.config.amendment_sha256,
            resume=True, write_receipt=False, network="disabled",
        )
    assert error.value.code == "resume_binding_mismatch"
    serialized = " ".join(path.read_text(encoding="utf-8").lower() for path in root.rglob("*.json*"))
    for forbidden in (
        "system_prompt", "user_prompt", "api_key", "provider_metadata",
        "ground_truth_label", "source_independence_label",
    ):
        assert forbidden not in serialized


class MalformedProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            request_id=request.request_id, model_name=request.model_name,
            content='{"stance":"uncertain"', finish_reason="stop",
            prompt_tokens=5, completion_tokens=5, total_tokens=10, latency_ms=1.0,
        )


class FailedOnceProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request) -> LLMResponse:
        self.calls += 1
        raise LLMProviderError(ProviderErrorCode.TIMEOUT, "safe timeout")


def test_hg1_provider_failure_can_resume_but_parser_invalid_cannot(tmp_path: Path) -> None:
    runner = HG1PilotRunner(CONFIG)
    spec = runner.config.runs[0]
    failed_root = tmp_path / "provider-failed"
    failed = FailedOnceProvider()
    with pytest.raises(CascadeRealAgentRunError) as error:
        runner.run_one(spec, provider=failed, root=failed_root, model_name="hg1-fake")
    assert error.value.code == "timeout"
    recovered = runner.run_one(
        spec, provider=HG1FakeProvider(), root=failed_root,
        model_name="hg1-fake", resume=True,
    )
    assert recovered.replay is not None and recovered.replay.status.value == "passed"
    summary = RequestLedger(failed_root / spec.run_id / "request_ledger.jsonl").summary(
        request_cap=18, completion_reservation_cap=9216,
    )
    assert summary.unique_logical_request_count == 18
    assert summary.transport_attempt_count == 19

    parser_root = tmp_path / "parser-invalid"
    malformed = MalformedProvider()
    with pytest.raises(CascadeRealAgentRunError) as error:
        runner.run_one(spec, provider=malformed, root=parser_root, model_name="hg1-fake")
    assert error.value.code == "malformed_json"
    assert malformed.calls == 1
    with pytest.raises(CascadeRealAgentRunError) as error:
        runner.run_one(
            spec, provider=HG1FakeProvider(), root=parser_root,
            model_name="hg1-fake", resume=True,
        )
    assert error.value.code == "completed_request_fingerprint_exists"
    ledger = (parser_root / spec.run_id / "request_ledger.jsonl").read_text(encoding="utf-8")
    assert "parser_recovery" not in ledger


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({}, "allow_network_required"),
        ({"allow_network": True}, "confirm_run_required"),
        ({"allow_network": True, "confirm_run": True}, "confirm_request_cap_must_equal_864"),
        ({"allow_network": True, "confirm_run": True, "confirm_request_cap": 864}, "confirm_completion_reservation_cap_must_equal_442368"),
    ],
)
def test_hg1_real_gate_blocks_before_provider_construction(monkeypatch, kwargs, code) -> None:
    def fail_from_env(*args, **kwargs):
        raise AssertionError("Provider must not be constructed")
    monkeypatch.setattr("evicon.provenance_cascade_hg1_pilot.ProviderConfig.from_env", fail_from_env)
    result = execute_hg1_real_pilot(config_path=CONFIG, approval_path=APPROVAL, **kwargs)
    assert result.status == "blocked"
    assert result.error_code == code
    assert result.provider_constructed_count == 0
    assert result.network == "disabled"


def test_hg1_preflight_failure_blocks_before_environment_read(monkeypatch) -> None:
    def fail_from_env(*args, **kwargs):
        raise AssertionError("Provider must not be constructed")
    monkeypatch.setattr("evicon.provenance_cascade_hg1_pilot.ProviderConfig.from_env", fail_from_env)
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg1_pilot.final_preflight_hg1",
        lambda *args, **kwargs: {"ready_for_real_pilot": False, "blocking_reasons": ["hash_mismatch"]},
    )
    result = execute_hg1_real_pilot(
        config_path=CONFIG, approval_path=APPROVAL, allow_network=True,
        confirm_run=True, confirm_request_cap=864,
        confirm_completion_reservation_cap=442368,
    )
    assert result.status == "blocked"
    assert result.error_code == "hash_mismatch"
    assert result.provider_constructed_count == 0
    assert result.network == "disabled"


def test_hg1_real_entry_executes_full_run_all_with_injected_fake_provider(monkeypatch, tmp_path: Path) -> None:
    from evicon.openai_provider import ProviderConfig
    import evicon.provenance_cascade_hg1_pilot as module

    output = tmp_path / "real-entry-fake"
    original_runner = module.HG1PilotRunner

    class TemporaryRunner(original_runner):
        def __init__(self, config_path=CONFIG):
            super().__init__(config_path)
            self.config = self.config.model_copy(update={"output_root": str(output)})

    monkeypatch.setattr(module, "HG1PilotRunner", TemporaryRunner)
    monkeypatch.setattr(module, "DEFAULT_OUTPUT_ROOT", str(output))
    monkeypatch.setattr(
        module,
        "final_preflight_hg1",
        lambda *args, **kwargs: {"ready_for_real_pilot": True, "blocking_reasons": []},
    )
    monkeypatch.setattr(
        module.ProviderConfig,
        "from_env",
        lambda *args, **kwargs: ProviderConfig(
            base_url="https://provider.invalid/v1", model_name="hg1-fake",
            allow_network=True, timeout_seconds=15, max_retries=1,
            max_tokens=512, temperature=0.2, seed=20261001,
        ),
    )
    monkeypatch.setattr(module, "OpenAICompatibleProvider", lambda *args, **kwargs: HG1FakeProvider())
    result = module.execute_hg1_real_pilot(
        config_path=CONFIG, approval_path=APPROVAL,
        allow_network=True, confirm_run=True,
        confirm_request_cap=864,
        confirm_completion_reservation_cap=442368,
        environment={
            "EVICON_LLM_BASE_URL": "https://provider.invalid/v1",
            "EVICON_LLM_MODEL": "hg1-fake",
            "EVICON_LLM_API_KEY": "synthetic-test-key",
        },
    )
    assert result.status == "completed"
    assert result.completed_run_count == 48
    assert result.logical_request_count == 864
    assert result.transport_attempt_count == 864
    assert result.directive_applied_count == 24
    assert result.provider_constructed_count == 48
    assert len(result.replay_statuses) == 48
    assert set(result.replay_statuses.values()) == {"passed"}
    assert (output / "pilot_receipt.json").is_file()
    serialized = " ".join(path.read_text(encoding="utf-8").lower() for path in output.rglob("*.json*"))
    for forbidden in (
        "synthetic-test-key", "system_prompt", "user_prompt", "api_key",
        "provider_metadata", "ground_truth_label", "source_independence_label",
    ):
        assert forbidden not in serialized
    assert OFFICIAL_OUTPUT.name == "provenance-cascade-pilot-hg1-v1"


def test_hg1_batch_failure_writes_no_receipt_and_resume_completes(tmp_path: Path) -> None:
    runner = HG1PilotRunner(CONFIG)
    root = tmp_path / "resumable"
    first = True

    def fail_first(_spec):
        nonlocal first
        if first:
            first = False
            return FailedOnceProvider()
        return HG1FakeProvider()

    with pytest.raises(CascadeRealAgentRunError) as error:
        runner.run_all(
            provider_factory=fail_first, root=root, model_name="hg1-fake",
            approval_sha256="a" * 64, compatibility_receipt_sha256="b" * 64,
            amendment_sha256=runner.config.amendment_sha256,
            write_receipt=True, network="disabled",
        )
    assert error.value.code == "timeout"
    assert (root / "pilot_batch_record.json").is_file()
    assert not (root / "pilot_receipt.json").exists()

    records, receipt = runner.run_all(
        provider_factory=lambda _spec: HG1FakeProvider(), root=root,
        model_name="hg1-fake", approval_sha256="a" * 64,
        compatibility_receipt_sha256="b" * 64,
        amendment_sha256=runner.config.amendment_sha256,
        resume=True, write_receipt=True, network="disabled",
    )
    assert len(records) == 48
    assert receipt.logical_request_count == 864
    assert receipt.transport_attempt_count == 865
    assert receipt.replay_passed_count == 48
    assert (root / "pilot_receipt.json").is_file()
