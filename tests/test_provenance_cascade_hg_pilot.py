from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.cascade_hg_compatibility import HGCompatibilityResult
from evicon.cascade_hg_compatibility_receipt import register_receipt
from evicon.cascade_real_agent_runner import CascadeRealAgentRunError
from evicon.cascade_outcome_replay import CascadeOutcomeReplayError, CascadeOutcomeReplayValidator
from evicon.cascade_hg_outcome_replay import HGOutcomeReplayValidator
from evicon.llm_contract import LLMResponse
from evicon.provenance_cascade_hg_pilot import (
    HGPilotRunner,
    execute_hg_real_pilot,
    run_hg_pilot_fake_smoke,
)
from evicon.provenance_cascade_identifiability import HGFakeProvider

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg.v1.toml"
APPROVAL = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg_approval_template.toml"
PROTOCOL = ROOT / "src/evicon/cascade_agent_protocol_hg.py"
AMENDMENT_SHA = "f840925cd46a27741c2245e011acf68725b4654c85a2a9ce899d69cb4b34e49a"


def test_hg_pilot_full_fake_smoke_has_complete_replay_and_budget() -> None:
    result = run_hg_pilot_fake_smoke(CONFIG)
    assert result["status"] == "fake_smoke_passed"
    assert result["run_count"] == 48
    assert result["matched_group_count"] == 12
    assert result["logical_request_count"] == 864
    assert result["provider_call_count"] == 864
    assert result["transport_attempt_count"] == 864
    assert result["replay_passed_count"] == 48
    assert result["directive_applied_count"] == 9
    assert result["completion_reservation_cap"] == 442368
    assert result["results_written"] is False


def test_hg_run_all_no_overwrite_resume_and_binding_protection(tmp_path: Path) -> None:
    runner = HGPilotRunner(CONFIG)
    root = tmp_path / "pilot"
    records, receipt = runner.run_all(
        provider_factory=lambda _spec: HGFakeProvider(),
        root=root,
        model_name="hg-fake",
        approval_sha256="a" * 64,
        compatibility_receipt_sha256="b" * 64,
        amendment_sha256=AMENDMENT_SHA,
        write_receipt=False,
        network="disabled",
    )
    assert len(records) == 48 and receipt.replay_passed_count == 48
    with pytest.raises(CascadeRealAgentRunError) as error:
        runner.run_all(
            provider_factory=lambda _spec: HGFakeProvider(),
            root=root,
            model_name="hg-fake",
            approval_sha256="a" * 64,
            compatibility_receipt_sha256="b" * 64,
            amendment_sha256=AMENDMENT_SHA,
            write_receipt=False,
            network="disabled",
        )
    assert error.value.code == "output_root_exists"

    calls = 0
    def fail_if_called(_spec):
        nonlocal calls
        calls += 1
        raise AssertionError("completed runs must not construct providers")

    resumed, resumed_receipt = runner.run_all(
        provider_factory=fail_if_called,
        root=root,
        model_name="hg-fake",
        approval_sha256="a" * 64,
        compatibility_receipt_sha256="b" * 64,
        amendment_sha256=AMENDMENT_SHA,
        resume=True,
        write_receipt=False,
        network="disabled",
    )
    assert calls == 0
    assert len(resumed) == 48 and resumed_receipt.logical_request_count == 864
    with pytest.raises(CascadeRealAgentRunError) as error:
        runner.run_all(
            provider_factory=fail_if_called,
            root=root,
            model_name="changed-model",
            approval_sha256="a" * 64,
            compatibility_receipt_sha256="b" * 64,
            amendment_sha256=AMENDMENT_SHA,
            resume=True,
            write_receipt=False,
            network="disabled",
        )
    assert error.value.code == "resume_binding_mismatch"
    serialized = " ".join(path.read_text(encoding="utf-8").lower() for path in root.rglob("*.json*"))
    for forbidden in ("system_prompt", "user_prompt", "api_key", "provider_metadata", "ground_truth_label", "source_independence_label"):
        assert forbidden not in serialized


class MalformedProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content='{"stance":"uncertain"',
            finish_reason="stop",
            prompt_tokens=5,
            completion_tokens=5,
            total_tokens=10,
            latency_ms=1.0,
        )


def test_parser_invalid_stops_without_recovery(tmp_path: Path) -> None:
    runner = HGPilotRunner(CONFIG)
    provider = MalformedProvider()
    with pytest.raises(CascadeRealAgentRunError) as error:
        runner.run_one(
            runner.config.runs[0],
            provider=provider,
            root=tmp_path,
            model_name="hg-fake",
        )
    assert error.value.code == "malformed_json"
    assert provider.calls == 1
    ledger = (tmp_path / runner.config.runs[0].run_id / "request_ledger.jsonl").read_text(encoding="utf-8")
    assert "parser_recovery" not in ledger


def test_real_gate_blocks_before_provider_environment_is_read(monkeypatch) -> None:
    def fail_from_env(*args, **kwargs):
        raise AssertionError("Provider must not be constructed")
    monkeypatch.setattr("evicon.provenance_cascade_hg_pilot.ProviderConfig.from_env", fail_from_env)
    monkeypatch.setattr(
        "evicon.provenance_cascade_hg_pilot.hg_preflight",
        lambda *args, **kwargs: {"ready_for_real_pilot": False, "blocking_reasons": ["test_preflight_blocked"]},
    )
    result = execute_hg_real_pilot(
        config_path=CONFIG,
        approval_path=APPROVAL,
        allow_network=True,
        confirm_run=True,
        confirm_request_cap=864,
        confirm_completion_reservation_cap=442368,
    )
    assert result.status == "blocked"
    assert result.error_code == "test_preflight_blocked"
    assert result.provider_constructed_count == 0
    assert result.network == "disabled"


def test_fake_smoke_does_not_touch_official_hg_output() -> None:
    official = ROOT / "results/provenance-cascade-pilot-hg-v1" / "pilot_batch_record.json"
    before = official.read_bytes() if official.exists() else None
    result = run_hg_pilot_fake_smoke(CONFIG)
    assert result["results_written"] is False
    after = official.read_bytes() if official.exists() else None
    assert after == before


class CrossClaimCorrectionProvider(HGFakeProvider):
    """Use a visible correction when taking a stance on the rumor claim."""

    def complete(self, request) -> LLMResponse:
        payload = json.loads(request.user_prompt)
        visible = {
            item["content_id"]
            for item in payload.get("visible_contents", [])
            if isinstance(item, dict) and isinstance(item.get("content_id"), str)
        }
        evidence = {
            item["evidence_id"]
            for item in payload.get("visible_evidence", [])
            if isinstance(item, dict) and isinstance(item.get("evidence_id"), str)
        }
        if (
            payload.get("target_claim_id") == "claim-hg-tmc-rumor"
            and "hg-tmc-content-correction" in visible
            and "evidence-hg-tmc-schedule" in evidence
        ):
            self.calls += 1
            return LLMResponse(
                request_id=request.request_id,
                model_name=request.model_name,
                content=json.dumps(
                    {
                        "stance": "rejects",
                        "content_ids_used": ["hg-tmc-content-correction"],
                        "evidence_ids_used": ["evidence-hg-tmc-schedule"],
                        "share_content_id": "hg-tmc-content-correction",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                finish_reason="stop",
                prompt_tokens=24,
                completion_tokens=20,
                total_tokens=44,
                latency_ms=1.0,
            )
        return super().complete(request)


def test_hg_replay_allows_visible_correction_context_for_rumor_stance(tmp_path: Path) -> None:
    runner = HGPilotRunner(CONFIG)
    spec = next(
        item
        for item in runner.config.runs
        if item.scenario_id == "cascade-hg-true-minority-correction"
        and item.seed == 20260911
        and item.condition.value == "no_intervention"
    )
    record = runner.run_one(
        spec,
        provider=CrossClaimCorrectionProvider(),
        root=tmp_path,
        model_name="hg-cross-claim-fake",
    )
    assert record.replay is not None
    assert record.replay.status.value == "passed"
    assert record.exposure_ledger is not None
    assert record.outcome_ledger is not None
    assert record.application_ledger is not None
    with pytest.raises(CascadeOutcomeReplayError):
        CascadeOutcomeReplayValidator.validate(
            runner.scenarios[spec.scenario_id].graph,
            record.exposure_ledger,
            record.outcome_ledger,
            record.application_ledger,
            record.round_contexts,
        )
    first = record.outcome_ledger.outcomes[0]
    bad = first.model_copy(update={"content_ids": ("hg-tmc-content-correction",)})
    bad_ledger = record.outcome_ledger.model_copy(
        update={"outcomes": (bad, *record.outcome_ledger.outcomes[1:])}
    )
    with pytest.raises(CascadeOutcomeReplayError):
        HGOutcomeReplayValidator.validate(
            runner.scenarios[spec.scenario_id].graph,
            record.exposure_ledger,
            bad_ledger,
            record.application_ledger,
            record.round_contexts,
        )
