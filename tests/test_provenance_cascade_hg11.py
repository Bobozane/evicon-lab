from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.cascade_agent_prompts import CascadeAgentRuntimeConfig
from evicon.cascade_agent_protocol_hg1 import render_cascade_agent_turn_hg1
from evicon.cascade_agent_protocol_hg11 import render_cascade_agent_turn_hg11
from evicon.cascade_real_agent_runner import CascadeRealAgentRunError
from evicon.llm_contract import LLMResponse
from evicon.provenance_cascade_hg1_compatibility import build_minimal_public_context
from evicon.provenance_cascade_hg11 import HG11FakeProvider, load_hg11_config, sha256_file
from evicon.provenance_cascade_hg11_compatibility import run_fake_compatibility
from evicon.provenance_cascade_hg11_pilot import (
    HG11PilotRunner,
    final_preflight_hg11,
    run_hg11_pilot_fake_smoke,
)
from evicon.request_ledger import request_fingerprint_facts

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg11.v1.toml"
APPROVAL = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg11_approval_template.toml"
OLD_BATCH = ROOT / "results/provenance-cascade-pilot-hg1-v1/pilot_batch_record.json"
OLD_LEDGER = ROOT / "results/provenance-cascade-pilot-hg1-v1/hg1-cascade-hg1-false-majority-20261003-no_intervention/request_ledger.jsonl"
OLD_CHECKPOINT = ROOT / "results/provenance-cascade-pilot-hg1-v1/hg1-cascade-hg1-false-majority-20261003-no_intervention/agent_checkpoint.json"
NEW_OUTPUT = ROOT / "results/provenance-cascade-pilot-hg11-v1"


def test_hg11_config_is_new_1024_token_plan_and_parent_is_frozen() -> None:
    config, scenarios, truths, schedules = load_hg11_config(CONFIG)
    assert config.agent_max_tokens == 1024
    assert config.request_cap == 864
    assert config.completion_reservation_cap == 884736
    assert len(config.runs) == 48
    assert len({item.matched_group_id for item in config.runs}) == 12
    assert all(item.run_id.startswith("hg11-") for item in config.runs)
    assert len(scenarios) == len(truths) == len(schedules) == 4
    assert sha256_file(OLD_BATCH) == config.parent_batch_sha256
    assert sha256_file(OLD_LEDGER) == config.parent_failed_ledger_sha256
    assert sha256_file(OLD_CHECKPOINT) == config.parent_failed_checkpoint_sha256
    assert config.parent_failed_fingerprint == "5830e470f6fc742714b427a14fc1a058dfd35496757c9a8e95298a3d83601213"
    assert NEW_OUTPUT.name == "provenance-cascade-pilot-hg11-v1"


def test_hg11_prompt_semantics_match_hg1_except_version_binding_and_token_limit() -> None:
    base = build_minimal_public_context(max_tokens=512, seed=20261001)
    amended = base.model_copy(update={
        "runtime_config": CascadeAgentRuntimeConfig(
            model_name=base.runtime_config.model_name,
            temperature=base.runtime_config.temperature,
            max_tokens=1024,
            seed=base.runtime_config.seed,
        )
    })
    old = render_cascade_agent_turn_hg1(base)
    new = render_cascade_agent_turn_hg11(amended)
    assert old.system_prompt == new.system_prompt
    old_payload = json.loads(old.user_prompt)
    new_payload = json.loads(new.user_prompt)
    assert old_payload.pop("template_version") != new_payload.pop("template_version")
    assert old_payload.pop("protocol_version") != new_payload.pop("protocol_version")
    assert old_payload == new_payload
    assert old.max_tokens == 512
    assert new.max_tokens == 1024
    assert request_fingerprint_facts(new)["phase"] == "agent_turn"
    assert request_fingerprint_facts(old)["fingerprint"] != request_fingerprint_facts(new)["fingerprint"]


def test_hg11_compatibility_fake_smoke_covers_1024_and_length() -> None:
    result = run_fake_compatibility()
    assert result["status"] == "passed"
    assert result["max_tokens"] == 1024
    assert result["valid_parser"] is True
    assert result["length_parser"] is False
    assert result["provider_call_count"] == 2
    assert result["network"] == "disabled"


def test_hg11_preflight_prioritizes_no_overwrite_for_existing_root() -> None:
    result = final_preflight_hg11(CONFIG, APPROVAL)
    assert result["status"] == "blocked"
    assert result["blocking_reasons"] == ["output_root_exists"]
    assert result["logical_request_cap"] == 864
    assert result["completion_reservation_cap"] == 884736
    assert result["network"] == "disabled"
    assert result["provider_constructed"] is False
    assert result["api_key_read"] is False
    assert result["results_written"] is False


def test_hg11_full_fake_smoke_has_48_replays_and_new_budget() -> None:
    result = run_hg11_pilot_fake_smoke(CONFIG)
    assert result["status"] == "fake_smoke_passed"
    assert result["run_count"] == 48
    assert result["matched_group_count"] == 12
    assert result["logical_request_count"] == 864
    assert result["provider_call_count"] == 864
    assert result["transport_attempt_count"] == 864
    assert result["replay_passed_count"] == 48
    assert result["directive_applied_count"] == 24
    assert result["completion_reservation_cap"] == 884736
    assert result["results_written"] is False
    # The smoke uses a temporary root; the authorized historical HG11 output
    # may legitimately exist and must remain untouched.
    assert NEW_OUTPUT.name == "provenance-cascade-pilot-hg11-v1"


def test_hg11_run_all_no_overwrite_and_completed_resume(tmp_path: Path) -> None:
    runner = HG11PilotRunner(CONFIG)
    root = tmp_path / "pilot"
    records, receipt = runner.run_all(
        provider_factory=lambda _spec: HG11FakeProvider(), root=root,
        model_name="hg11-fake", approval_sha256="a" * 64,
        compatibility_receipt_sha256="b" * 64,
        amendment_sha256=runner.config.amendment_sha256,
        write_receipt=False, network="disabled",
    )
    assert len(records) == 48 and receipt.logical_request_count == 864
    with pytest.raises(CascadeRealAgentRunError, match="output_root_exists"):
        runner.run_all(
            provider_factory=lambda _spec: HG11FakeProvider(), root=root,
            model_name="hg11-fake", approval_sha256="a" * 64,
            compatibility_receipt_sha256="b" * 64,
            amendment_sha256=runner.config.amendment_sha256,
            write_receipt=False, network="disabled",
        )
    calls = 0
    def fail_if_called(_spec):
        nonlocal calls
        calls += 1
        raise AssertionError("completed run must not replay")
    resumed, resumed_receipt = runner.run_all(
        provider_factory=fail_if_called, root=root, model_name="hg11-fake",
        approval_sha256="a" * 64, compatibility_receipt_sha256="b" * 64,
        amendment_sha256=runner.config.amendment_sha256,
        resume=True, write_receipt=False, network="disabled",
    )
    assert calls == 0
    assert len(resumed) == 48 and resumed_receipt.logical_request_count == 864


class MalformedAtLimitProvider:
    def complete(self, request) -> LLMResponse:
        return LLMResponse(
            request_id=request.request_id, model_name=request.model_name,
            content='{"stance":"uncertain"', finish_reason="length",
            prompt_tokens=50, completion_tokens=1024, total_tokens=1074,
            latency_ms=1.0,
        )


def test_hg11_parser_invalid_stops_without_recovery(tmp_path: Path) -> None:
    runner = HG11PilotRunner(CONFIG)
    spec = runner.config.runs[0]
    with pytest.raises(CascadeRealAgentRunError, match="malformed_json"):
        runner.run_one(spec, provider=MalformedAtLimitProvider(), root=tmp_path, model_name="hg11-fake")
    with pytest.raises(CascadeRealAgentRunError, match="completed_request_fingerprint_exists"):
        runner.run_one(
            spec, provider=HG11FakeProvider(), root=tmp_path,
            model_name="hg11-fake", resume=True,
        )
    ledger = (tmp_path / spec.run_id / "request_ledger.jsonl").read_text(encoding="utf-8")
    assert "parser_recovery" not in ledger
