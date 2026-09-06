from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.llm_contract import LLMResponse

from evicon.cascade_agent_protocol_v2_smoke import V2FakeProvider
from evicon.cascade_agent_protocol_v2_pilot import (
    HD2_CONFIG_RELATIVE,
    HD2_PROTOCOL_RELATIVE,
    HD2_TEMPLATE_RELATIVE,
    execute_hd2_real_pilot,
    final_preflight,
    main,
    run_fake_smoke,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / HD2_CONFIG_RELATIVE
APPROVAL = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd2_approval_template.toml"


def test_h_e_fake_smoke_is_complete_and_temporary_only() -> None:
    result = run_fake_smoke(CONFIG)
    assert result["status"] == "fake_smoke_passed"
    assert result["run_count"] == 48
    assert result["matched_group_count"] == 12
    assert result["logical_request_count"] == 864
    assert result["provider_call_count"] == 864
    assert result["replay_passed_count"] == 48
    assert result["directive_applied_count"] > 0
    assert result["completion_reservation_cap"] == 221184
    assert result["network"] == "disabled"
    assert result["results_written"] is False
    assert result["old_pilot_touched"] is False
    assert result["private_truth_exposed"] is False
    assert result["pilot_only"] is True
    assert result["parser_recovery_enabled"] is False
    assert result["safety"]["network_disabled"] is True


def test_h_e_default_preflight_is_ready_but_remains_offline() -> None:
    result = final_preflight(CONFIG, APPROVAL, resume=True)
    assert result["status"] == "ready_for_real_pilot"
    assert result["network"] == "disabled"
    assert result["provider_constructed"] is False
    assert result["api_key_read"] is False
    assert result["transport_called"] is False
    assert result["results_written"] is False
    assert result["ready_for_network_authorization"] is True
    assert result["blocking_reasons"] == []
    assert result["compatibility_receipt_sha256"]
    assert result["template_sha256"] == sha256_file(ROOT / HD2_TEMPLATE_RELATIVE)


def test_h_e_bindings_and_scope_are_new_and_fixed() -> None:
    result = final_preflight(CONFIG, APPROVAL, resume=True)
    assert result["config_sha256"] == sha256_file(CONFIG)
    assert result["protocol_sha256"] == sha256_file(ROOT / HD2_PROTOCOL_RELATIVE)
    assert result["template_sha256"] == sha256_file(ROOT / HD2_TEMPLATE_RELATIVE)
    assert result["response_format"] == "json_schema"
    assert result["schema_name"] == "cascade_agent_response_v2"
    assert result["output_root"] == "results/provenance-cascade-pilot-hd2-v1"
    assert result["run_count"] == 48
    assert result["matched_group_count"] == 12
    assert result["logical_request_count"] == 864
    assert result["completion_reservation_cap"] == 221184


def test_h_e_receipt_json_is_content_free() -> None:
    result = run_fake_smoke(CONFIG)
    dumped = json.dumps(result, sort_keys=True)
    for forbidden in ("user_prompt", "system_prompt", "api_key", "provider_metadata", "ground_truth_label", "source_independence_label"):
        assert forbidden not in dumped.lower()


def test_h_e_resume_reuses_completed_checkpoint_without_provider_replay(tmp_path) -> None:
    from evicon.cascade_agent_protocol_v2_pilot import HD2PilotRunner

    pilot = HD2PilotRunner(CONFIG)
    spec = pilot.config.runs[0]
    first = V2FakeProvider()
    pilot.run_one(spec, provider=first, root=tmp_path, model_name="hd2-fake-v2", resume=False)
    second = V2FakeProvider()
    resumed = pilot.run_one(spec, provider=second, root=tmp_path, model_name="hd2-fake-v2", resume=True)
    assert first.calls == 18
    assert second.calls == 0
    assert resumed.replay is not None and resumed.replay.status.value == "passed"
    ledger = (tmp_path / spec.run_id / "request_ledger.jsonl").read_text(encoding="utf-8")
    assert "system_prompt" not in ledger.lower()
    assert "user_prompt" not in ledger.lower()
    assert "provider_metadata" not in ledger


class _InvalidV2Provider:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content=json.dumps({
                "stance": "uncertain",
                "content_ids_used": [],
                "evidence_ids_used": [],
                "share_content_id": None,
                "unexpected_control": True,
            }),
            finish_reason="stop",
            prompt_tokens=3,
            completion_tokens=2,
            total_tokens=5,
            latency_ms=1.0,
        )


def _confirmed_args(*, resume: bool = False) -> list[str]:
    args = [
        "--mode", "real-pilot",
        "--allow-network",
        "--confirm-run",
        "--confirm-request-cap", "864",
        "--confirm-completion-reservation-cap", "221184",
    ]
    if resume:
        args.append("--resume")
    return args


def test_real_cli_missing_confirmation_never_constructs_provider(tmp_path, capsys) -> None:
    constructed = 0

    def factory(_spec):
        nonlocal constructed
        constructed += 1
        return V2FakeProvider()

    exit_code = main(
        ["--mode", "real-pilot", "--allow-network"],
        provider_factory=factory,
        test_output_root=tmp_path / "pilot",
        test_model_name="hd2-v2-fake",
    )
    result = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert result["status"] == "blocked"
    assert result["error_code"] == "confirm_run_required"
    assert result["provider_constructed_count"] == 0
    assert constructed == 0
    assert not (tmp_path / "pilot").exists()


def test_real_cli_hash_mismatch_never_constructs_provider(tmp_path, capsys) -> None:
    approval = tmp_path / "approval.toml"
    approval.write_text(
        APPROVAL.read_text(encoding="utf-8").replace(
            'config_sha256 = "9b26d7ebc19a8bf24c7458f90667d0ca416bad860c94f2fac6968703f7703416"',
            'config_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"',
        ),
        encoding="utf-8",
    )
    constructed = 0

    def factory(_spec):
        nonlocal constructed
        constructed += 1
        return V2FakeProvider()

    exit_code = main(
        [*_confirmed_args(), "--approval", str(approval)],
        provider_factory=factory,
        test_output_root=tmp_path / "pilot",
        test_model_name="hd2-v2-fake",
    )
    result = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert result["status"] == "blocked"
    assert result["error_code"] == "config_hash_mismatch"
    assert result["provider_constructed_count"] == 0
    assert constructed == 0
    assert not (tmp_path / "pilot").exists()


def test_real_cli_calls_run_all_and_resume_skips_completed_runs(tmp_path, capsys) -> None:
    root = tmp_path / "pilot"
    providers: list[V2FakeProvider] = []

    def factory(_spec):
        provider = V2FakeProvider()
        providers.append(provider)
        return provider

    exit_code = main(
        _confirmed_args(),
        provider_factory=factory,
        test_output_root=root,
        test_model_name="hd2-v2-fake",
    )
    first = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert first["status"] == "completed"
    assert first["completed_run_count"] == 48
    assert first["failed_run_count"] == 0
    assert first["logical_request_count"] == 864
    assert first["transport_attempt_count"] == 864
    assert first["actual_total_token_count"] == 15552
    assert first["directive_applied_count"] == 9
    assert set(first["replay_statuses"].values()) == {"passed"}
    assert first["provider_constructed_count"] == 48
    assert sum(provider.calls for provider in providers) == 864
    assert Path(first["receipt_path"]).exists()
    assert (root / "pilot_batch_record.json").exists()
    assert len(list(root.glob("*/request_ledger.jsonl"))) == 48

    blocked_calls = 0

    def blocked_factory(_spec):
        nonlocal blocked_calls
        blocked_calls += 1
        return V2FakeProvider()

    blocked = execute_hd2_real_pilot(
        allow_network=True,
        confirm_run=True,
        confirm_request_cap=864,
        confirm_completion_reservation_cap=221184,
        provider_factory=blocked_factory,
        test_output_root=root,
        test_model_name="hd2-v2-fake",
    )
    assert blocked.status == "blocked"
    assert blocked.error_code == "output_root_exists"
    assert blocked_calls == 0

    resumed_providers: list[V2FakeProvider] = []

    def resumed_factory(_spec):
        provider = V2FakeProvider()
        resumed_providers.append(provider)
        return provider

    resume_code = main(
        _confirmed_args(resume=True),
        provider_factory=resumed_factory,
        test_output_root=root,
        test_model_name="hd2-v2-fake",
    )
    resumed = json.loads(capsys.readouterr().out)
    assert resume_code == 0
    assert resumed["status"] == "completed"
    assert resumed["provider_constructed_count"] == 0
    assert resumed_providers == []
    assert resumed["logical_request_count"] == 864


def test_parser_invalid_stops_and_has_no_automatic_recovery(tmp_path) -> None:
    root = tmp_path / "pilot"
    invalid = _InvalidV2Provider()
    first = execute_hd2_real_pilot(
        allow_network=True,
        confirm_run=True,
        confirm_request_cap=864,
        confirm_completion_reservation_cap=221184,
        provider_factory=lambda _spec: invalid,
        test_output_root=root,
        test_model_name="hd2-v2-fake",
    )
    assert first.status == "failed"
    assert first.error_code == "extra_field"
    assert first.completed_run_count == 0
    assert first.failed_run_count == 1
    assert invalid.calls == 1
    assert not (root / "pilot_receipt.json").exists()

    resumed_provider = V2FakeProvider()
    resumed = execute_hd2_real_pilot(
        allow_network=True,
        confirm_run=True,
        confirm_request_cap=864,
        confirm_completion_reservation_cap=221184,
        resume=True,
        provider_factory=lambda _spec: resumed_provider,
        test_output_root=root,
        test_model_name="hd2-v2-fake",
    )
    assert resumed.status == "failed"
    assert resumed.error_code == "completed_request_fingerprint_exists"
    assert resumed_provider.calls == 0
    assert resumed.transport_attempt_count == 1
    assert not (root / "pilot_receipt.json").exists()


def test_resume_binding_change_is_rejected_before_provider_construction(tmp_path) -> None:
    root = tmp_path / "pilot"
    invalid = _InvalidV2Provider()
    first = execute_hd2_real_pilot(
        allow_network=True,
        confirm_run=True,
        confirm_request_cap=864,
        confirm_completion_reservation_cap=221184,
        provider_factory=lambda _spec: invalid,
        test_output_root=root,
        test_model_name="hd2-v2-fake",
    )
    assert first.status == "failed"
    constructed = 0

    def factory(_spec):
        nonlocal constructed
        constructed += 1
        return V2FakeProvider()

    changed = execute_hd2_real_pilot(
        allow_network=True,
        confirm_run=True,
        confirm_request_cap=864,
        confirm_completion_reservation_cap=221184,
        resume=True,
        provider_factory=factory,
        test_output_root=root,
        test_model_name="different-model",
    )
    assert changed.status == "blocked"
    assert changed.error_code == "resume_binding_mismatch"
    assert constructed == 0
