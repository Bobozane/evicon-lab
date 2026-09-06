from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import evicon.provenance_cascade_amendment as hd
from evicon.cascade_agent_integration_smoke import _FakeProvider
from evicon.cascade_amendment_integration_smoke import run_smoke
from evicon.cascade_protocol import CascadeScenarioLoader
from evicon.cascade_real_agent_runner import CascadeRealAgentRunError, CascadeRealAgentRunner
from evicon.provenance_cascade_amendment import HDPilotError, load_hd_config, preflight_hd, write_amendment_receipt
from evicon.provenance_cascade_exposure import ControllerPublicView
from evicon.provenance_cascade_preregistration import CascadeCondition
from evicon.validate_provenance_cascade_amendment import main
from evicon.llm_contract import LLMProviderError, ProviderErrorCode

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd.v2.toml"
OLD_CONFIG = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hb.v1.toml"
OLD_SCENARIO = ROOT / "configs/provenance_cascade/scenarios/false_majority.toml"
OLD_GRAPH = ROOT / "configs/provenance_cascade/fixtures/false_majority.public.json"
NEW_SCENARIO = ROOT / "configs/provenance_cascade/pilot/amendments/hb_trigger_observability_v2/false_majority.v2.toml"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hd_plan_is_versioned_and_historical_hb_material_is_unchanged() -> None:
    config, amendment = load_hd_config(CONFIG)
    assert _sha(OLD_CONFIG) == "133c23b725a30105dd64ad283d44303390e8e5c7c70554455c171dc048bcd670"
    assert _sha(OLD_SCENARIO) == "3b6436412ec38f0eba62a9ac75362f62b87299f6fa260ff9effceac77f1823a2"
    assert _sha(OLD_GRAPH) == "9db8f10fa924a84eb5190d249dfcacfc142c4964cbe33e8c6ea676cd85373b49"
    assert config.parent_config_sha256 == _sha(OLD_CONFIG)
    assert amendment.status == "pending_human_approval"
    assert len(config.runs) == 48
    assert len(config.matched_groups) == 12
    assert config.budget.request_cap == 864
    assert config.budget.completion_reservation_cap == 221184


def test_amended_snapshot_has_two_same_root_contents_at_round_one() -> None:
    config, amendment = load_hd_config(CONFIG)
    material = next(item for item in config.scenario_materials if item.scenario_id == "cascade-false-majority")
    scenario = CascadeScenarioLoader.load((CONFIG.parent / material.config_path).resolve())
    ledger = hd.ExposureLedgerLoader.load((CONFIG.parent / material.exposure_ledger_path).resolve())
    snapshot = next(item for item in ledger.snapshots if item.agent_id == "network-agent-01" and item.round_id == 1)
    view = ControllerPublicView.from_snapshot(snapshot, scenario.graph)
    assert snapshot.visible_content_ids == amendment.trigger_observation.visible_content_ids
    assert len(view.provenance_nodes) == 2
    assert view.root_count_for_claim("claim-fm") == 1
    assert amendment.trigger_observation.proposal_created_round_id == 1
    assert amendment.trigger_observation.directive_effective_round_id == 2


def test_preflight_is_technically_valid_but_not_real_pilot_ready(capsys: pytest.CaptureFixture[str]) -> None:
    original = Path.exists
    output_root = (ROOT / "results/provenance-cascade-pilot-hd-v2").resolve()

    def available(path: Path) -> bool:
        if path.resolve() == output_root:
            return False
        return original(path)

    with patch.object(Path, "exists", available):
        report = preflight_hd(CONFIG)
    assert report.status == "ready_for_human_approval"
    assert report.ready_for_real_pilot is False
    assert report.blocking_reasons == ("human_approval_required",)
    assert report.exposure_replay_status == "passed"
    assert report.credential_value_read is False
    assert report.provider_constructed is False
    with patch.object(Path, "exists", available):
        assert main(["--config", str(CONFIG)]) == 0
    output = capsys.readouterr().out.lower()
    assert "ground_truth_label" not in output
    assert "source_independence_label" not in output
    assert "api_key" not in output
    assert "prompt" not in output


def test_material_hash_tampering_is_rejected() -> None:
    original = hd._sha256

    def tampered(path: Path) -> str:
        if path.name == "false_majority.public.v2.json":
            return "0" * 64
        return original(path)

    with patch.object(hd, "_sha256", side_effect=tampered):
        with pytest.raises(HDPilotError) as error:
            load_hd_config(CONFIG)
    assert error.value.code in {"modified_file_hash_mismatch", "public_graph_hash_mismatch"}


def test_source_paths_cannot_escape_the_repository(tmp_path: Path) -> None:
    with pytest.raises(HDPilotError) as absolute:
        hd._resolve(CONFIG.parent, str(tmp_path / "outside.json"))
    assert absolute.value.code == "path_invalid"
    with pytest.raises(HDPilotError) as traversal:
        hd._resolve(CONFIG.parent, "../../../../outside.json")
    assert traversal.value.code == "path_outside_repository"


def test_missing_condition_and_calibration_reference_are_rejected() -> None:
    config, _ = load_hd_config(CONFIG)
    changed = config.runs[0].model_copy(update={"condition": CascadeCondition.GENERIC_DISSENT})
    with pytest.raises(HDPilotError) as error:
        hd._validate_runs(config.model_copy(update={"runs": (changed,) + config.runs[1:]}))
    assert error.value.code == "run_coordinate_set_mismatch"
    with pytest.raises(HDPilotError) as forbidden:
        hd._safe_material_path("../calibration/false_majority.toml")
    assert forbidden.value.code == "forbidden_material_reference"


def test_output_path_existing_blocks_real_readiness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    monkeypatch.setattr(hd, "_output_path", lambda config: existing)
    report = preflight_hd(CONFIG)
    assert report.status == "blocked"
    assert "output_root_exists" in report.blocking_reasons
    assert report.ready_for_real_pilot is False


class _FailOnceProvider(_FakeProvider):
    def __init__(self, fail_at: int) -> None:
        super().__init__()
        self.fail_at = fail_at

    def complete(self, request):
        if self.calls + 1 == self.fail_at:
            self.calls += 1
            raise LLMProviderError(ProviderErrorCode.TIMEOUT, "safe timeout")
        return super().complete(request)


def test_amended_scenario_request_ledger_resume_and_fingerprint_protection(tmp_path: Path) -> None:
    scenario = CascadeScenarioLoader.load(NEW_SCENARIO)
    runner = CascadeRealAgentRunner()
    kwargs = dict(
        scenario=scenario,
        seed=20260901,
        condition=CascadeCondition.PROVENANCE_AWARE_CONTROLLER,
        run_id="hd-resume-test",
        ledger_path=tmp_path / "ledger.jsonl",
        checkpoint_path=tmp_path / "checkpoint.json",
        request_cap=18,
        completion_reservation_cap=4608,
    )
    with pytest.raises(CascadeRealAgentRunError) as failure:
        runner.run_scenario(provider=_FailOnceProvider(5), **kwargs)
    assert failure.value.code == "timeout"
    resumed_provider = _FakeProvider()
    record = runner.run_scenario(provider=resumed_provider, resume=True, **kwargs)
    assert resumed_provider.calls == 14
    assert record.replay is not None and record.replay.status.value == "passed"
    with pytest.raises(CascadeRealAgentRunError) as changed:
        runner.run_scenario(provider=_FakeProvider(), resume=True, **{**kwargs, "seed": 20260902})
    assert changed.value.code == "checkpoint_binding_mismatch"


def test_full_48_run_fake_provider_smoke_is_complete_and_safe() -> None:
    summary = run_smoke(CONFIG)
    assert summary["run_count"] == 48
    assert summary["matched_group_count"] == 12
    assert summary["provider_call_count"] == 864
    assert summary["logical_request_count"] == 864
    assert summary["replay_status"] == "passed"
    assert summary["scheduled_count"] == summary["applied_count"] == 9
    assert summary["rejected_count"] == 0
    assert summary["private_truth_exposed"] is False
    assert summary["network"] == "disabled"
    rows = {(item["scenario_id"], item["condition"]): item for item in summary["condition_summaries"]}
    aware = rows[("cascade-false-majority", "provenance_aware_controller")]
    assert aware["scheduled_count"] == aware["applied_count"] == 3
    assert aware["reason_codes"] == ["visible_unverified_same_root_repetition"]
    for scenario_id in ("cascade-independent-true-consensus", "cascade-unresolved-disagreement"):
        assert all(rows[(scenario_id, condition.value)]["scheduled_count"] == 0 for condition in CascadeCondition)


def test_safe_receipt_is_idempotent_and_contains_no_sensitive_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hd, "_ROOT", tmp_path)
    output = tmp_path / "outputs/study-locks/receipt.json"
    summary = {
        "run_count": 48,
        "replay_status": "passed",
        "ledger_sha256": "a" * 64,
        "proposal_evaluation_count": 27,
        "scheduled_count": 9,
        "applied_count": 9,
    }
    receipt = write_amendment_receipt(CONFIG, summary, output)
    assert write_amendment_receipt(CONFIG, summary, output) == receipt
    raw = output.read_text(encoding="utf-8").lower()
    assert receipt.ready_for_real_pilot is False
    assert "prompt" not in raw
    assert "api_key" not in raw
    assert "provider_metadata" not in raw
    assert "ground_truth_label" not in raw
    assert "source_independence_label" not in raw
