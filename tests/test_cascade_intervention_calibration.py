from __future__ import annotations

import json
from pathlib import Path

from evicon.cascade_controller import CascadeAction
from evicon.cascade_intervention_calibration import (
    CascadeInterventionCalibrationRunner,
    run_calibration,
)
from evicon.cascade_intervention_calibration_smoke import main as calibration_smoke_main, run_smoke
from evicon.cascade_protocol import CascadeScenarioLoader
from evicon.provenance_cascade_preregistration import CascadeCondition

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/provenance_cascade/calibration/provenance_cascade_intervention_calibration.v1.toml"


def _runner() -> CascadeInterventionCalibrationRunner:
    return CascadeInterventionCalibrationRunner.from_file(MANIFEST)


def test_calibration_runs_all_scenarios_seeds_conditions_and_replays() -> None:
    rows = run_calibration(MANIFEST)
    assert len(rows) == 4 * 3 * 4
    assert all(row.replay_status == "passed" for row in rows)
    assert all(row.application_replay_status == "passed" for row in rows)
    assert all(row.development_only and row.calibration_only and row.not_paper_result and row.no_causal_conclusion for row in rows)


def test_false_majority_trigger_is_next_round_only_and_one_root() -> None:
    runner = _runner()
    scenario = next(item.scenario for item in runner.scenarios if item.binding.scenario_type.value == "false_majority")
    summary, record = runner.run_one(scenario, 20260901, CascadeCondition.PROVENANCE_AWARE_CONTROLLER)
    assert summary.proposal_count == 1
    assert summary.schedule_count == 1
    assert summary.applied_count == 1
    assert summary.action == (CascadeAction.REQUEST_INDEPENDENT_SOURCE.value,)
    assert "visible_unverified_same_root_repetition" in summary.reason_codes
    assert summary.same_root_target_agent_id == "network-agent-01"
    assert summary.same_root_visible_round == 1
    assert summary.same_root_visible_repetition_count == 2
    assert summary.same_root_visible_root_count == 1
    schedule = record.application_ledger.schedules[0]
    assert schedule.created_round_id == 1
    assert schedule.effective_round_id == 2
    target_contexts = [context for context in record.round_contexts if context.snapshot.agent_id == "network-agent-01"]
    assert len(target_contexts[0].directives) == 0
    assert len(target_contexts[1].directives) == 0
    assert len(target_contexts[2].directives) == 1


def test_round_one_cannot_trigger_before_same_root_content_is_visible() -> None:
    runner = _runner()
    scenario = next(item.scenario for item in runner.scenarios if item.binding.scenario_type.value == "false_majority")
    events: list[object] = []
    round_zero = runner.runner.build_round_snapshot(scenario, tuple(events), 0)
    assert all(len(snapshot.visible_content_ids) == 0 for snapshot in round_zero)
    runner.runner.base_runner._append_initial_content(scenario, 0, events)
    round_one = runner.runner.build_round_snapshot(scenario, tuple(events), 1)
    target = next(snapshot for snapshot in round_one if snapshot.agent_id == "network-agent-01")
    assert len(target.visible_content_ids) == 2


def test_current_round_public_outcomes_and_snapshots_are_unchanged_by_proposal() -> None:
    runner = _runner()
    scenario = next(item.scenario for item in runner.scenarios if item.binding.scenario_type.value == "false_majority")
    _, baseline = runner.run_one(scenario, 20260901, CascadeCondition.NO_INTERVENTION)
    _, controlled = runner.run_one(scenario, 20260901, CascadeCondition.PROVENANCE_AWARE_CONTROLLER)
    assert baseline.snapshot_sha256 == controlled.snapshot_sha256
    assert [item.model_dump(mode="json") for item in baseline.outcome_ledger.outcomes[:12]] == [item.model_dump(mode="json") for item in controlled.outcome_ledger.outcomes[:12]]
    assert baseline.outcome_ledger.outcomes[12:] != controlled.outcome_ledger.outcomes[12:]


def test_protected_scenarios_and_source_blind_boundary() -> None:
    runner = _runner()
    for spec in runner.scenarios:
        for condition in (CascadeCondition.GENERIC_DISSENT, CascadeCondition.SOURCE_BLIND_CONTROLLER, CascadeCondition.PROVENANCE_AWARE_CONTROLLER):
            summary, record = runner.run_one(spec.scenario, 20260901, condition)
            if spec.binding.scenario_type.value == "false_majority":
                if condition is CascadeCondition.SOURCE_BLIND_CONTROLLER:
                    assert all(not schedule.visible_source_root_ids for schedule in record.application_ledger.schedules)
            else:
                assert summary.applied_count == 0
                assert summary.action == ()


def test_smoke_is_safe_and_does_not_expose_private_truth(capsys) -> None:
    assert calibration_smoke_main([]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["run_count"] == 48
    assert payload["network"] == "disabled"
    assert payload["results_written"] is False
    assert "ground_truth_label" not in output
    assert "source_independence_label" not in output
    assert "prompt" not in output
    assert "results/" not in output
