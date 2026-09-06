from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.cascade_actors import (
    ActorProfile,
    ActorSafetyError,
    DeterministicActor,
    DeterministicActorSchedule,
)
from evicon.cascade_intervention_application import (
    CascadeDirectiveType,
    StructuredCascadeDirective,
)
from evicon.cascade_outcome_runner import CascadeOutcomeRunner
from evicon.cascade_outcome_smoke import main as outcome_smoke_main, run_smoke
from evicon.cascade_protocol import CascadeScenarioLoader
from evicon.provenance_cascade_preregistration import CascadeCondition
from evicon.provenance_cascade_exposure import ControllerPublicView
from evicon.cascade_controller_smoke import fixture_view

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "configs/provenance_cascade/scenarios"
NAMES = ("false_majority", "true_minority_correction", "independent_true_consensus", "unresolved_disagreement")


def load(name: str):
    return CascadeScenarioLoader.load(SCENARIOS / f"{name}.toml")


def test_outcome_runner_covers_four_scenarios_and_replay() -> None:
    for name in NAMES:
        record = CascadeOutcomeRunner().run_scenario(load(name), 20260901, CascadeCondition.PROVENANCE_AWARE_CONTROLLER)
        assert record.replay.status.value == "passed"
        assert record.replay.application_replay.status.value == "passed"
        assert record.agent_ids == tuple(f"network-agent-{i:02d}" for i in range(1, 7))
        assert record.round_count == 3
        assert record.outcome_count == 18
        assert record.snapshot_count == 18


def test_outcome_runner_is_bitwise_deterministic() -> None:
    scenario = load("false_majority")
    first = CascadeOutcomeRunner().run_scenario(scenario, 20260901, CascadeCondition.GENERIC_DISSENT)
    second = CascadeOutcomeRunner().run_scenario(scenario, 20260901, CascadeCondition.GENERIC_DISSENT)
    assert first.outcome_ledger_sha256 == second.outcome_ledger_sha256
    assert first.snapshot_sha256 == second.snapshot_sha256
    assert first.exposure_ledger.model_dump(mode="json") == second.exposure_ledger.model_dump(mode="json")
    assert first.outcome_ledger.model_dump(mode="json") == second.outcome_ledger.model_dump(mode="json")


def test_no_intervention_is_zero_effect() -> None:
    record = CascadeOutcomeRunner().run_scenario(load("false_majority"), 20260901)
    assert record.directive_applied_count == 0
    assert record.proposal_count == 0
    assert record.application_ledger.schedules == ()
    assert record.application_ledger.audit_events == ()


def test_actor_consumes_next_round_directive_without_private_data() -> None:
    view = fixture_view("false_majority")
    directive = StructuredCascadeDirective(
        directive_id="synthetic-schedule-directive",
        directive_type=CascadeDirectiveType.VERIFICATION_REQUEST,
        scenario_id=view.scenario_id,
        target_agent_id=view.agent_id,
        claim_id="claim-fm",
        effective_round_id=view.round_id,
        visible_content_ids=tuple(node.content_id for node in view.provenance_nodes),
        visible_evidence_ids=tuple(card.evidence_id for card in view.evidence_cards),
        reason_codes=("visible_unverified_same_root_repetition",),
    )
    decision = DeterministicActor(ActorProfile.VERIFICATION_RESPONSIVE_ACTOR).act(
        view.agent_id,
        view.round_id,
        view,
        directive,
        DeterministicActorSchedule(profile=ActorProfile.VERIFICATION_RESPONSIVE_ACTOR),
    )
    assert decision.stance.value in {"uncertain", "endorses", "rejects", "no_position"}
    serialized = json.dumps(decision.model_dump(mode="json"), sort_keys=True)
    assert "ground_truth_label" not in serialized
    assert "source_independence_label" not in serialized
    assert "prompt" not in serialized


def test_actor_rejects_wrong_target_and_unexposed_directive() -> None:
    view = fixture_view("false_majority")
    actor = DeterministicActor(ActorProfile.VERIFICATION_RESPONSIVE_ACTOR)
    bad_target = StructuredCascadeDirective(
        directive_id="bad-target",
        directive_type=CascadeDirectiveType.VERIFICATION_REQUEST,
        scenario_id=view.scenario_id,
        target_agent_id="network-agent-06",
        claim_id="claim-fm",
        effective_round_id=view.round_id,
    )
    with pytest.raises(ActorSafetyError) as exc:
        actor.act(view.agent_id, view.round_id, view, bad_target, DeterministicActorSchedule())
    assert exc.value.code == "directive_target_mismatch"
    bad_content = StructuredCascadeDirective(
        directive_id="bad-content",
        directive_type=CascadeDirectiveType.REASONING_REQUEST,
        scenario_id=view.scenario_id,
        target_agent_id=view.agent_id,
        claim_id="claim-fm",
        effective_round_id=view.round_id,
        visible_content_ids=("not-visible",),
    )
    with pytest.raises(ActorSafetyError) as exc:
        actor.act(view.agent_id, view.round_id, view, bad_content, DeterministicActorSchedule())
    assert exc.value.code == "directive_content_not_visible"


def test_outcome_smoke_is_offline_and_safe(capsys: pytest.CaptureFixture[str]) -> None:
    rows = run_smoke()
    assert len(rows) == 4 * 3 * 4
    assert all(row["replay_status"] == "passed" for row in rows)
    assert all(row["application_replay_status"] == "passed" for row in rows)
    assert outcome_smoke_main([]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["network"] == "disabled"
    assert payload["results_written"] is False
    assert payload["private_truth_exposed"] is False
    assert "ground_truth_label" not in output
    assert "source_independence_label" not in output
    assert "prompt" not in output
    assert "results/" not in output
