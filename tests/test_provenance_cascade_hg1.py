from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evicon.cascade_agent_protocol_hg1 import CascadeAgentProtocolHG1Runtime, build_identifiable_prompt_context
from evicon.cascade_controller import CascadeAction, CascadeControllerPolicyLoader
from evicon.cascade_outcomes import ClaimStance
from evicon.cascade_controller_hg1 import propose_hg1, validate_hg1_proposal
from evicon.cascade_hg1_replay import HG1CascadeProtocolRunner, HG1OutcomeReplayValidator
from evicon.cascade_protocol import CascadeProtocolRunner
from evicon.cascade_real_agent_runner import CascadeRealAgentRunError, CascadeRealAgentRunner
from evicon.provenance_cascade_exposure import ControllerPublicView
from evicon.provenance_cascade_hg1 import (
    DEFAULT_CONFIG, HG1Error, HG1FakeProvider, _claim_selector,
    _post_round_events, hg1_preflight, load_hg1_config,
    run_hg1_fake_smoke, sha256_file,
)
from evicon.provenance_cascade_preregistration import CascadeCondition


@pytest.fixture(scope="module")
def loaded():
    return load_hg1_config()


@pytest.fixture(scope="module")
def smoke():
    return run_hg1_fake_smoke()


def _record(tmp_path: Path, scenario_id: str, condition: CascadeCondition, seed: int = 20261001):
    config, scenarios, _, schedules = load_hg1_config()
    scenario = scenarios[scenario_id]
    schedule = schedules[scenario_id]
    base = HG1CascadeProtocolRunner()
    policy = CascadeControllerPolicyLoader.load(Path(config.policy_path))
    provider = HG1FakeProvider()
    record = CascadeRealAgentRunner(base_runner=base, policy_config=policy).run_scenario(
        scenario, seed, condition, provider=provider,
        run_id=f"test-{scenario_id}-{seed}-{condition.value}",
        ledger_path=tmp_path/"request_ledger.jsonl",
        checkpoint_path=tmp_path/"agent_checkpoint.json",
        model_name="hg1-test-fake", agent_temperature=0.2,
        agent_max_tokens=512, request_cap=18, completion_reservation_cap=9216,
        runtime=CascadeAgentProtocolHG1Runtime(), context_builder=build_identifiable_prompt_context,
        outcome_replay_validator=HG1OutcomeReplayValidator,
        initial_events_builder=base.preload_initial_events,
        claim_selector=lambda s, snap, rnd: _claim_selector(schedule, s, snap, rnd),
        proposal_builder=propose_hg1, proposal_validator=validate_hg1_proposal,
        post_round_events_builder=lambda s, rnd, events: _post_round_events(s, schedule, rnd, events),
        actor_schedule_provider=lambda s, rnd: (schedule.propagation_slots if rnd == 0 else ()),
        policy_config=policy,
    )
    return record, provider


def test_config_is_new_complete_and_budgeted(loaded) -> None:
    config, scenarios, truths, schedules = loaded
    assert len(config.runs) == 48
    assert len({item.matched_group_id for item in config.runs}) == 12
    assert config.seeds == (20261001, 20261002, 20261003)
    assert config.request_cap == 864
    assert config.completion_reservation_cap == 442368
    assert config.output_root == "results/provenance-cascade-pilot-hg1-v1"
    assert len(scenarios) == len(truths) == len(schedules) == 4
    # This configuration owns a versioned root. Its historical Pilot may exist
    # after an authorized run; a later evaluator must not treat that as a
    # configuration collision or overwrite it.
    assert config.output_root != "results/provenance-cascade-pilot-hg11-v1"


def test_round_zero_preload_is_opt_in_and_old_runner_stays_empty(loaded) -> None:
    _, scenarios, _, _ = loaded
    scenario = scenarios["cascade-hg1-false-majority"]
    old = CascadeProtocolRunner().build_round_snapshot(scenario, (), 0)
    assert all(not item.visible_content_ids for item in old)
    base = HG1CascadeProtocolRunner()
    events = base.preload_initial_events(scenario)
    new = base.build_round_snapshot(scenario, events, 0)
    assert all(len(item.visible_content_ids) == 1 for item in new)
    assert next(item for item in new if item.agent_id == "network-agent-02").visible_content_ids == ("hg1-fm-content-a",)
    assert next(item for item in new if item.agent_id == "network-agent-03").visible_content_ids == ("hg1-fm-content-b",)


def test_paired_surface_messages_differ_only_in_visible_root_structure(loaded, tmp_path: Path) -> None:
    config, scenarios, _, _ = loaded
    policy = CascadeControllerPolicyLoader.load(Path(config.policy_path))
    proposals = {}
    for scenario_id in ("cascade-hg1-false-majority", "cascade-hg1-independent-true-consensus"):
        scenario = scenarios[scenario_id]
        record, _ = _record(tmp_path/scenario_id, scenario_id, CascadeCondition.PROVENANCE_AWARE_CONTROLLER)
        snapshot = next(item for item in record.exposure_ledger.snapshots if item.agent_id == "network-agent-02" and item.round_id == 1)
        assert len(snapshot.visible_content_ids) == 2
        view = ControllerPublicView.from_snapshot(snapshot, scenario.graph)
        proposals[(scenario_id, "blind")] = propose_hg1(CascadeCondition.SOURCE_BLIND_CONTROLLER, view, policy)
        proposals[(scenario_id, "aware")] = propose_hg1(CascadeCondition.PROVENANCE_AWARE_CONTROLLER, view, policy)
    assert proposals[("cascade-hg1-false-majority", "blind")].used_source_root_ids == ()
    assert proposals[("cascade-hg1-independent-true-consensus", "blind")].action is CascadeAction.REQUEST_INDEPENDENT_SOURCE
    assert proposals[("cascade-hg1-false-majority", "aware")].action is CascadeAction.REQUEST_INDEPENDENT_SOURCE
    assert len(proposals[("cascade-hg1-false-majority", "aware")].used_source_root_ids) == 1
    assert proposals[("cascade-hg1-independent-true-consensus", "aware")].action is CascadeAction.ABSTAIN
    assert len(proposals[("cascade-hg1-independent-true-consensus", "aware")].used_source_root_ids) == 2


def test_directive_is_created_after_round_one_and_applied_round_two(tmp_path: Path) -> None:
    record, provider = _record(tmp_path, "cascade-hg1-false-majority", CascadeCondition.PROVENANCE_AWARE_CONTROLLER)
    assert record.directive_applied_count == 1
    schedule = record.application_ledger.schedules[0]
    assert schedule.created_round_id == 1
    social = [event for event in record.exposure_ledger.events if event.source_actor_id == "network-agent-03"]
    assert len(social) == 1
    assert social[0].round_id == 0
    assert social[0].target_agent_id == "network-agent-02"
    assert schedule.effective_round_id == 2
    assert provider.directive_seen == 1
    round_one = next(item for item in record.round_contexts if item.snapshot.agent_id == schedule.target_agent_id and item.snapshot.round_id == 1)
    round_two = next(item for item in record.round_contexts if item.snapshot.agent_id == schedule.target_agent_id and item.snapshot.round_id == 2)
    assert round_one.directives == ()
    assert len(round_two.directives) == 1


def test_resume_reuses_completed_checkpoint_without_provider_call(tmp_path: Path) -> None:
    record, provider = _record(tmp_path, "cascade-hg1-false-majority", CascadeCondition.NO_INTERVENTION)
    assert provider.calls == 18 and record.logical_request_count == 18
    config, scenarios, _, schedules = load_hg1_config()
    scenario = scenarios["cascade-hg1-false-majority"]
    schedule = schedules[scenario.scenario_id]
    base = HG1CascadeProtocolRunner()
    policy = CascadeControllerPolicyLoader.load(Path(config.policy_path))
    second = HG1FakeProvider()
    resumed = CascadeRealAgentRunner(base_runner=base, policy_config=policy).run_scenario(
        scenario, 20261001, CascadeCondition.NO_INTERVENTION, provider=second,
        run_id=f"test-{scenario.scenario_id}-20261001-no_intervention",
        ledger_path=tmp_path/"request_ledger.jsonl", checkpoint_path=tmp_path/"agent_checkpoint.json",
        model_name="hg1-test-fake", agent_temperature=0.2, agent_max_tokens=512,
        request_cap=18, completion_reservation_cap=9216, resume=True,
        runtime=CascadeAgentProtocolHG1Runtime(), context_builder=build_identifiable_prompt_context,
        outcome_replay_validator=HG1OutcomeReplayValidator,
        initial_events_builder=base.preload_initial_events,
        claim_selector=lambda s, snap, rnd: _claim_selector(schedule, s, snap, rnd),
        proposal_builder=propose_hg1, proposal_validator=validate_hg1_proposal,
        post_round_events_builder=lambda s, rnd, events: _post_round_events(s, schedule, rnd, events),
        actor_schedule_provider=lambda s, rnd: (schedule.propagation_slots if rnd == 0 else ()),
        policy_config=policy,
    )
    assert second.calls == 0
    assert resumed.logical_request_count == 18


def test_resume_binding_changes_are_rejected(tmp_path: Path) -> None:
    _record(tmp_path, "cascade-hg1-false-majority", CascadeCondition.NO_INTERVENTION)
    config, scenarios, _, schedules = load_hg1_config()
    scenario = scenarios["cascade-hg1-false-majority"]
    schedule = schedules[scenario.scenario_id]
    base = HG1CascadeProtocolRunner()
    policy = CascadeControllerPolicyLoader.load(Path(config.policy_path))
    with pytest.raises(CascadeRealAgentRunError, match="checkpoint_binding_mismatch"):
        CascadeRealAgentRunner(base_runner=base, policy_config=policy).run_scenario(
            scenario, 20261002, CascadeCondition.NO_INTERVENTION, provider=HG1FakeProvider(),
            run_id=f"test-{scenario.scenario_id}-20261001-no_intervention",
            ledger_path=tmp_path/"request_ledger.jsonl", checkpoint_path=tmp_path/"agent_checkpoint.json",
            model_name="hg1-test-fake", agent_temperature=0.2, agent_max_tokens=512,
            request_cap=18, completion_reservation_cap=9216, resume=True,
            runtime=CascadeAgentProtocolHG1Runtime(), context_builder=build_identifiable_prompt_context,
            outcome_replay_validator=HG1OutcomeReplayValidator,
            initial_events_builder=base.preload_initial_events,
            claim_selector=lambda s, snap, rnd: _claim_selector(schedule, s, snap, rnd),
            proposal_builder=propose_hg1, proposal_validator=validate_hg1_proposal,
            post_round_events_builder=lambda s, rnd, events: _post_round_events(s, schedule, rnd, events),
            actor_schedule_provider=lambda s, rnd: (schedule.propagation_slots if rnd == 0 else ()),
            policy_config=policy,
        )


def test_full_fake_smoke_has_identifiable_eligibility_and_replay(smoke) -> None:
    assert smoke["run_count"] == smoke["replay_passed_count"] == 48
    assert smoke["logical_request_count"] == smoke["provider_call_count"] == 864
    eligibility = smoke["eligibility"]
    assert eligibility["round0_substantive_stance"] > 0
    assert eligibility["correction_transition_eligible"] > 0
    assert eligibility["harmful_cascade_eligible"] > 0
    assert eligibility["supported_correction_eligible"] > 0
    assert eligibility["share_decision_eligible"] > 0
    assert smoke["effectiveness_claimed"] is False


def test_condition_operations_preserve_protection_and_information_difference(smoke) -> None:
    rows = smoke["scenario_condition_operations"]
    assert rows["cascade-hg1-false-majority|provenance_aware_controller"]["applied_count"] == 3
    assert rows["cascade-hg1-independent-true-consensus|source_blind_controller"]["applied_count"] == 3
    assert rows["cascade-hg1-independent-true-consensus|provenance_aware_controller"]["applied_count"] == 0
    assert rows["cascade-hg1-unresolved-disagreement|provenance_aware_controller"]["applied_count"] == 0


def test_safe_smoke_and_public_materials_have_no_private_truth(smoke, loaded) -> None:
    config, _, _, _ = loaded
    safe = json.dumps(smoke, sort_keys=True).lower()
    assert "ground_truth_label" not in safe
    assert "source_independence_label" not in safe
    for material in config.scenario_materials:
        public = Path(material.graph_path).read_text().lower()
        assert "ground_truth_label" not in public
        assert "source_independence_label" not in public


def test_preflight_is_offline_and_blocks_an_existing_historical_root() -> None:
    summary = hg1_preflight()
    assert summary["status"] == "blocked"
    assert summary["offline_design_complete"] is False
    assert summary["blocking_reasons"] == ["output_root_exists"]
    assert summary["network"] == "disabled"
    assert summary["provider_constructed"] is False
    assert summary["api_key_read"] is False
    assert summary["results_written"] is False


def test_hash_tampering_is_rejected(tmp_path: Path) -> None:
    text = Path(DEFAULT_CONFIG).read_text()
    text = text.replace("5d0b7e7ee62d297fdbe0b3168762ef4bc0946c38a6619ea98d43c17c726eb219", "0"*64)
    path = tmp_path/"bad.toml"
    path.write_text(text)
    with pytest.raises(HG1Error, match="protocol_hash_mismatch"):
        load_hg1_config(path)


def test_proposal_does_not_change_the_round_that_created_it(tmp_path: Path) -> None:
    baseline, _ = _record(tmp_path/"baseline", "cascade-hg1-false-majority", CascadeCondition.NO_INTERVENTION)
    aware, _ = _record(tmp_path/"aware", "cascade-hg1-false-majority", CascadeCondition.PROVENANCE_AWARE_CONTROLLER)
    baseline_snapshot = next(item.snapshot for item in baseline.round_contexts if item.snapshot.agent_id == "network-agent-02" and item.snapshot.round_id == 1)
    aware_snapshot = next(item.snapshot for item in aware.round_contexts if item.snapshot.agent_id == "network-agent-02" and item.snapshot.round_id == 1)
    assert baseline_snapshot.visible_claim_ids == aware_snapshot.visible_claim_ids
    assert baseline_snapshot.visible_content_ids == aware_snapshot.visible_content_ids
    assert baseline_snapshot.visible_evidence_ids == aware_snapshot.visible_evidence_ids
    assert baseline_snapshot.visible_root_relations == aware_snapshot.visible_root_relations
    baseline_outcome = next(item for item in baseline.outcome_ledger.outcomes if item.agent_id == "network-agent-02" and item.round_id == 1)
    aware_outcome = next(item for item in aware.outcome_ledger.outcomes if item.agent_id == "network-agent-02" and item.round_id == 1)
    assert baseline_outcome == aware_outcome


def test_supported_correction_remains_visible_and_retained(tmp_path: Path) -> None:
    record, _ = _record(tmp_path, "cascade-hg1-true-minority-correction", CascadeCondition.PROVENANCE_AWARE_CONTROLLER)
    correction_events = [event for event in record.exposure_ledger.events if event.claim_id == "claim-hg1-tmc-correction"]
    assert len(correction_events) == 6
    assert {event.target_agent_id for event in correction_events} == set(record.agent_ids)
    correction_outcomes = [item for item in record.outcome_ledger.outcomes if item.claim_id == "claim-hg1-tmc-correction" and item.round_id == 2]
    assert len(correction_outcomes) == 3
    assert all(item.stance is ClaimStance.ENDORSES for item in correction_outcomes)
    assert all(item.action.value not in {"hide", "suppress", "downgrade", "remove"} for item in record.application_ledger.schedules)


def test_unresolved_disagreement_preserves_uncertainty(tmp_path: Path) -> None:
    record, _ = _record(tmp_path, "cascade-hg1-unresolved-disagreement", CascadeCondition.PROVENANCE_AWARE_CONTROLLER)
    assert record.directive_applied_count == 0
    assert all(item.stance is ClaimStance.UNCERTAIN for item in record.outcome_ledger.outcomes)


def test_prompt_contains_only_exposed_public_text(loaded) -> None:
    _, scenarios, _, _ = loaded
    scenario = scenarios["cascade-hg1-false-majority"]
    base = HG1CascadeProtocolRunner()
    snapshot = next(item for item in base.build_round_snapshot(scenario, base.preload_initial_events(scenario), 0) if item.agent_id == "network-agent-02")
    context = build_identifiable_prompt_context(ControllerPublicView.from_snapshot(snapshot, scenario.graph), claim_id="claim-hg1-fm")
    request = CascadeAgentProtocolHG1Runtime().render_request(context)
    prompt = request.user_prompt.lower()
    assert "cedarline pilot will extend evening service" in prompt
    assert "hg1-fm-content-b" not in prompt
    assert "ground_truth_label" not in prompt
    assert "source_independence_label" not in prompt


def test_parent_hg_artifacts_remain_immutable() -> None:
    expected = {
        "results/provenance-cascade-pilot-hg-v1/pilot_receipt.json": "28b9065eddcfb63197c50e81296ce2b60da30ee71ad757fc94a70f009b81a24d",
        "results/provenance-cascade-pilot-hg-v1/pilot_batch_record.json": "e50bc273d33e169938a9fba1ab36e1d14e8b5899892b85dc6ab055421d75114b",
        "results/analyses/provenance-cascade-hg-pilot-evaluator-v1/analysis_report.json": "ef9c4655fa282ba4a2197423d86321590926888a2c4bd807cddd8132321db06b",
    }
    assert {path: sha256_file(path) for path in expected} == expected
