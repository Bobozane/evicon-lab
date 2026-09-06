from evicon.provenance_cascade_hg12_design import hg12_preflight, load_hg12_design, run_hg12_fake_smoke


def test_hg12_design_binds_new_observable_schedules() -> None:
    config, _, _, schedules = load_hg12_design()
    assert config.request_cap == 864
    assert config.completion_reservation_cap == 884736
    assert schedules["cascade-hg1-false-majority"].post_round_exposures[0].after_round_id == 0
    assert set(schedules["cascade-hg1-true-minority-correction"].outcome_targets[2].values()) == {"claim-hg1-tmc-correction"}


def test_hg12_fake_smoke_establishes_timing_and_eligibility_opportunities() -> None:
    summary = run_hg12_fake_smoke()
    assert summary["run_count"] == 48
    assert summary["replay_passed_count"] == 48
    assert summary["eligibility"]["initial_false_endorsement_eligible"] > 0
    assert summary["eligibility"]["correction_transition_eligible"] > 0
    assert summary["eligibility"]["harmful_cascade_eligible"] > 0
    assert summary["paired_root_opportunities"]["cascade-hg1-false-majority"] == {"visible_content_count": 2, "visible_root_count": 1}


def test_hg12_preflight_remains_offline_and_awaits_new_authorization() -> None:
    summary = hg12_preflight()
    assert summary["status"] == "blocked"
    assert set(summary["blocking_reasons"]) == {"human_approval_required", "one_shot_compatibility_check_required", "output_root_exists"}
    assert summary["network"] == "disabled"
    assert summary["provider_constructed"] is False
