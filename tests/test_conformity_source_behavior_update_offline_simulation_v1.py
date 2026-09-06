from __future__ import annotations

import json
from collections import Counter, defaultdict

import pytest

import evicon.conformity_source_behavior_update_offline_simulation_v1 as protocol
from evicon.conformity_source_behavior_update_offline_simulation_v1 import (
    DEFAULT_CONFIG,
    FakeResponsePattern,
    PresentationCounterbalance,
    PublicPriorJudgment,
    RESPONSE_SCHEMA,
    RESPONSE_SCHEMA_SHA256,
    StatefulUpdateFakeProvider,
    UpdateCondition,
    UpdatePhase,
    build_initial_cases,
    build_update_cases,
    contrast_summaries,
    load_config,
    order_shortcut_summaries,
    parse_response,
    protocol_self_check,
    public_payload,
    public_prior_sha256,
    render_request,
    run_offline_smoke,
    run_pattern,
    safe_observation,
    validate_execution,
)


def _initial_case():
    return build_initial_cases()[0]


def _initial_response(case):
    provider = StatefulUpdateFakeProvider(FakeResponsePattern.CONSTANT)
    return parse_response(provider.complete(render_request(case)).content, case)


def _update_case(condition: UpdateCondition = UpdateCondition.SAME_ROOT_SHOWN):
    initial_case = _initial_case()
    initial_response = _initial_response(initial_case)
    return next(
        case
        for case in build_update_cases(initial_case, initial_response)
        if case.condition is condition
    )


def _valid_response(case, **updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "ordinal_judgment": "uncertain",
        "public_action": "withhold",
        "share_decision": "do_not_share",
        "confidence": 3,
        "content_ids_used": [case.visible_content_ids[0]],
        "share_content_id": None,
    }
    payload.update(updates)
    return payload


def test_hash_bound_config_is_strictly_offline() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert config.protocol_sha256 == protocol.sha256_file(config.protocol_path)
    assert config.response_schema_sha256 == RESPONSE_SCHEMA_SHA256
    assert config.network_enabled is False
    assert config.real_provider_permitted is False
    assert config.api_key_read_permitted is False
    assert config.results_write_permitted is False
    assert config.evaluator_private_truth_available is False
    assert config.historical_results_excluded is True


def test_initial_matrix_has_one_j0_per_group_and_two_counterbalanced_presentations() -> None:
    cases = build_initial_cases()
    assert len(cases) == 16
    assert len({case.case_id for case in cases}) == 16
    assert all(case.phase is UpdatePhase.INITIAL for case in cases)
    assert all(case.condition is None for case in cases)
    assert Counter(case.scenario_id for case in cases) == Counter({
        "stateful-update-v1-s01": 4,
        "stateful-update-v1-s02": 4,
        "stateful-update-v1-s03": 4,
        "stateful-update-v1-s04": 4,
    })
    assert Counter(case.presentation_counterbalance for case in cases) == Counter({
        PresentationCounterbalance.FORWARD: 8,
        PresentationCounterbalance.REVERSED: 8,
    })
    grouped = defaultdict(list)
    for case in cases:
        grouped[(case.scenario_id, case.seed)].append(case)
        assert case.visible_content_ids == ("content-initial",)
        assert case.visible_public_prior_judgment is None
        assert case.public_prior_judgment_sha256 is None
    assert len(grouped) == 8
    assert all(
        {item.presentation_counterbalance for item in group}
        == set(PresentationCounterbalance)
        for group in grouped.values()
    )


def test_each_j1_branch_uses_the_exact_actual_j0_and_has_one_source_unshown_call() -> None:
    initial = _initial_case()
    response = _initial_response(initial)
    branches = build_update_cases(initial, response)
    expected_prior = PublicPriorJudgment.from_initial_response(response)
    expected_hash = public_prior_sha256(expected_prior)
    assert len(branches) == 5
    assert {case.condition for case in branches} == set(UpdateCondition)
    assert all(case.phase is UpdatePhase.UPDATE for case in branches)
    assert all(case.visible_public_prior_judgment == expected_prior for case in branches)
    assert all(case.public_prior_judgment_sha256 == expected_hash for case in branches)
    assert sum(
        case.condition is UpdateCondition.SOURCE_UNSHOWN_REPETITION
        for case in branches
    ) == 1


def test_social_branches_are_text_and_order_matched_within_group_but_root_projection_changes() -> None:
    for initial in build_initial_cases():
        branches = {
            case.condition: case
            for case in build_update_cases(initial, _initial_response(initial))
        }
        social = [
            branches[UpdateCondition.SOURCE_UNSHOWN_REPETITION],
            branches[UpdateCondition.SAME_ROOT_SHOWN],
            branches[UpdateCondition.INDEPENDENT_ROOTS_SHOWN],
        ]
        expected_text = [
            (item.content_id, item.public_summary)
            for item in social[0].visible_content
        ]
        assert all(
            [(item.content_id, item.public_summary) for item in case.visible_content]
            == expected_text
            for case in social[1:]
        )
        assert social[0].visible_social_root_count == 0
        assert social[1].visible_social_root_count == 1
        assert social[2].visible_social_root_count == 2
        assert social[0].visible_evidence_card_ids == ()
        assert social[1].visible_evidence_card_ids == ()
        assert social[2].visible_evidence_card_ids == ()


def test_counterbalance_changes_social_order_and_root_codes_without_changing_the_material_set() -> None:
    pairs = defaultdict(list)
    for initial in build_initial_cases():
        pairs[(initial.scenario_id, initial.seed)].append(initial)
    for pair in pairs.values():
        initial_by_presentation = {
            case.presentation_counterbalance: case for case in pair
        }
        forward = build_update_cases(
            initial_by_presentation[PresentationCounterbalance.FORWARD],
            _initial_response(initial_by_presentation[PresentationCounterbalance.FORWARD]),
        )
        reversed_cases = build_update_cases(
            initial_by_presentation[PresentationCounterbalance.REVERSED],
            _initial_response(initial_by_presentation[PresentationCounterbalance.REVERSED]),
        )
        forward_by_condition = {case.condition: case for case in forward}
        reversed_by_condition = {case.condition: case for case in reversed_cases}
        for condition in (
            UpdateCondition.SOURCE_UNSHOWN_REPETITION,
            UpdateCondition.SAME_ROOT_SHOWN,
            UpdateCondition.INDEPENDENT_ROOTS_SHOWN,
        ):
            forward_case = forward_by_condition[condition]
            reversed_case = reversed_by_condition[condition]
            assert forward_case.social_content_ids == ("message-a", "message-b")
            assert reversed_case.social_content_ids == ("message-b", "message-a")
            assert {
                item.content_id: item.public_summary
                for item in forward_case.visible_content
            } == {
                item.content_id: item.public_summary
                for item in reversed_case.visible_content
            }
        assert (
            forward_by_condition[UpdateCondition.SAME_ROOT_SHOWN]
            .public_root_assignments[0]
            .source_root_id
            != reversed_by_condition[UpdateCondition.SAME_ROOT_SHOWN]
            .public_root_assignments[0]
            .source_root_id
        )
        assert {
            item.content_id: item.source_root_id
            for item in forward_by_condition[
                UpdateCondition.INDEPENDENT_ROOTS_SHOWN
            ].public_root_assignments
        } != {
            item.content_id: item.source_root_id
            for item in reversed_by_condition[
                UpdateCondition.INDEPENDENT_ROOTS_SHOWN
            ].public_root_assignments
        }


def test_rendered_prompt_exposes_j0_to_j1_but_no_internal_study_coordinate() -> None:
    initial = _initial_case()
    initial_request = render_request(initial)
    assert set(initial_request.metadata) == {"protocol_version", "template_version"}
    initial_payload = json.loads(initial_request.user_prompt)
    assert set(initial_payload) == {
        "template_version",
        "phase",
        "target_claim",
        "decision_task",
        "visible_content",
        "public_root_relations",
        "visible_evidence_card_ids",
    }
    assert "visible_public_prior_judgment" not in initial_payload

    update = _update_case()
    request = render_request(update)
    payload = json.loads(request.user_prompt)
    assert payload["visible_public_prior_judgment"] == update.visible_public_prior_judgment.model_dump(mode="json")
    assert payload["public_prior_judgment_sha256"] == update.public_prior_judgment_sha256
    assert payload["reflection_only"] is False
    rendered_prompt = request.system_prompt + request.user_prompt
    for hidden in (
        update.case_id,
        update.group_id,
        update.scenario_id,
        update.condition.value,
        "ground_truth",
        "evaluator_private",
        "latent",
        "api_key",
    ):
        assert hidden not in rendered_prompt.lower()


def test_parser_rejects_malformed_unknown_unexposed_and_inconsistent_output() -> None:
    initial = _initial_case()
    update = _update_case()
    assert parse_response(json.dumps(_valid_response(initial)), initial).ordinal_judgment.value == "uncertain"
    assert parse_response(json.dumps(_valid_response(update)), update).ordinal_judgment.value == "uncertain"

    with pytest.raises(ValueError, match="malformed_json"):
        parse_response("{", update)
    with pytest.raises(ValueError, match="top_level_type"):
        parse_response("[]", update)
    with pytest.raises(ValueError, match="malformed_json"):
        parse_response(
            '{"ordinal_judgment":"uncertain","ordinal_judgment":"uncertain"}',
            update,
        )
    with pytest.raises(ValueError, match="invalid_schema"):
        parse_response(json.dumps(_valid_response(update, extra="forbidden")), update)
    with pytest.raises(ValueError, match="unavailable_content_id"):
        parse_response(json.dumps(_valid_response(update, content_ids_used=["unseen"])), update)
    with pytest.raises(ValueError, match="duplicate_content_id"):
        parse_response(json.dumps(_valid_response(
            update,
            content_ids_used=["content-initial", "content-initial"],
        )), update)
    with pytest.raises(ValueError, match="judgment_action_mismatch"):
        parse_response(json.dumps(_valid_response(
            update,
            ordinal_judgment="lean_adopt",
            public_action="withhold",
        )), update)
    with pytest.raises(ValueError, match="share_content_id_unexpected"):
        parse_response(json.dumps(_valid_response(
            update, share_content_id="content-initial"
        )), update)


def test_case_contract_rejects_tampered_or_missing_prior_before_a_request_is_rendered() -> None:
    update = _update_case()
    payload = update.model_dump(mode="python")
    payload["public_prior_judgment_sha256"] = "0" * 64
    with pytest.raises(Exception, match="stateful_update_prior_hash_mismatch"):
        type(update).model_validate(payload)

    payload = update.model_dump(mode="python")
    payload["visible_public_prior_judgment"] = None
    with pytest.raises(Exception, match="stateful_update_prior_missing"):
        type(update).model_validate(payload)


def test_fake_matrix_makes_exactly_one_j0_then_five_j1_per_group_and_reuses_it() -> None:
    observations, calls = run_pattern(FakeResponsePattern.CONSTANT)
    assert calls == 96
    groups = defaultdict(list)
    for observation in observations:
        groups[observation.group_id].append(observation)
    assert len(groups) == 16
    for group in groups.values():
        initial = [item for item in group if item.phase is UpdatePhase.INITIAL]
        updates = [item for item in group if item.phase is UpdatePhase.UPDATE]
        assert len(initial) == 1
        assert len(updates) == 5
        assert {item.condition for item in updates} == set(UpdateCondition)
        expected_prior = public_prior_sha256(PublicPriorJudgment(
            ordinal_judgment=initial[0].ordinal_judgment,
            public_action=initial[0].public_action,
            share_decision=initial[0].share_decision,
            confidence=initial[0].confidence,
            content_ids_used=("content-initial",),
            share_content_id=None,
        ))
        assert {item.public_prior_judgment_sha256 for item in updates} == {expected_prior}
    validate_execution(observations)


def test_offline_analysis_recovers_known_patterns_and_flags_order_shortcut_without_claiming_effect() -> None:
    constant, _ = run_pattern(FakeResponsePattern.CONSTANT)
    known, _ = run_pattern(FakeResponsePattern.KNOWN_DIFFERENCE)
    shortcut, _ = run_pattern(FakeResponsePattern.ORDER_SHORTCUT)

    assert all(
        item.mean_ordinal_difference == 0.0
        for item in contrast_summaries(constant)
    )
    known_by_id = {item.contrast_id.value: item for item in contrast_summaries(known)}
    assert known_by_id["self_reflection_minus_initial"].mean_ordinal_difference == 0.0
    assert known_by_id["source_unshown_repetition_minus_self_reflection"].mean_ordinal_difference == 1.0
    assert known_by_id["same_root_shown_minus_source_unshown_repetition"].mean_ordinal_difference == -1.0
    assert known_by_id["independent_roots_shown_minus_same_root_shown"].mean_ordinal_difference == 1.0
    assert known_by_id["direct_evidence_minus_self_reflection"].mean_ordinal_difference == 2.0
    shortcut_by_condition = {item.condition: item for item in order_shortcut_summaries(shortcut)}
    assert shortcut_by_condition[UpdateCondition.SOURCE_UNSHOWN_REPETITION].differing_ordinal_pair_count == 8
    assert shortcut_by_condition[UpdateCondition.SAME_ROOT_SHOWN].differing_ordinal_pair_count == 8
    assert shortcut_by_condition[UpdateCondition.INDEPENDENT_ROOTS_SHOWN].differing_ordinal_pair_count == 8
    assert shortcut_by_condition[UpdateCondition.SELF_REFLECTION].differing_ordinal_pair_count == 0
    assert shortcut_by_condition[UpdateCondition.DIRECT_EVIDENCE].differing_ordinal_pair_count == 0


def test_safe_observation_and_smoke_exclude_public_text_roots_prompts_and_truth() -> None:
    update = _update_case(UpdateCondition.INDEPENDENT_ROOTS_SHOWN)
    observation = safe_observation(
        update,
        parse_response(json.dumps(_valid_response(update)), update),
    )
    serialized = observation.model_dump_json().lower()
    for forbidden in (
        "lineage-",
        "message-a",
        "message-b",
        "evidence-card-01",
        "public_summary",
        "system_prompt",
        "user_prompt",
        "api_key",
        "ground_truth",
    ):
        assert forbidden not in serialized

    report = run_offline_smoke()
    assert report["status"] == "stateful_update_offline_smoke_passed"
    assert report["group_count"] == 16
    assert report["branches_per_group"] == 5
    assert report["logical_request_count_per_pattern"] == 96
    assert report["fake_logical_request_count"] == 288
    assert report["constant_pattern_recovered"] is True
    assert report["known_difference_recovered"] is True
    assert report["order_shortcut_detected"] is True
    assert report["safety"] == {
        "network": "disabled",
        "real_provider_constructed": False,
        "api_key_read": False,
        "results_written": False,
        "prompt_saved": False,
        "full_response_saved": False,
        "historical_results_used": False,
        "evaluator_private_truth_exposed": False,
        "behavior_effect_estimated": False,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def test_schema_and_static_self_check_describe_stateful_public_update_only() -> None:
    assert set(protocol.StatefulUpdateResponse.model_fields) == {
        "ordinal_judgment",
        "public_action",
        "share_decision",
        "confidence",
        "content_ids_used",
        "share_content_id",
    }
    assert RESPONSE_SCHEMA["additionalProperties"] is False
    assert len(RESPONSE_SCHEMA_SHA256) == 64
    report = protocol_self_check()
    assert report["group_count"] == 16
    assert report["logical_request_count_per_pattern"] == 96
    assert report["all_initial_request_ids_unique"] is True
    assert report["j1_uses_exact_structured_j0"] is True
    assert report["social_text_and_order_matched_within_group"] is True
    assert report["same_root_and_independent_root_counts"] == [0, 1, 2]
