from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

import evicon.conformity_source_behavior_offline_simulation_v1 as protocol
from evicon.conformity_source_behavior_offline_simulation_v1 import (
    FactorCondition,
    FactorizationFakeProvider,
    FactorizationResponse,
    FakeResponsePattern,
    RESPONSE_SCHEMA,
    RESPONSE_SCHEMA_SHA256,
    build_cases,
    load_config,
    parse_factor_response,
    render_factor_request,
    run_offline_smoke,
    safe_preflight,
)


def _valid_response(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "ordinal_judgment": "uncertain",
        "public_action": "withhold",
        "share_decision": "do_not_share",
        "confidence": 3,
        "content_ids_used": ["content-initial"],
        "share_content_id": None,
    }
    payload.update(updates)
    return payload


def test_offline_config_is_hash_bound_and_preflight_stays_disabled() -> None:
    config = load_config()
    assert config.protocol_sha256 == protocol.sha256_file(config.protocol_path)
    assert config.network_enabled is False
    assert config.real_provider_permitted is False
    assert config.api_key_read_permitted is False
    assert config.results_write_permitted is False

    report = safe_preflight()
    assert report["status"] == "offline_factorization_design_ready"
    assert report["ready_for_network"] is False
    assert report["network"] == "disabled"
    assert report["real_provider_constructed"] is False
    assert report["api_key_read"] is False
    assert report["results_written"] is False


def test_matrix_has_twelve_paired_groups_and_eight_conditions() -> None:
    cases = build_cases()
    assert len(cases) == 96
    assert len({case.case_id for case in cases}) == 96
    assert len({case.group_id for case in cases}) == 12
    assert Counter(case.condition for case in cases) == Counter({
        condition: 12 for condition in FactorCondition
    })

    grouped: dict[str, list[object]] = {}
    for case in cases:
        grouped.setdefault(case.group_id, []).append(case)
    for group_cases in grouped.values():
        assert {case.condition for case in group_cases} == set(FactorCondition)
        by_condition = {case.condition: case for case in group_cases}
        source_free = by_condition[FactorCondition.SOURCE_FREE_REPETITION]
        same_hidden = by_condition[FactorCondition.SAME_ROOT_HIDDEN]
        independent_hidden = by_condition[FactorCondition.INDEPENDENT_ROOTS_HIDDEN]
        assert render_factor_request(source_free).user_prompt == render_factor_request(
            same_hidden
        ).user_prompt
        assert render_factor_request(source_free).user_prompt == render_factor_request(
            independent_hidden
        ).user_prompt
        assert same_hidden.latent_focal_root_ids != independent_hidden.latent_focal_root_ids

        same_shown = by_condition[FactorCondition.SAME_ROOT_SHOWN]
        independent_shown = by_condition[FactorCondition.INDEPENDENT_ROOTS_SHOWN]
        assert tuple(item.public_summary for item in same_shown.visible_content) == tuple(
            item.public_summary for item in independent_shown.visible_content
        )
        assert same_shown.visible_focal_root_count == 1
        assert independent_shown.visible_focal_root_count == 2


def test_rendered_request_hides_internal_coordinates_and_private_fields() -> None:
    for case in build_cases():
        request = render_factor_request(case)
        prompt = request.system_prompt + request.user_prompt
        for hidden in (
            case.case_id,
            case.group_id,
            case.scenario_id,
            case.condition.value,
            str(case.seed),
            "latent_focal_root_ids",
        ):
            assert hidden not in prompt
        assert set(request.metadata) == {"protocol_version", "template_version"}
        assert "api_key" not in request.model_dump_json().lower()


def test_parser_rejects_malformed_unknown_unavailable_and_inconsistent_outputs() -> None:
    case = build_cases()[0]
    parsed = parse_factor_response(json.dumps(_valid_response()), case)
    assert isinstance(parsed, FactorizationResponse)
    with pytest.raises(ValueError, match="malformed_json"):
        parse_factor_response("{", case)
    with pytest.raises(ValueError, match="invalid_schema"):
        parse_factor_response(json.dumps(_valid_response(unexpected="x")), case)
    with pytest.raises(ValueError, match="unavailable_content_id"):
        parse_factor_response(json.dumps(_valid_response(
            content_ids_used=["content-initial", "content-unseen"]
        )), case)
    with pytest.raises(ValueError, match="judgment_action_mismatch"):
        parse_factor_response(json.dumps(_valid_response(public_action="adopt")), case)
    with pytest.raises(ValueError, match="share_content_id_invalid"):
        parse_factor_response(json.dumps(_valid_response(
            ordinal_judgment="lean_adopt",
            public_action="adopt",
            share_decision="share_with_caveat",
            share_content_id="content-unseen",
        )), case)
    with pytest.raises(ValueError, match="malformed_json"):
        parse_factor_response(
            '{"ordinal_judgment":"uncertain","ordinal_judgment":"uncertain"}', case
        )


def test_fake_provider_only_returns_schema_valid_public_content() -> None:
    case = build_cases()[0]
    request = render_factor_request(case)
    provider = FactorizationFakeProvider(FakeResponsePattern.KNOWN_DIFFERENCE)
    response = provider.complete(request)
    parsed = parse_factor_response(response.content, case)
    assert provider.calls == 1
    assert set(parsed.content_ids_used).issubset(set(case.visible_content_ids))
    assert response.provider_metadata == {}
    assert response.prompt_tokens is None
    assert response.completion_tokens is None


def test_offline_smoke_recovers_known_patterns_without_claiming_effects() -> None:
    output_root = Path("outputs/conformity-source-behavior-offline-simulation-v1")
    existed_before = output_root.exists()
    report = run_offline_smoke()
    assert report["status"] == "offline_factorization_smoke_passed"
    assert report["logical_request_count"] == 192
    assert report["known_difference_request_count"] == 96
    assert report["constant_request_count"] == 96
    assert report["unique_request_count_per_pattern"] == 96
    assert report["shared_t0_group_count"] == 12
    assert report["shared_t0_reused_without_provider_replay"] is True
    assert report["known_difference_recovered"] is True
    assert report["known_no_difference_recovered"] is True
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
    assert output_root.exists() is existed_before


def test_response_schema_is_strict_and_contains_only_public_behavior_fields() -> None:
    assert set(FactorizationResponse.model_fields) == {
        "ordinal_judgment",
        "public_action",
        "share_decision",
        "confidence",
        "content_ids_used",
        "share_content_id",
    }
    assert RESPONSE_SCHEMA["additionalProperties"] is False
    assert len(RESPONSE_SCHEMA_SHA256) == 64

