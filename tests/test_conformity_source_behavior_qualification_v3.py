from __future__ import annotations

import json
from collections import Counter

import pytest

import evicon.conformity_source_behavior_qualification_v3 as protocol_v3
from evicon.conformity_source_behavior_qualification_v3 import (
    RESPONSE_SCHEMA_SHA256,
    SafeBehaviorCaseAuditV3,
    load_approval,
    load_config,
    parse_behavior_response_v3,
    render_behavior_request_v3,
    safe_case_audit_v3,
    safe_preflight,
)
from evicon.conformity_source_behavior_qualification_v3_smoke import (
    build_cases,
    run_fake_smoke,
)
from evicon.conformity_source_manipulation import SourceProjection


def _valid_response(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "ordinal_judgment": "uncertain",
        "public_action": "withhold",
        "share_decision": "do_not_share",
        "confidence": 3,
        "content_ids_used": ["content-03", "content-04"],
        "share_content_id": None,
    }
    payload.update(updates)
    return payload


def test_v3_is_separate_bound_and_accepted_offline_only() -> None:
    config, corpus = load_config()
    approval = load_approval()
    assert config.gate_id == "evicon-conformity-source-behavior-qualification-v3"
    assert config.v2_artifacts_frozen is True
    assert len(corpus.scenarios) == 4
    assert approval.acceptance_status == "accepted"
    assert approval.accepted_by == "researcher_user"
    assert approval.accepted_on == "2026-08-28"
    assert approval.network_execution_authorized is False
    assert approval.response_schema_sha256 == RESPONSE_SCHEMA_SHA256


def test_v3_has_36_unique_counterbalanced_cases() -> None:
    cases = build_cases()
    assert len(cases) == len({item.case_id for item in cases}) == 36
    assert Counter(item.projection for item in cases) == Counter({
        SourceProjection.SOURCE_FREE: 12,
        SourceProjection.SAME_ROOT: 12,
        SourceProjection.INDEPENDENT_ROOTS: 12,
    })
    for scenario_id in {item.scenario_id for item in cases}:
        selected = [item for item in cases if item.scenario_id == scenario_id]
        assert len(selected) == 9
        for position in (1, 2, 3):
            assert Counter(
                item.projection for item in selected if item.order_position == position
            ) == Counter(SourceProjection)


def test_projection_changes_only_opaque_focal_root_relations() -> None:
    cases = build_cases()
    for scenario_id in {item.scenario_id for item in cases}:
        selected = [item for item in cases if item.scenario_id == scenario_id]
        public_text = {
            (
                item.target_claim,
                item.decision_task,
                tuple(content.public_summary for content in item.visible_content),
            )
            for item in selected
        }
        assert len(public_text) == 1
        assert all(
            {root.content_id for root in item.public_root_assignments}
            == {"content-01", "content-02", "content-03", "content-04"}
            for item in selected
        )
        assert all(
            tuple(root.source_root_id for root in item.public_root_assignments[2:])
            == ("root-03", "root-04")
            for item in selected
        )
        assert {
            item.projection: len({
                root.source_root_id
                for root in item.public_root_assignments[:2]
                if root.source_root_id is not None
            })
            for item in selected
        } == {
            SourceProjection.SOURCE_FREE: 0,
            SourceProjection.SAME_ROOT: 1,
            SourceProjection.INDEPENDENT_ROOTS: 2,
        }


def test_prompt_exposes_task_counter_material_and_public_root_semantics() -> None:
    case = build_cases()[0]
    request = render_behavior_request_v3(case)
    payload = json.loads(request.user_prompt)
    assert payload["target_claim"] == case.target_claim
    assert payload["decision_task"] == case.decision_task
    assert len(payload["visible_content"]) == 4
    assert len(payload["public_root_relations"]) == 4
    assert "supplied public source-root relations" in request.system_prompt
    assert "null means provenance is not shown" in request.system_prompt


def test_prompt_hides_internal_condition_and_order_coordinates() -> None:
    for case in build_cases():
        request = render_behavior_request_v3(case)
        prompt = request.system_prompt + request.user_prompt
        for hidden in (
            case.case_id,
            case.scenario_id,
            case.projection.value,
            str(case.seed),
            "order_position",
            "false_majority",
            "true_minority",
        ):
            assert hidden not in prompt
        assert "root-01" in prompt or case.projection is SourceProjection.SOURCE_FREE
        assert "independent" not in request.user_prompt.lower()


def test_parser_requires_consistent_judgment_action_and_sharing() -> None:
    case = build_cases()[0]
    parsed = parse_behavior_response_v3(json.dumps(_valid_response()), case)
    assert parsed.public_action.value == "withhold"
    with pytest.raises(ValueError, match="judgment_action_mismatch"):
        parse_behavior_response_v3(json.dumps(_valid_response(public_action="adopt")), case)
    with pytest.raises(ValueError, match="share_content_id_invalid"):
        parse_behavior_response_v3(json.dumps(_valid_response(
            ordinal_judgment="lean_adopt",
            public_action="adopt",
            share_decision="share_with_caveat",
            share_content_id="content-01",
        )), case)
    with pytest.raises(ValueError, match="invalid_schema"):
        parse_behavior_response_v3(json.dumps(
            _valid_response(ground_truth_label="forbidden")
        ), case)


def test_safe_audit_has_ordinal_behavior_but_no_text_or_private_fields() -> None:
    case = build_cases()[0]
    parsed = parse_behavior_response_v3(json.dumps(_valid_response()), case)
    audit = safe_case_audit_v3(case, parsed)
    assert isinstance(audit, SafeBehaviorCaseAuditV3)
    assert audit.ordinal_judgment.value == "uncertain"
    serialized = audit.model_dump_json().lower()
    for forbidden in (
        "target_claim", "decision_task", "public_summary", "prompt", "response",
        "content_ids_used", "source_root_id", "ground_truth", "api_key",
    ):
        assert forbidden not in serialized


def test_fake_smoke_exercises_nonconstant_measurement_without_effect_claim() -> None:
    report = run_fake_smoke()
    assert report["status"] == "source_behavior_v3_fake_sensitivity_smoke_passed"
    assert report["logical_request_count"] == 36
    assert report["unique_request_count"] == 36
    assert report["judgment_counts"] == {
        "lean_adopt": 12,
        "lean_reject": 12,
        "uncertain": 12,
    }
    assert report["measurement_sensitivity_exercised"] is True
    assert report["behavior_effect_estimated"] is False
    assert report["network"] == "disabled"


def test_default_preflight_is_offline_accepted_and_writes_nothing() -> None:
    assert "os" not in protocol_v3.__dict__
    report = safe_preflight()
    assert report["status"] == "offline_source_behavior_v3_gate_ready"
    assert report["approval_status"] == "accepted"
    assert report["blocking_reasons"] == [
        "provider_compatibility_not_requested",
    ]
    assert report["network"] == "disabled"
    assert report["provider_constructed"] is False
    assert report["api_key_read"] is False
    assert report["results_written"] is False
