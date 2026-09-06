from __future__ import annotations

import json
from collections import Counter, defaultdict

import pytest

import evicon.conformity_source_relation_comprehension_calibration_v1 as protocol
from evicon.conformity_source_relation_comprehension_calibration_v1 import (
    DEFAULT_CONFIG,
    EvidenceState,
    FakeResponsePattern,
    PresentationCounterbalance,
    RESPONSE_SCHEMA,
    RESPONSE_SCHEMA_SHA256,
    SourceRelation,
    SourceRelationCalibrationFakeProvider,
    SourceRelationComprehensionResponse,
    build_cases,
    load_config,
    parse_response,
    protocol_self_check,
    render_request,
    run_offline_smoke,
    safe_case_audit,
    safe_preflight,
)


def _valid_response(case, **updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "visible_social_message_count": 2,
        "source_relation": case.source_relation.value,
        "visible_social_root_count": case.visible_social_root_count,
        "social_root_assignments": [
            item.model_dump(mode="json") for item in case.public_root_assignments
        ],
        "direct_evidence_card_ids": list(case.direct_evidence_card_ids),
        "support_kind": case.support_kind.value,
        "source_structure_alone_decides_truth": False,
    }
    payload.update(updates)
    return payload


def _case(
    relation: SourceRelation = SourceRelation.SAME_SHOWN_ROOT,
    evidence: EvidenceState = EvidenceState.ABSENT,
    counterbalance: PresentationCounterbalance = PresentationCounterbalance.FORWARD,
):
    return next(
        case
        for case in build_cases()
        if case.source_relation is relation
        and case.evidence_state is evidence
        and case.presentation_counterbalance is counterbalance
    )


def test_hash_bound_config_and_default_preflight_are_offline(tmp_path) -> None:
    config = load_config(DEFAULT_CONFIG)
    assert config.protocol_sha256 == protocol.sha256_file(config.protocol_path)
    assert config.response_schema_sha256 == RESPONSE_SCHEMA_SHA256
    assert config.network_enabled is False
    assert config.real_provider_permitted is False
    assert config.api_key_read_permitted is False
    assert config.results_write_permitted is False

    before = tuple(tmp_path.iterdir())
    report = safe_preflight()
    assert report["status"] == "source_relation_comprehension_calibration_ready"
    assert report["network"] == "disabled"
    assert report["real_provider_constructed"] is False
    assert report["api_key_read"] is False
    assert report["results_written"] is False
    assert tuple(tmp_path.iterdir()) == before


def test_matrix_has_exact_orthogonal_coverage_and_text_matching() -> None:
    cases = build_cases()
    assert len(cases) == 24
    assert len({case.case_id for case in cases}) == 24
    assert Counter(case.source_relation for case in cases) == Counter({
        SourceRelation.NOT_SHOWN: 8,
        SourceRelation.SAME_SHOWN_ROOT: 8,
        SourceRelation.DISTINCT_SHOWN_ROOTS: 8,
    })
    assert Counter(case.evidence_state for case in cases) == Counter({
        EvidenceState.ABSENT: 12,
        EvidenceState.PRESENT: 12,
    })
    assert Counter(case.presentation_counterbalance for case in cases) == Counter({
        PresentationCounterbalance.FORWARD: 12,
        PresentationCounterbalance.REVERSED: 12,
    })

    grouped = defaultdict(list)
    for case in cases:
        grouped[(
            case.scenario_id,
            case.evidence_state,
            case.presentation_counterbalance,
        )].append(case)
    assert len(grouped) == 8
    for group in grouped.values():
        assert {case.source_relation for case in group} == set(SourceRelation)
        public_material = [
            [item.model_dump(mode="json") for item in case.visible_material]
            for case in group
        ]
        assert all(item == public_material[0] for item in public_material[1:])
        by_relation = {case.source_relation: case for case in group}
        assert all(
            item.source_root_id is None
            for item in by_relation[SourceRelation.NOT_SHOWN].public_root_assignments
        )
        assert by_relation[SourceRelation.SAME_SHOWN_ROOT].visible_social_root_count == 1
        assert by_relation[SourceRelation.DISTINCT_SHOWN_ROOTS].visible_social_root_count == 2


def test_counterbalance_changes_order_and_opaque_root_labels_but_not_text() -> None:
    grouped = defaultdict(list)
    for case in build_cases():
        grouped[(case.scenario_id, case.evidence_state, case.source_relation)].append(case)

    for pair in grouped.values():
        by_counterbalance = {
            case.presentation_counterbalance: case for case in pair
        }
        forward = by_counterbalance[PresentationCounterbalance.FORWARD]
        reversed_case = by_counterbalance[PresentationCounterbalance.REVERSED]
        assert tuple(item.content_id for item in forward.social_material) == (
            "message-a", "message-b"
        )
        assert tuple(item.content_id for item in reversed_case.social_material) == (
            "message-b", "message-a"
        )
        assert {
            item.content_id: item.public_summary for item in forward.visible_material
        } == {
            item.content_id: item.public_summary for item in reversed_case.visible_material
        }
        if forward.source_relation is SourceRelation.SAME_SHOWN_ROOT:
            assert (
                forward.public_root_assignments[0].source_root_id
                != reversed_case.public_root_assignments[0].source_root_id
            )
        if forward.source_relation is SourceRelation.DISTINCT_SHOWN_ROOTS:
            assert {
                item.content_id: item.source_root_id
                for item in forward.public_root_assignments
            } != {
                item.content_id: item.source_root_id
                for item in reversed_case.public_root_assignments
            }


def test_rendered_requests_contain_only_public_material_and_safe_metadata() -> None:
    for case in build_cases():
        request = render_request(case)
        assert set(request.metadata) == {"protocol_version", "template_version"}
        assert case.case_id not in request.user_prompt
        assert case.scenario_id not in request.user_prompt
        assert case.evidence_state.value not in request.user_prompt
        assert case.presentation_counterbalance.value not in request.user_prompt
        assert str(case.seed) not in request.user_prompt
        for forbidden in ("ground_truth", "evaluator_private", "latent", "api_key"):
            assert forbidden not in request.model_dump_json().lower()
        payload = json.loads(request.user_prompt)
        assert set(payload) == {
            "template_version",
            "visible_material",
            "public_root_relations",
            "task_scope",
        }
        assert "source_relation" not in payload
        assert "evidence_state" not in payload


def test_parser_accepts_only_exact_public_structure() -> None:
    case = _case()
    parsed = parse_response(json.dumps(_valid_response(case)), case)
    assert isinstance(parsed, SourceRelationComprehensionResponse)

    with pytest.raises(ValueError, match="malformed_json"):
        parse_response("{", case)
    with pytest.raises(ValueError, match="top_level_type"):
        parse_response("[]", case)
    with pytest.raises(ValueError, match="invalid_schema"):
        parse_response(json.dumps(_valid_response(case, unexpected="forbidden")), case)
    with pytest.raises(ValueError, match="malformed_json"):
        parse_response(
            '{"visible_social_message_count":2,"visible_social_message_count":2}',
            case,
        )
    with pytest.raises(ValueError, match="root_count_mismatch"):
        parse_response(json.dumps(_valid_response(case, visible_social_root_count=2)), case)
    with pytest.raises(ValueError, match="source_relation_mismatch"):
        parse_response(json.dumps(_valid_response(
            case, source_relation=SourceRelation.DISTINCT_SHOWN_ROOTS.value
        )), case)
    with pytest.raises(ValueError, match="root_assignment_order_mismatch"):
        parse_response(json.dumps(_valid_response(
            case,
            social_root_assignments=list(reversed([
                item.model_dump(mode="json") for item in case.public_root_assignments
            ])),
        )), case)
    changed_assignment = [
        item.model_dump(mode="json") for item in case.public_root_assignments
    ]
    changed_assignment[0]["source_root_id"] = "lineage-p4"
    with pytest.raises(ValueError, match="root_assignment_mismatch"):
        parse_response(json.dumps(_valid_response(
            case, social_root_assignments=changed_assignment
        )), case)


def test_parser_keeps_social_roots_direct_evidence_and_truth_separate() -> None:
    no_evidence_independent = _case(
        SourceRelation.DISTINCT_SHOWN_ROOTS, EvidenceState.ABSENT
    )
    with pytest.raises(ValueError, match="direct_evidence_card_mismatch"):
        parse_response(json.dumps(_valid_response(
            no_evidence_independent,
            direct_evidence_card_ids=["evidence-card-01"],
            support_kind="direct_evidence_present",
        )), no_evidence_independent)

    with_evidence = _case(
        SourceRelation.SAME_SHOWN_ROOT, EvidenceState.PRESENT
    )
    with pytest.raises(ValueError, match="direct_evidence_card_mismatch"):
        parse_response(json.dumps(_valid_response(
            with_evidence, direct_evidence_card_ids=[]
        )), with_evidence)
    with pytest.raises(ValueError, match="support_kind_mismatch"):
        parse_response(json.dumps(_valid_response(
            with_evidence, support_kind="social_reports_only"
        )), with_evidence)
    with pytest.raises(ValueError, match="invalid_schema"):
        parse_response(json.dumps(_valid_response(
            with_evidence, source_structure_alone_decides_truth=True
        )), with_evidence)


def test_fake_provider_and_shortcuts_exercise_the_contract_without_real_provider() -> None:
    cases = build_cases()
    correct = SourceRelationCalibrationFakeProvider(FakeResponsePattern.CORRECT)
    for case in cases:
        response = correct.complete(render_request(case))
        assert parse_response(response.content, case).source_relation is case.source_relation
        assert response.provider_metadata == {}
        assert response.prompt_tokens is None
    assert correct.calls == 24

    shortcut = SourceRelationCalibrationFakeProvider(FakeResponsePattern.ORDER_SHORTCUT)
    reversed_case = _case(
        SourceRelation.DISTINCT_SHOWN_ROOTS,
        EvidenceState.ABSENT,
        PresentationCounterbalance.REVERSED,
    )
    with pytest.raises(ValueError, match="root_assignment_order_mismatch"):
        parse_response(shortcut.complete(render_request(reversed_case)).content, reversed_case)

    evidence_confusion = SourceRelationCalibrationFakeProvider(
        FakeResponsePattern.EVIDENCE_CONFUSION
    )
    with pytest.raises(ValueError, match="direct_evidence_card_mismatch"):
        parse_response(
            evidence_confusion.complete(render_request(_case(
                SourceRelation.NOT_SHOWN, EvidenceState.ABSENT
            ))).content,
            _case(SourceRelation.NOT_SHOWN, EvidenceState.ABSENT),
        )
    truth_inference = SourceRelationCalibrationFakeProvider(FakeResponsePattern.TRUTH_INFERENCE)
    with pytest.raises(ValueError, match="invalid_schema"):
        parse_response(truth_inference.complete(render_request(_case())).content, _case())


def test_safe_audit_and_smoke_expose_only_aggregate_structural_state(tmp_path) -> None:
    case = _case(SourceRelation.DISTINCT_SHOWN_ROOTS, EvidenceState.PRESENT)
    parsed = parse_response(json.dumps(_valid_response(case)), case)
    audit = safe_case_audit(case, parsed)
    serialized = audit.model_dump_json().lower()
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

    before = tuple(tmp_path.iterdir())
    report = run_offline_smoke()
    assert report["status"] == "source_relation_comprehension_calibration_offline_smoke_passed"
    assert report["case_count"] == 24
    assert report["correct_fake_request_count"] == 24
    assert report["order_shortcut_request_count"] == 24
    assert report["unique_request_count"] == 24
    assert report["order_shortcut_rejection_count"] == 12
    assert report["source_relation_case_counts"] == {
        "distinct_shown_roots": 8,
        "not_shown": 8,
        "same_shown_root": 8,
    }
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
    assert tuple(tmp_path.iterdir()) == before


def test_schema_and_protocol_self_check_are_structural_only() -> None:
    assert set(SourceRelationComprehensionResponse.model_fields) == {
        "visible_social_message_count",
        "source_relation",
        "visible_social_root_count",
        "social_root_assignments",
        "direct_evidence_card_ids",
        "support_kind",
        "source_structure_alone_decides_truth",
    }
    assert RESPONSE_SCHEMA["additionalProperties"] is False
    assert len(RESPONSE_SCHEMA_SHA256) == 64
    report = protocol_self_check()
    assert report["case_count"] == 24
    assert report["all_request_ids_unique"] is True
    assert report["source_structure_does_not_decide_truth_rule"] is True
    assert report["direct_evidence_is_distinct_from_social_roots_rule"] is True
