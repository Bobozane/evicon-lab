from __future__ import annotations

import json

import pytest

from evicon.conformity_source_behavior_qualification import (
    AdoptionDecision, SharingDecision, parse_behavior_response, safe_case_audit, safe_preflight,
)
from evicon.conformity_source_behavior_qualification_smoke import build_cases, run_fake_smoke


def _valid(case):
    return {
        "adoption_decision": "withhold", "sharing_decision": "do_not_share",
        "content_ids_used": [], "share_content_id": None,
    }


def test_text_matched_source_projections_have_zero_one_two_visible_roots():
    cases = build_cases()
    assert len(cases) == 12
    for scenario_id in {item.scenario_id for item in cases}:
        selected = [item for item in cases if item.scenario_id == scenario_id]
        assert len({item.visible_content_ids for item in selected}) == 1
        assert len({item.public_summaries for item in selected}) == 1
        assert {
            item.projection.value: len({root.source_root_id for root in item.public_root_assignments if root.source_root_id})
            for item in selected
        } == {"source_free": 0, "same_root": 1, "independent_roots": 2}


def test_parser_requires_visible_ids_and_consistent_sharing():
    case = build_cases()[0]
    parsed = parse_behavior_response(json.dumps(_valid(case)), case)
    audit = safe_case_audit(case, parsed)
    assert audit.parser_status == "valid"
    bad = _valid(case) | {"sharing_decision": "share_visible_content", "share_content_id": "not-visible"}
    with pytest.raises(ValueError, match="share_content_id_invalid"):
        parse_behavior_response(json.dumps(bad), case)
    invalid = _valid(case) | {"ground_truth_label": "forbidden"}
    with pytest.raises(ValueError, match="invalid_schema"):
        parse_behavior_response(json.dumps(invalid), case)


def test_safe_audit_has_decision_categories_but_no_private_or_raw_fields():
    case = build_cases()[0]
    response = parse_behavior_response(json.dumps(_valid(case)), case)
    serialized = json.dumps(safe_case_audit(case, response).model_dump(mode="json"), sort_keys=True).lower()
    for forbidden in ("prompt", "response", "ground_truth", "independence", "api_key", "summary"):
        assert forbidden not in serialized
    assert AdoptionDecision.WITHHOLD.value in serialized
    assert SharingDecision.DO_NOT_SHARE.value in serialized


def test_fake_smoke_only_checks_protocol_not_effectiveness():
    report = run_fake_smoke()
    assert report["status"] == "source_behavior_fake_smoke_passed"
    assert report["logical_request_count"] == 12
    assert report["safe_case_audit_count"] == 12
    assert report["behavior_effect_estimated"] is False
    assert report["private_truth_exposed"] is False


def test_preflight_is_offline_and_remains_approval_gated():
    report = safe_preflight()
    assert report["status"] == "offline_behavior_gate_ready"
    assert report["blocking_reasons"] == ["human_approval_required", "provider_compatibility_not_requested"]
    assert report["network"] == "disabled"
    assert report["provider_constructed"] is False
