from __future__ import annotations

import json
from pathlib import Path

from evicon.cascade_agent_protocol_hg2 import EpistemicStance
from evicon.cascade_agent_protocol_hg23 import (
    AdoptionDecision, HG23ParserError, SharingDecision,
    parse_hg23_response, render_hg23_turn,
)
from evicon.provenance_cascade_hg2_compatibility import build_minimal_hg2_context
from evicon.cascade_outcomes import ClaimStance
from evicon.provenance_cascade_hg2 import sha256_file
from evicon.provenance_cascade_hg23 import final_preflight, run_fake_smoke

ROOT = Path(__file__).resolve().parents[1]
OLD_RECEIPT = ROOT / "results/provenance-cascade-hg22-eligibility-calibration-v1/calibration_receipt.json"
OLD_REPORT = ROOT / "results/analyses/provenance-cascade-hg22-eligibility-evaluator-v1/analysis_report.json"
FORMAL_ROOT = ROOT / "results/provenance-cascade-hg23-adoption-identifiability-v1"


def _payload(context, **updates):
    content_id = context.public_context.contents[0].content_id
    value = {
        "epistemic_stance": "uncertain",
        "adoption_decision": "adopt_claim",
        "sharing_decision": "share_with_caveat",
        "content_ids_used": [content_id],
        "evidence_ids_used": [],
        "share_content_id": content_id,
    }
    value.update(updates)
    return value


def test_epistemic_uncertainty_does_not_erase_observable_adoption() -> None:
    context = build_minimal_hg2_context()
    parsed, error = parse_hg23_response(json.dumps(_payload(context)), context)
    assert error is None and parsed is not None
    assert parsed.epistemic_stance is EpistemicStance.UNCERTAIN
    assert parsed.adoption_decision is AdoptionDecision.ADOPT_CLAIM
    assert parsed.stance is ClaimStance.ENDORSES
    assert parsed.sharing_decision is SharingDecision.SHARE_WITH_CAVEAT


def test_adoption_and_sharing_are_independent() -> None:
    context = build_minimal_hg2_context()
    parsed, error = parse_hg23_response(json.dumps(_payload(
        context, sharing_decision="do_not_share", share_content_id=None,
    )), context)
    assert error is None and parsed is not None
    assert parsed.adoption_decision is AdoptionDecision.ADOPT_CLAIM
    assert parsed.sharing_decision is SharingDecision.DO_NOT_SHARE


def test_parser_rejects_unknown_unseen_and_share_conflicts() -> None:
    context = build_minimal_hg2_context()
    extra = _payload(context, controller_verdict="true")
    assert parse_hg23_response(json.dumps(extra), context)[1] is HG23ParserError.EXTRA_FIELD
    unseen = _payload(context, content_ids_used=["unseen"], share_content_id="unseen")
    assert parse_hg23_response(json.dumps(unseen), context)[1] is HG23ParserError.UNAVAILABLE_CONTENT_ID
    conflict = _payload(context, sharing_decision="do_not_share")
    assert parse_hg23_response(json.dumps(conflict), context)[1] is HG23ParserError.SHARE_BEHAVIOR_INCONSISTENT


def test_prompt_freezes_non_truth_adoption_semantics_without_private_fields() -> None:
    request = render_hg23_turn(build_minimal_hg2_context())
    payload = (request.system_prompt + request.user_prompt).lower()
    assert "adopt_claim does not certify objective truth" in payload
    assert "epistemic_confidence_is_separate_from_public_adoption" in payload
    assert "ground_truth_label" not in payload
    assert "source_independence_label" not in payload


def test_full_fake_smoke_creates_all_required_eligibility_without_effect_claim() -> None:
    report = run_fake_smoke()
    assert report["run_count"] == 16
    assert report["matched_group_count"] == 4
    assert report["logical_request_count"] == 288
    assert report["replay_passed_count"] == 16
    assert report["behavior_observation_count"] == 288
    assert report["false_initial_adoption_eligible_count"] > 0
    assert report["correction_transition_eligible_count"] > 0
    assert report["harmful_conformity_eligible_count"] > 0
    assert report["effectiveness_claimed"] is False
    operations = report["operations"]
    assert operations["cascade-hg1-false-majority|provenance_aware_controller"]["applied"] > 0
    assert operations["cascade-hg1-independent-true-consensus|provenance_aware_controller"]["applied"] == 0


def test_preflight_is_offline_and_blocked_only_on_new_approval_compatibility() -> None:
    report = final_preflight()
    assert report["ready_for_real_calibration"] is False
    assert report["network"] == "disabled"
    assert report["provider_constructed"] is False
    assert report["api_key_read"] is False
    assert report["blocking_reasons"] == ["compatibility_check_required"]
    assert not FORMAL_ROOT.exists()


def test_hg22_inputs_remain_hash_locked() -> None:
    assert sha256_file(OLD_RECEIPT) == "a939fee68762e777220d5c12b82dc8810056870a12b02ada09afffedbf966870"
    assert sha256_file(OLD_REPORT) == "7f651c299411c1deb1ac4c9ae548a3449fdb59ed928af458f3f9f8780b5859ba"

