from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.cascade_agent_prompts import CascadeAgentRuntimeConfig
from evicon.cascade_agent_protocol_hg import build_identifiable_prompt_context
from evicon.cascade_agent_protocol_hg2 import (
    HG2_RESPONSE_JSON_SCHEMA, HG2_SCHEMA_NAME,
    BehavioralDecision, EpistemicStance, HG2DecisionTask, HG2ParserError,
    HG2PromptContext, PublicDecisionRole, parse_hg2_response, render_hg2_turn,
)
from evicon.openai_provider import OpenAICompatibleProvider, ProviderConfig, ResponseFormatMode
from evicon.provenance_cascade_exposure import ControllerPublicView
from evicon.provenance_cascade_hg2 import (
    DEFAULT_CONFIG, hg2_preflight, load_hg2_config, run_hg2_fake_calibration,
)
from evicon.request_ledger import request_fingerprint_facts


def _context() -> HG2PromptContext:
    _, scenarios, _ = load_hg2_config()
    scenario = scenarios["cascade-hg1-false-majority"]
    from evicon.cascade_hg1_replay import HG1CascadeProtocolRunner

    base = HG1CascadeProtocolRunner()
    events = base.preload_initial_events(scenario)
    snapshot = next(item for item in base.build_round_snapshot(scenario, events, 0) if item.agent_id == "network-agent-02")
    view = ControllerPublicView.from_snapshot(snapshot, scenario.graph)
    public = build_identifiable_prompt_context(
        view, claim_id="claim-hg1-fm",
        runtime_config=CascadeAgentRuntimeConfig(model_name="fake-hg2", max_tokens=1024, seed=20261021),
    )
    return HG2PromptContext(
        public_context=public, role=PublicDecisionRole.RAPID_RESPONSE,
        decision_task=HG2DecisionTask(
            task_id="test-reversible-task",
            public_decision_summary="Choose a provisional public action or defer pending verification.",
            reversible_action="Use a limited notice that can be updated.",
            defer_action="Wait for more public verification.",
        ),
    )


def test_hg2_separates_epistemic_and_behavior_and_strictly_parses() -> None:
    context = _context()
    content_id = context.public_context.contents[0].content_id
    parsed, error = parse_hg2_response(json.dumps({
        "epistemic_stance": "uncertain",
        "behavioral_decision": "share_with_caveat",
        "content_ids_used": [content_id], "evidence_ids_used": [],
        "share_content_id": content_id,
    }), context)
    assert error is None and parsed is not None
    assert parsed.epistemic_stance is EpistemicStance.UNCERTAIN
    assert parsed.behavioral_decision is BehavioralDecision.SHARE_WITH_CAVEAT


@pytest.mark.parametrize(
    ("update", "error"),
    [
        ({"controller_action": "suppress"}, HG2ParserError.EXTRA_FIELD),
        ({"epistemic_stance": "true"}, HG2ParserError.INVALID_EPISTEMIC_STANCE),
        ({"behavioral_decision": "force_consensus"}, HG2ParserError.INVALID_BEHAVIORAL_DECISION),
        ({"content_ids_used": ["unseen-content"], "share_content_id": "unseen-content"}, HG2ParserError.UNAVAILABLE_CONTENT_ID),
        ({"evidence_ids_used": ["unseen-evidence"]}, HG2ParserError.UNAVAILABLE_EVIDENCE_ID),
    ],
)
def test_hg2_parser_rejects_control_truth_and_unseen_ids(update, error) -> None:
    context = _context()
    content_id = context.public_context.contents[0].content_id
    payload = {
        "epistemic_stance": "uncertain", "behavioral_decision": "share_with_caveat",
        "content_ids_used": [content_id], "evidence_ids_used": [], "share_content_id": content_id,
    }
    payload.update(update)
    parsed, actual = parse_hg2_response(json.dumps(payload), context)
    assert parsed is None and actual is error


def test_hg2_share_contract_requires_visible_used_content() -> None:
    context = _context()
    parsed, error = parse_hg2_response(json.dumps({
        "epistemic_stance": "uncertain", "behavioral_decision": "share_with_caveat",
        "content_ids_used": [], "evidence_ids_used": [], "share_content_id": None,
    }), context)
    assert parsed is None and error is HG2ParserError.SHARE_FIELD


def test_hg2_prompt_contains_public_meaning_and_no_private_truth() -> None:
    request = render_hg2_turn(_context())
    payload = json.loads(request.user_prompt)
    assert payload["decision_contract"]["epistemic_stance_is_not_behavior"] is True
    assert payload["public_role"] == "rapid_response"
    assert "CedarLine" in payload["visible_claims"][0]["public_summary"]
    serialized = (request.system_prompt + request.user_prompt + json.dumps(request.metadata)).lower()
    for forbidden in ("ground_truth_label", "source_independence_label", "private_truth", "api_key", "authorization"):
        assert forbidden not in serialized
    facts = request_fingerprint_facts(request)
    assert facts["phase"] == "agent_turn"
    assert facts["agent_id"] == "network-agent-02"
    assert facts["round_id"] == 0


def test_hg2_paired_surface_differs_only_in_public_root_structure() -> None:
    config, scenarios, _ = load_hg2_config()
    tasks = {item.scenario_id: item for item in config.decision_tasks}
    false_id = "cascade-hg1-false-majority"
    consensus_id = "cascade-hg1-independent-true-consensus"
    assert tasks[false_id].public_decision_summary == tasks[consensus_id].public_decision_summary
    assert tuple(node.public_statement for node in scenarios[false_id].graph.nodes[:2]) == tuple(node.public_statement for node in scenarios[consensus_id].graph.nodes[:2])
    false_roots = {root for node in scenarios[false_id].graph.nodes[:2] for root in scenarios[false_id].graph.root_sources_for_node(node.node_id)}
    consensus_roots = {root for node in scenarios[consensus_id].graph.nodes[:2] for root in scenarios[consensus_id].graph.root_sources_for_node(node.node_id)}
    assert len(false_roots) == 1 and len(consensus_roots) == 2


def test_hg2_fake_calibration_establishes_eligibility_without_effect_claim() -> None:
    summary = run_hg2_fake_calibration()
    assert summary["run_count"] == 16
    assert summary["matched_group_count"] == 4
    assert summary["logical_request_count"] == 288
    assert summary["replay_passed_count"] == 16
    assert 0 < summary["round0_substantive_behavior_count"] < 6
    assert summary["correction_transition_eligible_count"] > 0
    assert summary["harmful_cascade_eligible_count"] > 0
    assert summary["effectiveness_claimed"] is False
    ops = summary["operations"]
    assert ops["cascade-hg1-false-majority|provenance_aware_controller"]["applied_count"] > 0
    assert ops["cascade-hg1-independent-true-consensus|provenance_aware_controller"]["applied_count"] == 0
    assert ops["cascade-hg1-independent-true-consensus|source_blind_controller"]["applied_count"] > 0


def test_hg2_preflight_is_offline_blocked_and_writes_no_results() -> None:
    config, _, _ = load_hg2_config()
    root = Path(__file__).resolve().parents[1]
    output = root / config.output_root
    before = output.exists()
    report = hg2_preflight()
    assert report["status"] == "blocked"
    assert report["blocking_reasons"]
    assert report["network"] == "disabled"
    assert report["provider_constructed"] is False
    assert report["api_key_read"] is False
    assert output.exists() is before


def test_provider_custom_schema_is_opt_in_and_legacy_default_is_unchanged() -> None:
    request = render_hg2_turn(_context())
    custom = OpenAICompatibleProvider(ProviderConfig(base_url="https://provider.invalid/v1", model_name="fake", allow_network=False, response_format=ResponseFormatMode.JSON_SCHEMA, response_schema_name=HG2_SCHEMA_NAME, response_schema=HG2_RESPONSE_JSON_SCHEMA), environment={"EVICON_LLM_API_KEY": "not-used"})
    custom_schema = custom._payload(request)["response_format"]["json_schema"]["schema"]
    assert custom_schema == HG2_RESPONSE_JSON_SCHEMA
    assert set(custom_schema["required"]) == {"epistemic_stance", "behavioral_decision", "content_ids_used", "evidence_ids_used", "share_content_id"}
    legacy = OpenAICompatibleProvider(ProviderConfig(base_url="https://provider.invalid/v1", model_name="fake", allow_network=False, response_format=ResponseFormatMode.JSON_SCHEMA, response_schema_name="legacy"), environment={"EVICON_LLM_API_KEY": "not-used"})
    legacy_schema = legacy._payload(request)["response_format"]["json_schema"]["schema"]
    assert "stance" in legacy_schema["properties"] and "epistemic_stance" not in legacy_schema["properties"]


def test_hg2_config_rejects_unknown_fields(tmp_path: Path) -> None:
    raw = Path(DEFAULT_CONFIG).read_text(encoding="utf-8") + "\nunknown_private_truth = true\n"
    candidate = tmp_path / "invalid.toml"
    candidate.write_text(raw, encoding="utf-8")
    with pytest.raises(Exception, match="hg2_config_invalid"):
        load_hg2_config(candidate)
