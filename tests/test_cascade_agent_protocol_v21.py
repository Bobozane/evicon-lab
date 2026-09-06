from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evicon.cascade_agent_prompts import CascadeAgentPromptContext, CascadeAgentRuntimeConfig
from evicon.cascade_agent_protocol_v21 import (
    CascadeAgentProtocolV21Runtime,
    CascadeAgentV21Diagnostic,
    V21_PROTOCOL_VERSION,
    V21_SCHEMA_NAME,
    V21_TEMPLATE_VERSION,
    load_hd21_config,
    parse_cascade_agent_response_v21,
    render_cascade_agent_turn_v21,
)
from evicon.cascade_agent_protocol_v21_compatibility import (
    HD21FakeProvider,
    compatibility_preflight,
    run_fake_compatibility,
)
from evicon.cascade_agent_protocol_v21_pilot import (
    HD21PilotRunner,
    final_preflight,
    run_hd21_fake_smoke,
)
from evicon.cascade_agent_runtime import CascadeAgentRuntimeStatus
from evicon.cascade_outcomes import ClaimStance
from evicon.cascade_real_agent_runner import CascadeRealAgentRunError
from evicon.llm_contract import LLMResponse
from evicon.models import EvidenceCard
from evicon.provenance_cascade import ProvenanceNode, SourceCategory, VerificationStatus
from evicon.provenance_cascade_exposure import (
    ControllerClaimView,
    ControllerPublicView,
    ControllerSourceRootView,
    VisibleRootRelation,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd21.v1.toml"
APPROVAL = ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd21_approval_template.toml"
COMPATIBILITY = ROOT / "outputs/study-locks/provenance_cascade_agent_v21_compatibility_receipt.json"
ABORT_RECEIPT = ROOT / "results/provenance-cascade-pilot-hd2-v1/hd2_abort_receipt.json"


def _context() -> CascadeAgentPromptContext:
    view = ControllerPublicView(
        scenario_id="v21-test",
        agent_id="network-agent-01",
        round_id=1,
        claims=(
            ControllerClaimView(
                claim_id="claim-1",
                verification_status=VerificationStatus.UNVERIFIED,
                evidence_card_ids=("evidence-1",),
            ),
        ),
        evidence_cards=(
            EvidenceCard(
                evidence_id="evidence-1",
                claim="public evidence summary",
                source="public source",
                supports=["claim-1"],
                introduced_round=0,
                visible_to=["network-agent-01"],
                reliability=0.8,
            ),
        ),
        provenance_nodes=(
            ProvenanceNode(
                node_id="node-1",
                content_id="content-1",
                scenario_id="v21-test",
                claim_id="claim-1",
                source_root_id="root-1",
                round_id=0,
            ),
        ),
        source_roots=(
            ControllerSourceRootView(
                source_root_id="root-1",
                public_source_category=SourceCategory.PRIMARY_RECORD,
            ),
        ),
        root_relations=(
            VisibleRootRelation(
                provenance_node_id="node-1",
                source_root_ids=("root-1",),
            ),
        ),
    )
    return CascadeAgentPromptContext.from_public_view(
        view,
        claim_id="claim-1",
        directive=None,
        runtime_config=CascadeAgentRuntimeConfig(
            model_name="v21-fake",
            temperature=0.2,
            max_tokens=512,
            seed=20260911,
        ),
    )


def _payload(**overrides: object) -> str:
    value = {
        "stance": "uncertain",
        "content_ids_used": ["content-1"],
        "evidence_ids_used": ["evidence-1"],
        "share_content_id": "content-1",
    }
    value.update(overrides)
    return json.dumps(value)


def test_v21_contract_is_versioned_strict_and_uses_512_tokens() -> None:
    context = _context()
    request = render_cascade_agent_turn_v21(context)
    parsed = parse_cascade_agent_response_v21(_payload(), context)
    assert parsed.valid and parsed.stance is ClaimStance.UNCERTAIN
    assert request.max_tokens == 512
    assert request.metadata["template_version"] == V21_TEMPLATE_VERSION
    assert V21_PROTOCOL_VERSION == "provenance_cascade_agent_protocol.v2_1"
    assert V21_SCHEMA_NAME == "cascade_agent_response_v2_1"
    assert request.request_id.startswith("v21-")


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ("[]", CascadeAgentV21Diagnostic.TOP_LEVEL_TYPE),
        ("{}", CascadeAgentV21Diagnostic.MISSING_FIELD),
        (_payload(action="abstain"), CascadeAgentV21Diagnostic.EXTRA_FIELD),
        (_payload(content_ids_used="content-1"), CascadeAgentV21Diagnostic.FIELD_TYPE),
        (_payload(stance="invalid"), CascadeAgentV21Diagnostic.INVALID_STANCE),
        (_payload(content_ids_used=["not-visible"]), CascadeAgentV21Diagnostic.UNAVAILABLE_CONTENT_ID),
        (_payload(evidence_ids_used=["not-visible"]), CascadeAgentV21Diagnostic.UNAVAILABLE_EVIDENCE_ID),
        (_payload(share_content_id="not-visible"), CascadeAgentV21Diagnostic.SHARE_FIELD),
        ('{"stance":"uncertain"', CascadeAgentV21Diagnostic.MALFORMED_JSON),
    ],
)
def test_v21_parser_diagnostics_are_stable_and_redacted(body: str, code: CascadeAgentV21Diagnostic) -> None:
    parsed = parse_cascade_agent_response_v21(body, _context())
    assert not parsed.valid
    assert parsed.validation_errors == (code,)
    assert "not-visible" not in parsed.model_dump_json()


def test_v21_config_and_completed_pilot_bind_the_new_scope() -> None:
    config, _ = load_hd21_config(CONFIG)
    assert len(config.runs) == 48
    assert len({item.matched_group_id for item in config.runs}) == 12
    assert config.agent_max_tokens == 512
    assert config.request_cap == 864
    assert config.completion_reservation_cap == 48 * 18 * 512 == 442368
    assert config.output_root == "results/provenance-cascade-pilot-hd21-v1"
    assert all(item.run_id.startswith("hd21-") for item in config.runs)
    abort = json.loads(ABORT_RECEIPT.read_text(encoding="utf-8"))
    assert abort["status"] == "aborted_protocol_truncation"
    assert abort["resume_prohibited"] is True
    assert abort["merge_with_hd21_prohibited"] is True
    completed_root = ROOT / config.output_root
    assert completed_root.is_dir()
    assert (completed_root / "pilot_receipt.json").is_file()
    assert (completed_root / "pilot_batch_record.json").is_file()


def test_v21_registered_compatibility_is_ready_and_fake_regression_passes() -> None:
    report = compatibility_preflight(config_path=CONFIG, receipt_path=COMPATIBILITY)
    assert report["status"] == "ready"
    assert report["blocking_reasons"] == []
    assert report["network"] == "disabled"
    fake = run_fake_compatibility()
    assert fake["status"] == "passed"
    assert fake["valid_parser"] is True
    assert fake["valid_finish_reason"] == "stop"
    assert fake["truncation_parser"] is False
    assert fake["truncation_finish_reason"] == "length"
    assert fake["truncation_completion_tokens"] == 512


def test_v21_completed_output_blocks_a_second_non_resume_preflight() -> None:
    report = final_preflight(CONFIG, APPROVAL)
    assert report["status"] == "blocked"
    assert report["ready_for_network_authorization"] is False
    assert report["blocking_reasons"] == ["output_root_exists"]
    assert report["network"] == "disabled"
    assert report["provider_constructed"] is False
    assert report["api_key_read"] is False
    assert report["results_written"] is False


def test_v21_full_fake_smoke_has_48_replayed_runs_and_864_requests() -> None:
    result = run_hd21_fake_smoke(CONFIG)
    assert result["status"] == "fake_smoke_passed"
    assert result["completed_run_count"] == 48
    assert result["matched_group_count"] == 12
    assert result["logical_request_count"] == 864
    assert result["provider_call_count"] == 864
    assert result["transport_attempt_count"] == 864
    assert result["replay_passed_count"] == 48
    assert set(result["replay_statuses"].values()) == {"passed"}
    assert result["directive_applied_count"] == 9
    assert result["completion_reservation_cap"] == 442368
    assert result["results_written"] is False
    dumped = json.dumps(result, sort_keys=True).lower()
    for forbidden in ("user_prompt", "system_prompt", "api_key", "provider_metadata", "ground_truth_label"):
        assert forbidden not in dumped


def test_v21_completed_fingerprint_is_not_replayed_on_resume(tmp_path: Path) -> None:
    pilot = HD21PilotRunner(CONFIG)
    spec = pilot.config.runs[0]
    first = HD21FakeProvider()
    record = pilot.run_one(spec, provider=first, root=tmp_path, model_name="v21-fake", resume=False)
    second = HD21FakeProvider()
    resumed = pilot.run_one(spec, provider=second, root=tmp_path, model_name="v21-fake", resume=True)
    assert first.calls == 18
    assert second.calls == 0
    assert record.replay is not None and record.replay.status.value == "passed"
    assert resumed.replay is not None and resumed.replay.status.value == "passed"
    ledger = (tmp_path / spec.run_id / "request_ledger.jsonl").read_text(encoding="utf-8").lower()
    assert "user_prompt" not in ledger and "system_prompt" not in ledger
