"""Offline H-G.2.3 adoption-identifiability amendment and FakeProvider gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from .cascade_agent_prompts import CascadeAgentRuntimeConfig
from .cascade_agent_protocol_hg import build_identifiable_prompt_context
from .cascade_agent_protocol_hg2 import EpistemicStance, HG2DecisionTask, HG2PromptContext, PublicDecisionRole
from .cascade_agent_protocol_hg23 import (
    AdoptionDecision, HG23Runtime, HG23_PROTOCOL_VERSION, HG23_SCHEMA_NAME,
    HG23_TEMPLATE_VERSION, SharingDecision,
)
from .cascade_controller import CascadeControllerPolicyLoader
from .cascade_controller_hg1 import propose_hg1, validate_hg1_proposal
from .cascade_hg1_replay import HG1CascadeProtocolRunner, HG1OutcomeReplayValidator
from .cascade_real_agent_runner import CascadeRealAgentRunner
from .llm_contract import LLMRequest, LLMResponse
from .provenance_cascade_hg1 import _claim_selector, _post_round_events
from .provenance_cascade_hg12_design import load_hg12_design
from .provenance_cascade_hg2 import HG2CalibrationConfig, load_hg2_config, sha256_file
from .provenance_cascade_preregistration import CascadeCondition

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg23_calibration.v1.toml"
DEFAULT_APPROVAL = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg23_approval_template.toml"
DEFAULT_RECEIPT = "outputs/study-locks/provenance_cascade_hg23_adoption_identifiability_amendment_receipt.json"
EXPECTED_SCENARIOS = (
    "cascade-hg1-false-majority", "cascade-hg1-true-minority-correction",
    "cascade-hg1-independent-true-consensus", "cascade-hg1-unresolved-disagreement",
)


class HG23Error(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


class HG23Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    study_id: Literal["evicon-provenance-cascade-hg23-adoption-identifiability-calibration"]
    config_version: Literal["provenance_cascade_hg23_observable_adoption.v1"]
    status: Literal["offline_validated_pending_approval_and_compatibility"]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    parent_config_path: str
    parent_config_sha256: str
    parent_analysis_path: str
    parent_analysis_sha256: str
    parent_calibration_receipt_path: str
    parent_calibration_receipt_sha256: str
    base_design_config_path: str
    base_design_config_sha256: str
    schedule_config_path: str
    schedule_config_sha256: str
    protocol_path: str
    protocol_sha256: str
    controller_path: str
    controller_sha256: str
    replay_path: str
    replay_sha256: str
    amendment_path: str
    amendment_sha256: str
    protocol_version: Literal["provenance_cascade_agent_protocol.hg2_3_observable_adoption.v1"]
    template_version: Literal["cascade_agent_turn.hg2_3_observable_adoption.v1"]
    response_format: Literal["json_schema"]
    response_schema_name: Literal["cascade_agent_epistemic_adoption_sharing_v1"]
    outcome_contract_version: Literal["cascade_public_adoption_outcome.v1"]
    metric_eligibility_version: Literal["provenance_cascade_hg23_eligibility.v1"]
    epistemic_confidence_separate_from_adoption: Literal[True]
    sharing_separate_from_adoption: Literal[True]
    adoption_does_not_certify_truth: Literal[True]
    scenario_design_changed: Literal[False]
    exposure_schedule_changed: Literal[False]
    controller_rules_changed: Literal[False]
    evaluator_truth_changed: Literal[False]
    metric_names_changed: Literal[False]
    metric_operationalization_versioned: Literal[True]
    seed: Literal[20261031]
    scenario_count: Literal[4]
    condition_count: Literal[4]
    agent_count: Literal[6]
    round_count: Literal[3]
    run_count: Literal[16]
    matched_group_count: Literal[4]
    logical_requests_per_run: Literal[18]
    request_cap: Literal[288]
    agent_max_tokens: Literal[1024]
    completion_reservation_cap: Literal[294912]
    temperature: Literal[0.2]
    max_retries: Literal[0]
    timeout_seconds: Literal[15]
    run_id_namespace: Literal["hg23"]
    output_root: Literal["results/provenance-cascade-hg23-adoption-identifiability-v1"]
    approval_path: str
    compatibility_receipt_path: str
    full_pilot_authorized: Literal[False]

    @model_validator(mode="after")
    def fixed_scope(self) -> "HG23Config":
        if self.request_cap != self.run_count * self.logical_requests_per_run:
            raise ValueError("request cap mismatch")
        if self.completion_reservation_cap != self.request_cap * self.agent_max_tokens:
            raise ValueError("completion reservation mismatch")
        return self


class HG23Approval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    approval_id: Literal["provenance-cascade-hg23-adoption-identifiability-approval-v1"]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    accepted_on: str
    config_sha256: str
    protocol_sha256: str
    amendment_sha256: str
    amendment_receipt_sha256: str
    compatibility_receipt_sha256: str
    confirm_observable_adoption_contract: bool
    confirm_epistemic_adoption_separation: bool
    confirm_sharing_adoption_separation: bool
    confirm_metric_operationalization_version: bool
    confirm_hg22_immutable_and_excluded: bool
    confirm_development_only: bool
    network_execution_authorized: Literal[False]

    @model_validator(mode="after")
    def acceptance(self) -> "HG23Approval":
        checks = (
            self.confirm_observable_adoption_contract,
            self.confirm_epistemic_adoption_separation,
            self.confirm_sharing_adoption_separation,
            self.confirm_metric_operationalization_version,
            self.confirm_hg22_immutable_and_excluded,
            self.confirm_development_only,
        )
        if self.acceptance_status == "accepted" and (not self.accepted_by or not self.accepted_on or not all(checks)):
            raise ValueError("accepted approval incomplete")
        return self


class HG23AmendmentReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    receipt_id: Literal["provenance-cascade-hg23-adoption-identifiability-amendment-v1"]
    status: Literal["offline_validated_pending_approval_and_compatibility"]
    config_sha256: str
    protocol_sha256: str
    amendment_sha256: str
    parent_config_sha256: str
    parent_analysis_sha256: str
    parent_calibration_receipt_sha256: str
    run_count: Literal[16]
    matched_group_count: Literal[4]
    logical_request_count: Literal[288]
    completion_reservation_cap: Literal[294912]
    replay_passed_count: Literal[16]
    behavior_observation_count: Literal[288]
    false_initial_adoption_eligible_count: int
    correction_transition_eligible_count: int
    harmful_conformity_eligible_count: int
    source_structure_opportunity_count: int
    network: Literal["disabled"]
    provider_constructed: Literal[False]
    results_written: Literal[False]
    private_truth_exposed: Literal[False]
    effectiveness_claimed: Literal[False]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]


def load_hg23_config(path: str | Path = DEFAULT_CONFIG) -> tuple[HG23Config, Path, HG2CalibrationConfig, dict, dict]:
    config_path = _path(path)
    try:
        config = HG23Config.model_validate(tomllib.loads(config_path.read_text(encoding="utf-8")))
    except (OSError, ValidationError, tomllib.TOMLDecodeError, ValueError) as exc:
        raise HG23Error("hg23_config_invalid") from exc
    bindings = (
        (config.parent_config_path, config.parent_config_sha256, "parent_config_hash_mismatch"),
        (config.parent_analysis_path, config.parent_analysis_sha256, "parent_analysis_hash_mismatch"),
        (config.parent_calibration_receipt_path, config.parent_calibration_receipt_sha256, "parent_receipt_hash_mismatch"),
        (config.base_design_config_path, config.base_design_config_sha256, "base_design_hash_mismatch"),
        (config.schedule_config_path, config.schedule_config_sha256, "schedule_hash_mismatch"),
        (config.protocol_path, config.protocol_sha256, "protocol_hash_mismatch"),
        (config.controller_path, config.controller_sha256, "controller_hash_mismatch"),
        (config.replay_path, config.replay_sha256, "replay_hash_mismatch"),
        (config.amendment_path, config.amendment_sha256, "amendment_hash_mismatch"),
    )
    for value, digest, code in bindings:
        if not _path(value).is_file() or sha256_file(_path(value)) != digest:
            raise HG23Error(code)
    parent, scenarios, schedules = load_hg2_config(config.base_design_config_path)
    if tuple(scenarios) != EXPECTED_SCENARIOS:
        raise HG23Error("scenario_set_mismatch")
    return config, config_path, parent, scenarios, schedules


class HG23FakeProvider:
    """Condition-neutral fixture that proves observability, not effectiveness."""
    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        payload = json.loads(request.user_prompt)
        claim_id = payload["target_claim_id"]
        role = PublicDecisionRole(payload["public_role"])
        claims = {item["claim_id"]: item for item in payload["visible_claims"]}
        contents = [item for item in payload["visible_contents"] if item["claim_id"] == claim_id]
        evidence = [item for item in payload["visible_evidence"] if claim_id in item["supports"] or claim_id in item["contradicts"]]
        supports = any(claim_id in item["supports"] for item in evidence)
        contradicts = any(claim_id in item["contradicts"] for item in evidence)
        content_ids = [item["content_id"] for item in contents]
        evidence_ids = [item["evidence_id"] for item in evidence]
        if claims.get(claim_id, {}).get("verification_status") == "contested" or (supports and contradicts):
            epistemic, adoption = EpistemicStance.UNCERTAIN, AdoptionDecision.WITHHOLD_JUDGMENT
        elif contradicts:
            epistemic, adoption = EpistemicStance.UNSUPPORTED, AdoptionDecision.REJECT_CLAIM
        elif supports:
            epistemic, adoption = EpistemicStance.SUPPORTED, AdoptionDecision.ADOPT_CLAIM
        elif role in {PublicDecisionRole.RAPID_RESPONSE, PublicDecisionRole.PROCEDURAL_FAIRNESS, PublicDecisionRole.COMMUNICATION_STABILITY}:
            epistemic, adoption = EpistemicStance.UNCERTAIN, AdoptionDecision.ADOPT_CLAIM
        else:
            epistemic, adoption = EpistemicStance.UNCERTAIN, AdoptionDecision.WITHHOLD_JUDGMENT
        should_share = bool(content_ids) and role in {PublicDecisionRole.RAPID_RESPONSE, PublicDecisionRole.COMMUNICATION_STABILITY} and adoption is not AdoptionDecision.REJECT_CLAIM
        sharing = SharingDecision.SHARE_WITH_CAVEAT if should_share else SharingDecision.DO_NOT_SHARE
        body = json.dumps({
            "epistemic_stance": epistemic.value,
            "adoption_decision": adoption.value,
            "sharing_decision": sharing.value,
            "content_ids_used": content_ids,
            "evidence_ids_used": evidence_ids,
            "share_content_id": content_ids[0] if should_share else None,
        }, sort_keys=True)
        return LLMResponse(
            request_id=request.request_id, model_name=request.model_name, content=body,
            finish_reason="stop", prompt_tokens=100, completion_tokens=45,
            total_tokens=145, latency_ms=1.0,
        )


def run_fake_smoke(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    config, _, parent, scenarios, schedules = load_hg23_config(path)
    policy = CascadeControllerPolicyLoader.load(_path("configs/provenance_cascade/pilot/amendments/hg1_identifiability_v1/cascade_controller_policy.hg1.v1.toml"))
    base = HG1CascadeProtocolRunner()
    tasks = {item.scenario_id: HG2DecisionTask.model_validate(item.model_dump(exclude={"scenario_id"})) for item in parent.decision_tasks}
    observations = []
    operations = defaultdict(lambda: Counter(proposal=0, scheduled=0, applied=0))
    replay_passed = logical_requests = 0
    with tempfile.TemporaryDirectory(prefix="evicon-hg23-fake-") as temporary:
        root = Path(temporary)
        for scenario_id in EXPECTED_SCENARIOS:
            scenario, schedule = scenarios[scenario_id], schedules[scenario_id]
            for condition in CascadeCondition:
                provider, runtime = HG23FakeProvider(), HG23Runtime()
                run_id = f"hg23-{scenario_id}-{config.seed}-{condition.value}"

                def context_builder(view, *, claim_id, directive, runtime_config, _task=tasks[scenario_id]):
                    return HG2PromptContext(
                        public_context=build_identifiable_prompt_context(view, claim_id=claim_id, directive=directive, runtime_config=runtime_config),
                        role=parent.roles[view.agent_id], decision_task=_task,
                    )

                record = CascadeRealAgentRunner(base_runner=base, policy_config=policy).run_scenario(
                    scenario, config.seed, condition, provider=provider, run_id=run_id,
                    ledger_path=root / run_id / "request_ledger.jsonl", checkpoint_path=None,
                    model_name="hg23-fake", agent_temperature=config.temperature,
                    agent_max_tokens=config.agent_max_tokens, request_cap=18,
                    completion_reservation_cap=18432, runtime=runtime,
                    context_builder=context_builder, outcome_replay_validator=HG1OutcomeReplayValidator,
                    initial_events_builder=base.preload_initial_events,
                    claim_selector=lambda s, snap, rnd, schedule=schedule: _claim_selector(schedule, s, snap, rnd),
                    proposal_builder=propose_hg1, proposal_validator=validate_hg1_proposal,
                    post_round_events_builder=lambda s, rnd, events, schedule=schedule: _post_round_events(s, schedule, rnd, events),
                    actor_schedule_provider=lambda _s, rnd, schedule=schedule: schedule.propagation_slots if rnd == 0 else (),
                    policy_config=policy,
                )
                if record.replay is None or record.replay.status.value != "passed":
                    raise HG23Error("fake_replay_failed")
                replay_passed += 1
                logical_requests += record.logical_request_count
                key = f"{scenario_id}|{condition.value}"
                operations[key]["proposal"] += record.proposal_count
                operations[key]["scheduled"] += len(record.application_ledger.schedules)
                operations[key]["applied"] += record.directive_applied_count
                for context, response in runtime.decisions:
                    observations.append((scenario_id, condition, context, response))
    if replay_passed != 16 or logical_requests != 288 or len(observations) != 288:
        raise HG23Error("fake_coordinate_incomplete")
    false_initial_agents = {
        context.public_context.agent_id for sid, condition, context, response in observations
        if sid.endswith("false-majority") and condition is CascadeCondition.NO_INTERVENTION
        and context.public_context.round_id == 0 and response.adoption_decision is AdoptionDecision.ADOPT_CLAIM
    }
    correction_initial_agents = {
        context.public_context.agent_id for sid, condition, context, response in observations
        if sid.endswith("true-minority-correction") and condition is CascadeCondition.NO_INTERVENTION
        and context.public_context.round_id == 0 and response.adoption_decision is AdoptionDecision.ADOPT_CLAIM
    }
    correction_final_agents = {
        context.public_context.agent_id for sid, condition, context, response in observations
        if sid.endswith("true-minority-correction") and condition is CascadeCondition.NO_INTERVENTION
        and context.public_context.round_id == 2 and response.adoption_decision is AdoptionDecision.ADOPT_CLAIM
    }
    harmful_agents = {
        context.public_context.agent_id for sid, condition, context, response in observations
        if sid.endswith("false-majority") and condition is CascadeCondition.NO_INTERVENTION
        and context.public_context.round_id == 1 and context.public_context.agent_id in false_initial_agents
        and len(context.public_context.contents) >= 2 and not context.public_context.evidence
    }
    transition_agents = correction_initial_agents & correction_final_agents
    if not false_initial_agents or not transition_agents or not harmful_agents:
        raise HG23Error("eligibility_chain_missing")
    # Round-zero behavior is condition-neutral; the fixture cannot encode an effect.
    for scenario_id in EXPECTED_SCENARIOS:
        signatures = []
        for condition in CascadeCondition:
            signatures.append(tuple(
                (context.public_context.agent_id, response.epistemic_stance.value, response.adoption_decision.value, response.sharing_decision.value)
                for sid, cond, context, response in observations
                if sid == scenario_id and cond is condition and context.public_context.round_id == 0
            ))
        if len(set(signatures)) != 1:
            raise HG23Error("fake_condition_effect_encoded")
    false_aware = operations["cascade-hg1-false-majority|provenance_aware_controller"]
    consensus_aware = operations["cascade-hg1-independent-true-consensus|provenance_aware_controller"]
    if false_aware["applied"] <= 0 or consensus_aware["applied"] != 0:
        raise HG23Error("controller_opportunity_missing")
    return {
        "status": "fake_smoke_passed",
        "run_count": 16,
        "matched_group_count": 4,
        "logical_request_count": 288,
        "replay_passed_count": replay_passed,
        "behavior_observation_count": len(observations),
        "false_initial_adoption_eligible_count": len(false_initial_agents),
        "correction_transition_eligible_count": len(transition_agents),
        "harmful_conformity_eligible_count": len(harmful_agents),
        "source_structure_opportunity_count": 2,
        "operations": {key: dict(value) for key, value in sorted(operations.items())},
        "completion_reservation_cap": config.completion_reservation_cap,
        "network": "disabled",
        "provider_constructed": False,
        "results_written": False,
        "private_truth_exposed": False,
        "effectiveness_claimed": False,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def write_amendment_receipt(path: str | Path = DEFAULT_RECEIPT) -> tuple[HG23AmendmentReceipt, str]:
    config, config_path, _, _, _ = load_hg23_config()
    smoke = run_fake_smoke()
    receipt = HG23AmendmentReceipt(
        receipt_id="provenance-cascade-hg23-adoption-identifiability-amendment-v1",
        status="offline_validated_pending_approval_and_compatibility",
        config_sha256=sha256_file(config_path), protocol_sha256=config.protocol_sha256,
        amendment_sha256=config.amendment_sha256, parent_config_sha256=config.parent_config_sha256,
        parent_analysis_sha256=config.parent_analysis_sha256,
        parent_calibration_receipt_sha256=config.parent_calibration_receipt_sha256,
        run_count=16, matched_group_count=4, logical_request_count=288,
        completion_reservation_cap=294912, replay_passed_count=16,
        behavior_observation_count=288,
        false_initial_adoption_eligible_count=int(smoke["false_initial_adoption_eligible_count"]),
        correction_transition_eligible_count=int(smoke["correction_transition_eligible_count"]),
        harmful_conformity_eligible_count=int(smoke["harmful_conformity_eligible_count"]),
        source_structure_opportunity_count=2,
        network="disabled", provider_constructed=False, results_written=False,
        private_truth_exposed=False, effectiveness_claimed=False,
        development_only=True, calibration_only=True, not_paper_result=True,
        no_causal_conclusion=True,
    )
    target = _path(path)
    if target.exists():
        existing = HG23AmendmentReceipt.model_validate_json(target.read_text(encoding="utf-8"))
        if existing != receipt:
            raise HG23Error("amendment_receipt_exists_with_different_content")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(receipt.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return receipt, hashlib.sha256(target.read_bytes()).hexdigest()


def final_preflight() -> dict[str, object]:
    reasons: list[str] = []
    try:
        config, config_path, _, _, _ = load_hg23_config()
        approval = HG23Approval.model_validate(tomllib.loads(_path(config.approval_path).read_text(encoding="utf-8")))
        if approval.config_sha256 != sha256_file(config_path) or approval.protocol_sha256 != config.protocol_sha256 or approval.amendment_sha256 != config.amendment_sha256:
            reasons.append("approval_binding_mismatch")
        if approval.acceptance_status != "accepted":
            reasons.append("approval_pending")
        receipt_path = _path(DEFAULT_RECEIPT)
        if not receipt_path.is_file():
            reasons.append("amendment_receipt_missing")
        elif approval.amendment_receipt_sha256 != sha256_file(receipt_path):
            reasons.append("amendment_receipt_binding_mismatch")
        compatibility_path = _path(config.compatibility_receipt_path)
        if not compatibility_path.is_file():
            reasons.append("compatibility_check_required")
        elif approval.compatibility_receipt_sha256 != sha256_file(compatibility_path):
            reasons.append("compatibility_receipt_binding_mismatch")
        if _path(config.output_root).exists():
            reasons.append("output_root_exists")
        return {
            "status": "ready_for_approval_and_compatibility" if set(reasons) <= {"approval_pending", "compatibility_check_required"} else "blocked",
            "ready_for_real_calibration": False,
            "blocking_reasons": sorted(set(reasons)),
            "config_sha256": sha256_file(config_path),
            "protocol_sha256": config.protocol_sha256,
            "amendment_sha256": config.amendment_sha256,
            "run_count": 16, "matched_group_count": 4, "logical_request_cap": 288,
            "completion_reservation_cap": 294912,
            "network": "disabled", "provider_constructed": False, "api_key_read": False,
            "results_written": False, "private_truth_exposed": False,
            "not_paper_result": True, "no_causal_conclusion": True,
        }
    except Exception as exc:
        return {
            "status": "blocked", "ready_for_real_calibration": False,
            "blocking_reasons": [getattr(exc, "code", "hg23_preflight_failed")],
            "network": "disabled", "provider_constructed": False, "api_key_read": False,
            "results_written": False, "private_truth_exposed": False,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline H-G.2.3 adoption-identifiability gate")
    parser.add_argument("--mode", choices=("preflight", "fake-smoke", "write-amendment-receipt"), default="preflight")
    args = parser.parse_args(argv)
    if args.mode == "fake-smoke":
        payload = run_fake_smoke()
    elif args.mode == "write-amendment-receipt":
        receipt, digest = write_amendment_receipt()
        payload = {"status": receipt.status, "receipt_sha256": digest, "network": "disabled", "results_written": False}
    else:
        payload = final_preflight()
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if payload["status"] in {"fake_smoke_passed", "offline_validated_pending_approval_and_compatibility", "ready_for_approval_and_compatibility"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

