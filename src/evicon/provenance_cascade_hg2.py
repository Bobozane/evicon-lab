"""Offline H-G.2 eligibility calibration and preflight.

The fake calibration validates timing and eligibility only. It never estimates
an intervention effect and has no real-provider entry point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .cascade_agent_prompts import CascadeAgentRuntimeConfig
from .cascade_agent_protocol_hg import build_identifiable_prompt_context
from .cascade_agent_protocol_hg2 import (
    BehavioralDecision, EpistemicStance, HG2AgentResponse, HG2DecisionTask,
    HG2PromptContext, HG2Runtime, HG2_PROTOCOL_VERSION, HG2_SCHEMA_NAME,
    HG2_TEMPLATE_VERSION, PublicDecisionRole,
)
from .cascade_controller import CascadeControllerPolicyLoader
from .cascade_controller_hg1 import propose_hg1, validate_hg1_proposal
from .cascade_hg1_replay import HG1CascadeProtocolRunner, HG1OutcomeReplayValidator
from .cascade_real_agent_runner import CascadeRealAgentRunner
from .llm_contract import LLMRequest, LLMResponse
from .provenance_cascade_hg1 import _claim_selector, _post_round_events
from .provenance_cascade_hg12_design import load_hg12_design
from .provenance_cascade_preregistration import CascadeCondition

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = _ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg2_calibration.v1.toml"
EXPECTED_AGENTS = tuple(f"network-agent-{index:02d}" for index in range(1, 7))
EXPECTED_SCENARIOS = (
    "cascade-hg1-false-majority", "cascade-hg1-true-minority-correction",
    "cascade-hg1-independent-true-consensus", "cascade-hg1-unresolved-disagreement",
)


class HG2DesignError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def sha256_file(value: str | Path) -> str:
    return hashlib.sha256(_path(value).read_bytes()).hexdigest()


class HG2TaskBinding(HG2DecisionTask):
    scenario_id: str


class HG2CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    study_id: Literal["evicon-provenance-cascade-hg2-eligibility-calibration"]
    config_version: Literal["provenance_cascade_hg2_epistemic_behavior.v1"]
    status: Literal["offline_design_pending_human_approval"]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    parent_pilot_id: Literal["evicon-provenance-cascade-pilot-hg12"]
    parent_pilot_status: Literal["non_identifiable_primary_outcome"]
    parent_config_path: str
    parent_config_sha256: str
    parent_receipt_path: str
    parent_receipt_sha256: str
    parent_analysis_path: str
    parent_analysis_sha256: str
    base_material_config_path: str
    base_material_config_sha256: str
    schedule_config_path: str
    schedule_config_sha256: str
    amendment_path: str
    amendment_sha256: str
    protocol_path: str
    protocol_sha256: str
    provider_path: str
    provider_sha256: str
    protocol_version: Literal["provenance_cascade_agent_protocol.hg2_epistemic_behavior.v1"]
    template_version: Literal["cascade_agent_turn.hg2_epistemic_behavior.v1"]
    response_schema_name: Literal["cascade_agent_epistemic_behavior_response_v1"]
    response_format: Literal["json_schema"]
    seed: Literal[20261021]
    agent_ids: tuple[str, ...]
    conditions: tuple[CascadeCondition, ...]
    scenario_ids: tuple[str, ...]
    max_rounds: Literal[3]
    topology_id: Literal["ring_6_bidirectional"]
    run_count: Literal[16]
    matched_group_count: Literal[4]
    logical_requests_per_run: Literal[18]
    request_cap: Literal[288]
    agent_max_tokens: Literal[1024]
    completion_reservation_per_run: Literal[18432]
    completion_reservation_cap: Literal[294912]
    temperature: Literal[0.2]
    max_retries: Literal[0]
    timeout_seconds: Literal[15]
    output_root: Literal["results/provenance-cascade-hg2-eligibility-calibration-v1"]
    real_eligibility_calibration_required: Literal[True]
    full_pilot_authorized: Literal[False]
    roles: dict[str, PublicDecisionRole]
    decision_tasks: tuple[HG2TaskBinding, ...]

    @model_validator(mode="after")
    def frozen_design(self) -> "HG2CalibrationConfig":
        if self.agent_ids != EXPECTED_AGENTS or self.scenario_ids != EXPECTED_SCENARIOS:
            raise ValueError("coordinate set mismatch")
        if self.conditions != tuple(CascadeCondition):
            raise ValueError("condition set mismatch")
        if set(self.roles) != set(self.agent_ids) or set(self.roles.values()) != set(PublicDecisionRole):
            raise ValueError("role assignment mismatch")
        if len(self.decision_tasks) != 4 or {item.scenario_id for item in self.decision_tasks} != set(self.scenario_ids):
            raise ValueError("decision task set mismatch")
        if self.request_cap != self.run_count * self.logical_requests_per_run:
            raise ValueError("request cap mismatch")
        if self.completion_reservation_cap != self.request_cap * self.agent_max_tokens:
            raise ValueError("completion reservation mismatch")
        false_task = next(item for item in self.decision_tasks if item.scenario_id.endswith("false-majority"))
        consensus_task = next(item for item in self.decision_tasks if item.scenario_id.endswith("independent-true-consensus"))
        if (
            false_task.task_id, false_task.public_decision_summary,
            false_task.reversible_action, false_task.defer_action,
        ) != (
            consensus_task.task_id, consensus_task.public_decision_summary,
            consensus_task.reversible_action, consensus_task.defer_action,
        ):
            raise ValueError("paired decision surface mismatch")
        return self


def load_hg2_config(path: str | Path = DEFAULT_CONFIG):
    config_path = _path(path)
    try:
        config = HG2CalibrationConfig.model_validate(tomllib.loads(config_path.read_text(encoding="utf-8")))
    except (OSError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise HG2DesignError("hg2_config_invalid") from exc
    for value, expected, code in (
        (config.parent_config_path, config.parent_config_sha256, "parent_config_hash_mismatch"),
        (config.parent_receipt_path, config.parent_receipt_sha256, "parent_receipt_hash_mismatch"),
        (config.parent_analysis_path, config.parent_analysis_sha256, "parent_analysis_hash_mismatch"),
        (config.base_material_config_path, config.base_material_config_sha256, "base_material_hash_mismatch"),
        (config.schedule_config_path, config.schedule_config_sha256, "schedule_config_hash_mismatch"),
        (config.amendment_path, config.amendment_sha256, "amendment_hash_mismatch"),
        (config.protocol_path, config.protocol_sha256, "protocol_hash_mismatch"),
        (config.provider_path, config.provider_sha256, "provider_hash_mismatch"),
    ):
        candidate = _path(value)
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise HG2DesignError(code)
    _, scenarios, truths, schedules = load_hg12_design(_path(config.schedule_config_path))
    if tuple(scenarios) != config.scenario_ids:
        raise HG2DesignError("scenario_order_mismatch")
    # The private fixtures are returned by the legacy material loader but are
    # deliberately dropped before any runtime or controller is constructed.
    del truths
    false_graph = scenarios[config.scenario_ids[0]].graph
    consensus_graph = scenarios[config.scenario_ids[2]].graph
    false_statements = tuple(node.public_statement for node in false_graph.nodes[:2])
    consensus_statements = tuple(node.public_statement for node in consensus_graph.nodes[:2])
    if false_statements != consensus_statements:
        raise HG2DesignError("paired_public_statement_mismatch")
    false_roots = {root for node in false_graph.nodes[:2] for root in false_graph.root_sources_for_node(node.node_id)}
    consensus_roots = {root for node in consensus_graph.nodes[:2] for root in consensus_graph.root_sources_for_node(node.node_id)}
    if len(false_roots) != 1 or len(consensus_roots) != 2:
        raise HG2DesignError("paired_provenance_structure_mismatch")
    return config, scenarios, schedules


class HG2BehaviorObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    scenario_id: str
    condition: CascadeCondition
    seed: int
    agent_id: str
    round_id: int
    claim_id: str | None
    role: PublicDecisionRole
    epistemic_stance: EpistemicStance
    behavioral_decision: BehavioralDecision
    visible_content_count: int = Field(ge=0)
    visible_evidence_count: int = Field(ge=0)
    visible_root_count: int = Field(ge=0)
    directive_present: bool
    shared: bool


class HG2FakeProvider:
    """Deterministic contract fixture; it does not encode condition effects."""

    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        payload = json.loads(request.user_prompt)
        role = PublicDecisionRole(payload["public_role"])
        claims = {item["claim_id"]: item for item in payload["visible_claims"]}
        claim_id = payload["target_claim_id"]
        claim = claims.get(claim_id, {})
        contents = [item for item in payload["visible_contents"] if item["claim_id"] == claim_id]
        evidence = [item for item in payload["visible_evidence"] if claim_id in item["supports"] or claim_id in item["contradicts"]]
        supports = any(claim_id in item["supports"] for item in evidence)
        contradicts = any(claim_id in item["contradicts"] for item in evidence)
        content_ids = [item["content_id"] for item in contents]
        evidence_ids = [item["evidence_id"] for item in evidence]
        if claim.get("verification_status") == "contested" or (supports and contradicts):
            epistemic = EpistemicStance.UNCERTAIN
            behavior = BehavioralDecision.DEFER_ACTION
        elif contradicts:
            epistemic = EpistemicStance.UNSUPPORTED
            behavior = BehavioralDecision.DO_NOT_SHARE
        elif supports:
            epistemic = EpistemicStance.SUPPORTED
            behavior = BehavioralDecision.SHARE_WITH_CAVEAT if content_ids else BehavioralDecision.SUPPORT_REVERSIBLE_ACTION
        elif role in {PublicDecisionRole.RAPID_RESPONSE, PublicDecisionRole.COMMUNICATION_STABILITY} and content_ids:
            epistemic = EpistemicStance.UNCERTAIN
            behavior = BehavioralDecision.SHARE_WITH_CAVEAT
        elif role in {PublicDecisionRole.HARM_AVOIDANCE, PublicDecisionRole.REVERSIBILITY_FOCUSED}:
            epistemic = EpistemicStance.UNCERTAIN
            behavior = BehavioralDecision.SUPPORT_REVERSIBLE_ACTION
        else:
            epistemic = EpistemicStance.UNCERTAIN
            behavior = BehavioralDecision.DEFER_ACTION
        share = content_ids[0] if behavior is BehavioralDecision.SHARE_WITH_CAVEAT and content_ids else None
        body = json.dumps({
            "epistemic_stance": epistemic.value,
            "behavioral_decision": behavior.value,
            "content_ids_used": content_ids,
            "evidence_ids_used": evidence_ids,
            "share_content_id": share,
        }, sort_keys=True)
        return LLMResponse(
            request_id=request.request_id, model_name=request.model_name,
            content=body, finish_reason="stop", prompt_tokens=100,
            completion_tokens=40, total_tokens=140, latency_ms=1.0,
        )


def run_hg2_fake_calibration(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    config, scenarios, schedules = load_hg2_config(path)
    _, _, _, parent_schedules = load_hg12_design(_path(config.schedule_config_path))
    policy = CascadeControllerPolicyLoader.load(_path("configs/provenance_cascade/pilot/amendments/hg1_identifiability_v1/cascade_controller_policy.hg1.v1.toml"))
    base = HG1CascadeProtocolRunner()
    tasks = {item.scenario_id: HG2DecisionTask.model_validate(item.model_dump(exclude={"scenario_id"})) for item in config.decision_tasks}
    observations: list[HG2BehaviorObservation] = []
    operations = defaultdict(lambda: Counter(run_count=0, proposal_count=0, scheduled_count=0, applied_count=0))
    replay_count = 0
    logical_requests = 0
    with tempfile.TemporaryDirectory(prefix="evicon-hg2-fake-calibration-") as temporary:
        root = Path(temporary)
        for scenario_id in config.scenario_ids:
            scenario = scenarios[scenario_id]
            schedule = parent_schedules[scenario_id]
            for condition in config.conditions:
                provider = HG2FakeProvider()
                runtime = HG2Runtime()
                run_id = f"hg2-{scenario_id}-{config.seed}-{condition.value}"

                def context_builder(view, *, claim_id, directive, runtime_config, _task=tasks[scenario_id]):
                    public = build_identifiable_prompt_context(
                        view, claim_id=claim_id, directive=directive,
                        runtime_config=runtime_config,
                    )
                    return HG2PromptContext(
                        public_context=public,
                        role=config.roles[view.agent_id],
                        decision_task=_task,
                    )

                record = CascadeRealAgentRunner(base_runner=base, policy_config=policy).run_scenario(
                    scenario, config.seed, condition, provider=provider, run_id=run_id,
                    ledger_path=root / run_id / "request_ledger.jsonl",
                    checkpoint_path=None, model_name="hg2-fake-calibration",
                    agent_temperature=config.temperature, agent_max_tokens=config.agent_max_tokens,
                    request_cap=config.logical_requests_per_run,
                    completion_reservation_cap=config.completion_reservation_per_run,
                    runtime=runtime, context_builder=context_builder,
                    outcome_replay_validator=HG1OutcomeReplayValidator,
                    initial_events_builder=base.preload_initial_events,
                    claim_selector=lambda s, snap, rnd, schedule=schedule: _claim_selector(schedule, s, snap, rnd),
                    proposal_builder=propose_hg1, proposal_validator=validate_hg1_proposal,
                    post_round_events_builder=lambda s, rnd, events, schedule=schedule: _post_round_events(s, schedule, rnd, events),
                    actor_schedule_provider=lambda _scenario, rnd, schedule=schedule: schedule.propagation_slots if rnd == 0 else (),
                    policy_config=policy,
                )
                if record.replay is None or record.replay.status.value != "passed":
                    raise HG2DesignError("fake_replay_failed")
                replay_count += 1
                logical_requests += record.logical_request_count
                row = operations[f"{scenario_id}|{condition.value}"]
                row["run_count"] += 1
                row["proposal_count"] += record.proposal_count
                row["scheduled_count"] += len(record.application_ledger.schedules) if record.application_ledger else 0
                row["applied_count"] += record.directive_applied_count
                for context, response in runtime.decisions:
                    public = context.public_context
                    observations.append(HG2BehaviorObservation(
                        run_id=run_id, scenario_id=scenario_id, condition=condition,
                        seed=config.seed, agent_id=public.agent_id, round_id=public.round_id,
                        claim_id=public.claim_id, role=context.role,
                        epistemic_stance=response.epistemic_stance,
                        behavioral_decision=response.behavioral_decision,
                        visible_content_count=len(public.contents),
                        visible_evidence_count=len(public.evidence),
                        visible_root_count=len(public.roots),
                        directive_present=public.directive is not None,
                        shared=response.share_content_id is not None,
                    ))
    if replay_count != 16 or logical_requests != 288 or len(observations) != 288:
        raise HG2DesignError("fake_coordinate_incomplete")
    risk_behaviors = {
        BehavioralDecision.SHARE, BehavioralDecision.SHARE_WITH_CAVEAT,
        BehavioralDecision.SUPPORT_REVERSIBLE_ACTION,
    }
    false_initial = [item for item in observations if item.scenario_id.endswith("false-majority") and item.condition is CascadeCondition.NO_INTERVENTION and item.round_id == 0 and item.behavioral_decision in risk_behaviors]
    correction_initial = [item for item in observations if item.scenario_id.endswith("true-minority-correction") and item.condition is CascadeCondition.NO_INTERVENTION and item.round_id == 0 and item.behavioral_decision in risk_behaviors]
    correction_final_agents = {
        item.agent_id for item in observations
        if item.scenario_id.endswith("true-minority-correction")
        and item.condition is CascadeCondition.NO_INTERVENTION and item.round_id == 2
        and item.epistemic_stance is EpistemicStance.SUPPORTED
    }
    transition_eligible = len({item.agent_id for item in correction_initial} & correction_final_agents)
    harmful_eligible = [
        item for item in observations
        if item.scenario_id.endswith("false-majority")
        and item.condition is CascadeCondition.NO_INTERVENTION and item.round_id == 1
        and item.agent_id in {initial.agent_id for initial in false_initial}
        and item.visible_content_count >= 2 and item.visible_evidence_count == 0
    ]
    if not (0 < len(false_initial) < 6):
        raise HG2DesignError("initial_behavior_not_identifiable")
    if transition_eligible <= 0 or not harmful_eligible:
        raise HG2DesignError("behavioral_eligibility_chain_missing")
    false_aware = operations["cascade-hg1-false-majority|provenance_aware_controller"]
    consensus_aware = operations["cascade-hg1-independent-true-consensus|provenance_aware_controller"]
    if false_aware["applied_count"] <= 0 or consensus_aware["applied_count"] != 0:
        raise HG2DesignError("provenance_mechanism_opportunity_missing")
    # The fake behavior is condition-neutral before a directive becomes visible.
    for scenario_id in config.scenario_ids:
        signatures = {}
        for condition in config.conditions:
            signatures[condition.value] = tuple(
                (item.agent_id, item.epistemic_stance.value, item.behavioral_decision.value)
                for item in observations
                if item.scenario_id == scenario_id and item.condition is condition and item.round_id == 0
            )
        if len(set(signatures.values())) != 1:
            raise HG2DesignError("fake_provider_encodes_condition_effect")
    serialized = json.dumps([item.model_dump(mode="json") for item in observations], sort_keys=True).lower()
    for forbidden in ("ground_truth_label", "source_independence_label", "private_truth", "api_key", "provider_metadata", "system_prompt", "user_prompt"):
        if forbidden in serialized:
            raise HG2DesignError("private_or_prompt_data_exposed")
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    return {
        "status": "offline_fake_calibration_passed",
        "run_count": 16,
        "matched_group_count": 4,
        "logical_request_count": 288,
        "replay_passed_count": replay_count,
        "round0_substantive_behavior_count": len(false_initial),
        "correction_transition_eligible_count": transition_eligible,
        "harmful_cascade_eligible_count": len(harmful_eligible),
        "operations": {key: dict(value) for key, value in sorted(operations.items())},
        "behavior_observation_sha256": digest,
        "network": "disabled",
        "provider_kind": "fake_only",
        "private_truth_exposed": False,
        "effectiveness_claimed": False,
        "development_only": True,
        "calibration_only": True,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def hg2_preflight(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    try:
        config, _, _ = load_hg2_config(path)
        smoke = run_hg2_fake_calibration(path)
        from .provenance_cascade_hg2_governance import validate_hg2_governance
        _, _, reasons = validate_hg2_governance(config)
        if _path(config.output_root).exists():
            reasons.append("output_root_exists")
        return {
            "status": "blocked",
            "blocking_reasons": reasons,
            "ready_for_real_eligibility_calibration": False,
            "full_pilot_authorized": False,
            "run_count": config.run_count,
            "matched_group_count": config.matched_group_count,
            "logical_request_cap": config.request_cap,
            "completion_reservation_cap": config.completion_reservation_cap,
            "protocol_version": HG2_PROTOCOL_VERSION,
            "template_version": HG2_TEMPLATE_VERSION,
            "schema_name": HG2_SCHEMA_NAME,
            "fake_replay_passed_count": smoke["replay_passed_count"],
            "round0_substantive_behavior_count": smoke["round0_substantive_behavior_count"],
            "correction_transition_eligible_count": smoke["correction_transition_eligible_count"],
            "harmful_cascade_eligible_count": smoke["harmful_cascade_eligible_count"],
            "network": "disabled", "provider_constructed": False,
            "api_key_read": False, "results_written": False,
            "development_only": True, "calibration_only": True,
            "not_paper_result": True, "no_causal_conclusion": True,
        }
    except Exception as exc:
        code = exc.code if isinstance(exc, HG2DesignError) else "hg2_preflight_failed"
        return {
            "status": "blocked", "blocking_reasons": [code],
            "ready_for_real_eligibility_calibration": False,
            "full_pilot_authorized": False, "network": "disabled",
            "provider_constructed": False, "api_key_read": False,
            "results_written": False, "development_only": True,
            "calibration_only": True, "not_paper_result": True,
            "no_causal_conclusion": True,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline H-G.2 epistemic/behavior eligibility calibration")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--fake-smoke", action="store_true")
    args = parser.parse_args(argv)
    result = run_hg2_fake_calibration(args.config) if args.fake_smoke else hg2_preflight(args.config)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_CONFIG", "HG2BehaviorObservation", "HG2CalibrationConfig",
    "HG2DesignError", "HG2FakeProvider", "HG2TaskBinding", "hg2_preflight",
    "load_hg2_config", "run_hg2_fake_calibration", "sha256_file",
]


if __name__ == "__main__":
    raise SystemExit(main())
