"""H-G.1 identifiability amendment, offline smoke, and final preflight."""
from __future__ import annotations

import hashlib
import json
import tempfile
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .cascade_agent_protocol_hg1 import (
    HG1_PROTOCOL_VERSION, HG1_SCHEMA_NAME, HG1_TEMPLATE_VERSION,
    CascadeAgentProtocolHG1Runtime, build_identifiable_prompt_context,
)
from .cascade_controller import CascadeControllerPolicyConfig, CascadeControllerPolicyLoader
from .cascade_controller_hg1 import propose_hg1, validate_hg1_proposal
from .cascade_hg1_replay import HG1CascadeProtocolRunner, HG1OutcomeReplayValidator
from .cascade_outcomes import ClaimStance
from .cascade_protocol import ActorAction, CascadeScenarioLoader, CascadeScenarioSpec
from .cascade_real_agent_runner import CascadeRealAgentRunRecord, CascadeRealAgentRunner
from .llm_contract import LLMResponse
from .provenance_cascade import (
    EvaluatorTruthFixture, EvaluatorTruthLoader, GroundTruthLabel,
    SourceIndependenceLabel, validate_fixture_pair,
)
from .provenance_cascade_exposure import (
    ExposureChannel, ExposureEvent, VisibilityReason,
)
from .provenance_cascade_preregistration import CascadeCondition, CascadeScenario

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = _ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg1.v1.toml"
DEFAULT_RECEIPT = _ROOT / "outputs/study-locks/provenance_cascade_hg1_identifiability_amendment_receipt.v5.json"
_EXPECTED_CONDITIONS = tuple(CascadeCondition)
_EXPECTED_SEEDS = (20261001, 20261002, 20261003)
_EXPECTED_AGENTS = tuple(f"network-agent-{i:02d}" for i in range(1, 7))


class HG1Error(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HG1PostRoundExposure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    after_round_id: Literal[0, 1]
    provenance_node_id: str
    target_agent_ids: tuple[str, ...]
    evidence_card_ids: tuple[str, ...] = ()


class HG1PropagationSlot(BaseModel):
    """A pre-registered social propagation opportunity after round 0."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    round_id: Literal[0]
    actor_id: str
    target_agent_ids: tuple[str, ...]
    action: Literal[ActorAction.REPOST]
    provenance_node_id: str
    parent_content_id: None = None
    evidence_card_ids: tuple[str, ...] = ()


class HG1Schedule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_id: str
    schedule_version: Literal["hg1_identifiability_schedule.v1"]
    post_round_exposures: tuple[HG1PostRoundExposure, ...]
    propagation_slots: tuple[HG1PropagationSlot, ...]
    outcome_targets: dict[int, dict[str, str]]
    development_only: Literal[True]
    not_paper_result: Literal[True]

    @model_validator(mode="after")
    def fixed_shape(self) -> "HG1Schedule":
        if set(self.outcome_targets) != {0, 1, 2}:
            raise ValueError("outcome target rounds incomplete")
        if any(set(row) != set(_EXPECTED_AGENTS) for row in self.outcome_targets.values()):
            raise ValueError("outcome target agents incomplete")
        if len(self.propagation_slots) != 1:
            raise ValueError("H-G.1 requires one matched round-zero social propagation slot")
        return self


class HG1MaterialBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_id: str
    scenario_type: CascadeScenario
    scenario_path: str
    scenario_sha256: str
    graph_path: str
    graph_sha256: str
    truth_path: str
    truth_sha256: str
    schedule_path: str
    schedule_sha256: str


class HG1RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    matched_group_id: str
    scenario_id: str
    condition: CascadeCondition
    seed: int
    expected_provider_requests: Literal[18] = 18
    completion_reservation: Literal[9216] = 9216


class HG1PilotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    study_id: Literal["evicon-provenance-cascade-pilot-hg1"]
    config_version: Literal["provenance_cascade_hg1_identifiability.v1"]
    protocol_version: Literal["provenance_cascade_agent_protocol.hg1_identifiable.v1"]
    template_version: Literal["cascade_agent_turn.hg1_public_content.v1"]
    response_schema_name: Literal["cascade_agent_response_v2_1"]
    status: Literal["offline_preflight"]
    development_only: Literal[True]
    pilot_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    parent_pilot_status: Literal["nonseparating_identifiability_failure"]
    parent_config_path: str
    parent_config_sha256: str
    parent_receipt_path: str
    parent_receipt_sha256: str
    parent_batch_path: str
    parent_batch_sha256: str
    parent_analysis_path: str
    parent_analysis_sha256: str
    preregistration_path: str
    preregistration_sha256: str
    amendment_path: str
    amendment_sha256: str
    protocol_path: str
    protocol_sha256: str
    controller_path: str
    controller_sha256: str
    replay_path: str
    replay_sha256: str
    runner_adapter_path: str
    runner_adapter_sha256: str
    policy_path: str
    policy_sha256: str
    approval_path: str
    compatibility_receipt_path: str
    conditions: tuple[CascadeCondition, ...]
    seeds: tuple[int, ...]
    agent_ids: tuple[str, ...]
    max_rounds: Literal[3]
    topology_id: Literal["ring_6_bidirectional"]
    run_count: Literal[48]
    matched_group_count: Literal[12]
    logical_requests_per_run: Literal[18]
    request_cap: Literal[864]
    agent_max_tokens: Literal[512]
    completion_reservation_per_run: Literal[9216]
    completion_reservation_cap: Literal[442368]
    response_format: Literal["json_schema"]
    max_retries: Literal[1]
    timeout_seconds: Literal[15]
    output_root: Literal["results/provenance-cascade-pilot-hg1-v1"]
    scenario_materials: tuple[HG1MaterialBinding, ...]

    @model_validator(mode="after")
    def fixed_design(self) -> "HG1PilotConfig":
        if self.protocol_version != HG1_PROTOCOL_VERSION or self.template_version != HG1_TEMPLATE_VERSION or self.response_schema_name != HG1_SCHEMA_NAME:
            raise ValueError("H-G.1 protocol binding mismatch")
        if self.conditions != _EXPECTED_CONDITIONS or self.seeds != _EXPECTED_SEEDS or self.agent_ids != _EXPECTED_AGENTS:
            raise ValueError("H-G.1 matched design mismatch")
        if len(self.scenario_materials) != 4 or {m.scenario_type for m in self.scenario_materials} != set(CascadeScenario):
            raise ValueError("H-G.1 scenario set incomplete")
        if self.request_cap != self.run_count * self.logical_requests_per_run:
            raise ValueError("H-G.1 request cap mismatch")
        if self.completion_reservation_cap != self.request_cap * self.agent_max_tokens:
            raise ValueError("H-G.1 completion reservation mismatch")
        return self

    @property
    def runs(self) -> tuple[HG1RunSpec, ...]:
        return tuple(
            HG1RunSpec(
                run_id=f"hg1-{material.scenario_id}-{seed}-{condition.value}",
                matched_group_id=f"hg1-{material.scenario_id}-{seed}",
                scenario_id=material.scenario_id, condition=condition, seed=seed,
            )
            for material in self.scenario_materials
            for seed in self.seeds
            for condition in self.conditions
        )


class HG1Approval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    approval_id: str
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str = ""
    accepted_on: str = ""
    config_sha256: str
    protocol_sha256: str
    controller_sha256: str
    replay_sha256: str
    amendment_sha256: str
    amendment_receipt_sha256: str
    compatibility_receipt_sha256: str = ""
    response_format: Literal["json_schema"]
    response_schema_name: Literal["cascade_agent_response_v2_1"]
    agent_max_tokens: Literal[512]
    run_count: Literal[48]
    logical_request_cap: Literal[864]
    completion_reservation_cap: Literal[442368]
    max_retries: Literal[1]
    timeout_seconds: Literal[15]
    confirm_round_zero_preload: bool = False
    confirm_public_view_only_controller: bool = False
    confirm_paired_root_design: bool = False
    confirm_eligibility_contract: bool = False
    confirm_parent_results_excluded: bool = False
    confirm_no_overwrite: bool = False
    confirm_append_only_ledger: bool = False
    confirm_resume_rules: bool = False
    network_execution_authorized: Literal[False] = False

    @property
    def ready(self) -> bool:
        return self.acceptance_status == "accepted" and bool(self.accepted_by and self.accepted_on) and all((
            self.confirm_round_zero_preload, self.confirm_public_view_only_controller,
            self.confirm_paired_root_design, self.confirm_eligibility_contract,
            self.confirm_parent_results_excluded, self.confirm_no_overwrite,
            self.confirm_append_only_ledger, self.confirm_resume_rules,
        ))


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (_ROOT / path).resolve()


def _load_schedule(path: Path) -> HG1Schedule:
    try:
        return HG1Schedule.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HG1Error("hg1_schedule_invalid") from exc


def load_hg1_config(path: str | Path = DEFAULT_CONFIG):
    config_path = _rooted(path)
    try:
        config = HG1PilotConfig.model_validate(tomllib.loads(config_path.read_text(encoding="utf-8")))
    except (OSError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise HG1Error("hg1_config_invalid") from exc
    bindings = (
        (config.parent_config_path, config.parent_config_sha256, "parent_config_hash_mismatch"),
        (config.parent_receipt_path, config.parent_receipt_sha256, "parent_receipt_hash_mismatch"),
        (config.parent_batch_path, config.parent_batch_sha256, "parent_batch_hash_mismatch"),
        (config.parent_analysis_path, config.parent_analysis_sha256, "parent_analysis_hash_mismatch"),
        (config.preregistration_path, config.preregistration_sha256, "preregistration_hash_mismatch"),
        (config.amendment_path, config.amendment_sha256, "amendment_hash_mismatch"),
        (config.protocol_path, config.protocol_sha256, "protocol_hash_mismatch"),
        (config.controller_path, config.controller_sha256, "controller_hash_mismatch"),
        (config.replay_path, config.replay_sha256, "replay_hash_mismatch"),
        (config.runner_adapter_path, config.runner_adapter_sha256, "runner_adapter_hash_mismatch"),
        (config.policy_path, config.policy_sha256, "policy_hash_mismatch"),
    )
    for value, expected, code in bindings:
        candidate = _rooted(value)
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise HG1Error(code)
    scenarios = {}; truths = {}; schedules = {}
    for material in config.scenario_materials:
        paths = (
            (material.scenario_path, material.scenario_sha256, "scenario_hash_mismatch"),
            (material.graph_path, material.graph_sha256, "graph_hash_mismatch"),
            (material.truth_path, material.truth_sha256, "truth_hash_mismatch"),
            (material.schedule_path, material.schedule_sha256, "schedule_hash_mismatch"),
        )
        for value, expected, code in paths:
            candidate = _rooted(value)
            if not candidate.is_file() or sha256_file(candidate) != expected:
                raise HG1Error(code)
        scenario = CascadeScenarioLoader.load(_rooted(material.scenario_path))
        truth = EvaluatorTruthLoader.load(_rooted(material.truth_path))
        schedule = _load_schedule(_rooted(material.schedule_path))
        validate_fixture_pair(scenario.graph, truth)
        if scenario.scenario_id != material.scenario_id or schedule.scenario_id != material.scenario_id or truth.scenario_id != material.scenario_id:
            raise HG1Error("material_coordinate_mismatch")
        if scenario.scenario_type is not material.scenario_type:
            raise HG1Error("scenario_type_mismatch")
        known_nodes = {node.node_id for node in scenario.graph.nodes}
        preloaded_by_agent = {
            agent_id: {
                item.provenance_node_id
                for item in scenario.initial_public_content
                if item.round_id == 0 and agent_id in item.target_agent_ids
            }
            for agent_id in scenario.agent_ids
        }
        neighbors = {
            agent_id: {
                right if left == agent_id else left
                for left, right in scenario.network_topology.edges
                if agent_id in {left, right}
            }
            for agent_id in scenario.agent_ids
        }
        for slot in schedule.propagation_slots:
            if slot.actor_id not in scenario.agent_ids or not set(slot.target_agent_ids).issubset(scenario.agent_ids):
                raise HG1Error("propagation_agent_unknown")
            if slot.provenance_node_id not in known_nodes or slot.provenance_node_id not in preloaded_by_agent[slot.actor_id]:
                raise HG1Error("propagation_content_not_preloaded")
            if not set(slot.target_agent_ids).issubset(neighbors[slot.actor_id]):
                raise HG1Error("propagation_target_outside_topology")
        public_text = _rooted(material.graph_path).read_text(encoding="utf-8").lower()
        if any(marker in public_text for marker in ("ground_truth_label", "source_independence_label", "evaluator_truth", "private_fixture")):
            raise HG1Error("private_truth_in_public_material")
        scenarios[material.scenario_id] = scenario
        truths[material.scenario_id] = truth
        schedules[material.scenario_id] = schedule
    if len(config.runs) != 48 or len({item.run_id for item in config.runs}) != 48 or len({item.matched_group_id for item in config.runs}) != 12:
        raise HG1Error("run_plan_incomplete")
    return config, scenarios, truths, schedules


def _post_round_events(scenario: CascadeScenarioSpec, schedule: HG1Schedule, round_id: int, existing) -> tuple[ExposureEvent, ...]:
    nodes = {node.node_id: node for node in scenario.graph.nodes}
    events = []
    existing_ids = {item.event_id for item in existing}
    for index, item in enumerate(schedule.post_round_exposures):
        if item.after_round_id != round_id:
            continue
        node = nodes[item.provenance_node_id]
        for target_index, target in enumerate(item.target_agent_ids):
            event_id = f"{scenario.scenario_id}-hg1-post-r{round_id}-i{index}-t{target_index}"
            if event_id in existing_ids:
                raise HG1Error("post_round_event_duplicate")
            events.append(ExposureEvent(
                event_id=event_id, target_agent_id=target, source_actor_id="system",
                round_id=round_id, content_id=node.content_id, claim_id=node.claim_id,
                provenance_node_id=node.node_id, channel=ExposureChannel.RECOMMENDATION,
                visibility_reason=VisibilityReason.RECOMMENDATION_DELIVERY,
                evidence_card_ids=item.evidence_card_ids,
            ))
    return tuple(events)


def _claim_selector(schedule: HG1Schedule, scenario: CascadeScenarioSpec, snapshot, round_id: int) -> str | None:
    claim_id = schedule.outcome_targets[round_id][snapshot.agent_id]
    return claim_id if claim_id in snapshot.visible_claim_ids else None


class HG1FakeProvider:
    """Uniform public-input rule for flow and eligibility validation only."""
    def __init__(self) -> None:
        self.calls = 0
        self.directive_seen = 0
        self.share_count = 0

    def complete(self, request) -> LLMResponse:
        self.calls += 1
        payload = json.loads(request.user_prompt)
        claims = payload.get("visible_claims", [])
        contents = payload.get("visible_contents", [])
        evidence = payload.get("visible_evidence", [])
        target_id = payload.get("target_claim_id")
        target = next((item for item in claims if item.get("claim_id") == target_id), None)
        if payload.get("directive") is not None:
            self.directive_seen += 1
        target_contents = [item["content_id"] for item in contents if item.get("claim_id") == target_id]
        contradicting = [item["evidence_id"] for item in evidence if target_id in item.get("contradicts", [])]
        supporting = [item["evidence_id"] for item in evidence if target_id in item.get("supports", [])]
        status = (target or {}).get("verification_status")
        if target is None:
            stance = "no_position"
        elif status == "contested":
            stance = "uncertain"
        elif contradicting:
            stance = "rejects"
        elif supporting or status == "supported":
            stance = "endorses"
        else:
            stance = "endorses"
        share = target_contents[-1] if target_contents else None
        if share is not None:
            self.share_count += 1
        body = json.dumps({
            "stance": stance,
            "content_ids_used": target_contents,
            "evidence_ids_used": (contradicting or supporting),
            "share_content_id": share,
        }, sort_keys=True, separators=(",", ":"))
        return LLMResponse(
            request_id=request.request_id, model_name=request.model_name, content=body,
            finish_reason="stop", prompt_tokens=30, completion_tokens=20,
            total_tokens=50, latency_ms=1.0,
        )


def _outcomes(record, agent_id, claim_id):
    return [item for item in record.outcome_ledger.outcomes if item.agent_id == agent_id and item.claim_id == claim_id]


def _eligibility(record: CascadeRealAgentRunRecord, scenario: CascadeScenarioSpec, truth: EvaluatorTruthFixture, share_count: int) -> dict[str, int]:
    false_claims = {item.claim_id for item in truth.records if item.ground_truth_label is GroundTruthLabel.FALSE}
    corrections = {item.claim_id for item in truth.records if item.ground_truth_label is GroundTruthLabel.TRUE and item.source_independence_label is SourceIndependenceLabel.INDEPENDENT}
    cards = {card.evidence_id: card for card in scenario.graph.evidence_cards}
    exposures = defaultdict(lambda: {"content": set(), "evidence": set()})
    for event in record.exposure_ledger.events:
        key = (event.target_agent_id, event.claim_id)
        exposures[key]["content"].add(event.content_id)
        exposures[key]["evidence"].update(event.evidence_card_ids)
    counts = Counter({"share_decision_eligible": share_count})
    for outcome in record.outcome_ledger.outcomes:
        if outcome.round_id == 0 and outcome.stance is not ClaimStance.NO_POSITION:
            counts["round0_substantive_stance"] += 1
    for agent in record.agent_ids:
        for false_claim in false_claims:
            history = _outcomes(record, agent, false_claim)
            initial_endorse = bool(history and history[0].stance is ClaimStance.ENDORSES)
            if initial_endorse:
                counts["initial_false_endorsement_eligible"] += 1
            correction_evidence = {
                eid for (a, _claim), value in exposures.items() if a == agent
                for eid in value["evidence"] if false_claim in cards[eid].contradicts
            }
            correction_content_seen = any(
                a == agent and claim in corrections and value["content"]
                for (a, claim), value in exposures.items()
            )
            if initial_endorse and (correction_evidence or correction_content_seen):
                counts["correction_transition_eligible"] += 1
                final_false = history[-1].stance if history else ClaimStance.NO_POSITION
                correction_endorsed = any(
                    (_outcomes(record, agent, claim) and _outcomes(record, agent, claim)[-1].stance is ClaimStance.ENDORSES)
                    for claim in corrections
                )
                if final_false is ClaimStance.REJECTS or correction_endorsed:
                    counts["correction_transition_observed"] += 1
            visible_false = exposures[(agent, false_claim)]["content"]
            if initial_endorse and len(visible_false) >= 2 and not correction_evidence:
                counts["harmful_cascade_eligible"] += 1
                if history and history[-1].stance is ClaimStance.ENDORSES:
                    counts["harmful_cascade_endorsement"] += 1
        for correction in corrections:
            history = _outcomes(record, agent, correction)
            related_evidence = {
                eid for eid in exposures[(agent, correction)]["evidence"] if correction in cards[eid].supports
            }
            if history and related_evidence:
                counts["supported_correction_eligible"] += 1
                if history[-1].stance is ClaimStance.ENDORSES:
                    counts["supported_correction_retained"] += 1
    return dict(counts)


def run_hg1_fake_smoke(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    config, scenarios, truths, schedules = load_hg1_config(path)
    policy = CascadeControllerPolicyLoader.load(_rooted(config.policy_path))
    base = HG1CascadeProtocolRunner()
    records = []
    eligibility = Counter()
    scenario_eligibility = defaultdict(Counter)
    operations = defaultdict(lambda: {"run_count": 0, "proposal_count": 0, "scheduled_count": 0, "applied_count": 0, "directive_prompt_count": 0})
    scenario_operations = defaultdict(lambda: {"run_count": 0, "proposal_count": 0, "scheduled_count": 0, "applied_count": 0})
    root_opportunities = {}
    with tempfile.TemporaryDirectory(prefix="evicon-hg1-identifiability-") as temporary:
        temp = Path(temporary)
        for spec in config.runs:
            scenario = scenarios[spec.scenario_id]
            schedule = schedules[spec.scenario_id]
            provider = HG1FakeProvider()
            record = CascadeRealAgentRunner(base_runner=base, policy_config=policy).run_scenario(
                scenario, spec.seed, spec.condition, provider=provider, run_id=spec.run_id,
                ledger_path=temp/spec.run_id/"request_ledger.jsonl",
                checkpoint_path=temp/spec.run_id/"agent_checkpoint.json",
                model_name="hg1-identifiability-fake", agent_temperature=0.2,
                agent_max_tokens=config.agent_max_tokens,
                request_cap=spec.expected_provider_requests,
                completion_reservation_cap=spec.completion_reservation,
                runtime=CascadeAgentProtocolHG1Runtime(),
                context_builder=build_identifiable_prompt_context,
                outcome_replay_validator=HG1OutcomeReplayValidator,
                initial_events_builder=base.preload_initial_events,
                claim_selector=lambda s, snap, rnd, schedule=schedule: _claim_selector(schedule, s, snap, rnd),
                proposal_builder=propose_hg1, proposal_validator=validate_hg1_proposal,
                post_round_events_builder=lambda s, rnd, events, schedule=schedule: _post_round_events(s, schedule, rnd, events),
                actor_schedule_provider=lambda s, rnd, schedule=schedule: (schedule.propagation_slots if rnd == 0 else ()),
                policy_config=policy,
            )
            if record.replay is None or record.replay.status.value != "passed":
                raise HG1Error("fake_replay_failed")
            records.append(record)
            eligibility_row = _eligibility(record, scenario, truths[spec.scenario_id], provider.share_count)
            eligibility.update(eligibility_row)
            scenario_eligibility[spec.scenario_id].update(eligibility_row)
            schedules_count = sum(item.status.value != "rejected" for item in record.application_ledger.schedules)
            row = operations[spec.condition.value]
            srow = scenario_operations[f"{spec.scenario_id}|{spec.condition.value}"]
            for target in (row, srow):
                target["run_count"] += 1
                target["proposal_count"] += record.proposal_count
                target["scheduled_count"] += schedules_count
                target["applied_count"] += record.directive_applied_count
            row["directive_prompt_count"] += provider.directive_seen
            if spec.condition is CascadeCondition.PROVENANCE_AWARE_CONTROLLER and spec.seed == config.seeds[0]:
                snap = next(item for item in record.exposure_ledger.snapshots if item.agent_id == "network-agent-02" and item.round_id == 1)
                root_opportunities[spec.scenario_id] = {
                    "visible_content_count": len(snap.visible_content_ids),
                    "visible_root_count": len({rid for rel in snap.visible_root_relations for rid in rel.source_root_ids}),
                }
    if len(records) != 48 or sum(item.logical_request_count for item in records) != 864:
        raise HG1Error("fake_run_plan_incomplete")
    if eligibility["round0_substantive_stance"] <= 0:
        raise HG1Error("round0_substantive_stance_missing")
    if eligibility["correction_transition_eligible"] <= 0:
        raise HG1Error("correction_transition_not_identifiable")
    if eligibility["harmful_cascade_eligible"] <= 0:
        raise HG1Error("harmful_cascade_not_identifiable")
    if eligibility["share_decision_eligible"] <= 0:
        raise HG1Error("share_decision_not_identifiable")
    false_roots = root_opportunities["cascade-hg1-false-majority"]
    consensus_roots = root_opportunities["cascade-hg1-independent-true-consensus"]
    if false_roots != {"visible_content_count": 2, "visible_root_count": 1} or consensus_roots != {"visible_content_count": 2, "visible_root_count": 2}:
        raise HG1Error("paired_root_opportunity_invalid")
    if scenario_operations["cascade-hg1-independent-true-consensus|provenance_aware_controller"]["applied_count"] != 0:
        raise HG1Error("independent_consensus_protection_failed")
    if scenario_operations["cascade-hg1-unresolved-disagreement|provenance_aware_controller"]["applied_count"] != 0:
        raise HG1Error("unresolved_abstain_failed")
    return {
        "status": "fake_smoke_passed",
        "run_count": len(records),
        "matched_group_count": len({(item.scenario_id, item.seed) for item in records}),
        "logical_request_count": sum(item.logical_request_count for item in records),
        "provider_call_count": sum(item.provider_call_count for item in records),
        "replay_passed_count": sum(item.replay.status.value == "passed" for item in records),
        "cascade_replay_passed_count": sum(item.replay.cascade_replay.status.value == "passed" for item in records),
        "application_replay_passed_count": sum(item.replay.application_replay.status.value == "passed" for item in records),
        "outcome_replay_passed_count": sum(item.replay.status.value == "passed" for item in records),
        "eligibility": dict(sorted(eligibility.items())),
        "scenario_eligibility": {key: dict(sorted(value.items())) for key, value in sorted(scenario_eligibility.items())},
        "condition_operations": dict(sorted(operations.items())),
        "scenario_condition_operations": dict(sorted(scenario_operations.items())),
        "paired_root_opportunities": root_opportunities,
        "request_cap": config.request_cap,
        "completion_reservation_cap": config.completion_reservation_cap,
        "network": "disabled", "results_written": False, "private_truth_exposed": False,
        "development_only": True, "pilot_only": True, "not_paper_result": True,
        "no_causal_conclusion": True, "effectiveness_claimed": False,
    }


def write_amendment_receipt(summary: dict[str, object], path: str | Path = DEFAULT_RECEIPT, config_path: str | Path = DEFAULT_CONFIG) -> Path:
    config, _, _, _ = load_hg1_config(config_path)
    destination = _rooted(path)
    payload = {
        "receipt_version": "provenance_cascade_hg1_identifiability_receipt.v5",
        "status": "offline_validated_pending_approval_and_compatibility",
        "config_sha256": sha256_file(_rooted(config_path)),
        "protocol_sha256": config.protocol_sha256,
        "controller_sha256": config.controller_sha256,
        "replay_sha256": config.replay_sha256,
        "runner_adapter_sha256": config.runner_adapter_sha256,
        "amendment_sha256": config.amendment_sha256,
        "parent_receipt_sha256": config.parent_receipt_sha256,
        "parent_analysis_sha256": config.parent_analysis_sha256,
        "run_count": summary["run_count"], "matched_group_count": summary["matched_group_count"],
        "logical_request_count": summary["logical_request_count"],
        "request_cap": summary["request_cap"],
        "completion_reservation_cap": summary["completion_reservation_cap"],
        "eligibility": summary["eligibility"],
        "scenario_eligibility": summary["scenario_eligibility"],
        "condition_operations": summary["condition_operations"],
        "scenario_condition_operations": summary["scenario_condition_operations"],
        "material_sha256": {
            item.scenario_id: {
                "scenario": item.scenario_sha256, "graph": item.graph_sha256,
                "schedule": item.schedule_sha256, "truth": item.truth_sha256,
            } for item in config.scenario_materials
        },
        "timing_contract": {
            "round_0": "one_pre_session_public_message_then_substantive_stance",
            "round_1": "social_repost_visible_then_proposal_after_outcome",
            "round_2": "correction_or_evidence_visible_and_prior_directive_effective",
        },
        "paired_root_opportunities": summary["paired_root_opportunities"],
        "replay_passed_count": summary["replay_passed_count"],
        "parent_results_excluded": True, "calibration_results_excluded": True,
        "network": "disabled", "private_truth_exposed": False,
        "development_only": True, "pilot_only": True, "not_paper_result": True,
        "no_causal_conclusion": True,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != encoded:
        raise HG1Error("amendment_receipt_no_overwrite")
    destination.write_text(encoded, encoding="utf-8")
    return destination


def hg1_preflight(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    reasons = []
    try:
        config, scenarios, _, _ = load_hg1_config(path)
    except HG1Error as exc:
        return {"status":"blocked","ready_for_real_pilot":False,"blocking_reasons":[exc.code],"network":"disabled","provider_constructed":False,"api_key_read":False,"results_written":False}
    config_path = _rooted(path)
    receipt_sha = None
    if not DEFAULT_RECEIPT.is_file():
        reasons.append("amendment_receipt_missing")
    else:
        try:
            receipt = json.loads(DEFAULT_RECEIPT.read_text(encoding="utf-8"))
            receipt_sha = sha256_file(DEFAULT_RECEIPT)
            expected_materials = {
                item.scenario_id: {
                    "scenario": item.scenario_sha256, "graph": item.graph_sha256,
                    "schedule": item.schedule_sha256, "truth": item.truth_sha256,
                } for item in config.scenario_materials
            }
            if receipt.get("receipt_version") != "provenance_cascade_hg1_identifiability_receipt.v5" or receipt.get("config_sha256") != sha256_file(config_path) or receipt.get("material_sha256") != expected_materials or receipt.get("replay_passed_count") != 48 or receipt.get("eligibility",{}).get("round0_substantive_stance",0) <= 0 or receipt.get("eligibility",{}).get("correction_transition_eligible",0) <= 0 or receipt.get("eligibility",{}).get("harmful_cascade_eligible",0) <= 0:
                reasons.append("amendment_receipt_invalid")
        except Exception:
            reasons.append("amendment_receipt_invalid")
    approval_path = _rooted(config.approval_path)
    approval = None
    if not approval_path.is_file():
        reasons.append("approval_missing")
    else:
        try:
            approval = HG1Approval.model_validate(tomllib.loads(approval_path.read_text(encoding="utf-8")))
            if not approval.ready:
                reasons.append("approval_pending")
            expected = (sha256_file(config_path), config.protocol_sha256, config.controller_sha256, config.replay_sha256, config.amendment_sha256, receipt_sha)
            actual = (approval.config_sha256, approval.protocol_sha256, approval.controller_sha256, approval.replay_sha256, approval.amendment_sha256, approval.amendment_receipt_sha256)
            if expected != actual:
                reasons.append("approval_hash_mismatch")
        except Exception:
            reasons.append("approval_invalid")
    compatibility = _rooted(config.compatibility_receipt_path)
    if not compatibility.is_file():
        reasons.append("compatibility_check_required")
    elif approval is None or not approval.compatibility_receipt_sha256 or sha256_file(compatibility) != approval.compatibility_receipt_sha256:
        reasons.append("compatibility_receipt_hash_mismatch")
    if _rooted(config.output_root).exists():
        reasons.append("output_root_exists")
    only_expected = set(reasons).issubset({"approval_pending", "compatibility_check_required"})
    return {
        "status": "blocked" if reasons else "ready_for_real_pilot",
        "ready_for_real_pilot": not reasons,
        "offline_design_complete": only_expected,
        "blocking_reasons": sorted(set(reasons)),
        "study_id": config.study_id,
        "config_sha256": sha256_file(config_path),
        "protocol_sha256": config.protocol_sha256,
        "controller_sha256": config.controller_sha256,
        "replay_sha256": config.replay_sha256,
        "runner_adapter_sha256": config.runner_adapter_sha256,
        "amendment_sha256": config.amendment_sha256,
        "amendment_receipt_sha256": receipt_sha,
        "scenario_count": len(scenarios), "condition_count": 4, "seed_count": 3,
        "run_count": len(config.runs), "matched_group_count": 12,
        "logical_request_cap": config.request_cap,
        "completion_reservation_cap": config.completion_reservation_cap,
        "output_root": config.output_root,
        "network": "disabled", "provider_constructed": False, "api_key_read": False,
        "results_written": False, "private_truth_exposed": False,
        "development_only": True, "pilot_only": True, "not_paper_result": True,
        "no_causal_conclusion": True,
    }


__all__ = [
    "HG1Approval", "HG1Error", "HG1MaterialBinding", "HG1PilotConfig", "HG1RunSpec",
    "HG1Schedule", "HG1PropagationSlot", "HG1FakeProvider", "load_hg1_config", "run_hg1_fake_smoke",
    "write_amendment_receipt", "hg1_preflight", "sha256_file",
]
