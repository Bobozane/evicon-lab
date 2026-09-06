"""Offline structural validation for the H-G1.2 observability amendment.

This module intentionally has no real-provider entry point.  It binds a new
public exposure schedule to the already-audited H-G1 materials and uses a fake
provider only to verify the eligibility chain and the three replay layers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from .cascade_agent_protocol_hg11 import CascadeAgentProtocolHG11Runtime, build_identifiable_prompt_context
from .cascade_controller import CascadeControllerPolicyLoader
from .cascade_controller_hg1 import propose_hg1, validate_hg1_proposal
from .cascade_hg1_replay import HG1CascadeProtocolRunner, HG1OutcomeReplayValidator
from .cascade_real_agent_runner import CascadeRealAgentRunner
from .provenance_cascade_hg1 import (
    DEFAULT_CONFIG as HG1_CONFIG,
    HG1FakeProvider,
    HG1Schedule,
    _claim_selector,
    _eligibility,
    _post_round_events,
    _rooted,
    load_hg1_config,
)
from .provenance_cascade_preregistration import CascadeCondition, CascadeScenario

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = _ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg12.v1.toml"
EXPECTED_CONDITIONS = tuple(CascadeCondition)
EXPECTED_AGENTS = tuple(f"network-agent-{index:02d}" for index in range(1, 7))
EXPECTED_SEEDS = (20261011, 20261012, 20261013)


class HG12DesignError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HG12ScheduleBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_id: str
    scenario_type: CascadeScenario
    schedule_path: str
    schedule_sha256: str


class HG12DesignConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    study_id: Literal["evicon-provenance-cascade-pilot-hg12"]
    config_version: Literal["provenance_cascade_hg1_2_observability.v1"]
    status: Literal["offline_design_pending_human_approval"]
    development_only: Literal[True]
    pilot_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    parent_pilot_id: Literal["evicon-provenance-cascade-pilot-hg11"]
    parent_pilot_status: Literal["non_identifiable_pilot"]
    parent_config_path: str
    parent_config_sha256: str
    base_material_config_path: str
    base_material_config_sha256: str
    amendment_path: str
    amendment_sha256: str
    protocol_path: str
    protocol_sha256: str
    protocol_version: Literal["provenance_cascade_agent_protocol.hg1_1_truncation.v1"]
    template_version: Literal["cascade_agent_turn.hg1_1_public_content.v1"]
    response_schema_name: Literal["cascade_agent_response_v2_1"]
    controller_path: str
    controller_sha256: str
    replay_path: str
    replay_sha256: str
    policy_path: str
    policy_sha256: str
    conditions: tuple[CascadeCondition, ...]
    seeds: tuple[int, ...]
    agent_ids: tuple[str, ...]
    max_rounds: Literal[3]
    topology_id: Literal["ring_6_bidirectional"]
    run_count: Literal[48]
    matched_group_count: Literal[12]
    logical_requests_per_run: Literal[18]
    request_cap: Literal[864]
    agent_max_tokens: Literal[1024]
    completion_reservation_per_run: Literal[18432]
    completion_reservation_cap: Literal[884736]
    response_format: Literal["json_schema"]
    max_retries: Literal[1]
    timeout_seconds: Literal[15]
    output_root: Literal["results/provenance-cascade-pilot-hg12-v1"]
    schedule_materials: tuple[HG12ScheduleBinding, ...]

    @model_validator(mode="after")
    def fixed_design(self) -> "HG12DesignConfig":
        if self.conditions != EXPECTED_CONDITIONS or self.seeds != EXPECTED_SEEDS:
            raise ValueError("matched design mismatch")
        if self.agent_ids != EXPECTED_AGENTS or len(self.schedule_materials) != 4:
            raise ValueError("agent or schedule shape mismatch")
        if {item.scenario_type for item in self.schedule_materials} != set(CascadeScenario):
            raise ValueError("scenario coverage mismatch")
        if self.request_cap != self.run_count * self.logical_requests_per_run:
            raise ValueError("request cap mismatch")
        if self.completion_reservation_cap != self.request_cap * self.agent_max_tokens:
            raise ValueError("completion reservation mismatch")
        return self


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (_ROOT / path).resolve()


def load_hg12_design(path: str | Path = DEFAULT_CONFIG):
    config_path = _path(path)
    try:
        config = HG12DesignConfig.model_validate(tomllib.loads(config_path.read_text(encoding="utf-8")))
    except (OSError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise HG12DesignError("hg12_config_invalid") from exc
    bindings = (
        (config.parent_config_path, config.parent_config_sha256, "parent_config_hash_mismatch"),
        (config.base_material_config_path, config.base_material_config_sha256, "base_material_config_hash_mismatch"),
        (config.amendment_path, config.amendment_sha256, "amendment_hash_mismatch"),
        (config.protocol_path, config.protocol_sha256, "protocol_hash_mismatch"),
        (config.controller_path, config.controller_sha256, "controller_hash_mismatch"),
        (config.replay_path, config.replay_sha256, "replay_hash_mismatch"),
        (config.policy_path, config.policy_sha256, "policy_hash_mismatch"),
    )
    for value, expected, code in bindings:
        candidate = _path(value)
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise HG12DesignError(code)
    _, scenarios, truths, _ = load_hg1_config(_path(config.base_material_config_path))
    schedules: dict[str, HG1Schedule] = {}
    for binding in config.schedule_materials:
        schedule_path = _path(binding.schedule_path)
        if not schedule_path.is_file() or sha256_file(schedule_path) != binding.schedule_sha256:
            raise HG12DesignError("schedule_hash_mismatch")
        try:
            schedule = HG1Schedule.model_validate_json(schedule_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HG12DesignError("schedule_invalid") from exc
        scenario = scenarios.get(binding.scenario_id)
        if scenario is None or scenario.scenario_type is not binding.scenario_type or schedule.scenario_id != binding.scenario_id:
            raise HG12DesignError("schedule_scenario_mismatch")
        if len(schedule.post_round_exposures) != 1 or schedule.post_round_exposures[0].after_round_id != 0:
            raise HG12DesignError("second_public_exposure_missing")
        exposure = schedule.post_round_exposures[0]
        if "network-agent-02" not in exposure.target_agent_ids:
            raise HG12DesignError("paired_root_target_missing")
        schedules[binding.scenario_id] = schedule
    _validate_opportunities(scenarios, schedules)
    return config, scenarios, truths, schedules


def _validate_opportunities(scenarios, schedules: dict[str, HG1Schedule]) -> None:
    for scenario_id in ("cascade-hg1-false-majority", "cascade-hg1-independent-true-consensus"):
        scenario = scenarios[scenario_id]
        schedule = schedules[scenario_id]
        initial = {
            item.provenance_node_id for item in scenario.initial_public_content
            if item.round_id == 0 and "network-agent-02" in item.target_agent_ids
        }
        second = schedule.post_round_exposures[0].provenance_node_id
        if len(initial) != 1 or second in initial:
            raise HG12DesignError("paired_root_surface_invalid")
        initial_root = scenario.graph.root_sources_for_node(next(iter(initial)))
        second_root = scenario.graph.root_sources_for_node(second)
        same_root = initial_root == second_root
        if (scenario_id.endswith("false-majority") and not same_root) or (scenario_id.endswith("independent-true-consensus") and same_root):
            raise HG12DesignError("paired_root_relation_invalid")
    correction = schedules["cascade-hg1-true-minority-correction"]
    if len(set(correction.outcome_targets[2].values())) != 1 or next(iter(correction.outcome_targets[2].values())) != "claim-hg1-tmc-correction":
        raise HG12DesignError("correction_round_two_target_invalid")


def run_hg12_fake_smoke(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    config, scenarios, truths, schedules = load_hg12_design(path)
    policy = CascadeControllerPolicyLoader.load(_path(config.policy_path))
    base = HG1CascadeProtocolRunner()
    eligibility = Counter()
    operations = defaultdict(lambda: Counter(run_count=0, proposal_count=0, scheduled_count=0, applied_count=0))
    paired_roots = {}
    replay_count = 0
    with tempfile.TemporaryDirectory(prefix="evicon-hg12-observability-") as temporary:
        temporary_path = Path(temporary)
        for material in config.schedule_materials:
            schedule = schedules[material.scenario_id]
            scenario = scenarios[material.scenario_id]
            for seed in config.seeds:
                for condition in config.conditions:
                    provider = HG1FakeProvider()
                    run_id = f"hg12-{material.scenario_id}-{seed}-{condition.value}"
                    record = CascadeRealAgentRunner(base_runner=base, policy_config=policy).run_scenario(
                        scenario, seed, condition, provider=provider, run_id=run_id,
                        ledger_path=temporary_path / run_id / "request_ledger.jsonl",
                        checkpoint_path=temporary_path / run_id / "agent_checkpoint.json",
                        model_name="hg12-observability-fake", agent_temperature=0.2,
                        agent_max_tokens=config.agent_max_tokens, request_cap=18,
                        completion_reservation_cap=config.completion_reservation_per_run,
                        runtime=CascadeAgentProtocolHG11Runtime(),
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
                        raise HG12DesignError("fake_replay_failed")
                    replay_count += 1
                    eligibility.update(_eligibility(record, scenario, truths[material.scenario_id], provider.share_count))
                    row = operations[f"{material.scenario_id}|{condition.value}"]
                    row["run_count"] += 1
                    row["proposal_count"] += record.proposal_count
                    row["scheduled_count"] += sum(item.status.value != "rejected" for item in record.application_ledger.schedules)
                    row["applied_count"] += record.directive_applied_count
                    if condition is CascadeCondition.PROVENANCE_AWARE_CONTROLLER and seed == config.seeds[0]:
                        snapshot = next(item for item in record.exposure_ledger.snapshots if item.agent_id == "network-agent-02" and item.round_id == 1)
                        paired_roots[material.scenario_id] = {
                            "visible_content_count": len(snapshot.visible_content_ids),
                            "visible_root_count": len({root for relation in snapshot.visible_root_relations for root in relation.source_root_ids}),
                        }
    if replay_count != 48:
        raise HG12DesignError("fake_run_count_invalid")
    for key in ("round0_substantive_stance", "initial_false_endorsement_eligible", "correction_transition_eligible", "harmful_cascade_eligible"):
        if eligibility[key] <= 0:
            raise HG12DesignError(f"{key}_missing")
    if paired_roots.get("cascade-hg1-false-majority") != {"visible_content_count": 2, "visible_root_count": 1}:
        raise HG12DesignError("same_root_opportunity_missing")
    if paired_roots.get("cascade-hg1-independent-true-consensus") != {"visible_content_count": 2, "visible_root_count": 2}:
        raise HG12DesignError("independent_root_opportunity_missing")
    false_aware = operations["cascade-hg1-false-majority|provenance_aware_controller"]
    if false_aware["applied_count"] <= 0:
        raise HG12DesignError("provenance_aware_application_missing")
    for scenario_id in ("cascade-hg1-independent-true-consensus", "cascade-hg1-unresolved-disagreement"):
        if operations[f"{scenario_id}|provenance_aware_controller"]["applied_count"] != 0:
            raise HG12DesignError("protection_abstain_failed")
    return {
        "status": "fake_smoke_passed",
        "run_count": 48,
        "matched_group_count": 12,
        "logical_request_count": 864,
        "completion_reservation_cap": config.completion_reservation_cap,
        "replay_passed_count": replay_count,
        "eligibility": dict(sorted(eligibility.items())),
        "scenario_condition_operations": {key: dict(value) for key, value in sorted(operations.items())},
        "paired_root_opportunities": paired_roots,
        "network": "disabled",
        "results_written": False,
        "private_truth_exposed": False,
        "development_only": True,
        "pilot_only": True,
        "not_paper_result": True,
        "no_causal_conclusion": True,
        "effectiveness_claimed": False,
    }


def hg12_preflight(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    try:
        config, _, _, _ = load_hg12_design(path)
    except HG12DesignError as exc:
        return {"status": "blocked", "blocking_reasons": [exc.code], "network": "disabled", "provider_constructed": False, "results_written": False}
    reasons = ["human_approval_required", "one_shot_compatibility_check_required"]
    if _path(config.output_root).exists():
        reasons.append("output_root_exists")
    return {
        "status": "blocked",
        "ready_for_real_pilot": False,
        "blocking_reasons": reasons,
        "study_id": config.study_id,
        "config_sha256": sha256_file(_path(path)),
        "protocol_sha256": config.protocol_sha256,
        "amendment_sha256": config.amendment_sha256,
        "scenario_count": 4,
        "condition_count": 4,
        "seed_count": 3,
        "run_count": config.run_count,
        "matched_group_count": config.matched_group_count,
        "logical_request_cap": config.request_cap,
        "completion_reservation_cap": config.completion_reservation_cap,
        "network": "disabled",
        "provider_constructed": False,
        "api_key_read": False,
        "results_written": False,
        "private_truth_exposed": False,
        "development_only": True,
        "pilot_only": True,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline H-G.1.2 design validation")
    parser.add_argument("--fake-smoke", action="store_true")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    payload = run_hg12_fake_smoke(args.config) if args.fake_smoke else hg12_preflight(args.config)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
