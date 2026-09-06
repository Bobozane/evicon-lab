"""Offline H-G.1.1 truncation amendment, FakeProvider smoke, and preflight."""
from __future__ import annotations

import hashlib
import json
import tempfile
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from .cascade_agent_protocol_hg11 import (
    HG11_PROTOCOL_VERSION, HG11_SCHEMA_NAME, HG11_TEMPLATE_VERSION,
    CascadeAgentProtocolHG11Runtime, build_identifiable_prompt_context,
)
from .cascade_controller import CascadeControllerPolicyLoader
from .cascade_controller_hg1 import propose_hg1, validate_hg1_proposal
from .cascade_hg1_replay import HG1CascadeProtocolRunner, HG1OutcomeReplayValidator
from .cascade_real_agent_runner import CascadeRealAgentRunner
from .provenance_cascade_hg1 import (
    HG1FakeProvider, _claim_selector, _post_round_events, load_hg1_config,
)
from .provenance_cascade_preregistration import CascadeCondition

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = _ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg11.v1.toml"
DEFAULT_RECEIPT = _ROOT / "outputs/study-locks/provenance_cascade_hg11_truncation_amendment_receipt.json"
_EXPECTED_CONDITIONS = tuple(CascadeCondition)
_EXPECTED_SEEDS = (20261001, 20261002, 20261003)
_EXPECTED_AGENTS = tuple(f"network-agent-{index:02d}" for index in range(1, 7))


class HG11FakeProvider(HG1FakeProvider):
    """Deterministic provider used only by the H-G.1.1 offline smoke."""


class HG11Error(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HG11RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    matched_group_id: str
    scenario_id: str
    condition: CascadeCondition
    seed: int
    expected_provider_requests: Literal[18] = 18
    completion_reservation: Literal[18432] = 18432


class HG11PilotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    study_id: Literal["evicon-provenance-cascade-pilot-hg11"]
    config_version: Literal["provenance_cascade_hg1_1_truncation.v1"]
    protocol_version: Literal["provenance_cascade_agent_protocol.hg1_1_truncation.v1"]
    template_version: Literal["cascade_agent_turn.hg1_1_public_content.v1"]
    response_schema_name: Literal["cascade_agent_response_v2_1"]
    status: Literal["offline_preflight"]
    development_only: Literal[True]
    pilot_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    parent_pilot_status: Literal["paused_protocol_truncation_risk"]
    parent_config_path: str
    parent_config_sha256: str
    parent_failure_diagnostic_path: str
    parent_failure_diagnostic_sha256: str
    amendment_path: str
    amendment_sha256: str
    protocol_path: str
    protocol_sha256: str
    controller_path: str
    controller_sha256: str
    replay_path: str
    replay_sha256: str
    policy_path: str
    policy_sha256: str
    approval_path: str
    compatibility_receipt_path: str
    amendment_receipt_path: str
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
    output_root: Literal["results/provenance-cascade-pilot-hg11-v1"]

    @model_validator(mode="after")
    def fixed_design(self) -> "HG11PilotConfig":
        if (
            self.protocol_version != HG11_PROTOCOL_VERSION
            or self.template_version != HG11_TEMPLATE_VERSION
            or self.response_schema_name != HG11_SCHEMA_NAME
        ):
            raise ValueError("protocol_binding_mismatch")
        if self.conditions != _EXPECTED_CONDITIONS or self.seeds != _EXPECTED_SEEDS or self.agent_ids != _EXPECTED_AGENTS:
            raise ValueError("matched_design_mismatch")
        if self.request_cap != self.run_count * self.logical_requests_per_run:
            raise ValueError("request_cap_mismatch")
        if self.completion_reservation_cap != self.request_cap * self.agent_max_tokens:
            raise ValueError("completion_reservation_mismatch")
        return self

    def _parent_diagnostic(self) -> dict[str, object]:
        try:
            payload = json.loads(_rooted(self.parent_failure_diagnostic_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HG11Error("failure_diagnostic_invalid") from exc
        return payload

    @property
    def parent_batch_sha256(self) -> str:
        return str(self._parent_diagnostic()["batch_record_sha256"])

    @property
    def parent_failed_ledger_sha256(self) -> str:
        return str(self._parent_diagnostic()["failed_run_ledger_sha256"])

    @property
    def parent_failed_checkpoint_sha256(self) -> str:
        return str(self._parent_diagnostic()["failed_run_checkpoint_sha256"])

    @property
    def parent_failed_fingerprint(self) -> str:
        return str(self._parent_diagnostic()["failed_fingerprint"])

    @property
    def runs(self) -> tuple[HG11RunSpec, ...]:
        parent, _, _, _ = load_hg1_config(_rooted(self.parent_config_path))
        return tuple(
            HG11RunSpec(
                run_id=f"hg11-{material.scenario_id}-{seed}-{condition.value}",
                matched_group_id=f"hg11-{material.scenario_id}-{seed}",
                scenario_id=material.scenario_id,
                condition=condition,
                seed=seed,
            )
            for material in parent.scenario_materials
            for seed in self.seeds
            for condition in self.conditions
        )


class HG11Approval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    approval_id: str
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str = ""
    accepted_on: str = ""
    config_sha256: str
    protocol_sha256: str
    amendment_sha256: str
    amendment_receipt_sha256: str
    compatibility_receipt_sha256: str = ""
    controller_sha256: str
    replay_sha256: str
    response_format: Literal["json_schema"]
    response_schema_name: Literal["cascade_agent_response_v2_1"]
    agent_max_tokens: Literal[1024]
    run_count: Literal[48]
    logical_request_cap: Literal[864]
    completion_reservation_cap: Literal[884736]
    max_retries: Literal[1]
    timeout_seconds: Literal[15]
    confirm_parent_batch_frozen: bool = False
    confirm_max_tokens_amendment: bool = False
    confirm_prompt_semantics_unchanged: bool = False
    confirm_scope_and_budget: bool = False
    confirm_no_overwrite: bool = False
    confirm_append_only_ledger: bool = False
    confirm_resume_rules: bool = False
    confirm_parser_invalid_stops: bool = False
    network_execution_authorized: Literal[False] = False

    @property
    def ready(self) -> bool:
        return self.acceptance_status == "accepted" and bool(self.accepted_by and self.accepted_on) and all((
            self.confirm_parent_batch_frozen, self.confirm_max_tokens_amendment,
            self.confirm_prompt_semantics_unchanged, self.confirm_scope_and_budget,
            self.confirm_no_overwrite, self.confirm_append_only_ledger,
            self.confirm_resume_rules, self.confirm_parser_invalid_stops,
        ))


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (_ROOT / path).resolve()


def load_hg11_config(path: str | Path = DEFAULT_CONFIG):
    config_path = _rooted(path)
    try:
        config = HG11PilotConfig.model_validate(tomllib.loads(config_path.read_text(encoding="utf-8")))
    except (OSError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise HG11Error("config_invalid") from exc
    for value, expected, code in (
        (config.parent_config_path, config.parent_config_sha256, "parent_config_hash_mismatch"),
        (config.parent_failure_diagnostic_path, config.parent_failure_diagnostic_sha256, "failure_diagnostic_hash_mismatch"),
        (config.amendment_path, config.amendment_sha256, "amendment_hash_mismatch"),
        (config.protocol_path, config.protocol_sha256, "protocol_hash_mismatch"),
        (config.controller_path, config.controller_sha256, "controller_hash_mismatch"),
        (config.replay_path, config.replay_sha256, "replay_hash_mismatch"),
        (config.policy_path, config.policy_sha256, "policy_hash_mismatch"),
    ):
        target = _rooted(value)
        if not target.is_file() or sha256_file(target) != expected:
            raise HG11Error(code)
    parent, scenarios, truths, schedules = load_hg1_config(_rooted(config.parent_config_path))
    if (
        parent.conditions != config.conditions or parent.seeds != config.seeds
        or parent.agent_ids != config.agent_ids or parent.max_rounds != config.max_rounds
        or parent.topology_id != config.topology_id
    ):
        raise HG11Error("parent_design_mismatch")
    if len(config.runs) != 48 or len({item.run_id for item in config.runs}) != 48:
        raise HG11Error("run_plan_invalid")
    if any(item.run_id.startswith("hg1-") for item in config.runs):
        raise HG11Error("old_run_id_reused")
    return config, scenarios, truths, schedules


def run_hg11_fake_smoke(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    config, scenarios, _, schedules = load_hg11_config(path)
    policy = CascadeControllerPolicyLoader.load(_rooted(config.policy_path))
    protocol_runner = HG1CascadeProtocolRunner()
    real_runner = CascadeRealAgentRunner(base_runner=protocol_runner, policy_config=policy)
    records = []
    with tempfile.TemporaryDirectory(prefix="evicon-hg11-truncation-") as temporary:
        temporary_path = Path(temporary)
        for spec in config.runs:
            schedule = schedules[spec.scenario_id]
            record = real_runner.run_scenario(
                scenarios[spec.scenario_id], spec.seed, spec.condition,
                provider=HG1FakeProvider(), run_id=spec.run_id,
                ledger_path=temporary_path/spec.run_id/"request_ledger.jsonl",
                checkpoint_path=temporary_path/spec.run_id/"agent_checkpoint.json",
                model_name="hg11-truncation-fake", agent_temperature=0.2,
                agent_max_tokens=config.agent_max_tokens,
                request_cap=spec.expected_provider_requests,
                completion_reservation_cap=spec.completion_reservation,
                runtime=CascadeAgentProtocolHG11Runtime(),
                context_builder=build_identifiable_prompt_context,
                outcome_replay_validator=HG1OutcomeReplayValidator,
                initial_events_builder=protocol_runner.preload_initial_events,
                claim_selector=lambda scenario, snapshot, round_id, schedule=schedule: _claim_selector(
                    schedule, scenario, snapshot, round_id
                ),
                proposal_builder=propose_hg1,
                proposal_validator=validate_hg1_proposal,
                post_round_events_builder=lambda scenario, round_id, events, schedule=schedule: _post_round_events(
                    scenario, schedule, round_id, events
                ),
                actor_schedule_provider=lambda _scenario, round_id, schedule=schedule: (
                    schedule.propagation_slots if round_id == 0 else ()
                ),
                policy_config=policy,
            )
            if record.replay is None or record.replay.status.value != "passed":
                raise HG11Error("fake_replay_failed")
            records.append(record)
    logical = sum(item.logical_request_count for item in records)
    provider_calls = sum(item.provider_call_count for item in records)
    return {
        "status": "fake_smoke_passed" if len(records) == 48 and logical == 864 else "fake_smoke_failed",
        "run_count": len(records),
        "matched_group_count": len({(item.scenario_id, item.seed) for item in records}),
        "logical_request_count": logical,
        "provider_call_count": provider_calls,
        "replay_passed_count": sum(item.replay.status.value == "passed" for item in records),
        "directive_applied_count": sum(item.directive_applied_count for item in records),
        "agent_max_tokens": config.agent_max_tokens,
        "request_cap": config.request_cap,
        "completion_reservation_cap": config.completion_reservation_cap,
        "old_batch_resumed": False,
        "network": "disabled", "results_written": False,
        "private_truth_exposed": False, "development_only": True,
        "pilot_only": True, "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def write_amendment_receipt(summary: dict[str, object], path: str | Path = DEFAULT_RECEIPT, config_path: str | Path = DEFAULT_CONFIG) -> Path:
    config, _, _, _ = load_hg11_config(config_path)
    target = _rooted(path)
    payload = {
        "receipt_version": "provenance_cascade_hg11_truncation_amendment_receipt.v1",
        "status": "offline_validated_pending_approval_and_compatibility",
        "config_sha256": sha256_file(_rooted(config_path)),
        "protocol_sha256": config.protocol_sha256,
        "amendment_sha256": config.amendment_sha256,
        "parent_config_sha256": config.parent_config_sha256,
        "parent_failure_diagnostic_sha256": config.parent_failure_diagnostic_sha256,
        "run_count": summary["run_count"],
        "matched_group_count": summary["matched_group_count"],
        "logical_request_count": summary["logical_request_count"],
        "request_cap": summary["request_cap"],
        "agent_max_tokens": summary["agent_max_tokens"],
        "completion_reservation_cap": summary["completion_reservation_cap"],
        "replay_passed_count": summary["replay_passed_count"],
        "directive_applied_count": summary["directive_applied_count"],
        "old_batch_resumed": False,
        "old_results_merged": False,
        "network": "disabled", "private_truth_exposed": False,
        "development_only": True, "pilot_only": True,
        "not_paper_result": True, "no_causal_conclusion": True,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_text(encoding="utf-8") != encoded:
        raise HG11Error("amendment_receipt_no_overwrite")
    target.write_text(encoded, encoding="utf-8")
    return target


def hg11_preflight(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    reasons: list[str] = []
    try:
        config, scenarios, _, _ = load_hg11_config(path)
    except HG11Error as exc:
        return {"status":"blocked","blocking_reasons":[exc.code],"network":"disabled","provider_constructed":False,"api_key_read":False,"results_written":False}
    config_path = _rooted(path)
    receipt_path = _rooted(config.amendment_receipt_path)
    receipt_sha = sha256_file(receipt_path) if receipt_path.is_file() else None
    if receipt_sha is None:
        reasons.append("amendment_receipt_missing")
    approval_path = _rooted(config.approval_path)
    approval = None
    if not approval_path.is_file():
        reasons.append("approval_missing")
    else:
        try:
            approval = HG11Approval.model_validate(tomllib.loads(approval_path.read_text(encoding="utf-8")))
            if not approval.ready:
                reasons.append("approval_pending")
            if (
                approval.config_sha256 != sha256_file(config_path)
                or approval.protocol_sha256 != config.protocol_sha256
                or approval.amendment_sha256 != config.amendment_sha256
                or approval.amendment_receipt_sha256 != receipt_sha
            ):
                reasons.append("approval_hash_mismatch")
        except Exception:
            reasons.append("approval_invalid")
    compatibility_path = _rooted(config.compatibility_receipt_path)
    if not compatibility_path.is_file():
        reasons.append("compatibility_check_required")
    elif approval is None or not approval.compatibility_receipt_sha256 or sha256_file(compatibility_path) != approval.compatibility_receipt_sha256:
        reasons.append("compatibility_receipt_hash_mismatch")
    if _rooted(config.output_root).exists():
        reasons.append("output_root_exists")
    return {
        "status": "ready_for_real_pilot" if not reasons else "blocked",
        "ready_for_real_pilot": not reasons,
        "offline_design_complete": set(reasons).issubset({"approval_pending", "compatibility_check_required"}),
        "blocking_reasons": sorted(set(reasons)),
        "study_id": config.study_id,
        "config_sha256": sha256_file(config_path),
        "protocol_sha256": config.protocol_sha256,
        "amendment_sha256": config.amendment_sha256,
        "amendment_receipt_sha256": receipt_sha,
        "run_count": len(config.runs), "matched_group_count": 12,
        "scenario_count": len(scenarios), "condition_count": 4, "seed_count": 3,
        "logical_request_cap": config.request_cap,
        "agent_max_tokens": config.agent_max_tokens,
        "completion_reservation_cap": config.completion_reservation_cap,
        "output_root": config.output_root,
        "network": "disabled", "provider_constructed": False,
        "api_key_read": False, "results_written": False,
        "private_truth_exposed": False, "development_only": True,
        "pilot_only": True, "not_paper_result": True,
        "no_causal_conclusion": True,
    }


__all__ = [
    "DEFAULT_CONFIG", "DEFAULT_RECEIPT", "HG11Approval", "HG11Error", "HG11FakeProvider",
    "HG11PilotConfig", "HG11RunSpec", "hg11_preflight", "load_hg11_config",
    "run_hg11_fake_smoke", "sha256_file", "write_amendment_receipt",
]
