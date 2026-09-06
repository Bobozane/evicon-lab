"""Offline trigger calibration for provenance-cascade interventions.

This module is an isolated development fixture.  It deliberately does not
modify the historical cascade runner or preregistration.  The calibration
scenario places two public content nodes with one shared root into the same
agent's round-one start view; the existing controller can therefore exercise
its already-registered same-root repetition rule and schedule a next-round
structured directive.
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .cascade_actors import ActorProfile
from .cascade_controller import CascadeControllerPolicyConfig, CascadeControllerPolicyLoader, CascadeReasonCode
from .cascade_outcome_runner import CascadeOutcomeRunError, CascadeOutcomeRunRecord, CascadeOutcomeRunner
from .cascade_protocol import CascadeScenarioLoader, CascadeScenarioSpec
from .models._validation import identifier_list, normalized_text
from .provenance_cascade import CascadeScenario as ScenarioType, file_sha256
from .provenance_cascade_preregistration import CascadeCondition, NetworkTopology

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "configs/provenance_cascade/calibration/provenance_cascade_intervention_calibration.v1.toml"

_AGENT_IDS = tuple(f"network-agent-{index:02d}" for index in range(1, 7))
_EXPECTED_CONDITIONS = tuple(CascadeCondition)
_EXPECTED_SCENARIOS = tuple(ScenarioType)
_FORBIDDEN_PRIVATE_FIELDS = {
    "ground_truth_label", "source_independence_label", "evaluator_truth", "private_fixture", "private_path",
}


class CalibrationRunError(ValueError):
    """Stable, non-sensitive calibration error code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class CalibrationScenarioBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_type: ScenarioType
    scenario_id: str = Field(min_length=1)
    scenario_path: str = Field(min_length=1)
    scenario_config_sha256: str = Field(min_length=64, max_length=64)
    graph_sha256: str = Field(min_length=64, max_length=64)

    @field_validator("scenario_id", "scenario_path")
    @classmethod
    def ids(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "value"))

    @field_validator("scenario_config_sha256", "graph_sha256")
    @classmethod
    def hashes(cls, value: str) -> str:
        value = value.strip().lower()
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("calibration hashes must be SHA-256 digests")
        return value


class CascadeInterventionCalibrationManifest(BaseModel):
    """Frozen local plan for trigger calibration; no run files are implied."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    calibration_id: str = Field(min_length=1)
    calibration_version: Literal["provenance_cascade_intervention_calibration.v1"]
    status: Literal["calibration"]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    agent_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    max_rounds: Literal[3]
    seeds: tuple[int, ...] = Field(min_length=3, max_length=3)
    conditions: tuple[CascadeCondition, ...] = Field(min_length=4, max_length=4)
    network_topology: NetworkTopology
    policy_path: str = Field(min_length=1)
    policy_sha256: str = Field(min_length=64, max_length=64)
    scenarios: tuple[CalibrationScenarioBinding, ...] = Field(min_length=4, max_length=4)

    @field_validator("calibration_id", "policy_path")
    @classmethod
    def text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "value"))

    @field_validator("agent_ids")
    @classmethod
    def agents(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(identifier_list(list(value), "agent_ids"))

    @field_validator("seeds")
    @classmethod
    def seeds_valid(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0 for seed in value):
            raise ValueError("calibration seeds must be positive integers")
        if len(set(value)) != len(value):
            raise ValueError("calibration seeds must be unique")
        return value

    @field_validator("policy_sha256")
    @classmethod
    def policy_hash(cls, value: str) -> str:
        value = value.strip().lower()
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("policy_sha256 must be a SHA-256 digest")
        return value

    @model_validator(mode="after")
    def fixed_shape(self) -> "CascadeInterventionCalibrationManifest":
        if self.agent_ids != _AGENT_IDS:
            raise ValueError("calibration requires the fixed six-agent order")
        if self.network_topology.agent_ids != list(self.agent_ids):
            raise ValueError("topology agents must match calibration agents")
        if self.network_topology.topology_id != "ring_6_bidirectional" or self.network_topology.directed:
            raise ValueError("calibration requires ring_6_bidirectional")
        if self.conditions != _EXPECTED_CONDITIONS:
            raise ValueError("calibration conditions must use the fixed four-condition order")
        if tuple(item.scenario_type for item in self.scenarios) != _EXPECTED_SCENARIOS:
            raise ValueError("calibration scenarios must use the fixed four-scenario order")
        ids = [item.scenario_id for item in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("calibration scenario IDs must be unique")
        if len(set(self.seeds)) != 3:
            raise ValueError("calibration requires exactly three seeds")
        return self


class CalibrationScenarioSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    binding: CalibrationScenarioBinding
    scenario: CascadeScenarioSpec


class CalibrationRunSummary(BaseModel):
    """Safe summary for one in-memory calibration run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    scenario_type: ScenarioType
    seed: int
    condition: CascadeCondition
    proposal_count: int = Field(ge=0)
    schedule_count: int = Field(ge=0)
    applied_count: int = Field(ge=0)
    action: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    snapshot_sha256: str
    exposure_sha256: str
    application_sha256: str
    outcome_sha256: str
    replay_status: str
    application_replay_status: str
    same_root_target_agent_id: str | None = None
    same_root_visible_round: int | None = None
    same_root_visible_repetition_count: int | None = None
    same_root_visible_root_count: int | None = None
    development_only: Literal[True] = True
    calibration_only: Literal[True] = True
    not_paper_result: Literal[True] = True
    no_causal_conclusion: Literal[True] = True


class CascadeInterventionCalibrationRunner:
    """Load and execute only local deterministic calibration fixtures."""

    def __init__(self, manifest: CascadeInterventionCalibrationManifest, base_dir: Path) -> None:
        self.manifest = manifest
        self.base_dir = base_dir
        self.policy = self._load_policy()
        self.scenarios = self._load_scenarios()
        self.runner = CascadeOutcomeRunner(policy_config=self.policy)

    @classmethod
    def from_file(cls, path: str | Path = DEFAULT_MANIFEST) -> "CascadeInterventionCalibrationRunner":
        manifest_path = Path(path).resolve()
        try:
            payload = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = CascadeInterventionCalibrationManifest.model_validate(payload)
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
            raise CalibrationRunError("calibration_manifest_invalid") from exc
        return cls(manifest, manifest_path.parent)

    def _load_policy(self) -> CascadeControllerPolicyConfig:
        policy_path = (self.base_dir / self.manifest.policy_path).resolve()
        try:
            if file_sha256(policy_path) != self.manifest.policy_sha256:
                raise CalibrationRunError("policy_hash_mismatch")
            policy = CascadeControllerPolicyLoader.load(policy_path)
            expected_ids = {item.scenario_id for item in self.manifest.scenarios}
            if {rule.scenario_id for rule in policy.scenario_rules} != expected_ids:
                raise CalibrationRunError("policy_scenario_binding_mismatch")
            return policy
        except CalibrationRunError:
            raise
        except Exception as exc:
            raise CalibrationRunError("calibration_policy_invalid") from exc

    def _load_scenarios(self) -> tuple[CalibrationScenarioSpec, ...]:
        loaded: list[CalibrationScenarioSpec] = []
        for binding in self.manifest.scenarios:
            path = (self.base_dir / binding.scenario_path).resolve()
            try:
                if file_sha256(path) != binding.scenario_config_sha256:
                    raise CalibrationRunError("scenario_config_hash_mismatch")
                scenario = CascadeScenarioLoader.load(path)
                if scenario.scenario_id != binding.scenario_id or scenario.scenario_type is not binding.scenario_type:
                    raise CalibrationRunError("scenario_identity_mismatch")
                if scenario.graph_sha256 != binding.graph_sha256:
                    raise CalibrationRunError("scenario_graph_hash_mismatch")
                if len(scenario.agent_ids) != 6 or scenario.max_rounds != 3 or scenario.network_topology.topology_id != "ring_6_bidirectional":
                    raise CalibrationRunError("scenario_shape_invalid")
                _assert_no_private_fields(scenario.model_dump(mode="json"))
                loaded.append(CalibrationScenarioSpec(binding=binding, scenario=scenario))
            except CalibrationRunError:
                raise
            except Exception as exc:
                raise CalibrationRunError("calibration_scenario_invalid") from exc
        return tuple(loaded)

    def run_one(self, scenario: CascadeScenarioSpec, seed: int, condition: CascadeCondition) -> tuple[CascadeRunSummary, CascadeOutcomeRunRecord]:
        try:
            record = self.runner.run_scenario(
                scenario,
                seed,
                condition,
                actor_profile=ActorProfile.VERIFICATION_RESPONSIVE_ACTOR,
                run_id=f"{scenario.scenario_id}-{seed}-{condition.value}",
                policy_config=self.policy,
            )
        except CascadeOutcomeRunError as exc:
            raise CalibrationRunError(exc.code) from exc
        schedules = record.application_ledger.schedules
        actions = tuple(sorted({item.action.value for item in schedules}))
        reasons = tuple(sorted({code for item in schedules for code in item.reason_codes}))
        same_agent: str | None = None
        same_round: int | None = None
        repetition: int | None = None
        roots: int | None = None
        if scenario.scenario_type is ScenarioType.FALSE_MAJORITY:
            for context in record.round_contexts:
                if context.snapshot.round_id != 1 or not context.snapshot.visible_content_ids:
                    continue
                if len(context.snapshot.visible_content_ids) >= 2:
                    view = self._view(context.snapshot, scenario)
                    same_agent = context.snapshot.agent_id
                    same_round = context.snapshot.round_id
                    repetition = len(view.provenance_nodes)
                    roots = view.root_count_for_claim(scenario.graph.claims[0].claim_id)
                    break
        summary = CalibrationRunSummary(
            scenario_id=record.scenario_id,
            scenario_type=scenario.scenario_type,
            seed=seed,
            condition=condition,
            proposal_count=record.proposal_count,
            schedule_count=len(schedules),
            applied_count=record.directive_applied_count,
            action=actions,
            reason_codes=reasons,
            snapshot_sha256=record.snapshot_sha256,
            exposure_sha256=record.exposure_ledger_sha256,
            application_sha256=record.application_ledger_sha256,
            outcome_sha256=record.outcome_ledger_sha256,
            replay_status=record.replay.cascade_replay.status.value,
            application_replay_status=record.replay.application_replay.status.value,
            same_root_target_agent_id=same_agent,
            same_root_visible_round=same_round,
            same_root_visible_repetition_count=repetition,
            same_root_visible_root_count=roots,
        )
        return summary, record

    @staticmethod
    def _view(snapshot: object, scenario: CascadeScenarioSpec):
        from .provenance_cascade_exposure import ControllerPublicView
        return ControllerPublicView.from_snapshot(snapshot, scenario.graph)

    def run_all(self) -> tuple[tuple[CalibrationRunSummary, ...], tuple[CascadeOutcomeRunRecord, ...]]:
        summaries: list[CalibrationRunSummary] = []
        records: list[CascadeOutcomeRunRecord] = []
        for binding_spec in self.scenarios:
            for seed in self.manifest.seeds:
                for condition in self.manifest.conditions:
                    summary, record = self.run_one(binding_spec.scenario, seed, condition)
                    summaries.append(summary)
                    records.append(record)
        return tuple(summaries), tuple(records)


def _assert_no_private_fields(value: object) -> None:
    if isinstance(value, dict):
        if _FORBIDDEN_PRIVATE_FIELDS.intersection(value):
            raise CalibrationRunError("private_truth_in_public_fixture")
        for child in value.values():
            _assert_no_private_fields(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_private_fields(child)


def run_calibration(path: str | Path = DEFAULT_MANIFEST) -> tuple[CalibrationRunSummary, ...]:
    return CascadeInterventionCalibrationRunner.from_file(path).run_all()[0]


__all__ = [
    "CalibrationRunError", "CalibrationRunSummary", "CalibrationScenarioBinding", "CalibrationScenarioSpec",
    "CascadeInterventionCalibrationManifest", "CascadeInterventionCalibrationRunner", "DEFAULT_MANIFEST", "run_calibration",
]
