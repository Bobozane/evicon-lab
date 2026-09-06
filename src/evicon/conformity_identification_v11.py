"""Shared-T0 branch design for Conformity Identification Study v1.1."""
from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_identification import (
    ConformityIdentificationConfig,
    IdentificationCondition,
    IdentificationError,
    IdentificationStage,
    ScenarioBinding,
    ScenarioType,
    load_identification_config,
    sha256_file,
)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V11_CONFIG = "configs/provenance_cascade/identification/conformity_identification_study.v1_1.toml"


class BranchRunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    matched_group_id: str
    scenario_id: str
    seed: int
    condition: IdentificationCondition
    shared_t0_checkpoint_id: str
    continuation_request_count: Literal[24]


class SharedT0Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    checkpoint_id: str
    matched_group_id: str
    scenario_id: str
    seed: int
    stage: Literal[IdentificationStage.INITIAL_PRIVATE]
    request_count: Literal[6]
    immutable_after_branch: Literal[True]
    reused_by_condition_count: Literal[6]


class ConformityIdentificationV11Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    study_id: Literal["evicon-provenance-cascade-conformity-identification-v1-1"]
    config_version: Literal["conformity_identification_study.v1_1"]
    status: Literal["offline_design_validation"]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    network_enabled: Literal[False]
    controller_enabled: Literal[False]
    intervention_enabled: Literal[False]
    parent_config_path: str
    parent_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_path: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    branch_contract_path: str
    branch_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_version: Literal["conformity_identification_protocol.shared_t0.v1"]
    template_version: Literal["conformity_identification_turn.shared_t0.v1"]
    branch_contract_version: Literal["conformity_identification_shared_t0.v1"]
    conditions: tuple[IdentificationCondition, ...]
    continuation_stages: tuple[IdentificationStage, ...]
    scenario_types: tuple[ScenarioType, ...]
    seeds: tuple[int, ...]
    agent_ids: tuple[str, ...]
    topology_id: Literal["ring_6_bidirectional"]
    scenario_bindings: tuple[ScenarioBinding, ...]
    response_schema_name: Literal["conformity_identification_response_v1"]
    agent_max_tokens: Literal[512]
    matched_group_count: Literal[12]
    branch_run_count: Literal[72]
    shared_t0_requests_per_group: Literal[6]
    continuation_requests_per_branch: Literal[24]
    logical_request_cap: Literal[1800]
    completion_reservation_cap: Literal[921600]
    output_root: Literal["results/provenance-cascade-conformity-identification-v1-1"]
    initial_checkpoint_reuse_required: Literal[True]
    independent_request_fingerprints_after_branch: Literal[True]
    old_results_excluded: Literal[True]
    no_unregistered_thresholds: Literal[True]

    @model_validator(mode="after")
    def fixed_scope(self) -> "ConformityIdentificationV11Config":
        if self.conditions != tuple(IdentificationCondition):
            raise ValueError("condition order mismatch")
        if self.continuation_stages != (
            IdentificationStage.SELF_REFLECTION,
            IdentificationStage.SOCIAL_EXPOSURE,
            IdentificationStage.CORRECTION_EVIDENCE,
            IdentificationStage.FINAL_JUDGMENT,
        ):
            raise ValueError("continuation stages mismatch")
        if self.scenario_types != tuple(ScenarioType):
            raise ValueError("scenario order mismatch")
        if len(set(self.seeds)) != 3 or len(set(self.agent_ids)) != 6:
            raise ValueError("seed or agent uniqueness mismatch")
        if len(self.scenario_bindings) != 4:
            raise ValueError("scenario binding count mismatch")
        groups = len(self.scenario_bindings) * len(self.seeds)
        branches = groups * len(self.conditions)
        requests = groups * self.shared_t0_requests_per_group + branches * self.continuation_requests_per_branch
        if (groups, branches, requests) != (self.matched_group_count, self.branch_run_count, self.logical_request_cap):
            raise ValueError("shared-T0 request accounting mismatch")
        if self.completion_reservation_cap != self.logical_request_cap * self.agent_max_tokens:
            raise ValueError("completion reservation mismatch")
        return self

    @property
    def shared_t0_specs(self) -> tuple[SharedT0Spec, ...]:
        return tuple(
            SharedT0Spec(
                checkpoint_id=f"ident-v11-t0-{binding.scenario_id}-{seed}",
                matched_group_id=f"ident-v11-{binding.scenario_id}-{seed}",
                scenario_id=binding.scenario_id,
                seed=seed,
                stage=IdentificationStage.INITIAL_PRIVATE,
                request_count=6,
                immutable_after_branch=True,
                reused_by_condition_count=6,
            )
            for binding in self.scenario_bindings
            for seed in self.seeds
        )

    @property
    def branch_runs(self) -> tuple[BranchRunSpec, ...]:
        return tuple(
            BranchRunSpec(
                run_id=f"ident-v11-{binding.scenario_id}-{seed}-{condition.value}",
                matched_group_id=f"ident-v11-{binding.scenario_id}-{seed}",
                scenario_id=binding.scenario_id,
                seed=seed,
                condition=condition,
                shared_t0_checkpoint_id=f"ident-v11-t0-{binding.scenario_id}-{seed}",
                continuation_request_count=24,
            )
            for binding in self.scenario_bindings
            for seed in self.seeds
            for condition in self.conditions
        )


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def load_v11_config(path: str | Path = DEFAULT_V11_CONFIG) -> tuple[ConformityIdentificationV11Config, ConformityIdentificationConfig]:
    resolved = _resolve(path)
    if not resolved.is_file():
        raise IdentificationError("v11_config_missing")
    try:
        config = ConformityIdentificationV11Config.model_validate(tomllib.loads(resolved.read_text(encoding="utf-8")))
    except Exception as exc:
        raise IdentificationError("v11_config_invalid") from exc
    if sha256_file(config.parent_config_path) != config.parent_config_sha256:
        raise IdentificationError("parent_config_hash_mismatch")
    if sha256_file(config.protocol_path) != config.protocol_sha256:
        raise IdentificationError("protocol_hash_mismatch")
    if sha256_file(config.branch_contract_path) != config.branch_contract_sha256:
        raise IdentificationError("branch_contract_hash_mismatch")
    parent, scenarios = load_identification_config(config.parent_config_path)
    if config.scenario_bindings != parent.scenario_bindings:
        raise IdentificationError("scenario_binding_changed")
    if tuple(item.scenario_type for item in config.scenario_bindings) != config.scenario_types:
        raise IdentificationError("scenario_type_binding_mismatch")
    if len(scenarios) != 4:
        raise IdentificationError("scenario_count_mismatch")
    return config, parent


def validate_branch_plan(config: ConformityIdentificationV11Config) -> None:
    checkpoints = config.shared_t0_specs
    branches = config.branch_runs
    if len({item.checkpoint_id for item in checkpoints}) != config.matched_group_count:
        raise IdentificationError("duplicate_t0_checkpoint")
    if len({item.run_id for item in branches}) != config.branch_run_count:
        raise IdentificationError("duplicate_branch_run_id")
    by_group = {item.matched_group_id: item for item in checkpoints}
    for group_id, checkpoint in by_group.items():
        group_branches = [item for item in branches if item.matched_group_id == group_id]
        if {item.condition for item in group_branches} != set(IdentificationCondition):
            raise IdentificationError("incomplete_branch_group")
        if {item.shared_t0_checkpoint_id for item in group_branches} != {checkpoint.checkpoint_id}:
            raise IdentificationError("branch_t0_binding_mismatch")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "BranchRunSpec", "ConformityIdentificationV11Config", "DEFAULT_V11_CONFIG", "SharedT0Spec",
    "load_v11_config", "validate_branch_plan",
]
