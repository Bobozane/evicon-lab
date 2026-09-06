"""Offline preflight contract for the provenance-cascade real-pilot gate.

This module validates only local, public planning material. It never reads the
API-key value, constructs a Provider, calls a model, creates a results
directory, or loads evaluator-private truth.
"""
from __future__ import annotations

import hashlib
import os
import re
import tomllib
from enum import Enum
from pathlib import Path
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .cascade_protocol import CascadeScenarioLoader
from .provenance_cascade_preregistration import (
    CascadeCondition,
    CascadeScenario,
    load_and_validate_provenance_cascade_preregistration,
)

_ROOT = Path(__file__).resolve().parents[2]
_AGENT_IDS = tuple(f"network-agent-{index:02d}" for index in range(1, 7))
_CONDITIONS = tuple(CascadeCondition)
_SCENARIO_IDS = {
    CascadeScenario.FALSE_MAJORITY: "cascade-false-majority",
    CascadeScenario.TRUE_MINORITY_CORRECTION: "cascade-true-minority-correction",
    CascadeScenario.INDEPENDENT_TRUE_CONSENSUS: "cascade-independent-true-consensus",
    CascadeScenario.UNRESOLVED_DISAGREEMENT: "cascade-unresolved-disagreement",
}
_PRIMARY_METRICS = (
    "false_cascade_adoption_rate",
    "supported_correction_retention",
    "beneficial_receptivity",
    "harmful_conformity",
    "intervention_false_positive_rate",
    "provenance_diversity",
    "replay_audit_pass_rate",
    "token_cost",
    "latency_ms",
)
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


class HBPilotError(ValueError):
    """Stable, non-sensitive H-B validation error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HBSafetyContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    development_only: Literal[True]
    calibration_only: Literal[False]
    pilot_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    evaluator_truth_isolated: Literal[True]
    no_mediator: Literal[True]
    no_online_feedback: Literal[True]


class HBMaterialReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    scenario_type: CascadeScenario
    config_path: str = Field(min_length=1)
    config_sha256: str = Field(min_length=64, max_length=64)
    public_graph_path: str = Field(min_length=1)
    public_graph_sha256: str = Field(min_length=64, max_length=64)

    @field_validator("config_sha256", "public_graph_sha256")
    @classmethod
    def hash_shape(cls, value: str) -> str:
        value = value.strip().lower()
        if not _SHA256.fullmatch(value):
            raise ValueError("material hash must be a SHA-256 digest")
        return value


class HBProviderContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_kind: Literal["openai_compatible"]
    model_env_var: Literal["EVICON_LLM_MODEL"]
    base_url_env_var: Literal["EVICON_LLM_BASE_URL"]
    api_key_env_var: Literal["EVICON_LLM_API_KEY"]
    network_default: Literal["disabled"]
    agent_temperature: Literal[0.2]
    agent_max_tokens: Literal[256]
    max_retries: int = Field(ge=0, le=3)
    timeout_seconds: float = Field(gt=0.0, le=120.0)
    min_interval_seconds: float = Field(ge=0.0, le=60.0)


class HBBudgetContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    logical_requests_per_run: int = Field(gt=0)
    request_cap: int = Field(gt=0)
    completion_reservation_per_run: int = Field(gt=0)
    completion_reservation_cap: int = Field(gt=0)
    token_reservation_is_not_total_usage: Literal[True]
    token_reservation_is_not_price_cap: Literal[True]


class HBOutputContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    results_root: str = Field(min_length=1)
    ledger_filename: Literal["request_ledger.jsonl"]
    batch_record_filename: Literal["batch_record.json"]
    reject_existing_paths: Literal[True]


class HBRunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    matched_group_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    scenario_type: CascadeScenario
    condition: CascadeCondition
    seed: int = Field(gt=0)
    agent_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    round_count: Literal[3]
    topology_id: Literal["ring_6_bidirectional"]
    expected_provider_requests: Literal[18]
    completion_reservation: Literal[4608]
    output_relative_path: str = Field(min_length=1)

    @field_validator("agent_ids")
    @classmethod
    def unique_agents(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(value) != _AGENT_IDS:
            raise ValueError("run agent order must be the fixed six-agent order")
        return value


class HBMatchedGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    matched_group_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    scenario_type: CascadeScenario
    seed: int = Field(gt=0)
    condition_order: tuple[CascadeCondition, ...] = Field(min_length=4, max_length=4)
    run_ids: tuple[str, ...] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def fixed_conditions(self) -> "HBMatchedGroup":
        if self.condition_order != _CONDITIONS:
            raise ValueError("matched group condition order is not preregistered")
        if len(set(self.run_ids)) != 4:
            raise ValueError("matched group run IDs must be unique")
        return self


class ProvenanceCascadeHBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: Literal["evicon-provenance-cascade-pilot-hb"]
    config_version: Literal["provenance_cascade_hb.v1"]
    status: Literal["preflight"]
    preregistration_path: str = Field(min_length=1)
    preregistration_sha256: str = Field(min_length=64, max_length=64)
    scenario_materials: tuple[HBMaterialReference, ...] = Field(min_length=4, max_length=4)
    conditions: tuple[CascadeCondition, ...] = Field(min_length=4, max_length=4)
    seeds: tuple[int, ...] = Field(min_length=3, max_length=3)
    agent_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    max_rounds: Literal[3]
    topology_id: Literal["ring_6_bidirectional"]
    primary_metrics: tuple[str, ...] = Field(min_length=9, max_length=9)
    provider: HBProviderContract
    budget: HBBudgetContract
    output: HBOutputContract
    matched_groups: tuple[HBMatchedGroup, ...] = Field(min_length=12, max_length=12)
    runs: tuple[HBRunSpec, ...] = Field(min_length=48, max_length=48)
    safety: HBSafetyContract

    @field_validator("preregistration_sha256")
    @classmethod
    def prereg_hash_shape(cls, value: str) -> str:
        value = value.strip().lower()
        if not _SHA256.fullmatch(value):
            raise ValueError("preregistration_sha256 must be a SHA-256 digest")
        return value

    @field_validator("agent_ids")
    @classmethod
    def fixed_agents(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(value) != _AGENT_IDS:
            raise ValueError("agent order must be the fixed six-agent order")
        return value

    @field_validator("conditions")
    @classmethod
    def fixed_conditions(cls, value: tuple[CascadeCondition, ...]) -> tuple[CascadeCondition, ...]:
        if tuple(value) != _CONDITIONS:
            raise ValueError("conditions must contain the four fixed conditions in order")
        return value

    @field_validator("seeds")
    @classmethod
    def fixed_seeds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(value)) != len(value) or any(seed <= 0 for seed in value):
            raise ValueError("seeds must be unique positive integers")
        return value

    @model_validator(mode="after")
    def fixed_shape(self) -> "ProvenanceCascadeHBConfig":
        if self.primary_metrics != _PRIMARY_METRICS:
            raise ValueError("primary_metrics must use the fixed preregistered metric suite")
        if self.budget.logical_requests_per_run != self.max_rounds * len(self.agent_ids):
            raise ValueError("logical_requests_per_run does not match Agent/round scope")
        expected_runs = len(self.scenario_materials) * len(self.seeds) * len(self.conditions)
        if len(self.matched_groups) != len(self.scenario_materials) * len(self.seeds):
            raise ValueError("matched group count does not match scenario/seed scope")
        if len(self.runs) != expected_runs:
            raise ValueError("run count does not match scenario/condition/seed scope")
        if self.budget.request_cap != expected_runs * self.budget.logical_requests_per_run:
            raise ValueError("request_cap does not match the fixed run scope")
        if self.budget.completion_reservation_per_run != self.budget.logical_requests_per_run * self.provider.agent_max_tokens:
            raise ValueError("completion reservation per run does not match provider cap")
        if self.budget.completion_reservation_cap != expected_runs * self.budget.completion_reservation_per_run:
            raise ValueError("completion reservation cap does not match the fixed run scope")
        return self


class HBPilotPreflightReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ready_for_offline_preflight", "blocked", "network_gate_validated_no_execution"]
    study_id: str
    config_version: str
    config_sha256: str
    preregistration_sha256: str
    scenario_count: int = Field(ge=0)
    condition_count: int = Field(ge=0)
    seed_count: int = Field(ge=0)
    matched_group_count: int = Field(ge=0)
    run_count: int = Field(ge=0)
    agent_count: int = Field(ge=0)
    round_count: int = Field(ge=0)
    topology_id: str
    logical_requests_per_run: int = Field(ge=0)
    planned_logical_request_count: int = Field(ge=0)
    completion_reservation_per_run: int = Field(ge=0)
    completion_reservation_cap: int = Field(ge=0)
    token_reservation_is_not_total_usage: bool
    provider_contract_valid: bool
    model_env_configured: bool
    base_url_env_configured: bool
    credential_value_read: bool
    network_default: str
    output_root: str
    output_paths_available: bool
    evaluator_truth_loaded: bool
    safety: dict[str, bool]
    blocking_reasons: tuple[str, ...] = ()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise HBPilotError("absolute_path_not_allowed")
    return (root / path).resolve()


def _safe_source_path(value: str) -> None:
    lowered = value.lower()
    if any(token in lowered for token in ("evaluator", "private", "wvs", "002", "003", "results")):
        raise HBPilotError("forbidden_source_reference")


def _validate_local_materials(config: ProvenanceCascadeHBConfig, config_path: Path) -> None:
    root = config_path.parent
    expected = set(_SCENARIO_IDS.values())
    if {item.scenario_id for item in config.scenario_materials} != expected:
        raise HBPilotError("scenario_material_set_mismatch")
    if {item.scenario_type for item in config.scenario_materials} != set(CascadeScenario):
        raise HBPilotError("scenario_type_set_mismatch")
    prereg_path = _resolve(root, config.preregistration_path)
    if _sha256(prereg_path) != config.preregistration_sha256:
        raise HBPilotError("preregistration_hash_mismatch")
    try:
        prereg_summary = load_and_validate_provenance_cascade_preregistration(prereg_path)
    except Exception as exc:
        raise HBPilotError("preregistration_invalid") from exc
    if prereg_summary.scenario_count != 4 or prereg_summary.condition_count != 4 or prereg_summary.agent_count != 6 or prereg_summary.round_count != 3 or prereg_summary.seed_count != 3:
        raise HBPilotError("preregistration_scope_mismatch")
    for material in config.scenario_materials:
        _safe_source_path(material.config_path)
        _safe_source_path(material.public_graph_path)
        scenario_path = _resolve(root, material.config_path)
        graph_path = _resolve(root, material.public_graph_path)
        if _sha256(scenario_path) != material.config_sha256:
            raise HBPilotError("scenario_config_hash_mismatch")
        if _sha256(graph_path) != material.public_graph_sha256:
            raise HBPilotError("public_graph_hash_mismatch")
        try:
            spec = CascadeScenarioLoader.load(scenario_path)
        except Exception as exc:
            raise HBPilotError("scenario_material_invalid") from exc
        if spec.scenario_id != material.scenario_id or spec.scenario_type is not material.scenario_type:
            raise HBPilotError("scenario_identity_mismatch")
        if Path(spec.graph_path).resolve() != graph_path:
            raise HBPilotError("scenario_graph_path_mismatch")
        if spec.agent_ids != config.agent_ids or spec.max_rounds != config.max_rounds or spec.policy != "no_intervention":
            raise HBPilotError("scenario_scope_mismatch")


def _validate_runs(config: ProvenanceCascadeHBConfig) -> None:
    materials = {item.scenario_id: item.scenario_type for item in config.scenario_materials}
    runs_by_id = {run.run_id: run for run in config.runs}
    if len(runs_by_id) != len(config.runs):
        raise HBPilotError("duplicate_run_id")
    expected_coords = {(scenario_id, condition, seed) for scenario_id in materials for condition in config.conditions for seed in config.seeds}
    actual_coords = {(run.scenario_id, run.condition, run.seed) for run in config.runs}
    if actual_coords != expected_coords:
        raise HBPilotError("run_coordinate_set_mismatch")
    for run in config.runs:
        if run.scenario_type is not materials.get(run.scenario_id):
            raise HBPilotError("run_scenario_type_mismatch")
        if run.matched_group_id != f"hb-{run.scenario_id}-{run.seed}":
            raise HBPilotError("run_matched_group_id_mismatch")
        if run.output_relative_path.startswith("/") or ".." in Path(run.output_relative_path).parts:
            raise HBPilotError("run_output_path_invalid")
        if any(token in run.run_id.lower() for token in ("002", "003", "wvs")):
            raise HBPilotError("legacy_run_reference_forbidden")
    groups = {group.matched_group_id: group for group in config.matched_groups}
    if len(groups) != len(config.matched_groups):
        raise HBPilotError("duplicate_matched_group_id")
    expected_groups = {(f"hb-{scenario_id}-{seed}", scenario_id, seed) for scenario_id in materials for seed in config.seeds}
    actual_groups = {(group.matched_group_id, group.scenario_id, group.seed) for group in config.matched_groups}
    if actual_groups != expected_groups:
        raise HBPilotError("matched_group_coordinate_set_mismatch")
    for group in config.matched_groups:
        group_runs = [runs_by_id.get(run_id) for run_id in group.run_ids]
        if any(run is None for run in group_runs) or {run.condition for run in group_runs if run} != set(config.conditions):
            raise HBPilotError("matched_group_conditions_incomplete")
        if any(run.scenario_id != group.scenario_id or run.seed != group.seed or run.matched_group_id != group.matched_group_id for run in group_runs if run):
            raise HBPilotError("matched_group_alignment_mismatch")
        if tuple(run.run_id for run in group_runs if run) != group.run_ids:
            raise HBPilotError("matched_group_run_order_mismatch")


def load_hb_config(path: str | Path) -> ProvenanceCascadeHBConfig:
    config_path = Path(path).resolve()
    try:
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        config = ProvenanceCascadeHBConfig.model_validate(payload)
        _validate_local_materials(config, config_path)
        _validate_runs(config)
        return config
    except HBPilotError:
        raise
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise HBPilotError("hb_config_invalid") from exc


def _output_path(config: ProvenanceCascadeHBConfig) -> Path:
    value = Path(config.output.results_root)
    if value.is_absolute() or ".." in value.parts or not value.parts or value.parts[0] != "results":
        raise HBPilotError("output_root_must_be_under_results")
    return (_ROOT / value).resolve()


def _output_available(config: ProvenanceCascadeHBConfig) -> bool:
    return not _output_path(config).exists()


def preflight_hb(
    path: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
    allow_network: bool = False,
    confirm_run: bool = False,
    confirm_request_cap: int | None = None,
    confirm_token_cap: int | None = None,
) -> HBPilotPreflightReport:
    config_path = Path(path).resolve()
    reasons: list[str] = []
    try:
        config = load_hb_config(config_path)
    except HBPilotError as exc:
        return HBPilotPreflightReport(
            status="blocked", study_id="unknown", config_version="unknown", config_sha256="0" * 64,
            preregistration_sha256="0" * 64, scenario_count=0, condition_count=0, seed_count=0,
            matched_group_count=0, run_count=0, agent_count=0, round_count=0, topology_id="unknown",
            logical_requests_per_run=0, planned_logical_request_count=0, completion_reservation_per_run=0,
            completion_reservation_cap=0, token_reservation_is_not_total_usage=True,
            provider_contract_valid=False, model_env_configured=False, base_url_env_configured=False,
            credential_value_read=False, network_default="disabled", output_root="results/",
            output_paths_available=False, evaluator_truth_loaded=False,
            safety={"development_only": True, "pilot_only": True, "not_paper_result": True, "no_causal_conclusion": True},
            blocking_reasons=(exc.code,),
        )
    source = os.environ if environment is None else environment
    model_present = bool(source.get(config.provider.model_env_var))
    base_present = bool(source.get(config.provider.base_url_env_var))
    try:
        output_available = _output_available(config)
    except HBPilotError as exc:
        reasons.append(exc.code)
        output_available = False
    if not output_available:
        reasons.append("output_root_exists")
    if allow_network:
        if not confirm_run:
            reasons.append("confirm_run_required")
        if confirm_request_cap != config.budget.request_cap:
            reasons.append("confirm_request_cap_mismatch")
        if confirm_token_cap != config.budget.completion_reservation_cap:
            reasons.append("confirm_token_cap_mismatch")
        if not reasons:
            return HBPilotPreflightReport(
                status="network_gate_validated_no_execution", study_id=config.study_id,
                config_version=config.config_version, config_sha256=_sha256(config_path),
                preregistration_sha256=config.preregistration_sha256, scenario_count=4,
                condition_count=4, seed_count=3, matched_group_count=len(config.matched_groups),
                run_count=len(config.runs), agent_count=6, round_count=3, topology_id=config.topology_id,
                logical_requests_per_run=config.budget.logical_requests_per_run,
                planned_logical_request_count=config.budget.request_cap,
                completion_reservation_per_run=config.budget.completion_reservation_per_run,
                completion_reservation_cap=config.budget.completion_reservation_cap,
                token_reservation_is_not_total_usage=True, provider_contract_valid=True,
                model_env_configured=model_present, base_url_env_configured=base_present,
                credential_value_read=False, network_default=config.provider.network_default,
                output_root=config.output.results_root, output_paths_available=output_available,
                evaluator_truth_loaded=False,
                safety=config.safety.model_dump(mode="json"),
                blocking_reasons=("network_execution_deferred_until_hc",),
            )
        
    if not allow_network and (confirm_run or confirm_request_cap is not None or confirm_token_cap is not None):
        reasons.append("network_confirmation_requires_allow_network")
    return HBPilotPreflightReport(
        status="blocked" if reasons else "ready_for_offline_preflight", study_id=config.study_id,
        config_version=config.config_version, config_sha256=_sha256(config_path),
        preregistration_sha256=config.preregistration_sha256, scenario_count=4,
        condition_count=4, seed_count=3, matched_group_count=len(config.matched_groups),
        run_count=len(config.runs), agent_count=6, round_count=3, topology_id=config.topology_id,
        logical_requests_per_run=config.budget.logical_requests_per_run,
        planned_logical_request_count=config.budget.request_cap,
        completion_reservation_per_run=config.budget.completion_reservation_per_run,
        completion_reservation_cap=config.budget.completion_reservation_cap,
        token_reservation_is_not_total_usage=True, provider_contract_valid=True,
        model_env_configured=model_present, base_url_env_configured=base_present,
        credential_value_read=False, network_default=config.provider.network_default,
        output_root=config.output.results_root, output_paths_available=output_available,
        evaluator_truth_loaded=False,
        safety=config.safety.model_dump(mode="json"), blocking_reasons=tuple(reasons),
    )


__all__ = [
    "HBMaterialReference", "HBBudgetContract", "HBMatchedGroup", "HBOutputContract",
    "HBProviderContract", "HBPilotError", "HBPilotPreflightReport", "HBRunSpec",
    "HBSafetyContract", "ProvenanceCascadeHBConfig", "load_hb_config", "preflight_hb",
]
