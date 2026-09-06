"""Versioned H-D amendment and offline preflight for provenance-cascade.

The H-B v1 configuration remains immutable.  This module validates a separate
public-exposure amendment, its 48-run matched plan, and local replay fixtures.
It never constructs a Provider, reads a credential value, or creates results.
"""
from __future__ import annotations

import hashlib
import json
import re
import tomllib
from enum import Enum
from pathlib import Path
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .cascade_controller import (
    CascadeAction,
    CascadeControllerPolicyLoader,
    CascadeReasonCode,
    propose,
)
from .cascade_protocol import CascadeScenarioLoader
from .cascade_replay import CascadeReplayValidator, ExposureLedgerLoader
from .provenance_cascade import CascadeScenario, PublicScenarioLoader
from .provenance_cascade_exposure import ControllerPublicView
from .provenance_cascade_preflight import (
    HBBudgetContract,
    HBMatchedGroup,
    HBOutputContract,
    HBProviderContract,
    HBRunSpec,
    load_hb_config,
)
from .provenance_cascade_preregistration import CascadeCondition, load_and_validate_provenance_cascade_preregistration

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROOT = _REPO_ROOT
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
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
_FORBIDDEN_PUBLIC_KEYS = {
    "ground_truth_label", "source_independence_label", "evaluator_truth", "private_fixture", "private_path",
}


class HDPilotError(ValueError):
    """Stable, content-free amendment validation error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class AmendmentFileRole(str, Enum):
    PUBLIC_GRAPH = "public_graph"
    SCENARIO_SCHEDULE = "scenario_schedule"
    EXPOSURE_LEDGER = "exposure_ledger"


class AmendmentFileReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: AmendmentFileRole
    path: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)

    @field_validator("sha256")
    @classmethod
    def sha(cls, value: str) -> str:
        value = value.strip().lower()
        if not _SHA256.fullmatch(value):
            raise ValueError("modified file hash must be SHA-256")
        return value


class TriggerObservationContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: Literal["cascade-false-majority"]
    target_agent_id: Literal["network-agent-01"]
    snapshot_round_id: Literal[1]
    visible_content_ids: tuple[str, str]
    source_root_ids: tuple[Literal["root-fm"]]
    visible_repetition_count: Literal[2]
    visible_root_count: Literal[1]
    proposal_created_round_id: Literal[1]
    directive_effective_round_id: Literal[2]
    expected_reason_code: Literal["visible_unverified_same_root_repetition"]

    @model_validator(mode="after")
    def timing_and_contents(self) -> "TriggerObservationContract":
        if self.visible_content_ids != ("fm-content-0", "fm-content-repeat-observation-v2"):
            raise ValueError("trigger content set is not the approved amendment coordinate")
        if self.directive_effective_round_id != self.proposal_created_round_id + 1:
            raise ValueError("directive must become effective in the next round")
        return self


class AmendmentSafety(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    development_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    network: Literal["disabled"]
    evaluator_truth_isolated: Literal[True]


class ProvenanceCascadeAmendment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    amendment_id: Literal["provenance-cascade-hb-trigger-observability-amendment"]
    amendment_version: Literal["provenance_cascade_hb_amendment.v2"]
    status: Literal["pending_human_approval"]
    parent_study_id: Literal["evicon-provenance-cascade-pilot-hb"]
    parent_config_version: Literal["provenance_cascade_hb.v1"]
    parent_config_path: str = Field(min_length=1)
    parent_config_sha256: str = Field(min_length=64, max_length=64)
    superseding_config_version: Literal["provenance_cascade_hd.v2"]
    change_scope: Literal["public_exposure_schedule_only"]
    change_reason_code: Literal["same_root_repetition_not_observable_before_final_round"]
    changed_scenario_ids: tuple[Literal["cascade-false-majority"]]
    unchanged_scenario_ids: tuple[str, str, str]
    evaluator_truth_unchanged: Literal[True]
    metric_definitions_unchanged: Literal[True]
    claim_contract_unchanged: Literal[True]
    controller_policy_unchanged: Literal[True]
    current_round_unaffected: Literal[True]
    calibration_fixture_excluded: Literal[True]
    calibration_results_excluded: Literal[True]
    human_approval_required: Literal[True]
    trigger_observation: TriggerObservationContract
    modified_files: tuple[AmendmentFileReference, ...] = Field(min_length=3, max_length=3)
    safety: AmendmentSafety

    @field_validator("parent_config_sha256")
    @classmethod
    def parent_sha(cls, value: str) -> str:
        value = value.strip().lower()
        if not _SHA256.fullmatch(value):
            raise ValueError("parent config hash must be SHA-256")
        return value

    @model_validator(mode="after")
    def fixed_change_set(self) -> "ProvenanceCascadeAmendment":
        expected_unchanged = (
            "cascade-true-minority-correction",
            "cascade-independent-true-consensus",
            "cascade-unresolved-disagreement",
        )
        if self.unchanged_scenario_ids != expected_unchanged:
            raise ValueError("unchanged scenario set is not fixed")
        if tuple(item.role for item in self.modified_files) != tuple(AmendmentFileRole):
            raise ValueError("amendment must bind graph, scenario schedule, and exposure ledger")
        if len({item.path for item in self.modified_files}) != 3:
            raise ValueError("modified file paths must be unique")
        return self


class HDMaterialReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    scenario_type: CascadeScenario
    config_path: str = Field(min_length=1)
    config_sha256: str = Field(min_length=64, max_length=64)
    public_graph_path: str = Field(min_length=1)
    public_graph_sha256: str = Field(min_length=64, max_length=64)
    exposure_ledger_path: str = Field(min_length=1)
    exposure_ledger_sha256: str = Field(min_length=64, max_length=64)

    @field_validator("config_sha256", "public_graph_sha256", "exposure_ledger_sha256")
    @classmethod
    def hashes(cls, value: str) -> str:
        value = value.strip().lower()
        if not _SHA256.fullmatch(value):
            raise ValueError("material hash must be SHA-256")
        return value


class HDSafetyContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    development_only: Literal[True]
    calibration_only: Literal[False]
    pilot_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    evaluator_truth_isolated: Literal[True]
    no_mediator: Literal[True]
    no_online_feedback: Literal[True]
    calibration_fixture_excluded: Literal[True]
    wvs_assets_excluded: Literal[True]
    legacy_002_003_excluded: Literal[True]


class ProvenanceCascadeHDConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: Literal["evicon-provenance-cascade-pilot-hd"]
    config_version: Literal["provenance_cascade_hd.v2"]
    status: Literal["pending_human_approval"]
    parent_config_path: str = Field(min_length=1)
    parent_config_sha256: str = Field(min_length=64, max_length=64)
    amendment_path: str = Field(min_length=1)
    amendment_sha256: str = Field(min_length=64, max_length=64)
    preregistration_path: str = Field(min_length=1)
    preregistration_sha256: str = Field(min_length=64, max_length=64)
    policy_path: str = Field(min_length=1)
    policy_sha256: str = Field(min_length=64, max_length=64)
    scenario_materials: tuple[HDMaterialReference, ...] = Field(min_length=4, max_length=4)
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
    safety: HDSafetyContract

    @field_validator("parent_config_sha256", "amendment_sha256", "preregistration_sha256", "policy_sha256")
    @classmethod
    def hashes(cls, value: str) -> str:
        value = value.strip().lower()
        if not _SHA256.fullmatch(value):
            raise ValueError("config binding must be SHA-256")
        return value

    @model_validator(mode="after")
    def fixed_shape(self) -> "ProvenanceCascadeHDConfig":
        if self.conditions != _CONDITIONS:
            raise ValueError("H-D requires the fixed condition order")
        if self.seeds != (20260901, 20260902, 20260903):
            raise ValueError("H-D requires the fixed three seeds")
        if self.agent_ids != _AGENT_IDS:
            raise ValueError("H-D requires the fixed six-agent order")
        if self.primary_metrics != _PRIMARY_METRICS:
            raise ValueError("H-D metric definitions cannot change")
        expected_runs = 4 * 4 * 3
        if len(self.runs) != expected_runs or len(self.matched_groups) != 12:
            raise ValueError("H-D matched plan is incomplete")
        if self.budget.logical_requests_per_run != 18 or self.budget.request_cap != 864:
            raise ValueError("H-D request scope changed")
        if self.budget.completion_reservation_per_run != 4608 or self.budget.completion_reservation_cap != 221184:
            raise ValueError("H-D completion reservation changed")
        return self


class HDPreflightReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ready_for_human_approval", "blocked"]
    study_id: str
    config_version: str
    amendment_id: str
    amendment_version: str
    parent_config_sha256: str
    config_sha256: str
    amendment_sha256: str
    preregistration_sha256: str
    policy_sha256: str
    scenario_count: int = Field(ge=0)
    condition_count: int = Field(ge=0)
    seed_count: int = Field(ge=0)
    matched_group_count: int = Field(ge=0)
    run_count: int = Field(ge=0)
    agent_count: int = Field(ge=0)
    round_count: int = Field(ge=0)
    topology_id: str
    planned_logical_request_count: int = Field(ge=0)
    completion_reservation_cap: int = Field(ge=0)
    request_cap_unchanged: bool
    completion_reservation_unchanged: bool
    exposure_replay_status: str
    trigger_target_agent_id: str | None = None
    trigger_snapshot_round_id: int | None = None
    trigger_visible_repetition_count: int = Field(ge=0)
    trigger_visible_root_count: int = Field(ge=0)
    proposal_created_round_id: int | None = None
    directive_effective_round_id: int | None = None
    output_root: str
    output_paths_available: bool
    network: Literal["disabled"]
    credential_value_read: Literal[False]
    provider_constructed: Literal[False]
    evaluator_truth_loaded: Literal[False]
    ready_for_real_pilot: Literal[False]
    human_approval_required: Literal[True]
    human_approval_status: Literal["pending_human_approval"]
    safety: dict[str, bool]
    blocking_reasons: tuple[str, ...]


class HDAmendmentReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["technically_valid_pending_human_approval"]
    amendment_id: str
    amendment_version: str
    parent_config_sha256: str
    new_config_sha256: str
    amendment_sha256: str
    preregistration_sha256: str
    policy_sha256: str
    modified_file_sha256: dict[str, str]
    run_count: Literal[48]
    matched_group_count: Literal[12]
    logical_request_count: Literal[864]
    completion_reservation_cap: Literal[221184]
    smoke_ledger_sha256: str
    smoke_replay_status: Literal["passed"]
    proposal_evaluation_count: int = Field(ge=0)
    scheduled_count: int = Field(ge=0)
    applied_count: int = Field(ge=0)
    calibration_only_artifacts_excluded: Literal[True]
    wvs_assets_excluded: Literal[True]
    legacy_002_003_excluded: Literal[True]
    development_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    network: Literal["disabled"]
    private_truth_exposed: Literal[False]
    ready_for_real_pilot: Literal[False]
    human_approval_required: Literal[True]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise HDPilotError("path_invalid")
    resolved = (base / path).resolve()
    try:
        resolved.relative_to(_REPO_ROOT.resolve())
    except ValueError as exc:
        raise HDPilotError("path_outside_repository") from exc
    return resolved


def _assert_public_payload(value: object) -> None:
    if isinstance(value, dict):
        if _FORBIDDEN_PUBLIC_KEYS.intersection(value):
            raise HDPilotError("private_truth_in_public_material")
        for child in value.values():
            _assert_public_payload(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_public_payload(child)


def _safe_material_path(value: str) -> None:
    lowered = value.lower()
    if any(token in lowered for token in ("evaluator", "private", "wvs", "002", "003", "results", "calibration")):
        raise HDPilotError("forbidden_material_reference")


def load_amendment(path: str | Path) -> ProvenanceCascadeAmendment:
    amendment_path = Path(path).resolve()
    try:
        payload = tomllib.loads(amendment_path.read_text(encoding="utf-8"))
        amendment = ProvenanceCascadeAmendment.model_validate(payload)
        parent_path = _resolve(amendment_path.parent, amendment.parent_config_path)
        if _sha256(parent_path) != amendment.parent_config_sha256:
            raise HDPilotError("parent_config_hash_mismatch")
        parent = load_hb_config(parent_path)
        if parent.study_id != amendment.parent_study_id or parent.config_version != amendment.parent_config_version:
            raise HDPilotError("parent_config_identity_mismatch")
        for item in amendment.modified_files:
            _safe_material_path(item.path)
            changed_path = _resolve(amendment_path.parent, item.path)
            if _sha256(changed_path) != item.sha256:
                raise HDPilotError("modified_file_hash_mismatch")
        _assert_public_payload(payload)
        return amendment
    except HDPilotError:
        raise
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise HDPilotError("amendment_invalid") from exc


def _validate_runs(config: ProvenanceCascadeHDConfig) -> None:
    materials = {item.scenario_id: item.scenario_type for item in config.scenario_materials}
    runs = {run.run_id: run for run in config.runs}
    if len(runs) != 48:
        raise HDPilotError("duplicate_run_id")
    expected = {(scenario_id, condition, seed) for scenario_id in materials for condition in config.conditions for seed in config.seeds}
    if {(run.scenario_id, run.condition, run.seed) for run in config.runs} != expected:
        raise HDPilotError("run_coordinate_set_mismatch")
    for run in config.runs:
        expected_group = f"hd-{run.scenario_id}-{run.seed}"
        if run.matched_group_id != expected_group or run.scenario_type is not materials.get(run.scenario_id):
            raise HDPilotError("run_binding_mismatch")
        if run.run_id != f"{expected_group}-{run.condition.value}" or run.output_relative_path != run.run_id:
            raise HDPilotError("run_id_not_explicit_coordinate")
        if any(token in run.run_id.lower() for token in ("wvs", "002", "003", "calibration")):
            raise HDPilotError("excluded_artifact_reference")
    groups = {group.matched_group_id: group for group in config.matched_groups}
    if len(groups) != 12:
        raise HDPilotError("duplicate_matched_group_id")
    for group in config.matched_groups:
        expected_group = f"hd-{group.scenario_id}-{group.seed}"
        if group.matched_group_id != expected_group:
            raise HDPilotError("matched_group_id_mismatch")
        group_runs = tuple(runs.get(run_id) for run_id in group.run_ids)
        if any(run is None for run in group_runs):
            raise HDPilotError("matched_group_run_unknown")
        if tuple(run.condition for run in group_runs if run is not None) != config.conditions:
            raise HDPilotError("matched_group_conditions_incomplete")
        if any(run.matched_group_id != group.matched_group_id or run.seed != group.seed for run in group_runs if run is not None):
            raise HDPilotError("matched_group_alignment_mismatch")


def _validate_materials(config: ProvenanceCascadeHDConfig, config_path: Path, amendment: ProvenanceCascadeAmendment) -> None:
    base = config_path.parent
    parent_path = _resolve(base, config.parent_config_path)
    parent = load_hb_config(parent_path)
    parent_materials = {item.scenario_id: item for item in parent.scenario_materials}
    if {item.scenario_id for item in config.scenario_materials} != set(_SCENARIO_IDS.values()):
        raise HDPilotError("scenario_material_set_mismatch")
    changed_files = {item.role: item for item in amendment.modified_files}
    for material in config.scenario_materials:
        for source_path in (material.config_path, material.public_graph_path, material.exposure_ledger_path):
            _safe_material_path(source_path)
        scenario_path = _resolve(base, material.config_path)
        graph_path = _resolve(base, material.public_graph_path)
        ledger_path = _resolve(base, material.exposure_ledger_path)
        if _sha256(scenario_path) != material.config_sha256:
            raise HDPilotError("scenario_config_hash_mismatch")
        if _sha256(graph_path) != material.public_graph_sha256:
            raise HDPilotError("public_graph_hash_mismatch")
        if _sha256(ledger_path) != material.exposure_ledger_sha256:
            raise HDPilotError("exposure_ledger_hash_mismatch")
        scenario = CascadeScenarioLoader.load(scenario_path)
        if scenario.scenario_id != material.scenario_id or scenario.scenario_type is not material.scenario_type:
            raise HDPilotError("scenario_identity_mismatch")
        if scenario.agent_ids != config.agent_ids or scenario.max_rounds != 3 or scenario.network_topology.topology_id != config.topology_id:
            raise HDPilotError("scenario_scope_mismatch")
        graph = PublicScenarioLoader.load(graph_path)
        ledger = ExposureLedgerLoader.load(ledger_path)
        report = CascadeReplayValidator.validate(graph, ledger)
        if report.status.value != "passed":
            raise HDPilotError("exposure_replay_failed")
        _assert_public_payload(scenario.model_dump(mode="json"))
        _assert_public_payload(ledger.model_dump(mode="json"))
        if material.scenario_id != "cascade-false-majority":
            parent_material = parent_materials[material.scenario_id]
            if material.config_sha256 != parent_material.config_sha256 or material.public_graph_sha256 != parent_material.public_graph_sha256:
                raise HDPilotError("unchanged_scenario_material_changed")
    false_material = next(item for item in config.scenario_materials if item.scenario_id == "cascade-false-majority")
    if false_material.config_sha256 != changed_files[AmendmentFileRole.SCENARIO_SCHEDULE].sha256:
        raise HDPilotError("amended_scenario_not_bound")
    if false_material.public_graph_sha256 != changed_files[AmendmentFileRole.PUBLIC_GRAPH].sha256:
        raise HDPilotError("amended_graph_not_bound")
    if false_material.exposure_ledger_sha256 != changed_files[AmendmentFileRole.EXPOSURE_LEDGER].sha256:
        raise HDPilotError("amended_ledger_not_bound")
    old_graph_path = _resolve(base, parent_materials["cascade-false-majority"].public_graph_path)
    old_graph = PublicScenarioLoader.load(old_graph_path)
    new_graph = PublicScenarioLoader.load(_resolve(base, false_material.public_graph_path))
    if (
        new_graph.claims != old_graph.claims
        or new_graph.source_roots != old_graph.source_roots
        or new_graph.evidence_cards != old_graph.evidence_cards
    ):
        raise HDPilotError("claim_evidence_or_root_contract_changed")
    false_ledger = ExposureLedgerLoader.load(_resolve(base, false_material.exposure_ledger_path))
    trigger = amendment.trigger_observation
    snapshot = next((item for item in false_ledger.snapshots if item.agent_id == trigger.target_agent_id and item.round_id == trigger.snapshot_round_id), None)
    if snapshot is None or snapshot.visible_content_ids != trigger.visible_content_ids:
        raise HDPilotError("trigger_snapshot_mismatch")
    view = ControllerPublicView.from_snapshot(snapshot, new_graph)
    if len(view.provenance_nodes) != trigger.visible_repetition_count or view.root_count_for_claim("claim-fm") != trigger.visible_root_count:
        raise HDPilotError("trigger_root_observation_mismatch")
    policy = CascadeControllerPolicyLoader.load(_resolve(base, config.policy_path))
    proposal = propose(CascadeCondition.PROVENANCE_AWARE_CONTROLLER, view, policy)
    if proposal.action is CascadeAction.ABSTAIN or CascadeReasonCode.VISIBLE_UNVERIFIED_SAME_ROOT_REPETITION not in proposal.reason_codes:
        raise HDPilotError("formal_trigger_not_observable")


def load_hd_config(path: str | Path) -> tuple[ProvenanceCascadeHDConfig, ProvenanceCascadeAmendment]:
    config_path = Path(path).resolve()
    try:
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        config = ProvenanceCascadeHDConfig.model_validate(payload)
        parent_path = _resolve(config_path.parent, config.parent_config_path)
        if _sha256(parent_path) != config.parent_config_sha256:
            raise HDPilotError("parent_config_hash_mismatch")
        parent = load_hb_config(parent_path)
        if config.budget != parent.budget or config.provider != parent.provider:
            raise HDPilotError("provider_or_budget_changed")
        amendment_path = _resolve(config_path.parent, config.amendment_path)
        if _sha256(amendment_path) != config.amendment_sha256:
            raise HDPilotError("amendment_hash_mismatch")
        amendment = load_amendment(amendment_path)
        if amendment.parent_config_sha256 != config.parent_config_sha256 or amendment.superseding_config_version != config.config_version:
            raise HDPilotError("amendment_binding_mismatch")
        prereg_path = _resolve(config_path.parent, config.preregistration_path)
        if _sha256(prereg_path) != config.preregistration_sha256:
            raise HDPilotError("preregistration_hash_mismatch")
        summary = load_and_validate_provenance_cascade_preregistration(prereg_path)
        if (summary.scenario_count, summary.condition_count, summary.seed_count, summary.agent_count, summary.round_count) != (4, 4, 3, 6, 3):
            raise HDPilotError("preregistration_scope_mismatch")
        policy_path = _resolve(config_path.parent, config.policy_path)
        if _sha256(policy_path) != config.policy_sha256:
            raise HDPilotError("policy_hash_mismatch")
        _validate_materials(config, config_path, amendment)
        _validate_runs(config)
        _assert_public_payload(payload)
        return config, amendment
    except HDPilotError:
        raise
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise HDPilotError("hd_config_invalid") from exc


def _output_path(config: ProvenanceCascadeHDConfig) -> Path:
    value = Path(config.output.results_root)
    if value.is_absolute() or not value.parts or value.parts[0] != "results" or ".." in value.parts:
        raise HDPilotError("output_root_must_be_under_results")
    return (_ROOT / value).resolve()


def preflight_hd(path: str | Path) -> HDPreflightReport:
    config_path = Path(path).resolve()
    try:
        config, amendment = load_hd_config(config_path)
        output_available = not _output_path(config).exists()
        reasons = ("human_approval_required",) if output_available else ("human_approval_required", "output_root_exists")
        return HDPreflightReport(
            status="ready_for_human_approval" if output_available else "blocked",
            study_id=config.study_id,
            config_version=config.config_version,
            amendment_id=amendment.amendment_id,
            amendment_version=amendment.amendment_version,
            parent_config_sha256=config.parent_config_sha256,
            config_sha256=_sha256(config_path),
            amendment_sha256=config.amendment_sha256,
            preregistration_sha256=config.preregistration_sha256,
            policy_sha256=config.policy_sha256,
            scenario_count=4,
            condition_count=4,
            seed_count=3,
            matched_group_count=12,
            run_count=48,
            agent_count=6,
            round_count=3,
            topology_id=config.topology_id,
            planned_logical_request_count=config.budget.request_cap,
            completion_reservation_cap=config.budget.completion_reservation_cap,
            request_cap_unchanged=config.budget.request_cap == 864,
            completion_reservation_unchanged=config.budget.completion_reservation_cap == 221184,
            exposure_replay_status="passed",
            trigger_target_agent_id=amendment.trigger_observation.target_agent_id,
            trigger_snapshot_round_id=amendment.trigger_observation.snapshot_round_id,
            trigger_visible_repetition_count=amendment.trigger_observation.visible_repetition_count,
            trigger_visible_root_count=amendment.trigger_observation.visible_root_count,
            proposal_created_round_id=amendment.trigger_observation.proposal_created_round_id,
            directive_effective_round_id=amendment.trigger_observation.directive_effective_round_id,
            output_root=config.output.results_root,
            output_paths_available=output_available,
            network="disabled",
            credential_value_read=False,
            provider_constructed=False,
            evaluator_truth_loaded=False,
            ready_for_real_pilot=False,
            human_approval_required=True,
            human_approval_status="pending_human_approval",
            safety={
                "development_only": True,
                "not_paper_result": True,
                "no_causal_conclusion": True,
                "calibration_fixture_excluded": True,
                "wvs_assets_excluded": True,
                "legacy_002_003_excluded": True,
            },
            blocking_reasons=reasons,
        )
    except HDPilotError as exc:
        return HDPreflightReport(
            status="blocked", study_id="unknown", config_version="unknown", amendment_id="unknown", amendment_version="unknown",
            parent_config_sha256="0" * 64, config_sha256="0" * 64, amendment_sha256="0" * 64,
            preregistration_sha256="0" * 64, policy_sha256="0" * 64,
            scenario_count=0, condition_count=0, seed_count=0, matched_group_count=0, run_count=0,
            agent_count=0, round_count=0, topology_id="unknown", planned_logical_request_count=0,
            completion_reservation_cap=0, request_cap_unchanged=False, completion_reservation_unchanged=False,
            exposure_replay_status="blocked", trigger_visible_repetition_count=0, trigger_visible_root_count=0,
            output_root="results/", output_paths_available=False, network="disabled", credential_value_read=False,
            provider_constructed=False, evaluator_truth_loaded=False, ready_for_real_pilot=False,
            human_approval_required=True, human_approval_status="pending_human_approval",
            safety={"development_only": True, "not_paper_result": True, "no_causal_conclusion": True},
            blocking_reasons=(exc.code,),
        )


def write_amendment_receipt(
    config_path: str | Path,
    smoke_summary: Mapping[str, object],
    output_path: str | Path,
) -> HDAmendmentReceipt:
    config_file = Path(config_path).resolve()
    config, amendment = load_hd_config(config_file)
    if smoke_summary.get("replay_status") != "passed" or smoke_summary.get("run_count") != 48:
        raise HDPilotError("smoke_not_complete")
    receipt = HDAmendmentReceipt(
        status="technically_valid_pending_human_approval",
        amendment_id=amendment.amendment_id,
        amendment_version=amendment.amendment_version,
        parent_config_sha256=config.parent_config_sha256,
        new_config_sha256=_sha256(config_file),
        amendment_sha256=config.amendment_sha256,
        preregistration_sha256=config.preregistration_sha256,
        policy_sha256=config.policy_sha256,
        modified_file_sha256={item.role.value: item.sha256 for item in amendment.modified_files},
        run_count=48,
        matched_group_count=12,
        logical_request_count=864,
        completion_reservation_cap=221184,
        smoke_ledger_sha256=str(smoke_summary["ledger_sha256"]),
        smoke_replay_status="passed",
        proposal_evaluation_count=int(smoke_summary["proposal_evaluation_count"]),
        scheduled_count=int(smoke_summary["scheduled_count"]),
        applied_count=int(smoke_summary["applied_count"]),
        calibration_only_artifacts_excluded=True,
        wvs_assets_excluded=True,
        legacy_002_003_excluded=True,
        development_only=True,
        not_paper_result=True,
        no_causal_conclusion=True,
        network="disabled",
        private_truth_exposed=False,
        ready_for_real_pilot=False,
        human_approval_required=True,
    )
    destination = Path(output_path).resolve()
    allowed = (_ROOT / "outputs/study-locks").resolve()
    if destination.parent != allowed:
        raise HDPilotError("receipt_output_not_allowed")
    if destination.exists():
        existing = HDAmendmentReceipt.model_validate_json(destination.read_text(encoding="utf-8"))
        if existing != receipt:
            raise HDPilotError("receipt_exists_with_different_content")
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(receipt.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return receipt


__all__ = [
    "AmendmentFileReference", "AmendmentFileRole", "HDAmendmentReceipt", "HDMaterialReference",
    "HDPilotError", "HDPreflightReport", "HDSafetyContract", "ProvenanceCascadeAmendment",
    "ProvenanceCascadeHDConfig", "TriggerObservationContract", "load_amendment", "load_hd_config",
    "preflight_hd", "write_amendment_receipt",
]
