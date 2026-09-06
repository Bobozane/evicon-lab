"""Offline preregistration contracts for provenance-aware cascade pilots.

This module is intentionally a planning boundary.  It validates a local TOML
record and never constructs a provider, reads credentials, or creates results.
Evaluator-only truth labels are represented as names in the boundary contract,
not as values available to any controller.
"""

from __future__ import annotations

import tomllib
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .models._validation import identifier_list, normalized_text


class CascadeScenario(str, Enum):
    FALSE_MAJORITY = "false_majority"
    TRUE_MINORITY_CORRECTION = "true_minority_correction"
    INDEPENDENT_TRUE_CONSENSUS = "independent_true_consensus"
    UNRESOLVED_DISAGREEMENT = "unresolved_disagreement"


class CascadeCondition(str, Enum):
    NO_INTERVENTION = "no_intervention"
    GENERIC_DISSENT = "generic_dissent"
    SOURCE_BLIND_CONTROLLER = "source_blind_controller"
    PROVENANCE_AWARE_CONTROLLER = "provenance_aware_controller"


class DefaultControllerBehavior(str, Enum):
    CONSTRAIN_FALSE_CASCADE = "constrain_false_cascade"
    PROTECT_SUPPORTED_CORRECTION = "protect_supported_correction"
    ABSTAIN = "abstain"


class ControllerInputProfile(str, Enum):
    NONE = "none"
    PUBLIC_STATE = "public_state"
    PUBLIC_STATE_WITHOUT_SOURCE_ROOTS = "public_state_without_source_roots"
    PUBLIC_STATE_WITH_SOURCE_ROOTS = "public_state_with_source_roots"


class SafeStudyMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    language: Literal["en"]
    material_status: Literal["pilot"]
    source_kind: Literal["author_original"]
    synthetic_only: Literal[True]
    not_real_public_opinion: Literal[True]
    license_note: str = Field(min_length=1)

    @field_validator("license_note")
    @classmethod
    def clean_note(cls, value: str) -> str:
        return normalized_text(value, "license_note")


class ScenarioDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_type: CascadeScenario
    public_description: str = Field(min_length=1)
    default_controller_behavior: DefaultControllerBehavior
    evaluator_annotation_required: Literal[True]

    @field_validator("public_description")
    @classmethod
    def clean_description(cls, value: str) -> str:
        return normalized_text(value, "public_description")

    @model_validator(mode="after")
    def expected_default(self) -> "ScenarioDeclaration":
        expected = {
            CascadeScenario.FALSE_MAJORITY: DefaultControllerBehavior.CONSTRAIN_FALSE_CASCADE,
            CascadeScenario.TRUE_MINORITY_CORRECTION: DefaultControllerBehavior.PROTECT_SUPPORTED_CORRECTION,
            CascadeScenario.INDEPENDENT_TRUE_CONSENSUS: DefaultControllerBehavior.ABSTAIN,
            CascadeScenario.UNRESOLVED_DISAGREEMENT: DefaultControllerBehavior.ABSTAIN,
        }[self.scenario_type]
        if self.default_controller_behavior is not expected:
            raise ValueError("scenario default behavior does not match its preregistered safety rule")
        return self


class ConditionDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: CascadeCondition
    controller_input_profile: ControllerInputProfile
    uses_evaluator_truth: Literal[False]
    triggers_on_majority_count: Literal[False]
    triggers_on_minority_count: Literal[False]

    @model_validator(mode="after")
    def expected_profile(self) -> "ConditionDeclaration":
        expected = {
            CascadeCondition.NO_INTERVENTION: ControllerInputProfile.NONE,
            CascadeCondition.GENERIC_DISSENT: ControllerInputProfile.PUBLIC_STATE,
            CascadeCondition.SOURCE_BLIND_CONTROLLER: ControllerInputProfile.PUBLIC_STATE_WITHOUT_SOURCE_ROOTS,
            CascadeCondition.PROVENANCE_AWARE_CONTROLLER: ControllerInputProfile.PUBLIC_STATE_WITH_SOURCE_ROOTS,
        }[self.condition]
        if self.controller_input_profile is not expected:
            raise ValueError("condition input profile does not match its preregistered boundary")
        return self


class NetworkTopology(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    topology_id: Literal["ring_6_bidirectional"]
    directed: Literal[False]
    agent_ids: list[str] = Field(min_length=6, max_length=6)
    edges: list[tuple[str, str]] = Field(min_length=6, max_length=6)

    @field_validator("agent_ids")
    @classmethod
    def topology_agents(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "agent_ids")

    @model_validator(mode="after")
    def ring_shape(self) -> "NetworkTopology":
        if len(set(self.agent_ids)) != 6 or any(a == b or a not in self.agent_ids or b not in self.agent_ids for a, b in self.edges):
            raise ValueError("topology edges must be six unique non-self edges over the declared agents")
        normalized = {frozenset(edge) for edge in self.edges}
        expected = {
            frozenset((self.agent_ids[index], self.agent_ids[(index + 1) % 6]))
            for index in range(6)
        }
        if normalized != expected or len(normalized) != 6:
            raise ValueError("topology must be the declared six-agent ring")
        return self


class InformationBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluator_only_fields: list[str]
    controller_allowed_public_fields: list[str]
    forbidden_controller_fields: list[str]
    labels_never_in_prompt_or_events: Literal[True]

    @model_validator(mode="after")
    def boundary_shape(self) -> "InformationBoundary":
        if self.evaluator_only_fields != ["ground_truth_label", "source_independence_label"]:
            raise ValueError("evaluator-only labels must be explicitly declared")
        if self.controller_allowed_public_fields != [
            "public_verification_state", "public_evidence", "source_root_relations", "actual_exposure"
        ]:
            raise ValueError("controller public inputs must be explicitly limited")
        if self.forbidden_controller_fields != [
            "ground_truth_label", "source_independence_label", "evaluator_annotation", "private_probe_answer"
        ]:
            raise ValueError("controller forbidden fields must include evaluator-only data")
        return self


class SuccessCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_goal: Literal["reduce_false_cascade_without_material_correction_retention_loss"]
    false_cascade_must_decrease: Literal[True]
    supported_correction_retention_must_not_be_materially_harmed: Literal[True]
    abstention_is_correct_for: list[CascadeScenario]
    no_unregistered_thresholds: Literal[True]

    @model_validator(mode="after")
    def abstention_shape(self) -> "SuccessCriterion":
        if self.abstention_is_correct_for != [
            CascadeScenario.INDEPENDENT_TRUE_CONSENSUS,
            CascadeScenario.UNRESOLVED_DISAGREEMENT,
        ]:
            raise ValueError("the two uncertainty scenarios must explicitly default to abstain")
        return self


_SCENARIOS = [scenario.value for scenario in CascadeScenario]
_CONDITIONS = [condition.value for condition in CascadeCondition]
_METRICS = [
    "false_cascade_adoption_rate",
    "supported_correction_retention",
    "beneficial_receptivity",
    "harmful_conformity",
    "intervention_false_positive_rate",
    "provenance_diversity",
    "replay_audit_pass_rate",
    "token_cost",
    "latency_ms",
]


class ProvenanceCascadePilotPreregistration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: str = Field(min_length=1)
    preregistration_version: Literal["provenance_cascade_pilot.v1"]
    status: Literal["pilot_preregistration"]
    development_only: Literal[True]
    not_paper_result: Literal[True]
    scenario_types: list[ScenarioDeclaration] = Field(min_length=4, max_length=4)
    conditions: list[ConditionDeclaration] = Field(min_length=4, max_length=4)
    agent_ids: list[str] = Field(min_length=6, max_length=6)
    max_rounds: Literal[3]
    seeds: list[int] = Field(min_length=3, max_length=3)
    initial_network_topology: NetworkTopology
    primary_metrics: list[str] = Field(min_length=9, max_length=9)
    information_boundary: InformationBoundary
    success_criterion: SuccessCriterion
    metadata: SafeStudyMetadata

    @field_validator("study_id")
    @classmethod
    def clean_id(cls, value: str) -> str:
        return normalized_text(value, "study_id")

    @field_validator("agent_ids")
    @classmethod
    def agents(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "agent_ids")

    @field_validator("seeds")
    @classmethod
    def seeds_unique(cls, value: list[int]) -> list[int]:
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0 for seed in value):
            raise ValueError("seeds must be positive integers")
        if len(set(value)) != len(value):
            raise ValueError("seeds must be unique")
        return value

    @field_validator("primary_metrics")
    @classmethod
    def metrics(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "primary_metrics")

    @model_validator(mode="after")
    def fixed_shape(self) -> "ProvenanceCascadePilotPreregistration":
        if [item.scenario_type.value for item in self.scenario_types] != _SCENARIOS:
            raise ValueError("scenario_types must contain the four fixed cascade scenarios in order")
        if [item.condition.value for item in self.conditions] != _CONDITIONS:
            raise ValueError("conditions must contain the four fixed conditions in order")
        if self.initial_network_topology.agent_ids != self.agent_ids:
            raise ValueError("network topology agent IDs must match the study agent IDs")
        if self.primary_metrics != _METRICS:
            raise ValueError("primary_metrics must use the fixed provenance-cascade metric suite")
        return self


class ProvenanceCascadePilotSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: str
    status: str
    template_version: str
    scenario_count: int
    condition_count: int
    agent_count: int
    round_count: int
    seed_count: int
    planned_logical_run_count: int
    primary_metrics: list[str]
    topology_id: str
    evaluator_labels_isolated: bool
    controller_public_inputs_declared: bool
    abstain_scenarios: list[str]
    runnable_offline: bool


class ProvenanceCascadePreregistrationError(ValueError):
    """Stable, safe local validation error."""


class ProvenanceCascadePreregistrationLoader:
    @staticmethod
    def load(path: str | Path) -> ProvenanceCascadePilotPreregistration:
        try:
            payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ProvenanceCascadePreregistrationError("cannot read provenance-cascade preregistration config") from exc
        try:
            return ProvenanceCascadePilotPreregistration.model_validate(payload)
        except ValidationError as exc:
            raise ProvenanceCascadePreregistrationError("provenance-cascade preregistration violates its fixed contract") from exc


def validate_provenance_cascade_preregistration(
    preregistration: ProvenanceCascadePilotPreregistration,
) -> ProvenanceCascadePilotSummary:
    return ProvenanceCascadePilotSummary(
        study_id=preregistration.study_id,
        status=preregistration.status,
        template_version=preregistration.preregistration_version,
        scenario_count=len(preregistration.scenario_types),
        condition_count=len(preregistration.conditions),
        agent_count=len(preregistration.agent_ids),
        round_count=preregistration.max_rounds,
        seed_count=len(preregistration.seeds),
        planned_logical_run_count=len(preregistration.seeds) * len(preregistration.conditions) * len(preregistration.scenario_types),
        primary_metrics=list(preregistration.primary_metrics),
        topology_id=preregistration.initial_network_topology.topology_id,
        evaluator_labels_isolated=preregistration.information_boundary.labels_never_in_prompt_or_events is True,
        controller_public_inputs_declared=True,
        abstain_scenarios=[scenario.scenario_type.value for scenario in preregistration.scenario_types if scenario.default_controller_behavior is DefaultControllerBehavior.ABSTAIN],
        runnable_offline=True,
    )


def load_and_validate_provenance_cascade_preregistration(path: str | Path) -> ProvenanceCascadePilotSummary:
    return validate_provenance_cascade_preregistration(ProvenanceCascadePreregistrationLoader.load(path))


__all__ = [
    "CascadeCondition", "CascadeScenario", "ConditionDeclaration", "ControllerInputProfile",
    "DefaultControllerBehavior", "InformationBoundary", "NetworkTopology", "ProvenanceCascadePilotPreregistration",
    "ProvenanceCascadePilotSummary", "ProvenanceCascadePreregistrationError", "ProvenanceCascadePreregistrationLoader",
    "SafeStudyMetadata", "ScenarioDeclaration", "SuccessCriterion", "load_and_validate_provenance_cascade_preregistration",
    "validate_provenance_cascade_preregistration",
]
