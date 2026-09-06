"""Offline contracts for a provenance-cascade conformity identification study."""
from __future__ import annotations

import hashlib
import json
import tomllib
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/provenance_cascade/identification/conformity_identification_study.v1.toml"


class IdentificationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class IdentificationCondition(str, Enum):
    PRIVATE_BASELINE = "private_baseline"
    SELF_REFLECTION = "self_reflection"
    SOURCE_FREE_REPETITION = "source_free_repetition"
    SAME_ROOT_SOCIAL = "same_root_social"
    INDEPENDENT_ROOTS = "independent_roots"
    VERIFIED_EVIDENCE = "verified_evidence"


class IdentificationStage(str, Enum):
    INITIAL_PRIVATE = "initial_private"
    SELF_REFLECTION = "self_reflection"
    SOCIAL_EXPOSURE = "social_exposure"
    CORRECTION_EVIDENCE = "correction_evidence"
    FINAL_JUDGMENT = "final_judgment"


class ScenarioType(str, Enum):
    FALSE_MAJORITY = "false_majority"
    TRUE_MINORITY_CORRECTION = "true_minority_correction"
    INDEPENDENT_TRUE_CONSENSUS = "independent_true_consensus"
    UNRESOLVED_DISAGREEMENT = "unresolved_disagreement"


class OrdinalJudgment(str, Enum):
    STRONGLY_REJECT = "strongly_reject"
    LEAN_REJECT = "lean_reject"
    UNCERTAIN = "uncertain"
    LEAN_ADOPT = "lean_adopt"
    STRONGLY_ADOPT = "strongly_adopt"


class PublicAction(str, Enum):
    ADOPT = "adopt"
    REJECT = "reject"
    WITHHOLD = "withhold"


class ShareDecision(str, Enum):
    SHARE = "share"
    SHARE_WITH_CAVEAT = "share_with_caveat"
    DO_NOT_SHARE = "do_not_share"


class PublicStimulus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    stimulus_id: str = Field(min_length=1)
    role: Literal["initial", "repeat_a", "repeat_b", "correction", "evidence"]
    public_summary: str = Field(min_length=24, max_length=360)
    source_root_id: str | None = None
    independent_source_root_id: str | None = None
    evidence_id: str | None = None

    @field_validator("public_summary")
    @classmethod
    def public_only(cls, value: str) -> str:
        lowered = value.lower()
        forbidden = ("ground_truth_label", "source_independence_label", "evaluator_private", "correct_answer")
        if any(token in lowered for token in forbidden):
            raise ValueError("private evaluator label in public material")
        return value.strip()


class IdentificationScenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_id: str = Field(min_length=1)
    scenario_type: ScenarioType
    version: Literal["conformity_identification_scenario.v1"]
    language: Literal["en"]
    status: Literal["development_only"]
    source_kind: Literal["author_original"]
    license_notes: str = Field(min_length=1)
    synthetic_disclaimer: Literal[True]
    target_claim_id: str = Field(min_length=1)
    decision_task: str = Field(min_length=24, max_length=360)
    stimuli: tuple[PublicStimulus, ...] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def material_shape(self) -> "IdentificationScenario":
        if len({item.stimulus_id for item in self.stimuli}) != len(self.stimuli):
            raise ValueError("duplicate stimulus ID")
        if Counter(item.role for item in self.stimuli) != Counter(
            {"initial": 1, "repeat_a": 1, "repeat_b": 1, "correction": 1, "evidence": 1}
        ):
            raise ValueError("scenario must define the five fixed public stimulus roles")
        repeats = [item for item in self.stimuli if item.role in {"repeat_a", "repeat_b"}]
        if any(item.source_root_id is None or item.independent_source_root_id is None for item in repeats):
            raise ValueError("repeat stimuli require same-root and independent-root projections")
        if len({item.independent_source_root_id for item in repeats}) != 2:
            raise ValueError("independent projection requires two distinct roots")
        evidence = next(item for item in self.stimuli if item.role == "evidence")
        if evidence.evidence_id is None:
            raise ValueError("evidence stimulus requires evidence ID")
        return self

    @property
    def same_root_repeat_count(self) -> int:
        roots = [item.source_root_id for item in self.stimuli if item.role in {"repeat_a", "repeat_b"}]
        return len(set(roots))

    @property
    def independent_repeat_count(self) -> int:
        roots = [item.independent_source_root_id for item in self.stimuli if item.role in {"repeat_a", "repeat_b"}]
        return len(set(roots))


class ScenarioBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_id: str
    scenario_type: ScenarioType
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class IdentificationContrast(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    contrast_id: Literal["social_conformity_increment", "evidence_receptivity_increment"]
    minuend: IdentificationCondition
    subtrahend: IdentificationCondition
    unit: Literal["scenario_seed_agent"]


_CONDITIONS = tuple(IdentificationCondition)
_STAGES = tuple(IdentificationStage)
_SCENARIOS = tuple(ScenarioType)
_METRICS = (
    "initial_substantive_judgment_rate",
    "ordinal_judgment_shift",
    "public_adoption_rate",
    "public_share_rate",
    "social_conformity_increment",
    "evidence_receptivity_increment",
)


class ConformityIdentificationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    study_id: Literal["evicon-provenance-cascade-conformity-identification-v1"]
    config_version: Literal["conformity_identification_study.v1"]
    protocol_version: Literal["conformity_identification_protocol.v1"]
    template_version: Literal["conformity_identification_turn.v1"]
    protocol_path: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["offline_design_validation"]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    network_enabled: Literal[False]
    controller_enabled: Literal[False]
    intervention_enabled: Literal[False]
    evaluator_truth_available_to_runtime: Literal[False]
    conditions: tuple[IdentificationCondition, ...]
    stages: tuple[IdentificationStage, ...]
    scenario_types: tuple[ScenarioType, ...]
    seeds: tuple[int, ...] = Field(min_length=3, max_length=3)
    agent_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    topology_id: Literal["ring_6_bidirectional"]
    scenario_bindings: tuple[ScenarioBinding, ...] = Field(min_length=4, max_length=4)
    primary_metrics: tuple[str, ...]
    contrasts: tuple[IdentificationContrast, ...] = Field(min_length=2, max_length=2)
    ordinal_scale: tuple[OrdinalJudgment, ...]
    logical_requests_per_run: Literal[30]
    agent_max_tokens: Literal[512]
    run_count: Literal[72]
    matched_group_count: Literal[12]
    logical_request_cap: Literal[2160]
    completion_reservation_cap: Literal[1105920]
    output_root: Literal["results/provenance-cascade-conformity-identification-v1"]
    old_results_excluded: Literal[True]
    no_unregistered_thresholds: Literal[True]

    @model_validator(mode="after")
    def frozen_design(self) -> "ConformityIdentificationConfig":
        if self.conditions != _CONDITIONS:
            raise ValueError("six identification conditions must be fixed and ordered")
        if self.stages != _STAGES:
            raise ValueError("five identification stages must be fixed and ordered")
        if self.scenario_types != _SCENARIOS:
            raise ValueError("four scenario types must be fixed and ordered")
        if self.ordinal_scale != tuple(OrdinalJudgment):
            raise ValueError("ordinal response scale must be fixed and ordered")
        if self.primary_metrics != _METRICS:
            raise ValueError("identification metrics must be fixed and ordered")
        if len(set(self.seeds)) != 3 or len(set(self.agent_ids)) != 6:
            raise ValueError("seeds and agents must be unique")
        if len({item.scenario_id for item in self.scenario_bindings}) != 4:
            raise ValueError("scenario bindings must be unique")
        if tuple(item.scenario_type for item in self.scenario_bindings) != self.scenario_types:
            raise ValueError("scenario binding types must match the fixed order")
        if self.run_count != len(self.scenario_types) * len(self.seeds) * len(self.conditions):
            raise ValueError("run count mismatch")
        if self.matched_group_count != len(self.scenario_types) * len(self.seeds):
            raise ValueError("matched group count mismatch")
        if self.logical_request_cap != self.run_count * self.logical_requests_per_run:
            raise ValueError("logical request cap mismatch")
        if self.completion_reservation_cap != self.logical_request_cap * self.agent_max_tokens:
            raise ValueError("completion reservation mismatch")
        contrasts = {item.contrast_id: item for item in self.contrasts}
        social = contrasts.get("social_conformity_increment")
        evidence = contrasts.get("evidence_receptivity_increment")
        if social is None or (social.minuend, social.subtrahend) != (
            IdentificationCondition.SAME_ROOT_SOCIAL, IdentificationCondition.SOURCE_FREE_REPETITION
        ):
            raise ValueError("social conformity contrast mismatch")
        if evidence is None or (evidence.minuend, evidence.subtrahend) != (
            IdentificationCondition.VERIFIED_EVIDENCE, IdentificationCondition.SELF_REFLECTION
        ):
            raise ValueError("evidence receptivity contrast mismatch")
        return self


class IdentificationObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_id: str
    seed: int
    condition: IdentificationCondition
    agent_id: str
    stage: IdentificationStage
    judgment: OrdinalJudgment
    action: PublicAction
    share: ShareDecision
    confidence: int = Field(ge=1, le=7)
    visible_stimulus_ids: tuple[str, ...]


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def load_scenario(path: str | Path) -> IdentificationScenario:
    resolved = _resolve(path)
    if not resolved.is_file():
        raise IdentificationError("scenario_file_missing")
    try:
        return IdentificationScenario.model_validate(tomllib.loads(resolved.read_text(encoding="utf-8")))
    except Exception as exc:
        raise IdentificationError("scenario_invalid") from exc


def load_identification_config(path: str | Path = DEFAULT_CONFIG) -> tuple[ConformityIdentificationConfig, tuple[IdentificationScenario, ...]]:
    resolved = _resolve(path)
    if not resolved.is_file():
        raise IdentificationError("config_missing")
    try:
        config = ConformityIdentificationConfig.model_validate(tomllib.loads(resolved.read_text(encoding="utf-8")))
    except Exception as exc:
        raise IdentificationError("config_invalid") from exc
    if sha256_file(config.protocol_path) != config.protocol_sha256:
        raise IdentificationError("protocol_hash_mismatch")
    scenarios: list[IdentificationScenario] = []
    for binding in config.scenario_bindings:
        scenario_path = _resolve(binding.path)
        if sha256_file(scenario_path) != binding.sha256:
            raise IdentificationError("scenario_hash_mismatch")
        scenario = load_scenario(scenario_path)
        if (scenario.scenario_id, scenario.scenario_type) != (binding.scenario_id, binding.scenario_type):
            raise IdentificationError("scenario_binding_mismatch")
        scenarios.append(scenario)
    return config, tuple(scenarios)


def visible_roles(condition: IdentificationCondition, stage: IdentificationStage) -> tuple[str, ...]:
    roles: list[str] = ["initial"]
    if stage in {IdentificationStage.SOCIAL_EXPOSURE, IdentificationStage.CORRECTION_EVIDENCE, IdentificationStage.FINAL_JUDGMENT}:
        if condition in {
            IdentificationCondition.SOURCE_FREE_REPETITION,
            IdentificationCondition.SAME_ROOT_SOCIAL,
            IdentificationCondition.INDEPENDENT_ROOTS,
        }:
            roles.extend(("repeat_a", "repeat_b"))
    if stage in {IdentificationStage.CORRECTION_EVIDENCE, IdentificationStage.FINAL_JUDGMENT}:
        if condition is IdentificationCondition.VERIFIED_EVIDENCE:
            roles.extend(("correction", "evidence"))
    return tuple(roles)


def validate_matched_inputs(config: ConformityIdentificationConfig, scenarios: tuple[IdentificationScenario, ...]) -> None:
    for scenario in scenarios:
        by_role = {item.role: item for item in scenario.stimuli}
        if len({by_role[role].public_summary for role in ("repeat_a", "repeat_b")}) != 2:
            raise IdentificationError("repeat_texts_not_distinct")
        if scenario.same_root_repeat_count != 1:
            raise IdentificationError("same_root_pair_not_single_root")
        if scenario.independent_repeat_count != 2:
            raise IdentificationError("independent_pair_not_two_roots")
        for stage in IdentificationStage:
            arms = {
                visible_roles(condition, stage)
                for condition in (
                    IdentificationCondition.SOURCE_FREE_REPETITION,
                    IdentificationCondition.SAME_ROOT_SOCIAL,
                    IdentificationCondition.INDEPENDENT_ROOTS,
                )
            }
            if len(arms) != 1:
                raise IdentificationError("repetition_arm_timing_mismatch")


def safe_preflight(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    try:
        config, scenarios = load_identification_config(path)
        validate_matched_inputs(config, scenarios)
        return {
            "status": "offline_design_ready",
            "ready_for_network": False,
            "blocking_reasons": ["human_approval_required", "provider_compatibility_not_requested"],
            "study_id": config.study_id,
            "config_sha256": sha256_file(path),
            "scenario_count": len(scenarios),
            "condition_count": len(config.conditions),
            "stage_count": len(config.stages),
            "run_count": config.run_count,
            "matched_group_count": config.matched_group_count,
            "logical_request_cap": config.logical_request_cap,
            "completion_reservation_cap": config.completion_reservation_cap,
            "primary_metrics": list(config.primary_metrics),
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "private_truth_exposed": False,
            "effect_estimated": False,
            "not_paper_result": True,
            "no_causal_conclusion": True,
        }
    except Exception as exc:
        return {
            "status": "blocked",
            "ready_for_network": False,
            "blocking_reasons": [getattr(exc, "code", "identification_preflight_failed")],
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "private_truth_exposed": False,
        }


def safe_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True)


__all__ = [
    "ConformityIdentificationConfig", "IdentificationCondition", "IdentificationContrast",
    "IdentificationError", "IdentificationObservation", "IdentificationScenario",
    "IdentificationStage", "OrdinalJudgment", "PublicAction", "ShareDecision",
    "DEFAULT_CONFIG", "load_identification_config", "safe_preflight", "sha256_file",
    "validate_matched_inputs", "visible_roles",
]
