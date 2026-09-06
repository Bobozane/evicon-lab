"""Frozen offline analysis plan for Conformity Identification Study v1.1."""
from __future__ import annotations

import json
import tomllib
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_identification import IdentificationCondition, IdentificationError, OrdinalJudgment, sha256_file
from .conformity_identification_v11 import DEFAULT_V11_CONFIG, load_v11_config, validate_branch_plan

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS_PLAN = "configs/provenance_cascade/identification/conformity_identification_analysis.v1_1.toml"


class ContrastId(str, Enum):
    NATURAL_INSTABILITY = "natural_instability"
    TEXT_REPETITION_EFFECT = "text_repetition_effect"
    SOCIAL_SOURCE_INCREMENT = "social_source_increment"
    INDEPENDENT_CORROBORATION_INCREMENT = "independent_corroboration_increment"
    EVIDENCE_RECEPTIVITY_INCREMENT = "evidence_receptivity_increment"


class PlannedContrast(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    contrast_id: ContrastId
    minuend: IdentificationCondition
    subtrahend: IdentificationCondition
    outcome: Literal["ordinal_judgment_shift"]
    pairing_unit: Literal["scenario_seed_agent"]
    inference_cluster: Literal["scenario_seed"]


class OrdinalScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    judgment: OrdinalJudgment
    score: Literal[-2, -1, 0, 1, 2]


class EligibilityRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    rule_id: Literal[
        "initial_substantive", "paired_final_observation", "correction_transition", "sharing_observation"
    ]
    required_coordinates: tuple[str, ...]
    exclusion_codes: tuple[str, ...]
    missing_returns_not_applicable: Literal[True]


_EXPECTED_CONTRASTS = (
    (ContrastId.NATURAL_INSTABILITY, IdentificationCondition.SELF_REFLECTION, IdentificationCondition.PRIVATE_BASELINE),
    (ContrastId.TEXT_REPETITION_EFFECT, IdentificationCondition.SOURCE_FREE_REPETITION, IdentificationCondition.SELF_REFLECTION),
    (ContrastId.SOCIAL_SOURCE_INCREMENT, IdentificationCondition.SAME_ROOT_SOCIAL, IdentificationCondition.SOURCE_FREE_REPETITION),
    (ContrastId.INDEPENDENT_CORROBORATION_INCREMENT, IdentificationCondition.INDEPENDENT_ROOTS, IdentificationCondition.SAME_ROOT_SOCIAL),
    (ContrastId.EVIDENCE_RECEPTIVITY_INCREMENT, IdentificationCondition.VERIFIED_EVIDENCE, IdentificationCondition.SELF_REFLECTION),
)


class IdentificationAnalysisPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    analysis_id: Literal["evicon-conformity-identification-analysis-v1-1"]
    analysis_version: Literal["conformity_identification_analysis.v1_1"]
    status: Literal["offline_preregistered"]
    development_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    study_config_path: str
    study_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    branch_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordinal_scores: tuple[OrdinalScore, ...]
    primary_contrasts: tuple[PlannedContrast, ...]
    diagnostic_metric: Literal["conventional_answer_flip_toward_visible_claim"]
    diagnostic_metric_is_primary: Literal[False]
    report_metric_disagreement: Literal[True]
    eligibility_rules: tuple[EligibilityRule, ...]
    agent_observations_are_independent_replicates: Literal[False]
    aggregate_agents_before_inference: Literal[True]
    descriptive_statistics: tuple[str, ...]
    paired_test: Literal["exact_sign_flip_on_scenario_seed_aggregates"]
    robustness: tuple[str, ...]
    no_significance_threshold_added: Literal[True]
    no_go_no_go_threshold_registered: Literal[True]
    evaluator_truth_runtime_access: Literal[False]
    safe_output_only: Literal[True]
    output_root: Literal["results/analyses/provenance-cascade-conformity-identification-v1-1"]

    @model_validator(mode="after")
    def frozen_analysis(self) -> "IdentificationAnalysisPlan":
        expected_scores = tuple(zip(tuple(OrdinalJudgment), (-2, -1, 0, 1, 2), strict=True))
        if tuple((item.judgment, item.score) for item in self.ordinal_scores) != expected_scores:
            raise ValueError("ordinal score mapping mismatch")
        if tuple((item.contrast_id, item.minuend, item.subtrahend) for item in self.primary_contrasts) != _EXPECTED_CONTRASTS:
            raise ValueError("primary contrast set mismatch")
        if len({item.rule_id for item in self.eligibility_rules}) != 4:
            raise ValueError("eligibility rules must be unique and complete")
        if self.descriptive_statistics != ("count", "denominator", "mean", "median", "standard_deviation"):
            raise ValueError("descriptive statistics mismatch")
        if self.robustness != ("leave_one_scenario_out", "by_scenario", "by_seed", "missingness_audit"):
            raise ValueError("robustness plan mismatch")
        return self


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def load_analysis_plan(path: str | Path = DEFAULT_ANALYSIS_PLAN) -> IdentificationAnalysisPlan:
    resolved = _resolve(path)
    if not resolved.is_file():
        raise IdentificationError("analysis_plan_missing")
    try:
        plan = IdentificationAnalysisPlan.model_validate(tomllib.loads(resolved.read_text(encoding="utf-8")))
    except Exception as exc:
        raise IdentificationError("analysis_plan_invalid") from exc
    config, _ = load_v11_config(plan.study_config_path)
    validate_branch_plan(config)
    if sha256_file(plan.study_config_path) != plan.study_config_sha256:
        raise IdentificationError("analysis_study_hash_mismatch")
    if config.branch_contract_sha256 != plan.branch_contract_sha256:
        raise IdentificationError("analysis_branch_hash_mismatch")
    if config.protocol_sha256 != plan.protocol_sha256:
        raise IdentificationError("analysis_protocol_hash_mismatch")
    return plan


def safe_analysis_preflight(path: str | Path = DEFAULT_ANALYSIS_PLAN) -> dict[str, object]:
    try:
        plan = load_analysis_plan(path)
        return {
            "status": "analysis_plan_ready",
            "analysis_id": plan.analysis_id,
            "study_config_sha256": plan.study_config_sha256,
            "primary_contrasts": [item.contrast_id.value for item in plan.primary_contrasts],
            "contrast_count": len(plan.primary_contrasts),
            "pairing_unit": "scenario_seed_agent",
            "inference_cluster": "scenario_seed",
            "traditional_metric_primary": False,
            "go_no_go_thresholds_registered": False,
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "private_truth_exposed": False,
            "not_paper_result": True,
            "no_causal_conclusion": True,
        }
    except Exception as exc:
        return {
            "status": "blocked",
            "blocking_reasons": [getattr(exc, "code", "analysis_preflight_failed")],
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "private_truth_exposed": False,
        }


def safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


__all__ = [
    "ContrastId", "DEFAULT_ANALYSIS_PLAN", "IdentificationAnalysisPlan", "PlannedContrast",
    "load_analysis_plan", "safe_analysis_preflight",
]
