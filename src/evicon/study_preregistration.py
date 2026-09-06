"""Offline, protocol-blind WVS v2 preregistration contracts.

This module plans a future calibration or confirmatory study.  It deliberately
does not construct providers, read API configuration, load probe text, or
create result directories.
"""

from __future__ import annotations

import tomllib
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .frozen_wvs7_manifest import FrozenWVS7ManifestError, load_frozen_manifest, sha256_file
from .models import ProtocolCondition
from .models._validation import identifier_list, normalized_text


_ROOT = Path(__file__).resolve().parents[2]
_EXPECTED_FROZEN_SHA256 = "181fc27816565e22d23689bb2fd3da5fe693db7943c734a0ccfc355aa40190f2"
_EXPECTED_PROBE_SET_ID = "wvs7-english-core-23"
_EXPECTED_FROZEN_PROBE_PATH = "outputs/wvs7-frozen/english_core_wvs7_23_frozen.json"
_EXPECTED_FROZEN_MANIFEST_PATH = "outputs/wvs7-frozen/english_core_wvs7_23_frozen_manifest.json"
_EXPECTED_VARIABLE_IDS = frozenset(
    {
        "Q106", "Q107", "Q108", "Q109", "Q110", "Q111", "Q149", "Q150",
        "Q158", "Q159", "Q160", "Q161", "Q162", "Q163", "Q196", "Q197",
        "Q198", "Q241", "Q243", "Q246", "Q247", "Q248", "Q249",
    }
)
_BASELINE_CONDITIONS = (
    ProtocolCondition.INDEPENDENT,
    ProtocolCondition.SOCIAL_ONLY,
    ProtocolCondition.EVIDENCE_ONLY,
    ProtocolCondition.EVIDENCE_SOCIAL,
)
_PRIMARY_METRICS = ("pairwise_diversity", "social_influence_loss", "profile_drift")
_EXPLANATORY_METRICS = ("structural_diversity", "value_dimension_coverage")
_OBSERVED_V1_002_TOTAL_TOKENS = 190_602
_COMPLETION_RESERVATION_CAP = 51_200

# Public receipts used by tests and later offline planning tools.  They are
# constants, not runtime provider or experiment settings.
EXPECTED_FROZEN_PROBE_SET_SHA256 = _EXPECTED_FROZEN_SHA256
COMPLETION_RESERVATION_CAP = _COMPLETION_RESERVATION_CAP
OBSERVATIONAL_V1_002_TOTAL_TOKENS = _OBSERVED_V1_002_TOTAL_TOKENS


class StudyStatus(str, Enum):
    """Planning state, not an experimental finding."""

    CALIBRATION = "calibration"
    CONFIRMATORY = "confirmatory"


class V1PilotExclusion(BaseModel):
    """Why the older condition-aware v1 pilot is not part of v2 analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    probe_template_version: str = Field(min_length=1)
    condition_aware: bool
    excluded_from_v2_main_analysis: bool
    observed_total_tokens: int = Field(gt=0)

    @field_validator("run_id", "probe_template_version")
    @classmethod
    def text(cls, value: str, info: object) -> str:
        return normalized_text(value, info.field_name)

    @model_validator(mode="after")
    def excluded_v1_contract(self) -> "V1PilotExclusion":
        if self.run_id != "wvs7-real-baseline-pilot-seed-002":
            raise ValueError("v1 exclusion must identify the audited 002 pilot")
        if self.probe_template_version != "contextual_value_probe.v1" or not self.condition_aware:
            raise ValueError("v1 exclusion must describe the condition-aware v1 pilot")
        if not self.excluded_from_v2_main_analysis:
            raise ValueError("v1 002 must be excluded from v2 main analysis")
        if self.observed_total_tokens != _OBSERVED_V1_002_TOTAL_TOKENS:
            raise ValueError("v1 observed total token reference must remain the audited 190602 value")
        return self


class CostPlan(BaseModel):
    """Separate bounded completion reservation from observed total-token cost."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requests_per_matched_group: int = Field(gt=0)
    completion_reservation_cap: int = Field(gt=0)
    observational_actual_total_tokens_reference: int = Field(gt=0)
    estimated_total_tokens_per_matched_group: int = Field(gt=0)
    reference_run_id: str = Field(min_length=1)
    reference_is_hard_cap: bool

    @field_validator("reference_run_id")
    @classmethod
    def reference_id(cls, value: str) -> str:
        return normalized_text(value, "reference_run_id")

    @model_validator(mode="after")
    def validate_cost_meaning(self) -> "CostPlan":
        if self.completion_reservation_cap != _COMPLETION_RESERVATION_CAP:
            raise ValueError("completion reservation must be 51200 for this fixed calibration plan")
        if self.observational_actual_total_tokens_reference != _OBSERVED_V1_002_TOTAL_TOKENS:
            raise ValueError("observational total-token reference must be the audited 190602 value")
        if self.estimated_total_tokens_per_matched_group != self.observational_actual_total_tokens_reference:
            raise ValueError("estimated total tokens must be explicitly tied to the observational reference")
        if self.reference_run_id != "wvs7-real-baseline-pilot-seed-002" or self.reference_is_hard_cap:
            raise ValueError("the 002 reference is observational and must not be represented as a hard cap")
        return self


class ConfirmatoryRequirements(BaseModel):
    """Requirements for a later study, separate from current calibration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_seed_count: int = Field(ge=3)
    development_scenario_ids: list[str] = Field(min_length=1)
    test_scenario_ids: list[str] = Field(min_length=1)

    @field_validator("development_scenario_ids", "test_scenario_ids")
    @classmethod
    def scenario_ids(cls, value: list[str], info: object) -> list[str]:
        return identifier_list(value, info.field_name)

    @model_validator(mode="after")
    def split_is_disjoint(self) -> "ConfirmatoryRequirements":
        overlap = set(self.development_scenario_ids) & set(self.test_scenario_ids)
        if overlap:
            raise ValueError("development and test scenario IDs must not overlap")
        return self


class StudyPreregistration(BaseModel):
    """Strict, no-network declaration for a future protocol-blind v2 study."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: str = Field(min_length=1)
    preregistration_version: str = Field(min_length=1)
    status: StudyStatus
    development_only: bool
    not_paper_result: bool
    probe_template_version: str = Field(min_length=1)
    frozen_probe_set_path: str = Field(min_length=1)
    frozen_manifest_path: str = Field(min_length=1)
    frozen_probe_set_sha256: str = Field(min_length=64, max_length=64)
    probe_set_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    scenario_path: str = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    agent_ids: list[str] = Field(min_length=2)
    max_rounds: int = Field(ge=1)
    conditions: list[ProtocolCondition] = Field(min_length=1)
    planned_calibration_run_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    agent_temperature: float = Field(ge=0.0, le=2.0)
    agent_max_tokens: int = Field(gt=0)
    probe_temperature: float = Field(ge=0.0, le=2.0)
    probe_max_tokens: int = Field(gt=0)
    reasoning_effort: Literal["none"]
    max_retries: int = Field(ge=0, le=5)
    request_interval_seconds: float = Field(ge=0.0, le=60.0)
    coverage_threshold: float = Field(ge=0.0, le=1.0)
    run_output_dir: str = Field(min_length=1)
    batch_output_dir: str = Field(min_length=1)
    primary_metrics: list[str] = Field(min_length=1)
    explanatory_metrics: list[str] = Field(min_length=1)
    v1_002_exclusion: V1PilotExclusion
    cost_plan: CostPlan
    future_confirmatory_requirements: ConfirmatoryRequirements

    @field_validator(
        "study_id", "preregistration_version", "probe_template_version", "frozen_probe_set_path",
        "frozen_manifest_path", "probe_set_id", "scenario_id", "scenario_path", "planned_calibration_run_id",
        "model_name", "run_output_dir", "batch_output_dir",
    )
    @classmethod
    def text(cls, value: str, info: object) -> str:
        return normalized_text(value, info.field_name)

    @field_validator("agent_ids")
    @classmethod
    def agents(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "agent_ids")

    @field_validator("seeds")
    @classmethod
    def seed_values(cls, value: list[int]) -> list[int]:
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in value):
            raise ValueError("seeds must contain integers")
        if len(value) != len(set(value)):
            raise ValueError("seeds must not contain duplicates")
        return value

    @field_validator("frozen_probe_set_sha256")
    @classmethod
    def frozen_hash(cls, value: str) -> str:
        normalized = value.lower().strip()
        if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
            raise ValueError("frozen_probe_set_sha256 must be a SHA-256 digest")
        return normalized

    @field_validator("primary_metrics", "explanatory_metrics")
    @classmethod
    def metric_names(cls, value: list[str], info: object) -> list[str]:
        return identifier_list(value, info.field_name)

    @model_validator(mode="after")
    def validate_study_shape(self) -> "StudyPreregistration":
        if self.probe_template_version != "contextual_value_probe.v2":
            raise ValueError("protocol-blind study must use contextual_value_probe.v2")
        if self.probe_set_id != _EXPECTED_PROBE_SET_ID:
            raise ValueError("study must reference the frozen English 23-item ProbeSet")
        if self.frozen_probe_set_sha256 != _EXPECTED_FROZEN_SHA256:
            raise ValueError("study must reference the audited frozen ProbeSet hash")
        if tuple(self.conditions) != _BASELINE_CONDITIONS:
            raise ValueError("conditions must list the four baseline protocols in fixed order")
        if tuple(self.primary_metrics) != _PRIMARY_METRICS:
            raise ValueError("primary metrics must be pairwise_diversity, social_influence_loss, and profile_drift")
        if tuple(self.explanatory_metrics) != _EXPLANATORY_METRICS:
            raise ValueError("structural diversity and coverage are explanatory metrics only")
        if self.planned_calibration_run_id != "wvs7-real-baseline-pilot-seed-003":
            raise ValueError("calibration may only declare the future 003 run ID")
        if self.max_rounds != 2 or len(self.agent_ids) != 2:
            raise ValueError("calibration plan must use two agents and two rounds")
        if (
            self.agent_temperature != 0.2
            or self.agent_max_tokens != 256
            or self.probe_temperature != 0.0
            or self.probe_max_tokens != 128
            or self.reasoning_effort != "none"
            or self.max_retries != 1
        ):
            raise ValueError("calibration must use its fixed v2 generation and retry parameters")
        if Path(self.run_output_dir).parts[:1] != ("results",) or Path(self.batch_output_dir).parts[:1] != ("results",):
            raise ValueError("calibration output paths must remain under results")
        if self.status is StudyStatus.CALIBRATION:
            if not self.development_only or not self.not_paper_result or len(self.seeds) != 1:
                raise ValueError("calibration must be development-only, non-paper, and use exactly one seed")
        else:
            if self.development_only or self.not_paper_result:
                raise ValueError("confirmatory studies cannot be marked development-only or not-paper")
            if len(self.seeds) < self.future_confirmatory_requirements.minimum_seed_count:
                raise ValueError("confirmatory study has fewer seeds than its preregistered minimum")
        return self


class StudyPreregistrationSummary(BaseModel):
    """Safe CLI report without source text, prompts, results, or secrets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: str
    status: StudyStatus
    template_version: str
    condition_count: int = Field(ge=0)
    seed_count: int = Field(ge=0)
    expected_provider_requests_per_matched_group: int = Field(ge=0)
    completion_reservation_cap_per_matched_group: int = Field(ge=0)
    estimated_total_tokens_per_matched_group: int = Field(ge=0)
    primary_metrics: list[str]
    calibration_runnable: bool


class StudyPreregistrationError(ValueError):
    """Stable, safe preregistration validation failure."""


class StudyPreregistrationLoader:
    """Read one local TOML preregistration without providers or network access."""

    @staticmethod
    def load(path: str | Path) -> StudyPreregistration:
        try:
            payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise StudyPreregistrationError("cannot read preregistration config") from exc
        try:
            return StudyPreregistration.model_validate(payload)
        except ValidationError as exc:
            raise StudyPreregistrationError("preregistration config violates its required contract") from exc


def _project_path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else _ROOT / candidate


def validate_study_preregistration(preregistration: StudyPreregistration) -> StudyPreregistrationSummary:
    """Perform safe local integrity checks without loading WVS probe text."""
    if (
        preregistration.frozen_probe_set_path != _EXPECTED_FROZEN_PROBE_PATH
        or preregistration.frozen_manifest_path != _EXPECTED_FROZEN_MANIFEST_PATH
    ):
        raise StudyPreregistrationError("frozen_probe_set_path_is_not_the_audited_english_receipt")
    probe_path = _project_path(preregistration.frozen_probe_set_path)
    manifest_path = _project_path(preregistration.frozen_manifest_path)
    try:
        frozen_manifest = load_frozen_manifest(manifest_path)
        actual_hash = sha256_file(probe_path)
    except FrozenWVS7ManifestError as exc:
        raise StudyPreregistrationError("frozen_probe_set_receipt_invalid") from exc
    if actual_hash != preregistration.frozen_probe_set_sha256:
        raise StudyPreregistrationError("frozen_probe_set_hash_mismatch")
    if frozen_manifest.probe_set_sha256 != preregistration.frozen_probe_set_sha256:
        raise StudyPreregistrationError("frozen_manifest_hash_mismatch")
    if (
        frozen_manifest.probe_set_id != preregistration.probe_set_id
        or frozen_manifest.item_count != 23
        or set(frozen_manifest.variable_ids) != _EXPECTED_VARIABLE_IDS
        or "Q48" in frozen_manifest.variable_ids
        or frozen_manifest.language_arm != "english_core"
        or frozen_manifest.probe_set_filename != probe_path.name
        or frozen_manifest.wvs_official_factor_structure is not False
        or frozen_manifest.q158_q163_secondary_analysis is not True
    ):
        raise StudyPreregistrationError("frozen_probe_set_metadata_mismatch")
    required_special_codes = {
        "Q111",
        "Q149_Q150",
        "Q241_Q243_Q246_Q247_Q248_Q249",
        "Q158_Q163",
        "Q246",
    }
    if not required_special_codes.issubset(frozen_manifest.special_code_summary):
        raise StudyPreregistrationError("frozen_probe_set_special_metadata_mismatch")

    planned_run_directory = _ROOT / "results" / preregistration.planned_calibration_run_id
    calibration_runnable = preregistration.status is StudyStatus.CALIBRATION and not planned_run_directory.exists()
    return StudyPreregistrationSummary(
        study_id=preregistration.study_id,
        status=preregistration.status,
        template_version=preregistration.probe_template_version,
        condition_count=len(preregistration.conditions),
        seed_count=len(preregistration.seeds),
        expected_provider_requests_per_matched_group=preregistration.cost_plan.requests_per_matched_group,
        completion_reservation_cap_per_matched_group=preregistration.cost_plan.completion_reservation_cap,
        estimated_total_tokens_per_matched_group=preregistration.cost_plan.estimated_total_tokens_per_matched_group,
        primary_metrics=list(preregistration.primary_metrics),
        calibration_runnable=calibration_runnable,
    )


def load_and_validate_study_preregistration(path: str | Path) -> StudyPreregistrationSummary:
    """Load and validate the local planning record without writing state."""
    return validate_study_preregistration(StudyPreregistrationLoader.load(path))


__all__ = [
    "ConfirmatoryRequirements",
    "COMPLETION_RESERVATION_CAP",
    "CostPlan",
    "EXPECTED_FROZEN_PROBE_SET_SHA256",
    "OBSERVATIONAL_V1_002_TOTAL_TOKENS",
    "StudyPreregistration",
    "StudyPreregistrationError",
    "StudyPreregistrationLoader",
    "StudyPreregistrationSummary",
    "StudyStatus",
    "V1PilotExclusion",
    "load_and_validate_study_preregistration",
    "validate_study_preregistration",
]
