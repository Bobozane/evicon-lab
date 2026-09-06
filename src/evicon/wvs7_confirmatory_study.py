from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .frozen_wvs7_manifest import load_frozen_manifest, sha256_file
from .models import ProtocolCondition
from .models._validation import identifier_list, normalized_text
from .wvs7_test_scenario import (
    TEST_SCENARIO_ID, TestScenarioAuthoringError, load_authoring, load_review, validate_test_scenario,
)

EXPECTED_V2_PROBE_SET_ID = "wvs7-english-core-23"
EXPECTED_V2_PROBE_SHA256 = "181fc27816565e22d23689bb2fd3da5fe693db7943c734a0ccfc355aa40190f2"
EXPECTED_V2_TEMPLATE = "contextual_value_probe.v2"
EXPECTED_CONDITIONS = (
    ProtocolCondition.INDEPENDENT,
    ProtocolCondition.SOCIAL_ONLY,
    ProtocolCondition.EVIDENCE_ONLY,
    ProtocolCondition.EVIDENCE_SOCIAL,
)
EXPECTED_PRIMARY_METRICS = ("pairwise_diversity", "social_influence_loss", "profile_drift")
EXPECTED_CONFIRMATORY_SEEDS = (20260820, 20260821, 20260822)
_ROOT = Path(__file__).resolve().parents[2]


class ConfirmatoryStudyError(ValueError):
    """Stable, safe manifest validation failure."""


class ConfirmatoryRunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    matched_group_id: str = Field(min_length=1)
    condition: ProtocolCondition
    scenario_id: str = Field(min_length=1)
    seed: int
    agent_ids: list[str] = Field(min_length=2)
    max_rounds: int = Field(ge=1)
    model_name: str = Field(min_length=1)
    probe_set_id: str = Field(min_length=1)
    probe_template_version: str = Field(min_length=1)

    @field_validator(
        "run_id", "matched_group_id", "scenario_id", "model_name",
        "probe_set_id", "probe_template_version",
    )
    @classmethod
    def text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("agent_ids")
    @classmethod
    def agents(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "agent_ids")

    @field_validator("seed")
    @classmethod
    def seed_value(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("seed must be an integer")
        return value


class ConfirmatoryStudyManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: str = Field(min_length=1)
    preregistration_version: str = Field(min_length=1)
    status: Literal["confirmatory"]
    development_only: Literal[False]
    not_paper_result: Literal[False]
    probe_template_version: Literal["contextual_value_probe.v2"]
    frozen_probe_set_path: str = Field(min_length=1)
    frozen_probe_set_sha256: str = Field(min_length=64, max_length=64)
    probe_set_id: Literal["wvs7-english-core-23"]
    development_scenario_ids: list[str] = Field(default_factory=list)
    test_scenario_ids: list[str] = Field(default_factory=list)
    test_scenario_path: str = Field(min_length=1)
    test_scenario_authoring_path: str = Field(min_length=1)
    test_scenario_review_path: str = Field(min_length=1)
    test_scenario_sha256: str = Field(min_length=64, max_length=64)
    agent_ids: list[str] = Field(min_length=2)
    max_rounds: int = Field(ge=1)
    model_name: str = Field(min_length=1)
    seeds: list[int] = Field(default_factory=list)
    conditions: list[ProtocolCondition]
    primary_metrics: list[str]
    runs: list[ConfirmatoryRunSpec] = Field(default_factory=list)
    test_scenario_review_status: Literal["pending", "accepted"] = "pending"
    execution_status: Literal["future_explicit_confirmation_required"] = "future_explicit_confirmation_required"
    excluded_development_run_ids: list[str] = Field(default_factory=list)
    logical_requests_per_matched_group: int = Field(default=384, ge=0)
    completion_reservation_per_matched_group: int = Field(default=51200, ge=0)
    observational_total_token_reference_per_matched_group: int = Field(default=190602, ge=0)
    total_logical_requests: int = Field(default=1152, ge=0)
    total_completion_reservation: int = Field(default=153600, ge=0)
    observational_total_token_reference: int = Field(default=571806, ge=0)

    @field_validator("study_id", "preregistration_version", "frozen_probe_set_path", "model_name", "test_scenario_path", "test_scenario_authoring_path", "test_scenario_review_path")
    @classmethod
    def text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator(
        "development_scenario_ids", "test_scenario_ids", "agent_ids",
        "excluded_development_run_ids",
    )
    @classmethod
    def identifiers(cls, value: list[str], info: object) -> list[str]:
        return identifier_list(value, getattr(info, "field_name", "identifiers"))

    @field_validator("seeds")
    @classmethod
    def seed_values(cls, value: list[int]) -> list[int]:
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in value):
            raise ValueError("seeds must contain integers")
        if len(value) != len(set(value)):
            raise ValueError("seeds must not contain duplicates")
        return value

    @field_validator("test_scenario_sha256")
    @classmethod
    def scenario_sha256(cls, value: str) -> str:
        normalized = value.lower().strip()
        if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
            raise ValueError("test_scenario_sha256 must be a SHA-256 digest")
        return normalized

    @field_validator("frozen_probe_set_sha256")
    @classmethod
    def sha256(cls, value: str) -> str:
        normalized = value.lower().strip()
        if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
            raise ValueError("frozen_probe_set_sha256 must be a SHA-256 digest")
        return normalized

    @model_validator(mode="after")
    def fixed_contract(self) -> "ConfirmatoryStudyManifest":
        if self.probe_template_version != EXPECTED_V2_TEMPLATE or self.probe_set_id != EXPECTED_V2_PROBE_SET_ID:
            raise ValueError("confirmatory study must use contextual_value_probe.v2 and the English frozen ProbeSet")
        if self.frozen_probe_set_sha256 != EXPECTED_V2_PROBE_SHA256:
            raise ValueError("confirmatory study frozen ProbeSet hash is not the audited hash")
        if not self.test_scenario_path or not self.test_scenario_authoring_path or not self.test_scenario_review_path:
            raise ValueError("confirmatory study must declare test scenario authoring and review paths")
        if self.test_scenario_ids and tuple(self.test_scenario_ids) != (TEST_SCENARIO_ID,):
            raise ValueError("confirmatory study may only declare the independently authored v2 test scenario")
        if tuple(self.conditions) != EXPECTED_CONDITIONS:
            raise ValueError("confirmatory conditions must list all four protocols in fixed order")
        if tuple(self.primary_metrics) != EXPECTED_PRIMARY_METRICS:
            raise ValueError("confirmatory primary metrics are fixed")
        if 20260819 in self.seeds:
            raise ValueError("calibration seed 20260819 cannot enter confirmatory study")
        if tuple(self.seeds) != EXPECTED_CONFIRMATORY_SEEDS:
            raise ValueError("confirmatory study must use the three preregistered seeds")
        required_exclusions = {
            "wvs7-real-baseline-pilot-seed-002",
            "wvs7-real-baseline-pilot-seed-003",
        }
        if not required_exclusions.issubset(self.excluded_development_run_ids):
            raise ValueError("confirmatory manifest must explicitly exclude both development pilot run IDs")
        return self


class ConfirmatoryGateReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["blocked", "ready_for_confirmatory"]
    study_id: str
    template_version: str
    seed_count: int = Field(ge=0)
    condition_count: int = Field(ge=0)
    test_scenario_count: int = Field(ge=0)
    primary_metrics: list[str]
    blocking_reasons: list[str] = Field(default_factory=list)


def load_confirmatory_manifest(path: str | Path) -> ConfirmatoryStudyManifest:
    try:
        payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        return ConfirmatoryStudyManifest.model_validate(payload)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise ConfirmatoryStudyError("confirmatory_manifest_invalid") from exc


def validate_confirmatory_manifest(
    manifest: ConfirmatoryStudyManifest,
    *,
    verify_frozen_file: bool = True,
) -> ConfirmatoryGateReport:
    reasons: list[str] = []
    if not manifest.development_scenario_ids:
        reasons.append("development_scenario_ids_required")
    if not manifest.test_scenario_ids:
        reasons.append("independent_test_scenario_ids_required")
    if set(manifest.development_scenario_ids) & set(manifest.test_scenario_ids):
        reasons.append("development_test_scenarios_must_be_disjoint")
    if manifest.test_scenario_review_status != "accepted":
        reasons.append("test_scenario_manual_review_required")
    if manifest.test_scenario_ids:
        try:
            authoring = load_authoring(_ROOT / manifest.test_scenario_authoring_path)
            review = load_review(_ROOT / manifest.test_scenario_review_path)
            scenario_report = validate_test_scenario(authoring, review, require_review=True)
            manifest_scenario_path = Path(manifest.test_scenario_path)
            if not manifest_scenario_path.is_absolute():
                manifest_scenario_path = _ROOT / manifest_scenario_path
            authoring_scenario_path = Path(authoring.scenario_path)
            if not authoring_scenario_path.is_absolute():
                authoring_scenario_path = _ROOT / authoring_scenario_path
            scenario_path_matches = manifest_scenario_path.resolve() == authoring_scenario_path.resolve()
            scenario_hash_matches = sha256_file(manifest_scenario_path) == manifest.test_scenario_sha256
            if not scenario_path_matches:
                reasons.append("test_scenario_path_mismatch")
            if not scenario_hash_matches:
                reasons.append("test_scenario_hash_mismatch")
            if (
                scenario_report.status != "ready"
                or authoring.scenario_id not in manifest.test_scenario_ids
                or authoring.scenario_sha256 != manifest.test_scenario_sha256
            ):
                reasons.extend(scenario_report.blocking_reasons or ["test_scenario_authoring_gate_blocked"])
        except (TestScenarioAuthoringError, OSError, ValueError):
            reasons.append("test_scenario_authoring_gate_invalid")
    if len(manifest.seeds) < 3:
        reasons.append("at_least_three_new_seeds_required")
    if verify_frozen_file:
        try:
            probe_path = Path(manifest.frozen_probe_set_path)
            if not probe_path.is_absolute():
                probe_path = _ROOT / probe_path
            manifest_path = probe_path.with_name("english_core_wvs7_23_frozen_manifest.json")
            frozen = load_frozen_manifest(manifest_path)
            if sha256_file(probe_path) != manifest.frozen_probe_set_sha256 or frozen.probe_set_sha256 != manifest.frozen_probe_set_sha256:
                reasons.append("frozen_probe_set_hash_mismatch")
        except Exception:
            reasons.append("frozen_probe_set_unavailable")
    if not manifest.runs:
        reasons.append("complete_four_condition_groups_required")
    else:
        by_seed: dict[int, list[ConfirmatoryRunSpec]] = {}
        seen_run_ids: set[str] = set()
        for raw_run in manifest.runs:
            try:
                run = raw_run if isinstance(raw_run, ConfirmatoryRunSpec) else ConfirmatoryRunSpec.model_validate(raw_run)
            except ValidationError:
                reasons.append("run_spec_invalid")
                continue
            if run.run_id in seen_run_ids:
                reasons.append("duplicate_run_id")
            seen_run_ids.add(run.run_id)
            if run.seed not in manifest.seeds:
                reasons.append("run_seed_not_preregistered")
            if run.scenario_id not in manifest.test_scenario_ids:
                reasons.append("confirmatory_run_must_use_test_scenario")
            if run.probe_template_version != EXPECTED_V2_TEMPLATE:
                reasons.append("v1_or_non_v2_run_rejected")
            if (
                run.probe_set_id != manifest.probe_set_id
                or run.agent_ids != manifest.agent_ids
                or run.max_rounds != manifest.max_rounds
                or run.model_name != manifest.model_name
            ):
                reasons.append("matched_run_contract_mismatch")
            if (
                run.run_id in {
                    "wvs7-real-baseline-pilot-seed-002",
                    "wvs7-real-baseline-pilot-seed-003",
                }
                or "seed-002" in run.run_id
                or "seed-003" in run.run_id
            ):
                reasons.append("development_run_cannot_enter_confirmatory_manifest")
            by_seed.setdefault(run.seed, []).append(run)
        matched_group_ids: list[str] = []
        for seed in manifest.seeds:
            group = by_seed.get(seed, [])
            if (
                len(group) != 4
                or {run.condition for run in group} != set(EXPECTED_CONDITIONS)
                or len({run.matched_group_id for run in group}) != 1
            ):
                reasons.append("complete_four_condition_groups_required")
            elif group:
                matched_group_ids.append(group[0].matched_group_id)
        if len(matched_group_ids) != len(set(matched_group_ids)):
            reasons.append("matched_group_id_must_be_unique")
    reasons = sorted(set(reasons))
    return ConfirmatoryGateReport(
        status="ready_for_confirmatory" if not reasons else "blocked",
        study_id=manifest.study_id,
        template_version=manifest.probe_template_version,
        seed_count=len(manifest.seeds),
        condition_count=len(manifest.conditions),
        test_scenario_count=len(manifest.test_scenario_ids),
        primary_metrics=list(manifest.primary_metrics),
        blocking_reasons=reasons,
    )


def validate_confirmatory_study(
    manifest: ConfirmatoryStudyManifest,
    *,
    verify_frozen_file: bool = True,
) -> ConfirmatoryGateReport:
    """Descriptive alias for :func:`validate_confirmatory_manifest`."""
    return validate_confirmatory_manifest(manifest, verify_frozen_file=verify_frozen_file)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a future v2 confirmatory study manifest offline.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = load_confirmatory_manifest(args.config)
        report = validate_confirmatory_manifest(manifest)
    except ConfirmatoryStudyError:
        print(json.dumps({"status": "blocked", "blocking_reasons": ["confirmatory_manifest_invalid"]}, sort_keys=True))
        return 1
    print(json.dumps(report.model_dump(mode="json"), sort_keys=True))
    return 0 if report.status == "ready_for_confirmatory" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ConfirmatoryGateReport",
    "ConfirmatoryRunSpec",
    "ConfirmatoryStudyError",
    "ConfirmatoryStudyManifest",
    "load_confirmatory_manifest",
    "validate_confirmatory_manifest",
    "validate_confirmatory_study",
]
