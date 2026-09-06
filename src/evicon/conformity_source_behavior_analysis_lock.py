"""Outcome-blind descriptive-analysis lock for source-behavior v2.

The lock is created after data collection, but before any adoption or sharing
distribution is computed. Its default CLI validates hashes and safe receipt
metadata only. It never parses the case-level audit or writes analysis output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_identification import IdentificationError, safe_json
from .conformity_source_behavior_runner import QualificationRunReceipt
from .conformity_source_manipulation import SourceProjection

_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ANALYSIS_LOCK = (
    "configs/provenance_cascade/identification/"
    "conformity_source_behavior_analysis_lock.v2.toml"
)
DEFAULT_ANALYSIS_APPROVAL = (
    "configs/provenance_cascade/identification/"
    "conformity_source_behavior_analysis_approval.v2.toml"
)
DEFAULT_QUALIFICATION_RECEIPT = (
    "outputs/conformity-source-behavior-qualification-v2/qualification_receipt.json"
)
DEFAULT_SAFE_CASE_AUDIT = (
    "outputs/conformity-source-behavior-qualification-v2/safe_case_audits.jsonl"
)


class DescriptiveContrastId(str, Enum):
    SOURCE_IDENTITY_DISCLOSURE = "same_root_minus_source_free"
    ROOT_INDEPENDENCE_INCREMENT = "independent_roots_minus_same_root"


class PlannedDescriptiveContrast(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contrast_id: DescriptiveContrastId
    minuend: SourceProjection
    subtrahend: SourceProjection
    pairing_unit: Literal["scenario_id"]


_EXPECTED_CONTRASTS = (
    (
        DescriptiveContrastId.SOURCE_IDENTITY_DISCLOSURE,
        SourceProjection.SAME_ROOT,
        SourceProjection.SOURCE_FREE,
    ),
    (
        DescriptiveContrastId.ROOT_INDEPENDENCE_INCREMENT,
        SourceProjection.INDEPENDENT_ROOTS,
        SourceProjection.SAME_ROOT,
    ),
)


class SourceBehaviorAnalysisLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis_id: Literal["evicon-conformity-source-behavior-analysis-v2"]
    analysis_version: Literal["conformity_source_behavior_analysis_lock.v2"]
    status: Literal["post_collection_outcome_blind_lock"]
    development_only: Literal[True]
    qualification_only: Literal[True]
    not_preregistered_before_data_collection: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    no_population_generalization: Literal[True]
    qualification_receipt_path: Literal[
        "outputs/conformity-source-behavior-qualification-v2/qualification_receipt.json"
    ]
    qualification_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    safe_case_audit_path: Literal[
        "outputs/conformity-source-behavior-qualification-v2/safe_case_audits.jsonl"
    ]
    safe_case_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runner_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: Literal["gpt-5.6-luna"]
    seed: Literal[20261110]
    scenario_count: Literal[4]
    projection_count: Literal[3]
    case_count: Literal[12]
    pairing_unit: Literal["scenario_id"]
    primary_outcome: Literal["adoption_decision"]
    secondary_outcome: Literal["sharing_decision"]
    adoption_positive_value: Literal["adopt_visible_claim"]
    adoption_negative_value: Literal["withhold"]
    sharing_positive_value: Literal["share_visible_content"]
    sharing_negative_value: Literal["do_not_share"]
    primary_contrasts: tuple[PlannedDescriptiveContrast, ...]
    report_fields: tuple[str, ...]
    missing_pair_policy: Literal["stop_without_report"]
    duplicate_case_policy: Literal["stop_without_report"]
    inferential_tests: tuple[str, ...]
    p_values_reported: Literal[False]
    confidence_intervals_reported: Literal[False]
    significance_threshold_registered: Literal[False]
    go_no_go_threshold_registered: Literal[False]
    fixed_projection_order: tuple[SourceProjection, ...]
    projection_order_randomized: Literal[False]
    projection_order_confounding_acknowledged: Literal[True]
    order_adjustment_possible: Literal[False]
    historical_results_joined: Literal[False]
    evaluator_private_truth_loaded: Literal[False]
    aggregate_safe_output_only: Literal[True]
    case_level_values_in_report: Literal[False]
    conclusion_scope: Literal[
        "single_model_single_seed_four_scenario_descriptive_qualification_only"
    ]
    output_root: Literal[
        "results/analyses/conformity-source-behavior-qualification-v2"
    ]

    @model_validator(mode="after")
    def exact_analysis_scope(self) -> "SourceBehaviorAnalysisLock":
        observed = tuple(
            (item.contrast_id, item.minuend, item.subtrahend)
            for item in self.primary_contrasts
        )
        if observed != _EXPECTED_CONTRASTS:
            raise ValueError("source-behavior contrast set mismatch")
        if self.report_fields != (
            "eligible_scenario_count",
            "positive_difference_count",
            "zero_difference_count",
            "negative_difference_count",
            "mean_paired_difference",
        ):
            raise ValueError("source-behavior report field set mismatch")
        if self.inferential_tests:
            raise ValueError("inferential tests are not allowed")
        if self.fixed_projection_order != tuple(SourceProjection):
            raise ValueError("fixed projection order mismatch")
        return self


class SourceBehaviorAnalysisApproval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: Literal["evicon-conformity-source-behavior-analysis-v2"]
    approval_version: Literal["conformity_source_behavior_analysis_approval.v2"]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    accepted_on: str
    analysis_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    safe_case_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirm_post_collection_not_preregistered: bool
    confirm_fixed_order_confounding: bool
    confirm_scenario_paired_contrasts: bool
    confirm_adoption_primary_sharing_secondary: bool
    confirm_aggregate_descriptive_output_only: bool
    confirm_no_inferential_tests_or_thresholds: bool
    confirm_no_causal_or_generalized_claim: bool
    confirm_no_historical_result_join: bool
    confirm_no_evaluator_private_truth: bool
    analysis_execution_authorized: Literal[False]

    @model_validator(mode="after")
    def accepted_fields_complete(self) -> "SourceBehaviorAnalysisApproval":
        if self.acceptance_status != "accepted":
            return self
        if not self.accepted_by.strip() or not self.accepted_on.strip():
            raise ValueError("accepted analysis approval requires reviewer and date")
        try:
            date.fromisoformat(self.accepted_on)
        except ValueError as exc:
            raise ValueError("accepted_on must be an ISO date") from exc
        confirmations = (
            self.confirm_post_collection_not_preregistered,
            self.confirm_fixed_order_confounding,
            self.confirm_scenario_paired_contrasts,
            self.confirm_adoption_primary_sharing_secondary,
            self.confirm_aggregate_descriptive_output_only,
            self.confirm_no_inferential_tests_or_thresholds,
            self.confirm_no_causal_or_generalized_claim,
            self.confirm_no_historical_result_join,
            self.confirm_no_evaluator_private_truth,
        )
        if not all(confirmations):
            raise ValueError("accepted analysis approval is incomplete")
        return self


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def _sha(path: str | Path) -> str:
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def analysis_lock_sha256(path: str | Path = DEFAULT_ANALYSIS_LOCK) -> str:
    return _sha(path)


def load_analysis_lock(
    path: str | Path = DEFAULT_ANALYSIS_LOCK,
) -> SourceBehaviorAnalysisLock:
    try:
        plan = SourceBehaviorAnalysisLock.model_validate(
            tomllib.loads(_resolve(path).read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise IdentificationError("source_behavior_analysis_lock_invalid") from exc

    try:
        receipt = QualificationRunReceipt.model_validate_json(
            _resolve(plan.qualification_receipt_path).read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise IdentificationError("source_behavior_analysis_receipt_invalid") from exc

    expected_hashes = {
        "qualification_receipt_sha256": _sha(plan.qualification_receipt_path),
        "safe_case_audit_sha256": _sha(plan.safe_case_audit_path),
    }
    if any(getattr(plan, key) != value for key, value in expected_hashes.items()):
        raise IdentificationError("source_behavior_analysis_input_hash_mismatch")
    receipt_bindings = {
        "safe_case_audit_sha256": receipt.safe_case_audit_sha256,
        "config_sha256": receipt.config_sha256,
        "protocol_sha256": receipt.protocol_sha256,
        "source_approval_sha256": receipt.source_approval_sha256,
        "compatibility_receipt_sha256": receipt.compatibility_receipt_sha256,
        "execution_approval_sha256": receipt.execution_approval_sha256,
        "runner_sha256": receipt.runner_sha256,
        "response_schema_sha256": receipt.response_schema_sha256,
        "model": receipt.model,
        "seed": receipt.generation_parameters.seed,
        "case_count": receipt.logical_request_count,
    }
    if any(getattr(plan, key) != value for key, value in receipt_bindings.items()):
        raise IdentificationError("source_behavior_analysis_receipt_binding_mismatch")
    if receipt.status != "completed" or receipt.parser_invalid_count != 0:
        raise IdentificationError("source_behavior_analysis_receipt_incomplete")
    return plan


def load_analysis_approval(
    path: str | Path = DEFAULT_ANALYSIS_APPROVAL,
    *,
    analysis_lock_path: str | Path = DEFAULT_ANALYSIS_LOCK,
) -> SourceBehaviorAnalysisApproval:
    plan = load_analysis_lock(analysis_lock_path)
    try:
        approval = SourceBehaviorAnalysisApproval.model_validate(
            tomllib.loads(_resolve(path).read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise IdentificationError("source_behavior_analysis_approval_invalid") from exc
    expected = {
        "analysis_lock_sha256": analysis_lock_sha256(analysis_lock_path),
        "qualification_receipt_sha256": plan.qualification_receipt_sha256,
        "safe_case_audit_sha256": plan.safe_case_audit_sha256,
    }
    if any(getattr(approval, key) != value for key, value in expected.items()):
        raise IdentificationError("source_behavior_analysis_approval_binding_mismatch")
    return approval


def safe_preflight(
    analysis_lock_path: str | Path = DEFAULT_ANALYSIS_LOCK,
    approval_path: str | Path = DEFAULT_ANALYSIS_APPROVAL,
) -> dict[str, object]:
    try:
        plan = load_analysis_lock(analysis_lock_path)
        approval = load_analysis_approval(
            approval_path, analysis_lock_path=analysis_lock_path
        )
    except Exception as exc:
        return {
            "status": "blocked",
            "blocking_reasons": [
                getattr(exc, "code", "source_behavior_analysis_preflight_failed")
            ],
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "case_outcomes_loaded": False,
            "descriptive_statistics_computed": False,
            "results_written": False,
            "private_truth_exposed": False,
        }
    if approval.acceptance_status == "accepted":
        reasons = ["source_behavior_analysis_execution_authorization_required"]
        ready = True
    else:
        reasons = ["source_behavior_analysis_human_approval_required"]
        ready = False
    return {
        "status": "source_behavior_analysis_lock_ready",
        "analysis_id": plan.analysis_id,
        "analysis_lock_sha256": analysis_lock_sha256(analysis_lock_path),
        "approval_status": approval.acceptance_status,
        "qualification_receipt_sha256": plan.qualification_receipt_sha256,
        "safe_case_audit_sha256": plan.safe_case_audit_sha256,
        "primary_contrasts": [item.contrast_id.value for item in plan.primary_contrasts],
        "primary_outcome": plan.primary_outcome,
        "secondary_outcome": plan.secondary_outcome,
        "scenario_count": plan.scenario_count,
        "case_count": plan.case_count,
        "fixed_order_confounding_acknowledged": True,
        "blocking_reasons": reasons,
        "ready_for_analysis_execution": ready,
        "network": "disabled",
        "provider_constructed": False,
        "api_key_read": False,
        "case_outcomes_loaded": False,
        "descriptive_statistics_computed": False,
        "results_written": False,
        "private_truth_exposed": False,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the outcome-blind source-behavior analysis lock"
    )
    parser.add_argument("--analysis-lock", default=DEFAULT_ANALYSIS_LOCK)
    parser.add_argument("--approval", default=DEFAULT_ANALYSIS_APPROVAL)
    arguments = parser.parse_args(argv)
    print(safe_json(safe_preflight(arguments.analysis_lock, arguments.approval)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ANALYSIS_APPROVAL",
    "DEFAULT_ANALYSIS_LOCK",
    "DescriptiveContrastId",
    "PlannedDescriptiveContrast",
    "SourceBehaviorAnalysisApproval",
    "SourceBehaviorAnalysisLock",
    "analysis_lock_sha256",
    "load_analysis_approval",
    "load_analysis_lock",
    "safe_preflight",
]
