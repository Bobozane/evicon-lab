"""Human approval boundary for the source-relation comprehension calibration.

The approval permits neither a behavior experiment nor the 24-case
calibration.  It records review of the offline protocol; any later network
action still needs its own explicit command and scope confirmation.
"""

from __future__ import annotations

import hashlib
import tomllib
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_identification import sha256_file
from .conformity_source_relation_comprehension_calibration_v1 import (
    DEFAULT_CONFIG,
    RESPONSE_SCHEMA_SHA256,
    load_config,
)


_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_APPROVAL = (
    "configs/provenance_cascade/identification/"
    "conformity_source_relation_comprehension_calibration_approval.v1.toml"
)


class SourceRelationCalibrationApprovalError(ValueError):
    """Stable, non-sensitive approval error boundary."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class SourceRelationCalibrationApproval(BaseModel):
    """Human review of the static calibration design only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: Literal["evicon-source-relation-comprehension-calibration-v1"]
    approval_version: Literal[
        "conformity_source_relation_comprehension_calibration_approval.v1"
    ]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    accepted_on: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirm_public_synthetic_content_only: bool
    confirm_source_structure_not_truth: bool
    confirm_evidence_distinct_from_social_roots: bool
    confirm_counterbalanced_presentation: bool
    confirm_historical_artifacts_excluded: bool
    confirm_fake_smoke_not_model_evidence: bool
    confirm_no_behavior_or_causal_claim: bool
    network_execution_authorized: Literal[False]

    @model_validator(mode="after")
    def accepted_review_is_complete(self) -> "SourceRelationCalibrationApproval":
        if self.acceptance_status != "accepted":
            return self
        if not self.accepted_by.strip() or not self.accepted_on.strip():
            raise ValueError("source_relation_calibration_approval_identity_missing")
        try:
            date.fromisoformat(self.accepted_on)
        except ValueError as exc:
            raise ValueError("source_relation_calibration_approval_date_invalid") from exc
        confirmations = (
            self.confirm_public_synthetic_content_only,
            self.confirm_source_structure_not_truth,
            self.confirm_evidence_distinct_from_social_roots,
            self.confirm_counterbalanced_presentation,
            self.confirm_historical_artifacts_excluded,
            self.confirm_fake_smoke_not_model_evidence,
            self.confirm_no_behavior_or_causal_claim,
        )
        if not all(confirmations):
            raise ValueError("source_relation_calibration_approval_incomplete")
        return self


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def approval_sha256(path: str | Path = DEFAULT_APPROVAL) -> str:
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def load_approval(
    path: str | Path = DEFAULT_APPROVAL,
) -> SourceRelationCalibrationApproval:
    try:
        approval = SourceRelationCalibrationApproval.model_validate(
            tomllib.loads(_resolve(path).read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise SourceRelationCalibrationApprovalError(
            "source_relation_calibration_approval_invalid"
        ) from exc
    try:
        config = load_config(DEFAULT_CONFIG)
    except Exception as exc:
        raise SourceRelationCalibrationApprovalError(
            "source_relation_calibration_approval_config_invalid"
        ) from exc
    if approval.config_sha256 != sha256_file(DEFAULT_CONFIG):
        raise SourceRelationCalibrationApprovalError(
            "source_relation_calibration_approval_config_hash_mismatch"
        )
    if approval.protocol_sha256 != config.protocol_sha256:
        raise SourceRelationCalibrationApprovalError(
            "source_relation_calibration_approval_protocol_hash_mismatch"
        )
    if approval.response_schema_sha256 != RESPONSE_SCHEMA_SHA256:
        raise SourceRelationCalibrationApprovalError(
            "source_relation_calibration_approval_schema_hash_mismatch"
        )
    return approval


def safe_preflight(path: str | Path = DEFAULT_APPROVAL) -> dict[str, object]:
    """Read review status without using environment, provider, or outputs."""

    try:
        approval = load_approval(path)
        config = load_config(DEFAULT_CONFIG)
    except SourceRelationCalibrationApprovalError as exc:
        return {
            "status": "blocked",
            "blocking_reasons": [exc.code],
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
        }
    reasons = []
    if approval.acceptance_status != "accepted":
        reasons.append("human_approval_required")
    reasons.append("provider_compatibility_not_requested")
    return {
        "status": "source_relation_calibration_approval_ready",
        "approval_status": approval.acceptance_status,
        "approval_sha256": approval_sha256(path),
        "config_sha256": approval.config_sha256,
        "protocol_sha256": approval.protocol_sha256,
        "response_schema_sha256": approval.response_schema_sha256,
        "case_count": config.case_count,
        "logical_request_cap": config.logical_request_cap,
        "completion_reservation_cap": config.completion_reservation_cap,
        "blocking_reasons": reasons,
        "ready_for_comprehension_calibration": False,
        "network": "disabled",
        "provider_constructed": False,
        "api_key_read": False,
        "results_written": False,
        "behavior_effect_estimated": False,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


__all__ = [
    "DEFAULT_APPROVAL",
    "SourceRelationCalibrationApproval",
    "SourceRelationCalibrationApprovalError",
    "approval_sha256",
    "load_approval",
    "safe_preflight",
]
