"""Static authorization bindings for the one-shot compatibility gate.

The source-relation comprehension calibration remains an offline-only protocol.
This module is deliberately separate: it binds the one-request Provider
compatibility check without changing the calibration protocol's network policy.
An accepted review still requires the caller to explicitly pass
``--allow-network`` for that one request.
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
    PROTOCOL_VERSION,
    RESPONSE_SCHEMA_SHA256,
    SCHEMA_NAME,
    TEMPLATE_VERSION,
    load_config,
)


_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_COMPATIBILITY_CONFIG = (
    "configs/provenance_cascade/identification/"
    "conformity_source_relation_comprehension_compatibility.v1.toml"
)
DEFAULT_COMPATIBILITY_APPROVAL = (
    "configs/provenance_cascade/identification/"
    "conformity_source_relation_comprehension_compatibility_approval.v1.toml"
)
DEFAULT_COMPATIBILITY_MODULE = (
    "src/evicon/conformity_source_relation_comprehension_compatibility_v1.py"
)


class SourceRelationCompatibilityApprovalError(ValueError):
    """Stable, redacted failure codes for the compatibility authorization path."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class SourceRelationCompatibilityConfigV1(BaseModel):
    """Fixed, public, one-request compatibility scope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    compatibility_id: Literal["evicon-source-relation-comprehension-compatibility-v1"]
    compatibility_version: Literal[
        "conformity_source_relation_comprehension_compatibility.v1"
    ]
    status: Literal["awaiting_explicit_authorization"]
    development_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    network_enabled_by_default: Literal[False]
    explicit_allow_network_required: Literal[True]
    one_shot_http_request_cap: Literal[1]
    receipt_write_permitted_on_completed_check: Literal[True]
    calibration_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_protocol_version: Literal[
        "conformity_source_relation_comprehension_calibration.v1"
    ]
    template_version: Literal["conformity_source_relation_comprehension_turn.v1"]
    response_schema_name: Literal["conformity_source_relation_comprehension_response_v1"]
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_module_path: Literal[
        "src/evicon/conformity_source_relation_comprehension_compatibility_v1.py"
    ]
    compatibility_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    max_tokens: Literal[128]
    temperature: Literal[0.0]
    seed: Literal[20261401]
    max_retries: Literal[0]
    timeout_seconds: Literal[5.0]
    reasoning_effort: Literal["none"]


class SourceRelationCompatibilityApprovalV1(BaseModel):
    """Human review of the compatibility scope, not standing network permission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: Literal[
        "evicon-source-relation-comprehension-compatibility-approval-v1"
    ]
    approval_version: Literal[
        "conformity_source_relation_comprehension_compatibility_approval.v1"
    ]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    accepted_on: str
    compatibility_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirm_one_shot_only: bool
    confirm_minimal_public_synthetic_content_only: bool
    confirm_no_behavior_calibration_or_experiment: bool
    confirm_redacted_receipt_only: bool
    confirm_no_historical_or_private_material: bool
    confirm_explicit_command_required: bool
    confirm_not_effect_or_causal_evidence: bool
    persistent_network_authorization_granted: Literal[False]
    explicit_command_authorization_required: Literal[True]

    @model_validator(mode="after")
    def accepted_review_is_complete(self) -> "SourceRelationCompatibilityApprovalV1":
        if self.acceptance_status != "accepted":
            return self
        if not self.accepted_by.strip() or not self.accepted_on.strip():
            raise ValueError("source_relation_compatibility_approval_identity_missing")
        try:
            date.fromisoformat(self.accepted_on)
        except ValueError as exc:
            raise ValueError("source_relation_compatibility_approval_date_invalid") from exc
        confirmations = (
            self.confirm_one_shot_only,
            self.confirm_minimal_public_synthetic_content_only,
            self.confirm_no_behavior_calibration_or_experiment,
            self.confirm_redacted_receipt_only,
            self.confirm_no_historical_or_private_material,
            self.confirm_explicit_command_required,
            self.confirm_not_effect_or_causal_evidence,
        )
        if not all(confirmations):
            raise ValueError("source_relation_compatibility_approval_incomplete")
        return self


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def compatibility_config_sha256(
    path: str | Path = DEFAULT_COMPATIBILITY_CONFIG,
) -> str:
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def compatibility_approval_sha256(
    path: str | Path = DEFAULT_COMPATIBILITY_APPROVAL,
) -> str:
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def load_compatibility_config(
    path: str | Path = DEFAULT_COMPATIBILITY_CONFIG,
) -> SourceRelationCompatibilityConfigV1:
    """Load and bind the compatibility-specific static scope."""

    try:
        config = SourceRelationCompatibilityConfigV1.model_validate(
            tomllib.loads(_resolve(path).read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise SourceRelationCompatibilityApprovalError(
            "source_relation_compatibility_config_invalid"
        ) from exc
    try:
        calibration = load_config(DEFAULT_CONFIG)
    except Exception as exc:
        raise SourceRelationCompatibilityApprovalError(
            "source_relation_compatibility_calibration_config_invalid"
        ) from exc
    expected = {
        "calibration_config_sha256": sha256_file(DEFAULT_CONFIG),
        "calibration_protocol_sha256": calibration.protocol_sha256,
        "calibration_protocol_version": PROTOCOL_VERSION,
        "template_version": TEMPLATE_VERSION,
        "response_schema_name": SCHEMA_NAME,
        "response_schema_sha256": RESPONSE_SCHEMA_SHA256,
        "compatibility_module_path": DEFAULT_COMPATIBILITY_MODULE,
        "compatibility_module_sha256": sha256_file(DEFAULT_COMPATIBILITY_MODULE),
    }
    observed = {name: getattr(config, name) for name in expected}
    if observed != expected:
        raise SourceRelationCompatibilityApprovalError(
            "source_relation_compatibility_config_binding_mismatch"
        )
    return config


def load_compatibility_approval(
    path: str | Path = DEFAULT_COMPATIBILITY_APPROVAL,
) -> SourceRelationCompatibilityApprovalV1:
    """Load the dedicated approval only when it matches the fixed scope."""

    try:
        approval = SourceRelationCompatibilityApprovalV1.model_validate(
            tomllib.loads(_resolve(path).read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise SourceRelationCompatibilityApprovalError(
            "source_relation_compatibility_approval_invalid"
        ) from exc
    config = load_compatibility_config()
    expected = {
        "compatibility_config_sha256": compatibility_config_sha256(),
        "calibration_config_sha256": config.calibration_config_sha256,
        "calibration_protocol_sha256": config.calibration_protocol_sha256,
        "response_schema_sha256": config.response_schema_sha256,
        "compatibility_module_sha256": config.compatibility_module_sha256,
    }
    observed = {name: getattr(approval, name) for name in expected}
    if observed != expected:
        raise SourceRelationCompatibilityApprovalError(
            "source_relation_compatibility_approval_binding_mismatch"
        )
    return approval


__all__ = [
    "DEFAULT_COMPATIBILITY_APPROVAL",
    "DEFAULT_COMPATIBILITY_CONFIG",
    "DEFAULT_COMPATIBILITY_MODULE",
    "SourceRelationCompatibilityApprovalError",
    "SourceRelationCompatibilityApprovalV1",
    "SourceRelationCompatibilityConfigV1",
    "compatibility_approval_sha256",
    "compatibility_config_sha256",
    "load_compatibility_approval",
    "load_compatibility_config",
]
