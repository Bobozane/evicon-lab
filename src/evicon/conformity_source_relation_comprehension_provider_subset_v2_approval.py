"""Hash-bound approval and configuration for the provider-subset amendment.

The original v1 compatibility attempt is intentionally a parent record.  This
module creates a new scope after that attempt was rejected for
``response_format_unsupported``.  Accepting this file reviews the amended
wire schema only; it is never standing network permission.
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
    DEFAULT_CONFIG as CALIBRATION_CONFIG,
    PROTOCOL_VERSION,
    RESPONSE_SCHEMA_SHA256,
    SCHEMA_NAME,
    TEMPLATE_VERSION,
    load_config as load_calibration_config,
)
from .conformity_source_relation_comprehension_provider_subset_v2 import (
    PROVIDER_RESPONSE_SCHEMA_SHA256,
    PROVIDER_SCHEMA_NAME,
    PROVIDER_SCHEMA_VERSION,
    REMOVED_PROVIDER_KEYWORDS,
)


_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = (
    "configs/provenance_cascade/identification/"
    "conformity_source_relation_comprehension_compatibility_provider_subset.v2.toml"
)
DEFAULT_APPROVAL = (
    "configs/provenance_cascade/identification/"
    "conformity_source_relation_comprehension_compatibility_provider_subset_approval.v2.toml"
)
DEFAULT_MODULE = (
    "src/evicon/conformity_source_relation_comprehension_compatibility_provider_subset_v2.py"
)
PROVIDER_SCHEMA_MODULE = (
    "src/evicon/conformity_source_relation_comprehension_provider_subset_v2.py"
)
PARENT_CONFIG = (
    "configs/provenance_cascade/identification/"
    "conformity_source_relation_comprehension_compatibility.v1.toml"
)
PARENT_APPROVAL = (
    "configs/provenance_cascade/identification/"
    "conformity_source_relation_comprehension_compatibility_approval.v1.toml"
)
PARENT_MODULE = "src/evicon/conformity_source_relation_comprehension_compatibility_v1.py"
PARENT_ATTEMPT_CLAIM = (
    "outputs/study-locks/conformity_source_relation_comprehension_compatibility.v1.claim"
)

COMPATIBILITY_VERSION = (
    "conformity_source_relation_comprehension_compatibility.provider_subset.v2"
)
APPROVAL_VERSION = (
    "conformity_source_relation_comprehension_compatibility_provider_subset_approval.v2"
)


class SourceRelationProviderSubsetApprovalError(ValueError):
    """Stable, redacted errors for this amendment's governance boundary."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class SourceRelationProviderSubsetConfigV2(BaseModel):
    """Immutable scope of the one-shot provider-subset compatibility check."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    compatibility_id: Literal[
        "evicon-source-relation-comprehension-compatibility-provider-subset-v2"
    ]
    compatibility_version: Literal[COMPATIBILITY_VERSION]
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
    calibration_protocol_version: Literal[PROTOCOL_VERSION]
    template_version: Literal[TEMPLATE_VERSION]
    canonical_schema_name: Literal[SCHEMA_NAME]
    canonical_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_schema_name: Literal[PROVIDER_SCHEMA_NAME]
    provider_schema_version: Literal[PROVIDER_SCHEMA_VERSION]
    provider_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_schema_module_path: Literal[PROVIDER_SCHEMA_MODULE]
    provider_schema_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    removed_provider_keywords: tuple[str, ...]
    local_parser_remains_strict: Literal[True]
    prompt_semantics_changed: Literal[False]
    public_material_changed: Literal[False]

    compatibility_module_path: Literal[DEFAULT_MODULE]
    compatibility_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    parent_compatibility_config_path: Literal[PARENT_CONFIG]
    parent_compatibility_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_compatibility_approval_path: Literal[PARENT_APPROVAL]
    parent_compatibility_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_compatibility_module_path: Literal[PARENT_MODULE]
    parent_compatibility_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_attempt_claim_path: Literal[PARENT_ATTEMPT_CLAIM]
    parent_attempt_claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_attempt_status: Literal["provider_error"]
    parent_failure_category: Literal["response_format_unsupported"]
    parent_receipt_created: Literal[False]

    max_tokens: Literal[128]
    temperature: Literal[0.0]
    seed: Literal[20261401]
    max_retries: Literal[0]
    timeout_seconds: Literal[5.0]
    reasoning_effort: Literal["none"]

    @model_validator(mode="after")
    def fixed_scope(self) -> "SourceRelationProviderSubsetConfigV2":
        if self.removed_provider_keywords != tuple(REMOVED_PROVIDER_KEYWORDS):
            raise ValueError("source_relation_provider_subset_removed_keyword_mismatch")
        if self.canonical_schema_sha256 != RESPONSE_SCHEMA_SHA256:
            raise ValueError("source_relation_provider_subset_canonical_schema_mismatch")
        if self.provider_schema_sha256 != PROVIDER_RESPONSE_SCHEMA_SHA256:
            raise ValueError("source_relation_provider_subset_schema_mismatch")
        return self


class SourceRelationProviderSubsetApprovalV2(BaseModel):
    """Researcher review of the amended wire contract, not network permission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: Literal[
        "evicon-source-relation-comprehension-compatibility-provider-subset-v2"
    ]
    approval_version: Literal[APPROVAL_VERSION]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    accepted_on: str
    compatibility_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_compatibility_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_compatibility_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_compatibility_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_attempt_claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_failure_category: Literal["response_format_unsupported"]
    provider_schema_name: Literal[PROVIDER_SCHEMA_NAME]
    provider_schema_version: Literal[PROVIDER_SCHEMA_VERSION]
    confirm_one_shot_only: bool
    confirm_provider_subset_only: bool
    confirm_local_parser_remains_strict: bool
    confirm_prompt_and_public_material_unchanged: bool
    confirm_parent_v1_immutable: bool
    confirm_no_calibration_or_behavior_execution: bool
    confirm_redacted_receipt_only: bool
    confirm_no_private_or_historical_material: bool
    confirm_explicit_command_required: bool
    confirm_not_effect_or_causal_evidence: bool
    persistent_network_authorization_granted: Literal[False]
    explicit_command_authorization_required: Literal[True]

    @model_validator(mode="after")
    def accepted_review_is_complete(
        self,
    ) -> "SourceRelationProviderSubsetApprovalV2":
        if self.acceptance_status != "accepted":
            return self
        if not self.accepted_by.strip() or not self.accepted_on.strip():
            raise ValueError("source_relation_provider_subset_approval_identity_missing")
        try:
            date.fromisoformat(self.accepted_on)
        except ValueError as exc:
            raise ValueError("source_relation_provider_subset_approval_date_invalid") from exc
        confirmations = (
            self.confirm_one_shot_only,
            self.confirm_provider_subset_only,
            self.confirm_local_parser_remains_strict,
            self.confirm_prompt_and_public_material_unchanged,
            self.confirm_parent_v1_immutable,
            self.confirm_no_calibration_or_behavior_execution,
            self.confirm_redacted_receipt_only,
            self.confirm_no_private_or_historical_material,
            self.confirm_explicit_command_required,
            self.confirm_not_effect_or_causal_evidence,
        )
        if not all(confirmations):
            raise ValueError("source_relation_provider_subset_approval_incomplete")
        return self


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def compatibility_config_sha256(path: str | Path = DEFAULT_CONFIG) -> str:
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def compatibility_approval_sha256(path: str | Path = DEFAULT_APPROVAL) -> str:
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def _parse_config(path: str | Path) -> SourceRelationProviderSubsetConfigV2:
    try:
        return SourceRelationProviderSubsetConfigV2.model_validate(
            tomllib.loads(_resolve(path).read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise SourceRelationProviderSubsetApprovalError(
            "source_relation_provider_subset_config_invalid"
        ) from exc


def load_compatibility_config(
    path: str | Path = DEFAULT_CONFIG,
) -> SourceRelationProviderSubsetConfigV2:
    """Load the amendment and verify every parent/static hash binding."""

    config_path = _resolve(path)
    config = _parse_config(config_path)
    try:
        calibration = load_calibration_config(CALIBRATION_CONFIG)
        parent_claim = _resolve(config.parent_attempt_claim_path)
        if not parent_claim.is_file():
            raise ValueError("parent claim missing")
        expected = {
            "calibration_config_sha256": sha256_file(CALIBRATION_CONFIG),
            "calibration_protocol_sha256": calibration.protocol_sha256,
            "parent_compatibility_config_sha256": sha256_file(PARENT_CONFIG),
            "parent_compatibility_approval_sha256": sha256_file(PARENT_APPROVAL),
            "parent_compatibility_module_sha256": sha256_file(PARENT_MODULE),
            "parent_attempt_claim_sha256": sha256_file(parent_claim),
            "provider_schema_module_sha256": sha256_file(PROVIDER_SCHEMA_MODULE),
            "compatibility_module_sha256": sha256_file(DEFAULT_MODULE),
        }
        observed = {name: getattr(config, name) for name in expected}
        if observed != expected:
            raise ValueError("source_relation_provider_subset_binding_mismatch")
    except Exception as exc:
        if isinstance(exc, SourceRelationProviderSubsetApprovalError):
            raise
        raise SourceRelationProviderSubsetApprovalError(
            "source_relation_provider_subset_binding_mismatch"
        ) from exc
    return config


def load_compatibility_approval(
    path: str | Path = DEFAULT_APPROVAL,
) -> SourceRelationProviderSubsetApprovalV2:
    """Load the separate amendment approval and check its exact bindings."""

    try:
        approval = SourceRelationProviderSubsetApprovalV2.model_validate(
            tomllib.loads(_resolve(path).read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise SourceRelationProviderSubsetApprovalError(
            "source_relation_provider_subset_approval_invalid"
        ) from exc
    try:
        config = load_compatibility_config()
        expected = {
            "compatibility_config_sha256": compatibility_config_sha256(),
            "calibration_config_sha256": config.calibration_config_sha256,
            "calibration_protocol_sha256": config.calibration_protocol_sha256,
            "canonical_schema_sha256": config.canonical_schema_sha256,
            "provider_schema_sha256": config.provider_schema_sha256,
            "compatibility_module_sha256": config.compatibility_module_sha256,
            "parent_compatibility_config_sha256": config.parent_compatibility_config_sha256,
            "parent_compatibility_approval_sha256": config.parent_compatibility_approval_sha256,
            "parent_compatibility_module_sha256": config.parent_compatibility_module_sha256,
            "parent_attempt_claim_sha256": config.parent_attempt_claim_sha256,
            "parent_failure_category": config.parent_failure_category,
            "provider_schema_name": config.provider_schema_name,
            "provider_schema_version": config.provider_schema_version,
        }
        observed = {name: getattr(approval, name) for name in expected}
        if observed != expected:
            raise ValueError("source_relation_provider_subset_approval_binding_mismatch")
    except Exception as exc:
        if isinstance(exc, SourceRelationProviderSubsetApprovalError):
            raise
        raise SourceRelationProviderSubsetApprovalError(
            "source_relation_provider_subset_approval_binding_mismatch"
        ) from exc
    return approval


__all__ = [
    "APPROVAL_VERSION",
    "COMPATIBILITY_VERSION",
    "DEFAULT_APPROVAL",
    "DEFAULT_CONFIG",
    "DEFAULT_MODULE",
    "PARENT_ATTEMPT_CLAIM",
    "PARENT_APPROVAL",
    "PARENT_CONFIG",
    "PARENT_MODULE",
    "PROVIDER_SCHEMA_MODULE",
    "SourceRelationProviderSubsetApprovalError",
    "SourceRelationProviderSubsetApprovalV2",
    "SourceRelationProviderSubsetConfigV2",
    "compatibility_approval_sha256",
    "compatibility_config_sha256",
    "load_compatibility_approval",
    "load_compatibility_config",
]
