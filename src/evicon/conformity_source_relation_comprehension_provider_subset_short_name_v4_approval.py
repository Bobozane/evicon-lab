"""Hash-bound approval boundary for the short-schema-name v4 amendment.

The v1, v2, and v3 source-relation compatibility attempts are immutable
parents.  Version 4 changes exactly one wire-level field relative to v3:
``response_format.json_schema.name``.  It remains offline by default and an
accepted review never grants standing network permission.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_identification import sha256_file
from .conformity_source_relation_comprehension_calibration_v1 import (
    RESPONSE_SCHEMA_SHA256 as CANONICAL_SCHEMA_SHA256,
)
from .conformity_source_relation_comprehension_provider_subset_v2 import (
    PROVIDER_RESPONSE_SCHEMA_SHA256,
    PROVIDER_SCHEMA_NAME,
    PROVIDER_SCHEMA_VERSION,
)


_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = (
    "configs/provenance_cascade/identification/"
    "conformity_source_relation_comprehension_compatibility_"
    "provider_subset_short_name.v4.toml"
)
DEFAULT_APPROVAL = (
    "configs/provenance_cascade/identification/"
    "conformity_source_relation_comprehension_compatibility_"
    "provider_subset_short_name_approval.v4.toml"
)
DEFAULT_MODULE = (
    "src/evicon/"
    "conformity_source_relation_comprehension_compatibility_"
    "provider_subset_short_name_v4.py"
)
PROVIDER_SCHEMA_MODULE = (
    "src/evicon/conformity_source_relation_comprehension_provider_subset_v2.py"
)

V1_CONFIG = (
    "configs/provenance_cascade/identification/"
    "conformity_source_relation_comprehension_compatibility.v1.toml"
)
V1_APPROVAL = (
    "configs/provenance_cascade/identification/"
    "conformity_source_relation_comprehension_compatibility_approval.v1.toml"
)
V1_MODULE = "src/evicon/conformity_source_relation_comprehension_compatibility_v1.py"
V1_ATTEMPT_CLAIM = (
    "outputs/study-locks/conformity_source_relation_comprehension_"
    "compatibility.v1.claim"
)
V1_RECEIPT = (
    "outputs/study-locks/conformity_source_relation_comprehension_"
    "compatibility.v1.json"
)

V2_CONFIG = (
    "configs/provenance_cascade/identification/"
    "conformity_source_relation_comprehension_compatibility_provider_subset.v2.toml"
)
V2_APPROVAL = (
    "configs/provenance_cascade/identification/"
    "conformity_source_relation_comprehension_compatibility_"
    "provider_subset_approval.v2.toml"
)
V2_MODULE = (
    "src/evicon/conformity_source_relation_comprehension_"
    "compatibility_provider_subset_v2.py"
)
V2_ATTEMPT_CLAIM = (
    "outputs/study-locks/conformity_source_relation_comprehension_"
    "compatibility_provider_subset.v2.claim"
)
V2_RECEIPT = (
    "outputs/study-locks/conformity_source_relation_comprehension_"
    "compatibility_provider_subset.v2.json"
)

V3_CONFIG = (
    "configs/provenance_cascade/identification/"
    "conformity_source_relation_comprehension_compatibility_"
    "provider_subset_timeout.v3.toml"
)
V3_APPROVAL = (
    "configs/provenance_cascade/identification/"
    "conformity_source_relation_comprehension_compatibility_"
    "provider_subset_timeout_approval.v3.toml"
)
V3_MODULE = (
    "src/evicon/conformity_source_relation_comprehension_"
    "compatibility_provider_subset_timeout_v3.py"
)
V3_ATTEMPT_CLAIM = (
    "outputs/study-locks/conformity_source_relation_comprehension_"
    "compatibility_provider_subset_timeout.v3.claim"
)
V3_RECEIPT = (
    "outputs/study-locks/conformity_source_relation_comprehension_"
    "compatibility_provider_subset_timeout.v3.json"
)

SHORT_PROVIDER_SCHEMA_NAME = "source_relation_comprehension_v1_subset_v4"
MAX_PROVIDER_SCHEMA_NAME_LENGTH = 64
COMPATIBILITY_VERSION = (
    "conformity_source_relation_comprehension_compatibility."
    "provider_subset_short_name.v4"
)
APPROVAL_VERSION = (
    "conformity_source_relation_comprehension_compatibility_"
    "provider_subset_short_name_approval.v4"
)


class SourceRelationProviderSubsetShortNameApprovalError(ValueError):
    """Stable, redacted errors for the v4 review boundary."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class SourceRelationProviderSubsetShortNameConfigV4(BaseModel):
    """Immutable scope for the one-field short-name amendment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    compatibility_id: Literal[
        "evicon-source-relation-comprehension-compatibility-"
        "provider-subset-short-name-v4"
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

    v3_payload_delta: tuple[Literal["response_format.json_schema.name"]]
    provider_schema_body_changed: Literal[False]
    canonical_parser_changed: Literal[False]
    prompt_semantics_changed: Literal[False]
    public_material_changed: Literal[False]
    generation_parameters_changed: Literal[False]
    strict_json_schema_retained: Literal[True]

    v3_provider_schema_name: Literal[PROVIDER_SCHEMA_NAME]
    v3_provider_schema_name_length: Literal[71]
    provider_schema_name: Literal[SHORT_PROVIDER_SCHEMA_NAME]
    provider_schema_name_length: Literal[42]
    provider_schema_name_max_length: Literal[MAX_PROVIDER_SCHEMA_NAME_LENGTH]
    provider_schema_version: Literal[PROVIDER_SCHEMA_VERSION]
    provider_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_schema_module_path: Literal[PROVIDER_SCHEMA_MODULE]
    provider_schema_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    compatibility_module_path: Literal[DEFAULT_MODULE]
    compatibility_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    v1_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v1_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v1_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v1_attempt_claim_sha256: Literal[
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ]
    v1_attempt_status: Literal["provider_error"]
    v1_failure_category: Literal["response_format_unsupported"]
    v1_attempt_count: Literal[1]
    v1_receipt_created: Literal[False]

    v2_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v2_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v2_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v2_attempt_claim_sha256: Literal[
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ]
    v2_attempt_status: Literal["provider_error"]
    v2_failure_category: Literal["timeout"]
    v2_attempt_count: Literal[1]
    v2_receipt_created: Literal[False]

    v3_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_attempt_claim_sha256: Literal[
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ]
    v3_attempt_status: Literal["provider_error"]
    v3_failure_category: Literal["http_client_error"]
    v3_attempt_count: Literal[1]
    v3_receipt_created: Literal[False]

    max_tokens: Literal[128]
    temperature: Literal[0.0]
    seed: Literal[20261401]
    max_retries: Literal[0]
    timeout_seconds: Literal[15.0]
    reasoning_effort: Literal["none"]

    @model_validator(mode="after")
    def fixed_scope(self) -> "SourceRelationProviderSubsetShortNameConfigV4":
        if self.v3_payload_delta != ("response_format.json_schema.name",):
            raise ValueError("source_relation_short_name_v4_delta_mismatch")
        if self.provider_schema_sha256 != PROVIDER_RESPONSE_SCHEMA_SHA256:
            raise ValueError("source_relation_short_name_v4_provider_schema_mismatch")
        if self.canonical_schema_sha256 != CANONICAL_SCHEMA_SHA256:
            raise ValueError("source_relation_short_name_v4_canonical_schema_mismatch")
        if self.provider_schema_name_length != len(self.provider_schema_name):
            raise ValueError("source_relation_short_name_v4_name_length_mismatch")
        if self.v3_provider_schema_name_length != len(self.v3_provider_schema_name):
            raise ValueError("source_relation_short_name_v4_parent_name_length_mismatch")
        if self.provider_schema_name_length > self.provider_schema_name_max_length:
            raise ValueError("source_relation_short_name_v4_name_too_long")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.provider_schema_name):
            raise ValueError("source_relation_short_name_v4_name_characters_invalid")
        return self


class SourceRelationProviderSubsetShortNameApprovalV4(BaseModel):
    """Researcher review of v4; it never grants persistent network access."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: Literal[
        "evicon-source-relation-comprehension-compatibility-"
        "provider-subset-short-name-v4"
    ]
    approval_version: Literal[APPROVAL_VERSION]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    accepted_on: str

    compatibility_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_schema_name: Literal[SHORT_PROVIDER_SCHEMA_NAME]
    provider_schema_name_length: Literal[42]
    provider_schema_name_max_length: Literal[MAX_PROVIDER_SCHEMA_NAME_LENGTH]
    provider_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_schema_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v1_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v1_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v1_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v1_attempt_claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v2_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v2_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v2_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v2_attempt_claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_attempt_claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    confirm_only_schema_name_changes: bool
    confirm_name_is_ascii_and_at_most_64_characters: bool
    confirm_one_shot_only: bool
    confirm_zero_retries: bool
    confirm_strict_schema_and_parser_unchanged: bool
    confirm_prompt_material_and_generation_parameters_unchanged: bool
    confirm_v1_v2_v3_immutable: bool
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
    ) -> "SourceRelationProviderSubsetShortNameApprovalV4":
        if self.acceptance_status != "accepted":
            return self
        if not self.accepted_by.strip() or not self.accepted_on.strip():
            raise ValueError("source_relation_short_name_v4_approval_identity_missing")
        try:
            date.fromisoformat(self.accepted_on)
        except ValueError as exc:
            raise ValueError("source_relation_short_name_v4_approval_date_invalid") from exc
        confirmations = (
            self.confirm_only_schema_name_changes,
            self.confirm_name_is_ascii_and_at_most_64_characters,
            self.confirm_one_shot_only,
            self.confirm_zero_retries,
            self.confirm_strict_schema_and_parser_unchanged,
            self.confirm_prompt_material_and_generation_parameters_unchanged,
            self.confirm_v1_v2_v3_immutable,
            self.confirm_no_calibration_or_behavior_execution,
            self.confirm_redacted_receipt_only,
            self.confirm_no_private_or_historical_material,
            self.confirm_explicit_command_required,
            self.confirm_not_effect_or_causal_evidence,
        )
        if not all(confirmations):
            raise ValueError("source_relation_short_name_v4_approval_incomplete")
        return self


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def compatibility_config_sha256(path: str | Path = DEFAULT_CONFIG) -> str:
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def compatibility_approval_sha256(path: str | Path = DEFAULT_APPROVAL) -> str:
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def _parse_config(path: str | Path) -> SourceRelationProviderSubsetShortNameConfigV4:
    try:
        return SourceRelationProviderSubsetShortNameConfigV4.model_validate(
            tomllib.loads(_resolve(path).read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise SourceRelationProviderSubsetShortNameApprovalError(
            "source_relation_short_name_v4_config_invalid"
        ) from exc


def _check_empty_claim(path: str) -> str:
    claim = _resolve(path)
    if not claim.is_file() or claim.stat().st_size != 0:
        raise ValueError("parent claim missing or nonempty")
    return sha256_file(claim)


def _check_no_receipt(path: str) -> None:
    if _resolve(path).exists():
        raise ValueError("parent receipt unexpectedly exists")


def load_compatibility_config(
    path: str | Path = DEFAULT_CONFIG,
) -> SourceRelationProviderSubsetShortNameConfigV4:
    """Load v4 and verify its implementation plus all consumed parents."""

    config = _parse_config(path)
    try:
        _check_no_receipt(V1_RECEIPT)
        _check_no_receipt(V2_RECEIPT)
        _check_no_receipt(V3_RECEIPT)
        expected = {
            "provider_schema_sha256": PROVIDER_RESPONSE_SCHEMA_SHA256,
            "provider_schema_module_sha256": sha256_file(PROVIDER_SCHEMA_MODULE),
            "canonical_schema_sha256": CANONICAL_SCHEMA_SHA256,
            "compatibility_module_sha256": sha256_file(DEFAULT_MODULE),
            "v1_config_sha256": sha256_file(V1_CONFIG),
            "v1_approval_sha256": sha256_file(V1_APPROVAL),
            "v1_module_sha256": sha256_file(V1_MODULE),
            "v1_attempt_claim_sha256": _check_empty_claim(V1_ATTEMPT_CLAIM),
            "v2_config_sha256": sha256_file(V2_CONFIG),
            "v2_approval_sha256": sha256_file(V2_APPROVAL),
            "v2_module_sha256": sha256_file(V2_MODULE),
            "v2_attempt_claim_sha256": _check_empty_claim(V2_ATTEMPT_CLAIM),
            "v3_config_sha256": sha256_file(V3_CONFIG),
            "v3_approval_sha256": sha256_file(V3_APPROVAL),
            "v3_module_sha256": sha256_file(V3_MODULE),
            "v3_attempt_claim_sha256": _check_empty_claim(V3_ATTEMPT_CLAIM),
        }
        observed = {name: getattr(config, name) for name in expected}
        if observed != expected:
            raise ValueError("binding mismatch")
    except Exception as exc:
        if isinstance(exc, SourceRelationProviderSubsetShortNameApprovalError):
            raise
        raise SourceRelationProviderSubsetShortNameApprovalError(
            "source_relation_short_name_v4_binding_mismatch"
        ) from exc
    return config


def load_compatibility_approval(
    path: str | Path = DEFAULT_APPROVAL,
) -> SourceRelationProviderSubsetShortNameApprovalV4:
    """Load v4 review and verify it binds the exact short-name scope."""

    try:
        approval = SourceRelationProviderSubsetShortNameApprovalV4.model_validate(
            tomllib.loads(_resolve(path).read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise SourceRelationProviderSubsetShortNameApprovalError(
            "source_relation_short_name_v4_approval_invalid"
        ) from exc
    try:
        config = load_compatibility_config()
        expected = {
            "compatibility_config_sha256": compatibility_config_sha256(),
            "compatibility_module_sha256": config.compatibility_module_sha256,
            "provider_schema_name": config.provider_schema_name,
            "provider_schema_name_length": config.provider_schema_name_length,
            "provider_schema_name_max_length": config.provider_schema_name_max_length,
            "provider_schema_sha256": config.provider_schema_sha256,
            "provider_schema_module_sha256": config.provider_schema_module_sha256,
            "canonical_schema_sha256": config.canonical_schema_sha256,
            "v1_config_sha256": config.v1_config_sha256,
            "v1_approval_sha256": config.v1_approval_sha256,
            "v1_module_sha256": config.v1_module_sha256,
            "v1_attempt_claim_sha256": config.v1_attempt_claim_sha256,
            "v2_config_sha256": config.v2_config_sha256,
            "v2_approval_sha256": config.v2_approval_sha256,
            "v2_module_sha256": config.v2_module_sha256,
            "v2_attempt_claim_sha256": config.v2_attempt_claim_sha256,
            "v3_config_sha256": config.v3_config_sha256,
            "v3_approval_sha256": config.v3_approval_sha256,
            "v3_module_sha256": config.v3_module_sha256,
            "v3_attempt_claim_sha256": config.v3_attempt_claim_sha256,
        }
        observed = {name: getattr(approval, name) for name in expected}
        if observed != expected:
            raise ValueError("approval binding mismatch")
    except Exception as exc:
        if isinstance(exc, SourceRelationProviderSubsetShortNameApprovalError):
            raise
        raise SourceRelationProviderSubsetShortNameApprovalError(
            "source_relation_short_name_v4_approval_binding_mismatch"
        ) from exc
    return approval


__all__ = [
    "APPROVAL_VERSION",
    "COMPATIBILITY_VERSION",
    "DEFAULT_APPROVAL",
    "DEFAULT_CONFIG",
    "DEFAULT_MODULE",
    "MAX_PROVIDER_SCHEMA_NAME_LENGTH",
    "PROVIDER_SCHEMA_MODULE",
    "SHORT_PROVIDER_SCHEMA_NAME",
    "SourceRelationProviderSubsetShortNameApprovalError",
    "SourceRelationProviderSubsetShortNameApprovalV4",
    "SourceRelationProviderSubsetShortNameConfigV4",
    "V1_APPROVAL",
    "V1_ATTEMPT_CLAIM",
    "V1_CONFIG",
    "V1_MODULE",
    "V1_RECEIPT",
    "V2_APPROVAL",
    "V2_ATTEMPT_CLAIM",
    "V2_CONFIG",
    "V2_MODULE",
    "V2_RECEIPT",
    "V3_APPROVAL",
    "V3_ATTEMPT_CLAIM",
    "V3_CONFIG",
    "V3_MODULE",
    "V3_RECEIPT",
    "compatibility_approval_sha256",
    "compatibility_config_sha256",
    "load_compatibility_approval",
    "load_compatibility_config",
]
