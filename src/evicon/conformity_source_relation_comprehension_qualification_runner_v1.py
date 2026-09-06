"""Bounded 24-case execution gate for source-relation structure qualification.

This is deliberately narrower than a behavior study.  It checks whether a
fixed model endpoint can return the required *public source-structure
description* for all 24 pre-written cases.  It does not measure adoption,
conformity, evidence use, truth, a network, or an intervention.

The default CLI path is inert.  A real run requires a separate accepted
approval, a current one-shot compatibility receipt, explicit command flags,
and an empty output location.  The only persisted case data are redacted
structural audits and an immutable attempt receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import tomllib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_identification import safe_json, sha256_file
from .conformity_source_relation_comprehension_calibration_v1 import (
    DEFAULT_CONFIG as CALIBRATION_CONFIG,
    PROTOCOL_VERSION,
    RESPONSE_SCHEMA,
    RESPONSE_SCHEMA_SHA256,
    SCHEMA_NAME,
    TEMPLATE_VERSION,
    EvidenceState,
    PresentationCounterbalance,
    SafeSourceRelationCalibrationAudit,
    SourceRelation,
    SourceRelationCalibrationContext,
    build_cases,
    load_config as load_calibration_config,
    parse_response,
    render_request,
    safe_case_audit,
)
from .conformity_source_relation_comprehension_compatibility_v1 import (
    DEFAULT_RECEIPT as COMPATIBILITY_RECEIPT,
    CompatibilityReceiptV1,
    load_receipt_v1,
)
from .conformity_source_relation_comprehension_compatibility_approval_v1 import (
    DEFAULT_COMPATIBILITY_APPROVAL,
    load_compatibility_approval,
)
from .llm_contract import LLMProviderError, ProviderErrorCode
from .openai_provider import (
    OpenAICompatibleProvider,
    OpenAICompatibleTransport,
    ProviderConfig,
    ReasoningEffort,
    ResponseFormatMode,
    UrllibTransport,
)


_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG: Final[str] = (
    "configs/provenance_cascade/identification/"
    "conformity_source_relation_comprehension_qualification.v1.toml"
)
DEFAULT_APPROVAL: Final[str] = (
    "configs/provenance_cascade/identification/"
    "conformity_source_relation_comprehension_qualification_approval.v1.toml"
)
DEFAULT_RUNNER: Final[str] = (
    "src/evicon/conformity_source_relation_comprehension_qualification_runner_v1.py"
)
DEFAULT_OUTPUT_ROOT: Final[str] = (
    "outputs/conformity-source-relation-comprehension-qualification-v1"
)
DEFAULT_ATTEMPT_CLAIM: Final[str] = (
    "outputs/study-locks/conformity_source_relation_comprehension_qualification.v1.claim"
)
SAFE_AUDIT_NAME: Final[str] = "safe_case_audits.jsonl"
ATTEMPT_RECEIPT_NAME: Final[str] = "qualification_attempt_receipt.json"
RUNNER_VERSION: Final[str] = "conformity_source_relation_comprehension_qualification_runner.v1"
RECEIPT_VERSION: Final[str] = "conformity_source_relation_comprehension_qualification_receipt.v1"
SCHEMA_VERSION: Final[str] = "conformity_source_relation_comprehension_response.v1"
REQUEST_CAP: Final[int] = 24
COMPLETION_RESERVATION_CAP: Final[int] = 3072
MAX_TOKENS: Final[int] = 128
TEMPERATURE: Final[float] = 0.0
SEED: Final[int] = 20261401
MAX_RETRIES: Final[int] = 0
TIMEOUT_SECONDS: Final[float] = 5.0
REASONING_EFFORT: Final[ReasoningEffort] = ReasoningEffort.NONE


class SourceRelationQualificationError(ValueError):
    """Stable and non-sensitive execution boundary."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class SourceRelationQualificationArtifactWriteError(RuntimeError):
    """Preserve the safe artifact state when receipt creation is interrupted."""

    def __init__(self, audit_path: Path | None, receipt_path: Path | None) -> None:
        self.audit_path = audit_path
        self.receipt_path = receipt_path
        super().__init__("source_relation_qualification_artifact_write_failed")


class SourceRelationQualificationConfig(BaseModel):
    """Hash-bound scope of the separate 24-case execution gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    qualification_id: Literal["evicon-source-relation-comprehension-qualification-v1"]
    qualification_version: Literal["conformity_source_relation_comprehension_qualification.v1"]
    runner_version: Literal[
        "conformity_source_relation_comprehension_qualification_runner.v1"
    ]
    status: Literal["awaiting_explicit_authorization"]
    development_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    network_enabled_by_default: Literal[False]
    explicit_allow_network_required: Literal[True]
    explicit_run_confirmation_required: Literal[True]
    real_provider_permitted_after_authorization: Literal[True]
    api_key_read_permitted_after_authorization: Literal[True]
    results_write_permitted_after_authorization: Literal[True]
    evaluator_private_truth_available: Literal[False]
    historical_artifacts_excluded: Literal[True]
    public_structure_task_only: Literal[True]
    compatibility_receipt_required: Literal[True]
    compatibility_receipt_path: Literal[
        "outputs/study-locks/"
        "conformity_source_relation_comprehension_compatibility.v1.json"
    ]
    compatibility_receipt_version: Literal[
        "conformity_source_relation_comprehension_compatibility_receipt.v1"
    ]
    calibration_config_path: Literal[
        "configs/provenance_cascade/identification/"
        "conformity_source_relation_comprehension_calibration.v1.toml"
    ]
    calibration_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_protocol_version: Literal[
        "conformity_source_relation_comprehension_calibration.v1"
    ]
    template_version: Literal["conformity_source_relation_comprehension_turn.v1"]
    response_schema_name: Literal["conformity_source_relation_comprehension_response_v1"]
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runner_path: Literal[
        "src/evicon/conformity_source_relation_comprehension_qualification_runner_v1.py"
    ]
    runner_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_model: str = Field(min_length=1, max_length=160)
    case_count: Literal[24]
    logical_request_cap: Literal[24]
    completion_reservation_cap: Literal[3072]
    max_tokens: Literal[128]
    temperature: Literal[0.0]
    seed: Literal[20261401]
    max_retries: Literal[0]
    timeout_seconds: Literal[5.0]
    reasoning_effort: Literal["none"]
    output_root: Literal[
        "outputs/conformity-source-relation-comprehension-qualification-v1"
    ]
    attempt_claim_path: Literal[
        "outputs/study-locks/conformity_source_relation_comprehension_qualification.v1.claim"
    ]

    @model_validator(mode="after")
    def fixed_execution_scope(self) -> "SourceRelationQualificationConfig":
        if self.logical_request_cap != self.case_count:
            raise ValueError("source_relation_qualification_request_cap_mismatch")
        if self.completion_reservation_cap != self.max_tokens * self.case_count:
            raise ValueError("source_relation_qualification_reservation_cap_mismatch")
        return self


class SourceRelationQualificationApproval(BaseModel):
    """Human review of this execution scope, without standing network permission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: Literal["evicon-source-relation-comprehension-qualification-v1"]
    approval_version: Literal[
        "conformity_source_relation_comprehension_qualification_approval.v1"
    ]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    accepted_on: str
    qualification_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runner_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_model: str = Field(min_length=1, max_length=160)
    confirm_public_structure_task_only: bool
    confirm_compatibility_receipt_required: bool
    confirm_full_24_case_coverage_required: bool
    confirm_no_retry_and_terminal_failure: bool
    confirm_no_private_or_historical_material: bool
    confirm_not_behavior_or_causal_evidence: bool
    persistent_network_authorization_granted: Literal[False]
    explicit_command_authorization_required: Literal[True]

    @model_validator(mode="after")
    def accepted_approval_is_complete(self) -> "SourceRelationQualificationApproval":
        if self.acceptance_status != "accepted":
            return self
        if not self.accepted_by.strip() or not self.accepted_on.strip():
            raise ValueError("source_relation_qualification_approval_identity_missing")
        try:
            from datetime import date

            date.fromisoformat(self.accepted_on)
        except ValueError as exc:
            raise ValueError("source_relation_qualification_approval_date_invalid") from exc
        confirmations = (
            self.confirm_public_structure_task_only,
            self.confirm_compatibility_receipt_required,
            self.confirm_full_24_case_coverage_required,
            self.confirm_no_retry_and_terminal_failure,
            self.confirm_no_private_or_historical_material,
            self.confirm_not_behavior_or_causal_evidence,
        )
        if not all(confirmations):
            raise ValueError("source_relation_qualification_approval_incomplete")
        return self


class QualificationSchemaContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    response_format: Literal["json_schema"] = "json_schema"
    schema_name: Literal["conformity_source_relation_comprehension_response_v1"] = SCHEMA_NAME
    schema_version: Literal["conformity_source_relation_comprehension_response.v1"] = (
        SCHEMA_VERSION
    )
    schema_sha256: str = Field(default=RESPONSE_SCHEMA_SHA256, pattern=r"^[0-9a-f]{64}$")
    template_version: Literal["conformity_source_relation_comprehension_turn.v1"] = (
        TEMPLATE_VERSION
    )


class QualificationGenerationParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tokens: Literal[128] = MAX_TOKENS
    temperature: Literal[0.0] = TEMPERATURE
    seed: Literal[20261401] = SEED
    max_retries: Literal[0] = MAX_RETRIES
    timeout_seconds: Literal[5.0] = TIMEOUT_SECONDS
    reasoning_effort: Literal["none"] = REASONING_EFFORT.value
    logical_request_cap: Literal[24] = REQUEST_CAP
    completion_reservation_cap: Literal[3072] = COMPLETION_RESERVATION_CAP


class QualificationTokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def complete_or_absent(self) -> "QualificationTokenUsage":
        values = (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        if any(value is None for value in values) and not all(value is None for value in values):
            raise ValueError("source_relation_qualification_token_usage_partial")
        if all(value is not None for value in values):
            assert self.prompt_tokens is not None
            assert self.completion_tokens is not None
            if self.total_tokens != self.prompt_tokens + self.completion_tokens:
                raise ValueError("source_relation_qualification_token_usage_mismatch")
        return self

    @property
    def complete(self) -> bool:
        return self.total_tokens is not None


class QualificationCaseAudit(BaseModel):
    """Text-free, root-code-free audit for one successful structural response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^source-relation-v1-c\d{2}$")
    scenario_id: Literal["source-relation-v1-s01", "source-relation-v1-s02"]
    source_relation: SourceRelation
    evidence_state: EvidenceState
    presentation_counterbalance: PresentationCounterbalance
    parser_status: Literal["valid"] = "valid"
    visible_social_message_count: Literal[2]
    visible_social_root_count: Literal[0, 1, 2]
    direct_evidence_card_count: Literal[0, 1]
    support_kind: Literal["social_reports_only", "direct_evidence_present"]
    source_structure_alone_decides_truth: Literal[False]
    finish_reason: Literal["stop"]
    http_status_class: Literal["2xx"]
    token_usage: QualificationTokenUsage
    latency_ms: float = Field(ge=0.0)
    attempt_count: Literal[1] = 1


class QualificationFailureAudit(BaseModel):
    """The only failure surface; it excludes all request and response text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str | None = Field(default=None, pattern=r"^source-relation-v1-c\d{2}$")
    stage: Literal["pre_request", "transport", "response", "binding"]
    error_category: str = Field(pattern=r"^[a-z0-9_]+$")
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls", "other"] | None = None
    http_status_class: Literal["2xx", "3xx", "4xx", "5xx"] | None = None
    parser_valid: bool | None = None
    token_usage: QualificationTokenUsage = Field(default_factory=QualificationTokenUsage)
    latency_ms: float | None = Field(default=None, ge=0.0)
    cumulative_transport_attempt_count: int = Field(ge=0, le=REQUEST_CAP)


class QualificationSafety(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    network_enabled: bool = False
    provider_constructed: bool = False
    api_key_read: bool = False
    compatibility_receipt_loaded: bool = False
    qualification_executed: bool = False
    safe_case_audits_written: bool = False
    attempt_receipt_written: bool = False
    api_key_saved: Literal[False] = False
    prompt_saved: Literal[False] = False
    full_response_saved: Literal[False] = False
    headers_saved: Literal[False] = False
    provider_metadata_saved: Literal[False] = False
    request_ledger_written: Literal[False] = False
    evaluator_private_truth_exposed: Literal[False] = False
    historical_pilot_material_used: Literal[False] = False
    wvs_material_used: Literal[False] = False
    behavior_effect_estimated: Literal[False] = False
    not_paper_result: Literal[True] = True
    no_causal_conclusion: Literal[True] = True


class QualificationRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    status: Literal["network_disabled", "blocked", "completed", "failed"]
    qualification_status: Literal[
        "not_run", "execution_blocked", "execution_failed", "source_structure_qualified"
    ]
    error_code: str | None = Field(default=None, pattern=r"^[a-z0-9_]+$")
    model: str | None = None
    logical_request_count: int = Field(default=0, ge=0, le=REQUEST_CAP)
    transport_attempt_count: int = Field(default=0, ge=0, le=REQUEST_CAP)
    completed_case_count: int = Field(default=0, ge=0, le=REQUEST_CAP)
    parser_invalid_count: int = Field(default=0, ge=0, le=REQUEST_CAP)
    receipt_path: str | None = None
    safe_case_audit_path: str | None = None
    attempt_claim_released: bool | None = None
    claim_cleanup_status: Literal[
        "not_applicable", "acquire_failed", "released", "retained_after_attempt", "release_failed"
    ] = "not_applicable"
    safety: QualificationSafety = Field(default_factory=QualificationSafety)


class QualificationReceipt(BaseModel):
    """Immutable, redacted binding receipt for either a completed or failed attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt_version: Literal[
        "conformity_source_relation_comprehension_qualification_receipt.v1"
    ]
    runner_version: Literal[
        "conformity_source_relation_comprehension_qualification_runner.v1"
    ]
    status: Literal["source_structure_qualified", "execution_failed"]
    model: str = Field(min_length=1, max_length=160)
    qualification_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runner_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_contract: QualificationSchemaContract = Field(alias="schema")
    generation_parameters: QualificationGenerationParameters
    logical_request_count: int = Field(ge=0, le=REQUEST_CAP)
    transport_attempt_count: int = Field(ge=0, le=REQUEST_CAP)
    completed_case_count: int = Field(ge=0, le=REQUEST_CAP)
    parser_invalid_count: int = Field(ge=0, le=REQUEST_CAP)
    safe_case_audit_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    failure_audit: QualificationFailureAudit | None = None
    bindings_unchanged_before_receipt: bool
    safety: QualificationSafety

    @model_validator(mode="after")
    def execution_state_is_internally_consistent(self) -> "QualificationReceipt":
        if self.status == "source_structure_qualified":
            if (
                self.failure_audit is not None
                or self.logical_request_count != REQUEST_CAP
                or self.transport_attempt_count != REQUEST_CAP
                or self.completed_case_count != REQUEST_CAP
                or self.parser_invalid_count != 0
                or self.safe_case_audit_sha256 is None
                or not self.bindings_unchanged_before_receipt
                or not self.safety.qualification_executed
                or not self.safety.safe_case_audits_written
                or not self.safety.attempt_receipt_written
            ):
                raise ValueError("source_relation_qualification_success_receipt_inconsistent")
        elif self.failure_audit is None:
            raise ValueError("source_relation_qualification_failure_receipt_missing_audit")
        if self.completed_case_count == 0 and self.safe_case_audit_sha256 is not None:
            raise ValueError("source_relation_qualification_empty_audit_hash_present")
        if self.completed_case_count > 0 and self.safe_case_audit_sha256 is None:
            raise ValueError("source_relation_qualification_completed_audit_hash_missing")
        if self.safety.safe_case_audits_written != (self.completed_case_count > 0):
            raise ValueError("source_relation_qualification_audit_safety_mismatch")
        if self.logical_request_count < self.completed_case_count:
            raise ValueError("source_relation_qualification_receipt_count_inconsistent")
        if self.transport_attempt_count < self.completed_case_count:
            raise ValueError("source_relation_qualification_receipt_transport_count_inconsistent")
        return self


class _CountingTransport:
    """Hard process-local transport cap, independent of retry settings."""

    def __init__(self, transport: OpenAICompatibleTransport, cap: int) -> None:
        self._transport = transport
        self._cap = cap
        self.calls = 0

    def post(self, url, headers, payload, timeout_seconds):  # type: ignore[no-untyped-def]
        if self.calls >= self._cap:
            raise LLMProviderError(
                ProviderErrorCode.PROVIDER_FAILURE,
                "source_relation_qualification_transport_cap_exceeded",
            )
        self.calls += 1
        return self._transport.post(url, headers, payload, timeout_seconds)


_ERROR_CATEGORY_MAP: Final[dict[str, str]] = {
    ProviderErrorCode.TIMEOUT.value: "timeout",
    ProviderErrorCode.CONNECTION_FAILURE.value: "connection_failure",
    ProviderErrorCode.DNS_OR_URL_FAILURE.value: "dns_or_url_failure",
    ProviderErrorCode.AUTHENTICATION_FAILED.value: "authentication_failed",
    ProviderErrorCode.RATE_LIMITED.value: "rate_limited",
    ProviderErrorCode.HTTP_CLIENT_ERROR.value: "http_client_error",
    ProviderErrorCode.HTTP_SERVER_ERROR.value: "http_server_error",
    ProviderErrorCode.RESPONSE_FORMAT_UNSUPPORTED.value: "response_format_unsupported",
    ProviderErrorCode.MALFORMED_RESPONSE.value: "malformed_provider_response",
    ProviderErrorCode.INVALID_RESPONSE.value: "invalid_provider_response",
    ProviderErrorCode.USAGE_MISMATCH.value: "invalid_token_usage",
    ProviderErrorCode.MISSING_API_KEY.value: "missing_api_key",
    ProviderErrorCode.MISSING_BASE_URL.value: "missing_base_url",
    ProviderErrorCode.MISSING_MODEL.value: "missing_model",
    ProviderErrorCode.PROVIDER_FAILURE.value: "provider_failure",
}


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_toml(path: str | Path, model_type: type[BaseModel], code: str) -> tuple[BaseModel, bytes]:
    try:
        raw = _resolve(path).read_bytes()
        return model_type.model_validate(tomllib.loads(raw.decode("utf-8"))), raw
    except Exception as exc:
        raise SourceRelationQualificationError(code) from exc


def load_qualification_config(
    path: str | Path = DEFAULT_CONFIG,
) -> SourceRelationQualificationConfig:
    parsed, raw = _read_toml(
        path, SourceRelationQualificationConfig, "source_relation_qualification_config_invalid"
    )
    assert isinstance(parsed, SourceRelationQualificationConfig)
    calibration = load_calibration_config(parsed.calibration_config_path)
    expected = {
        "calibration_config_sha256": _sha256_bytes(_resolve(parsed.calibration_config_path).read_bytes()),
        "calibration_protocol_sha256": calibration.protocol_sha256,
        "calibration_protocol_version": PROTOCOL_VERSION,
        "template_version": TEMPLATE_VERSION,
        "response_schema_name": SCHEMA_NAME,
        "response_schema_sha256": RESPONSE_SCHEMA_SHA256,
        "runner_path": DEFAULT_RUNNER,
        "runner_sha256": sha256_file(DEFAULT_RUNNER),
        "case_count": REQUEST_CAP,
        "logical_request_cap": REQUEST_CAP,
        "completion_reservation_cap": COMPLETION_RESERVATION_CAP,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "max_retries": MAX_RETRIES,
        "timeout_seconds": TIMEOUT_SECONDS,
        "reasoning_effort": REASONING_EFFORT.value,
        "output_root": DEFAULT_OUTPUT_ROOT,
        "attempt_claim_path": DEFAULT_ATTEMPT_CLAIM,
        "compatibility_receipt_path": COMPATIBILITY_RECEIPT,
        "compatibility_receipt_version": (
            "conformity_source_relation_comprehension_compatibility_receipt.v1"
        ),
    }
    if {key: getattr(parsed, key) for key in expected} != expected:
        raise SourceRelationQualificationError("source_relation_qualification_config_binding_mismatch")
    if raw != _resolve(path).read_bytes():
        raise SourceRelationQualificationError("source_relation_qualification_config_changed")
    return parsed


def load_qualification_approval(
    path: str | Path = DEFAULT_APPROVAL,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    require_accepted: bool = False,
) -> SourceRelationQualificationApproval:
    config = load_qualification_config(config_path)
    parsed, _ = _read_toml(
        path,
        SourceRelationQualificationApproval,
        "source_relation_qualification_approval_invalid",
    )
    assert isinstance(parsed, SourceRelationQualificationApproval)
    expected = {
        "qualification_config_sha256": _sha256_bytes(_resolve(config_path).read_bytes()),
        "calibration_config_sha256": config.calibration_config_sha256,
        "calibration_protocol_sha256": config.calibration_protocol_sha256,
        "response_schema_sha256": config.response_schema_sha256,
        "runner_sha256": config.runner_sha256,
        "required_model": config.required_model,
    }
    if {key: getattr(parsed, key) for key in expected} != expected:
        raise SourceRelationQualificationError("source_relation_qualification_approval_binding_mismatch")
    if require_accepted and parsed.acceptance_status != "accepted":
        raise SourceRelationQualificationError("source_relation_qualification_approval_not_accepted")
    return parsed


class _FrozenBindings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config: SourceRelationQualificationConfig
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_receipt: CompatibilityReceiptV1


def _freeze_bindings(
    *,
    config_path: str | Path,
    approval_path: str | Path,
    compatibility_receipt_path: str | Path,
    compatibility_approval_path: str | Path,
    require_accepted: bool,
) -> _FrozenBindings:
    config = load_qualification_config(config_path)
    approval = load_qualification_approval(
        approval_path, config_path=config_path, require_accepted=require_accepted
    )
    try:
        compatibility_approval = load_compatibility_approval(compatibility_approval_path)
        if require_accepted and compatibility_approval.acceptance_status != "accepted":
            raise SourceRelationQualificationError(
                "source_relation_qualification_compatibility_approval_not_accepted"
            )
        compatibility_receipt = load_receipt_v1(
            compatibility_receipt_path,
            approval_path=compatibility_approval_path,
        )
    except Exception as exc:
        if isinstance(exc, SourceRelationQualificationError):
            raise
        raise SourceRelationQualificationError(
            "source_relation_qualification_compatibility_receipt_invalid"
        ) from exc
    if compatibility_receipt.model != config.required_model:
        raise SourceRelationQualificationError(
            "source_relation_qualification_compatibility_model_mismatch"
        )
    return _FrozenBindings(
        config=config,
        config_sha256=_sha256_bytes(_resolve(config_path).read_bytes()),
        approval_sha256=_sha256_bytes(_resolve(approval_path).read_bytes()),
        compatibility_receipt_sha256=_sha256_bytes(
            _resolve(compatibility_receipt_path).read_bytes()
        ),
        compatibility_receipt=compatibility_receipt,
    )


def _bindings_current(
    bindings: _FrozenBindings,
    *,
    config_path: str | Path,
    approval_path: str | Path,
    compatibility_receipt_path: str | Path,
    compatibility_approval_path: str | Path,
) -> bool:
    try:
        return _freeze_bindings(
            config_path=config_path,
            approval_path=approval_path,
            compatibility_receipt_path=compatibility_receipt_path,
            compatibility_approval_path=compatibility_approval_path,
            require_accepted=True,
        ) == bindings
    except Exception:
        return False


def _http_status_class(status_code: int | None) -> Literal["2xx", "3xx", "4xx", "5xx"] | None:
    if status_code is None:
        return None
    if 200 <= status_code < 300:
        return "2xx"
    if 300 <= status_code < 400:
        return "3xx"
    if 400 <= status_code < 500:
        return "4xx"
    if 500 <= status_code < 600:
        return "5xx"
    return None


def _safe_finish_reason(value: str) -> Literal["stop", "length", "content_filter", "tool_calls", "other"]:
    if value in {"stop", "length", "content_filter", "tool_calls"}:
        return value  # type: ignore[return-value]
    return "other"


def _usage_from_response(response: object) -> QualificationTokenUsage:
    return QualificationTokenUsage(
        prompt_tokens=getattr(response, "prompt_tokens", None),
        completion_tokens=getattr(response, "completion_tokens", None),
        total_tokens=getattr(response, "total_tokens", None),
    )


def _provider_config(environment: Mapping[str, str] | None, config: SourceRelationQualificationConfig) -> ProviderConfig:
    source = os.environ if environment is None else environment
    base_url = source.get("EVICON_LLM_BASE_URL") or None
    if not base_url:
        raise SourceRelationQualificationError("missing_base_url")
    model = source.get("EVICON_LLM_MODEL") or None
    if not model:
        raise SourceRelationQualificationError("missing_model")
    if model != config.required_model:
        raise SourceRelationQualificationError("source_relation_qualification_model_mismatch")
    return ProviderConfig(
        base_url=base_url,
        model_name=model,
        allow_network=True,
        timeout_seconds=TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        seed=SEED,
        reasoning_effort=REASONING_EFFORT,
        response_format=ResponseFormatMode.JSON_SCHEMA,
        response_schema_name=SCHEMA_NAME,
        response_schema=RESPONSE_SCHEMA,
    )


def _acquire_claim(path: Path) -> None:
    descriptor: int | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.fsync(descriptor)
    except FileExistsError as exc:
        raise SourceRelationQualificationError(
            "source_relation_qualification_attempt_claim_exists"
        ) from exc
    except OSError as exc:
        raise SourceRelationQualificationError(
            "source_relation_qualification_claim_acquire_failed"
        ) from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _release_claim(path: Path) -> bool:
    """Best-effort cleanup after an attempt that did not issue a request."""

    try:
        path.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def _atomic_create(target: Path, payload: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise SourceRelationQualificationError(
                "source_relation_qualification_output_exists"
            ) from exc
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _safe_audit_from_success(
    context: SourceRelationCalibrationContext,
    structural_response: object,
    provider_response: object,
) -> QualificationCaseAudit:
    structural = safe_case_audit(context, structural_response)  # type: ignore[arg-type]
    assert isinstance(structural, SafeSourceRelationCalibrationAudit)
    usage = _usage_from_response(provider_response)
    if not usage.complete:
        raise SourceRelationQualificationError("invalid_token_usage")
    return QualificationCaseAudit(
        **structural.model_dump(mode="python"),
        finish_reason="stop",
        http_status_class="2xx",
        token_usage=usage,
        latency_ms=getattr(provider_response, "latency_ms"),
    )


def _has_complete_coverage(audits: Sequence[QualificationCaseAudit]) -> bool:
    """Require the fixed matrix, not merely 24 successful transport calls."""

    if len(audits) != REQUEST_CAP or len({item.case_id for item in audits}) != REQUEST_CAP:
        return False
    try:
        expected_cases = tuple(build_cases(CALIBRATION_CONFIG))
    except Exception:
        return False
    expected = {
        (
            item.case_id,
            item.scenario_id,
            item.source_relation,
            item.evidence_state,
            item.presentation_counterbalance,
        )
        for item in expected_cases
    }
    actual = {
        (
            item.case_id,
            item.scenario_id,
            item.source_relation,
            item.evidence_state,
            item.presentation_counterbalance,
        )
        for item in audits
    }
    return len(expected) == REQUEST_CAP and actual == expected


def _write_attempt_artifacts(
    *,
    output_root: Path,
    bindings: _FrozenBindings,
    audits: list[QualificationCaseAudit],
    failure: QualificationFailureAudit | None,
    bindings_current: bool,
    bindings_current_check: Callable[[], bool],
    safety: QualificationSafety,
) -> tuple[
    Path | None,
    Path,
    QualificationFailureAudit | None,
    bool,
]:
    audit_path: Path | None = None
    receipt_path: Path | None = None
    candidate_audit_path: Path | None = None
    candidate_receipt_path: Path | None = None
    try:
        output_root.mkdir(parents=True, exist_ok=False)
        audit_sha256: str | None = None
        if audits:
            serialized_audits = "".join(
                json.dumps(
                    item.model_dump(mode="json"),
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for item in audits
            )
            candidate_audit_path = output_root / SAFE_AUDIT_NAME
            _atomic_create(candidate_audit_path, serialized_audits)
            audit_path = candidate_audit_path
            audit_sha256 = hashlib.sha256(serialized_audits.encode("utf-8")).hexdigest()
        if not bindings_current_check():
            bindings_current = False
            if failure is None:
                failure = QualificationFailureAudit(
                    stage="binding",
                    error_category="source_relation_qualification_binding_changed_before_receipt",
                    cumulative_transport_attempt_count=len(audits),
                )
        receipt = QualificationReceipt(
            receipt_version=RECEIPT_VERSION,
            runner_version=RUNNER_VERSION,
            status=(
                "source_structure_qualified"
                if failure is None and bindings_current
                else "execution_failed"
            ),
            model=bindings.config.required_model,
            qualification_config_sha256=bindings.config_sha256,
            calibration_config_sha256=bindings.config.calibration_config_sha256,
            calibration_protocol_sha256=bindings.config.calibration_protocol_sha256,
            qualification_approval_sha256=bindings.approval_sha256,
            runner_sha256=bindings.config.runner_sha256,
            compatibility_receipt_sha256=bindings.compatibility_receipt_sha256,
            schema=QualificationSchemaContract(),
            generation_parameters=QualificationGenerationParameters(),
            logical_request_count=len(audits) + (1 if failure and failure.case_id else 0),
            transport_attempt_count=(
                failure.cumulative_transport_attempt_count if failure else len(audits)
            ),
            completed_case_count=len(audits),
            parser_invalid_count=int(failure is not None and failure.parser_valid is False),
            safe_case_audit_sha256=audit_sha256,
            failure_audit=failure,
            bindings_unchanged_before_receipt=bindings_current,
            safety=safety.model_copy(
                update={
                    "safe_case_audits_written": bool(audits),
                    "attempt_receipt_written": True,
                }
            ),
        )
        candidate_receipt_path = output_root / ATTEMPT_RECEIPT_NAME
        serialized_receipt = (
            json.dumps(
                receipt.model_dump(mode="json", by_alias=True),
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        _atomic_create(candidate_receipt_path, serialized_receipt)
        receipt_path = candidate_receipt_path
        return audit_path, receipt_path, failure, bindings_current
    except Exception as exc:
        actual_audit_path = next(
            (
                path
                for path in (audit_path, candidate_audit_path)
                if path is not None and path.exists()
            ),
            None,
        )
        actual_receipt_path = (
            next(
                (
                    path
                    for path in (receipt_path, candidate_receipt_path)
                    if path is not None and path.exists()
                ),
                None,
            )
        )
        raise SourceRelationQualificationArtifactWriteError(
            actual_audit_path,
            actual_receipt_path,
        ) from exc


def _unique_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate_json_field")
        payload[key] = value
    return payload


def load_qualification_receipt(
    path: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    approval_path: str | Path = DEFAULT_APPROVAL,
    compatibility_receipt_path: str | Path = COMPATIBILITY_RECEIPT,
    compatibility_approval_path: str | Path = DEFAULT_COMPATIBILITY_APPROVAL,
) -> QualificationReceipt:
    """Load only a text-free receipt whose current static bindings still match."""

    target = _resolve(path)
    try:
        payload = json.loads(
            target.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_pairs,
        )
        receipt = QualificationReceipt.model_validate(payload)
    except Exception as exc:
        raise SourceRelationQualificationError(
            "source_relation_qualification_receipt_invalid"
        ) from exc
    try:
        bindings = _freeze_bindings(
            config_path=config_path,
            approval_path=approval_path,
            compatibility_receipt_path=compatibility_receipt_path,
            compatibility_approval_path=compatibility_approval_path,
            require_accepted=False,
        )
    except Exception as exc:
        raise SourceRelationQualificationError(
            "source_relation_qualification_receipt_binding_mismatch"
        ) from exc
    if (
        receipt.model != bindings.config.required_model
        or receipt.qualification_config_sha256 != bindings.config_sha256
        or receipt.calibration_config_sha256 != bindings.config.calibration_config_sha256
        or receipt.calibration_protocol_sha256
        != bindings.config.calibration_protocol_sha256
        or receipt.qualification_approval_sha256 != bindings.approval_sha256
        or receipt.runner_sha256 != bindings.config.runner_sha256
        or receipt.compatibility_receipt_sha256
        != bindings.compatibility_receipt_sha256
        or receipt.schema_contract.schema_sha256
        != bindings.config.response_schema_sha256
    ):
        raise SourceRelationQualificationError(
            "source_relation_qualification_receipt_binding_mismatch"
        )
    audit_path = target.with_name(SAFE_AUDIT_NAME)
    if receipt.safe_case_audit_sha256 is None:
        if audit_path.exists():
            raise SourceRelationQualificationError(
                "source_relation_qualification_receipt_audit_mismatch"
            )
        return receipt
    try:
        audit_raw = audit_path.read_bytes()
        if hashlib.sha256(audit_raw).hexdigest() != receipt.safe_case_audit_sha256:
            raise ValueError("audit_hash_mismatch")
        audits = tuple(
            QualificationCaseAudit.model_validate(
                json.loads(line, object_pairs_hook=_unique_json_pairs)
            )
            for line in audit_raw.decode("utf-8").splitlines()
            if line
        )
        if len(audits) != receipt.completed_case_count:
            raise ValueError("audit_count_mismatch")
        if receipt.status == "source_structure_qualified" and not _has_complete_coverage(
            audits
        ):
            raise ValueError("audit_coverage_mismatch")
    except Exception as exc:
        raise SourceRelationQualificationError(
            "source_relation_qualification_receipt_audit_mismatch"
        ) from exc
    return receipt


def _blocked(
    error_code: str,
    *,
    model: str | None = None,
    safety: QualificationSafety | None = None,
    attempt_claim_released: bool | None = None,
    claim_cleanup_status: Literal[
        "not_applicable", "acquire_failed", "released", "retained_after_attempt", "release_failed"
    ] = "not_applicable",
) -> QualificationRunResult:
    return QualificationRunResult(
        status="blocked",
        qualification_status="execution_blocked",
        error_code=error_code,
        model=model,
        attempt_claim_released=attempt_claim_released,
        claim_cleanup_status=claim_cleanup_status,
        safety=safety or QualificationSafety(),
    )


def safe_preflight(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    approval_path: str | Path = DEFAULT_APPROVAL,
    compatibility_receipt_path: str | Path = COMPATIBILITY_RECEIPT,
    compatibility_approval_path: str | Path = DEFAULT_COMPATIBILITY_APPROVAL,
) -> dict[str, object]:
    """Read static bindings and a receipt if present; never inspect environment or write."""

    try:
        config = load_qualification_config(config_path)
        approval = load_qualification_approval(approval_path, config_path=config_path)
    except SourceRelationQualificationError as exc:
        return {
            "status": "blocked",
            "blocking_reasons": [exc.code],
            "network": "disabled",
            "api_key_read": False,
            "provider_constructed": False,
            "results_written": False,
        }
    compatibility_available = False
    compatibility_approval_accepted = False
    try:
        compatibility_approval = load_compatibility_approval(compatibility_approval_path)
        compatibility_approval_accepted = (
            compatibility_approval.acceptance_status == "accepted"
        )
        receipt = load_receipt_v1(
            compatibility_receipt_path,
            approval_path=compatibility_approval_path,
        )
        compatibility_available = receipt.model == config.required_model
    except Exception:
        pass
    reasons: list[str] = []
    if approval.acceptance_status != "accepted":
        reasons.append("human_approval_required")
    if not compatibility_approval_accepted:
        reasons.append("provider_compatibility_approval_required")
    if not compatibility_available:
        reasons.append("provider_compatibility_receipt_required")
    return {
        "status": "source_relation_qualification_preflight_ready",
        "approval_status": approval.acceptance_status,
        "compatibility_approval_accepted": compatibility_approval_accepted,
        "compatibility_receipt_available": compatibility_available,
        "case_count": config.case_count,
        "logical_request_cap": config.logical_request_cap,
        "completion_reservation_cap": config.completion_reservation_cap,
        "blocking_reasons": reasons,
        "network": "disabled",
        "api_key_read": False,
        "provider_constructed": False,
        "results_written": False,
        "behavior_effect_estimated": False,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def _execute_qualification_v1(
    *,
    allow_network: bool = False,
    confirm_run: bool = False,
    confirm_request_cap: int | None = None,
    confirm_completion_reservation_cap: int | None = None,
    environment: Mapping[str, str] | None = None,
    transport: OpenAICompatibleTransport | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    approval_path: str | Path = DEFAULT_APPROVAL,
    compatibility_receipt_path: str | Path = COMPATIBILITY_RECEIPT,
    compatibility_approval_path: str | Path = DEFAULT_COMPATIBILITY_APPROVAL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    attempt_claim_path: str | Path = DEFAULT_ATTEMPT_CLAIM,
) -> QualificationRunResult:
    """Injectable implementation used only by offline transport tests."""

    if not allow_network:
        return QualificationRunResult(
            status="network_disabled",
            qualification_status="not_run",
            error_code="network_disabled",
        )
    if not confirm_run:
        return _blocked("source_relation_qualification_run_confirmation_required")
    if confirm_request_cap != REQUEST_CAP:
        return _blocked("source_relation_qualification_request_cap_confirmation_mismatch")
    if confirm_completion_reservation_cap != COMPLETION_RESERVATION_CAP:
        return _blocked("source_relation_qualification_reservation_confirmation_mismatch")
    target = _resolve(output_root)
    claim = _resolve(attempt_claim_path)
    if target.exists():
        return _blocked("source_relation_qualification_output_exists")
    if claim.exists():
        return _blocked("source_relation_qualification_attempt_claim_exists")
    try:
        bindings = _freeze_bindings(
            config_path=config_path,
            approval_path=approval_path,
            compatibility_receipt_path=compatibility_receipt_path,
            compatibility_approval_path=compatibility_approval_path,
            require_accepted=True,
        )
    except SourceRelationQualificationError as exc:
        return _blocked(exc.code)
    try:
        _acquire_claim(claim)
    except SourceRelationQualificationError as exc:
        return _blocked(
            exc.code,
            claim_cleanup_status=(
                "acquire_failed"
                if exc.code == "source_relation_qualification_claim_acquire_failed"
                else "not_applicable"
            ),
        )

    # From this point onward the explicit run is authorized and its static
    # compatibility receipt has been loaded, even if no HTTP request occurs.
    safety = QualificationSafety(
        network_enabled=True,
        compatibility_receipt_loaded=True,
    )
    try:
        provider_config = _provider_config(environment, bindings.config)
    except SourceRelationQualificationError as exc:
        released = _release_claim(claim)
        return QualificationRunResult(
            status="failed",
            qualification_status="execution_failed",
            error_code=exc.code,
            model=bindings.config.required_model,
            attempt_claim_released=released,
            claim_cleanup_status="released" if released else "release_failed",
            safety=safety,
        )
    counted_transport = _CountingTransport(transport or UrllibTransport(), REQUEST_CAP)
    try:
        provider = OpenAICompatibleProvider(
            provider_config,
            transport=counted_transport,
            environment=environment,
        )
    except Exception:
        released = _release_claim(claim)
        return QualificationRunResult(
            status="failed",
            qualification_status="execution_failed",
            error_code="provider_construction_failed",
            model=bindings.config.required_model,
            attempt_claim_released=released,
            claim_cleanup_status="released" if released else "release_failed",
            safety=safety.model_copy(update={"api_key_read": True}),
        )
    safety = safety.model_copy(
        update={"provider_constructed": True, "api_key_read": True}
    )
    if not getattr(provider, "_api_key", None):
        released = _release_claim(claim)
        return QualificationRunResult(
            status="failed",
            qualification_status="execution_failed",
            error_code="missing_api_key",
            model=bindings.config.required_model,
            attempt_claim_released=released,
            claim_cleanup_status="released" if released else "release_failed",
            safety=safety,
        )
    if not _bindings_current(
        bindings,
        config_path=config_path,
        approval_path=approval_path,
        compatibility_receipt_path=compatibility_receipt_path,
        compatibility_approval_path=compatibility_approval_path,
    ):
        released = _release_claim(claim)
        return _blocked(
            "source_relation_qualification_binding_changed_before_request",
            model=bindings.config.required_model,
            safety=safety,
            attempt_claim_released=released,
            claim_cleanup_status="released" if released else "release_failed",
        )
    safety = safety.model_copy(
        update={"qualification_executed": True}
    )
    audits: list[QualificationCaseAudit] = []
    failure: QualificationFailureAudit | None = None
    try:
        cases = tuple(
            case.model_copy(update={"model_name": bindings.config.required_model})
            for case in build_cases(CALIBRATION_CONFIG)
        )
    except Exception:
        cases = ()
        failure = QualificationFailureAudit(
            stage="pre_request",
            error_category="internal_execution_failure",
            cumulative_transport_attempt_count=counted_transport.calls,
        )
    for context in cases:
        if failure is not None:
            break
        if not _bindings_current(
            bindings,
            config_path=config_path,
            approval_path=approval_path,
            compatibility_receipt_path=compatibility_receipt_path,
            compatibility_approval_path=compatibility_approval_path,
        ):
            failure = QualificationFailureAudit(
                stage="binding",
                error_category="source_relation_qualification_binding_changed_before_request",
                cumulative_transport_attempt_count=counted_transport.calls,
            )
            break
        try:
            request = render_request(context)
        except Exception:
            failure = QualificationFailureAudit(
                case_id=context.case_id,
                stage="pre_request",
                error_category="internal_execution_failure",
                cumulative_transport_attempt_count=counted_transport.calls,
            )
            break
        try:
            response = provider.complete(request)
        except LLMProviderError as exc:
            failure = QualificationFailureAudit(
                case_id=context.case_id,
                stage="transport",
                error_category=_ERROR_CATEGORY_MAP.get(exc.code.value, "provider_failure"),
                http_status_class=_http_status_class(exc.status_code),
                cumulative_transport_attempt_count=counted_transport.calls,
            )
            break
        except Exception:
            failure = QualificationFailureAudit(
                case_id=context.case_id,
                stage="transport",
                error_category="provider_failure",
                cumulative_transport_attempt_count=counted_transport.calls,
            )
            break
        finish_reason = _safe_finish_reason(response.finish_reason)
        metadata = response.provider_metadata if isinstance(response.provider_metadata, dict) else {}
        status_class = _http_status_class(metadata.get("status_code")) or "2xx"
        try:
            usage = _usage_from_response(response)
        except Exception:
            failure = QualificationFailureAudit(
                case_id=context.case_id,
                stage="response",
                error_category="invalid_token_usage",
                finish_reason=finish_reason,
                http_status_class=status_class,
                parser_valid=None,
                cumulative_transport_attempt_count=counted_transport.calls,
            )
            break
        try:
            parsed_response = parse_response(response.content, context)
            parser_valid = True
        except ValueError as exc:
            parser_valid = False
            parser_error = str(exc) if str(exc) in {
                "malformed_json", "top_level_type", "invalid_schema", "unavailable_content_id",
                "root_assignment_order_mismatch", "root_assignment_mismatch",
                "social_message_count_mismatch", "root_count_mismatch", "source_relation_mismatch",
                "direct_evidence_card_mismatch", "support_kind_mismatch", "source_structure_truth_inference",
            } else "invalid_structure_response"
            failure = QualificationFailureAudit(
                case_id=context.case_id,
                stage="response",
                error_category=parser_error,
                finish_reason=finish_reason,
                http_status_class=status_class,
                parser_valid=False,
                token_usage=usage,
                latency_ms=response.latency_ms,
                cumulative_transport_attempt_count=counted_transport.calls,
            )
            break
        except Exception:
            failure = QualificationFailureAudit(
                case_id=context.case_id,
                stage="response",
                error_category="internal_execution_failure",
                finish_reason=finish_reason,
                http_status_class=status_class,
                parser_valid=None,
                token_usage=usage,
                latency_ms=response.latency_ms,
                cumulative_transport_attempt_count=counted_transport.calls,
            )
            break
        if finish_reason != "stop" or status_class != "2xx" or not usage.complete:
            failure = QualificationFailureAudit(
                case_id=context.case_id,
                stage="response",
                error_category=(
                    "invalid_token_usage" if not usage.complete else "invalid_provider_response"
                ),
                finish_reason=finish_reason,
                http_status_class=status_class,
                parser_valid=parser_valid,
                token_usage=usage,
                latency_ms=response.latency_ms,
                cumulative_transport_attempt_count=counted_transport.calls,
            )
            break
        try:
            audits.append(_safe_audit_from_success(context, parsed_response, response))
        except Exception:
            failure = QualificationFailureAudit(
                case_id=context.case_id,
                stage="response",
                error_category="internal_execution_failure",
                finish_reason=finish_reason,
                http_status_class=status_class,
                parser_valid=True,
                token_usage=usage,
                latency_ms=response.latency_ms,
                cumulative_transport_attempt_count=counted_transport.calls,
            )
            break

    current = _bindings_current(
        bindings,
        config_path=config_path,
        approval_path=approval_path,
        compatibility_receipt_path=compatibility_receipt_path,
        compatibility_approval_path=compatibility_approval_path,
    )
    if failure is None and not current:
        failure = QualificationFailureAudit(
            stage="binding",
            error_category="source_relation_qualification_binding_changed_before_receipt",
            cumulative_transport_attempt_count=counted_transport.calls,
        )
    if failure is None and not _has_complete_coverage(audits):
        failure = QualificationFailureAudit(
            stage="binding",
            error_category="source_relation_qualification_coverage_incomplete",
            cumulative_transport_attempt_count=counted_transport.calls,
        )
    try:
        audit_path, receipt_path, failure, current = _write_attempt_artifacts(
            output_root=target,
            bindings=bindings,
            audits=audits,
            failure=failure,
            bindings_current=current,
            bindings_current_check=lambda: _bindings_current(
                bindings,
                config_path=config_path,
                approval_path=approval_path,
                compatibility_receipt_path=compatibility_receipt_path,
                compatibility_approval_path=compatibility_approval_path,
            ),
            safety=safety,
        )
    except SourceRelationQualificationArtifactWriteError as exc:
        partial_audit_path = exc.audit_path
        partial_receipt_path = exc.receipt_path
        partial_safety = safety.model_copy(
            update={
                "safe_case_audits_written": partial_audit_path is not None,
                "attempt_receipt_written": partial_receipt_path is not None,
            }
        )
        return QualificationRunResult(
            status="failed",
            qualification_status="execution_failed",
            error_code="source_relation_qualification_artifact_write_failed",
            model=bindings.config.required_model,
            logical_request_count=len(audits) + int(failure is not None and failure.case_id is not None),
            transport_attempt_count=counted_transport.calls,
            completed_case_count=len(audits),
            parser_invalid_count=int(failure is not None and failure.parser_valid is False),
            receipt_path=(
                str(partial_receipt_path.relative_to(_ROOT))
                if partial_receipt_path and partial_receipt_path.is_relative_to(_ROOT)
                else (str(partial_receipt_path) if partial_receipt_path else None)
            ),
            safe_case_audit_path=(
                str(partial_audit_path.relative_to(_ROOT))
                if partial_audit_path and partial_audit_path.is_relative_to(_ROOT)
                else (str(partial_audit_path) if partial_audit_path else None)
            ),
            safety=partial_safety,
            attempt_claim_released=False,
            claim_cleanup_status="retained_after_attempt",
        )
    result_safety = safety.model_copy(
        update={
            "safe_case_audits_written": bool(audits),
            "attempt_receipt_written": True,
        }
    )
    if failure is not None:
        return QualificationRunResult(
            status="failed",
            qualification_status="execution_failed",
            error_code=failure.error_category,
            model=bindings.config.required_model,
            logical_request_count=len(audits) + int(failure.case_id is not None),
            transport_attempt_count=counted_transport.calls,
            completed_case_count=len(audits),
            parser_invalid_count=int(failure.parser_valid is False),
            receipt_path=str(receipt_path.relative_to(_ROOT)) if receipt_path.is_relative_to(_ROOT) else str(receipt_path),
            safe_case_audit_path=(
                str(audit_path.relative_to(_ROOT)) if audit_path and audit_path.is_relative_to(_ROOT)
                else (str(audit_path) if audit_path else None)
            ),
            safety=result_safety,
            attempt_claim_released=False,
            claim_cleanup_status="retained_after_attempt",
        )
    claim_released = _release_claim(claim)
    return QualificationRunResult(
        status="completed",
        qualification_status="source_structure_qualified",
        model=bindings.config.required_model,
        logical_request_count=REQUEST_CAP,
        transport_attempt_count=counted_transport.calls,
        completed_case_count=len(audits),
        parser_invalid_count=0,
        receipt_path=str(receipt_path.relative_to(_ROOT)) if receipt_path.is_relative_to(_ROOT) else str(receipt_path),
        safe_case_audit_path=(
            str(audit_path.relative_to(_ROOT)) if audit_path and audit_path.is_relative_to(_ROOT)
            else (str(audit_path) if audit_path else None)
        ),
        safety=result_safety,
        attempt_claim_released=claim_released,
        claim_cleanup_status="released" if claim_released else "release_failed",
    )


def execute_qualification_v1(
    *,
    allow_network: bool = False,
    confirm_run: bool = False,
    confirm_request_cap: int | None = None,
    confirm_completion_reservation_cap: int | None = None,
    environment: Mapping[str, str] | None = None,
    transport: OpenAICompatibleTransport | None = None,
) -> QualificationRunResult:
    """Run only the configured gate; production callers cannot redirect artifacts."""

    return _execute_qualification_v1(
        allow_network=allow_network,
        confirm_run=confirm_run,
        confirm_request_cap=confirm_request_cap,
        confirm_completion_reservation_cap=confirm_completion_reservation_cap,
        environment=environment,
        transport=transport,
        config_path=DEFAULT_CONFIG,
        approval_path=DEFAULT_APPROVAL,
        compatibility_receipt_path=COMPATIBILITY_RECEIPT,
        compatibility_approval_path=DEFAULT_COMPATIBILITY_APPROVAL,
        output_root=DEFAULT_OUTPUT_ROOT,
        attempt_claim_path=DEFAULT_ATTEMPT_CLAIM,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the bounded source-relation structure qualification gate"
    )
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--confirm-request-cap", type=int)
    parser.add_argument("--confirm-completion-reservation-cap", type=int)
    arguments = parser.parse_args(argv)
    result = execute_qualification_v1(
        allow_network=arguments.allow_network,
        confirm_run=arguments.confirm_run,
        confirm_request_cap=arguments.confirm_request_cap,
        confirm_completion_reservation_cap=arguments.confirm_completion_reservation_cap,
    )
    print(safe_json(result.model_dump(mode="json")))
    return (
        0
        if result.status in {"network_disabled", "completed"}
        and result.claim_cleanup_status != "release_failed"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT_RECEIPT_NAME",
    "COMPLETION_RESERVATION_CAP",
    "DEFAULT_APPROVAL",
    "DEFAULT_ATTEMPT_CLAIM",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_ROOT",
    "MAX_RETRIES",
    "MAX_TOKENS",
    "QualificationCaseAudit",
    "QualificationFailureAudit",
    "QualificationReceipt",
    "QualificationRunResult",
    "QualificationSafety",
    "REQUEST_CAP",
    "RUNNER_VERSION",
    "SourceRelationQualificationApproval",
    "SourceRelationQualificationConfig",
    "SourceRelationQualificationError",
    "execute_qualification_v1",
    "load_qualification_approval",
    "load_qualification_config",
    "load_qualification_receipt",
    "safe_preflight",
]
