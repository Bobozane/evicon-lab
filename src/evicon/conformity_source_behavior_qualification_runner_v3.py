"""Bounded v3 source-behavior qualification runner.

This runner is deliberately separate from the older v1/v2 runners.  It binds
the accepted v3 behavior design to its one-shot compatibility receipt and to
the completed source-structure qualification.  It is a development gate only:
it does not estimate an effect, run a network cascade, or evaluate private
truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_identification import (
    IdentificationError,
    OrdinalJudgment,
    PublicAction,
    ShareDecision,
    safe_json,
    sha256_file,
)
from .conformity_source_behavior_compatibility_v3 import (
    DEFAULT_MODULE as COMPATIBILITY_MODULE,
    DEFAULT_RECEIPT as COMPATIBILITY_RECEIPT,
    CompatibilityReceiptV3,
    load_receipt_v3,
)
from .conformity_source_behavior_qualification_v3 import (
    DEFAULT_APPROVAL as DESIGN_APPROVAL,
    DEFAULT_CONFIG as DESIGN_CONFIG,
    PROTOCOL_VERSION,
    RESPONSE_SCHEMA,
    RESPONSE_SCHEMA_SHA256,
    SCHEMA_NAME,
    TEMPLATE_VERSION,
    BehaviorQualificationContextV3,
    load_config as load_design_config,
    load_approval as load_design_approval,
    parse_behavior_response_v3,
    render_behavior_request_v3,
    safe_case_audit_v3,
)
from .conformity_source_behavior_qualification_v3_smoke import build_cases
from .conformity_source_relation_comprehension_qualification_provider_subset_short_name_v2 import (
    DEFAULT_APPROVAL as SOURCE_RELATION_APPROVAL,
    DEFAULT_CONFIG as SOURCE_RELATION_CONFIG,
    DEFAULT_OUTPUT_ROOT as SOURCE_RELATION_OUTPUT,
    load_qualification_receipt as load_source_relation_receipt,
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
    "conformity_source_behavior_qualification_runner.v3.toml"
)
DEFAULT_APPROVAL: Final[str] = (
    "configs/provenance_cascade/identification/"
    "conformity_source_behavior_qualification_runner_approval.v3.toml"
)
DEFAULT_RUNNER: Final[str] = (
    "src/evicon/conformity_source_behavior_qualification_runner_v3.py"
)
DEFAULT_OUTPUT_ROOT: Final[str] = "outputs/conformity-source-behavior-qualification-v3"
DEFAULT_ATTEMPT_CLAIM: Final[str] = (
    "outputs/study-locks/conformity_source_behavior_qualification_runner.v3.claim"
)
SAFE_AUDIT_NAME: Final[str] = "safe_case_audits.jsonl"
RECEIPT_NAME: Final[str] = "qualification_receipt.json"
RUNNER_VERSION: Final[str] = "conformity_source_behavior_qualification_runner.v3"
RECEIPT_VERSION: Final[str] = "conformity_source_behavior_qualification_receipt.v3"
REQUEST_CAP: Final[int] = 36
COMPLETION_RESERVATION_CAP: Final[int] = 9216
MAX_TOKENS: Final[int] = 256
TEMPERATURE: Final[float] = 0.2
MAX_RETRIES: Final[int] = 0
TIMEOUT_SECONDS: Final[float] = 5.0
REASONING_EFFORT: Final[ReasoningEffort] = ReasoningEffort.NONE
SEEDS: Final[tuple[int, int, int]] = (20261201, 20261202, 20261203)


class SourceBehaviorV3RunnerError(IdentificationError):
    """Stable, non-sensitive runner errors."""


class V3RunnerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_id: Literal["evicon-conformity-source-behavior-qualification-runner-v3"]
    execution_version: Literal["conformity_source_behavior_qualification_runner.v3"]
    runner_version: Literal[RUNNER_VERSION]
    status: Literal["awaiting_explicit_authorization"]
    development_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    behavior_effect_estimated: Literal[False]
    network_enabled_by_default: Literal[False]
    explicit_allow_network_required: Literal[True]
    explicit_run_confirmation_required: Literal[True]
    evaluator_private_truth_available: Literal[False]
    historical_artifacts_excluded: Literal[True]
    v3_design_config_path: str
    v3_design_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_design_approval_path: str
    v3_design_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_compatibility_receipt_path: str
    v3_compatibility_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_compatibility_module_path: str
    v3_compatibility_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_relation_config_path: str
    source_relation_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_relation_approval_path: str
    source_relation_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_relation_receipt_path: str
    source_relation_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_path: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_path: str
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runner_path: Literal[DEFAULT_RUNNER]
    runner_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_model: str = Field(min_length=1, max_length=160)
    response_schema_name: Literal[SCHEMA_NAME]
    response_schema_version: Literal["conformity_source_behavior_response.v3"]
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_version: Literal[PROTOCOL_VERSION]
    template_version: Literal[TEMPLATE_VERSION]
    scenario_count: Literal[4]
    projection_count: Literal[3]
    seed_count: Literal[3]
    case_count: Literal[36]
    logical_request_cap: Literal[36]
    completion_reservation_cap: Literal[9216]
    max_tokens: Literal[256]
    temperature: Literal[0.2]
    max_retries: Literal[0]
    timeout_seconds: Literal[5]
    reasoning_effort: Literal["none"]
    output_root: Literal[DEFAULT_OUTPUT_ROOT]
    attempt_claim_path: Literal[DEFAULT_ATTEMPT_CLAIM]

    @model_validator(mode="after")
    def fixed_scope(self) -> "V3RunnerConfig":
        if self.case_count != self.scenario_count * self.projection_count * self.seed_count:
            raise ValueError("source_behavior_v3_runner_case_count_mismatch")
        if self.logical_request_cap != self.case_count:
            raise ValueError("source_behavior_v3_runner_request_cap_mismatch")
        if self.completion_reservation_cap != self.case_count * self.max_tokens:
            raise ValueError("source_behavior_v3_runner_reservation_mismatch")
        return self


class V3RunnerApproval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: Literal["evicon-conformity-source-behavior-qualification-runner-v3"]
    approval_version: Literal[
        "conformity_source_behavior_qualification_runner_approval.v3"
    ]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    accepted_on: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_design_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_design_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_compatibility_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_relation_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runner_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_model: str = Field(min_length=1, max_length=160)
    confirm_cross_sectional_scope_only: bool
    confirm_source_structure_qualification_required: bool
    confirm_three_seed_counterbalance: bool
    confirm_full_36_case_coverage: bool
    confirm_no_retry_and_terminal_failure: bool
    confirm_no_private_or_historical_material: bool
    confirm_not_effect_or_causal_evidence: bool
    persistent_network_authorization_granted: Literal[False]
    explicit_command_authorization_required: Literal[True]

    @model_validator(mode="after")
    def complete_if_accepted(self) -> "V3RunnerApproval":
        if self.acceptance_status != "accepted":
            return self
        if not self.accepted_by.strip() or not self.accepted_on.strip():
            raise ValueError("source_behavior_v3_runner_approval_identity_missing")
        from datetime import date

        try:
            date.fromisoformat(self.accepted_on)
        except ValueError as exc:
            raise ValueError("source_behavior_v3_runner_approval_date_invalid") from exc
        confirmations = (
            self.confirm_cross_sectional_scope_only,
            self.confirm_source_structure_qualification_required,
            self.confirm_three_seed_counterbalance,
            self.confirm_full_36_case_coverage,
            self.confirm_no_retry_and_terminal_failure,
            self.confirm_no_private_or_historical_material,
            self.confirm_not_effect_or_causal_evidence,
        )
        if not all(confirmations):
            raise ValueError("source_behavior_v3_runner_approval_incomplete")
        return self


class V3SchemaContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    response_format: Literal["json_schema"] = "json_schema"
    schema_name: Literal[SCHEMA_NAME] = SCHEMA_NAME
    schema_version: Literal["conformity_source_behavior_response.v3"] = (
        "conformity_source_behavior_response.v3"
    )
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_version: Literal[TEMPLATE_VERSION] = TEMPLATE_VERSION


class V3GenerationParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tokens: Literal[256] = MAX_TOKENS
    temperature: Literal[0.2] = TEMPERATURE
    max_retries: Literal[0] = MAX_RETRIES
    timeout_seconds: Literal[5.0] = TIMEOUT_SECONDS
    reasoning_effort: Literal["none"] = REASONING_EFFORT.value
    seeds: tuple[Literal[20261201], Literal[20261202], Literal[20261203]] = SEEDS
    logical_request_cap: Literal[36] = REQUEST_CAP
    completion_reservation_cap: Literal[9216] = COMPLETION_RESERVATION_CAP


class V3TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def complete_or_absent(self) -> "V3TokenUsage":
        values = (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        if any(value is None for value in values) and not all(value is None for value in values):
            raise ValueError("source_behavior_v3_runner_token_usage_partial")
        if all(value is not None for value in values):
            assert self.prompt_tokens is not None
            assert self.completion_tokens is not None
            if self.total_tokens != self.prompt_tokens + self.completion_tokens:
                raise ValueError("source_behavior_v3_runner_token_usage_mismatch")
        return self

    @property
    def complete(self) -> bool:
        return self.total_tokens is not None


class V3CaseAudit(BaseModel):
    """Safe semantic observation; no text, prompt, root ID, or raw reply."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^source-behavior-v3-q\d{3}$")
    scenario_id: str = Field(pattern=r"^source-behavior-v3-s0[1-4]$")
    projection: Literal["source_free", "same_root", "independent_roots"]
    seed: Literal[20261201, 20261202, 20261203]
    order_position: int = Field(ge=1, le=3)
    parser_status: Literal["valid"] = "valid"
    ordinal_judgment: OrdinalJudgment
    public_action: PublicAction
    share_decision: ShareDecision
    confidence: int = Field(ge=1, le=7)
    used_content_count: int = Field(ge=1, le=4)
    visible_focal_root_count: int = Field(ge=0, le=2)
    finish_reason: Literal["stop"] = "stop"
    http_status_class: Literal["2xx"] = "2xx"
    token_usage: V3TokenUsage
    latency_ms: float = Field(ge=0.0)
    attempt_count: Literal[1] = 1


class V3FailureAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str | None = Field(default=None, pattern=r"^source-behavior-v3-q\d{3}$")
    stage: Literal["pre_request", "transport", "response", "binding", "coverage"]
    error_category: str = Field(pattern=r"^[a-z0-9_]+$")
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls", "other"] | None = None
    http_status_class: Literal["2xx", "3xx", "4xx", "5xx"] | None = None
    parser_valid: bool | None = None
    token_usage: V3TokenUsage = Field(default_factory=V3TokenUsage)
    latency_ms: float | None = Field(default=None, ge=0.0)
    transport_attempt_count: int = Field(ge=0, le=REQUEST_CAP)


class V3RunnerSafety(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    network_enabled: bool = False
    provider_constructed: bool = False
    api_key_read: bool = False
    compatibility_receipt_loaded: bool = False
    source_structure_qualification_loaded: bool = False
    qualification_executed: bool = False
    safe_case_audits_written: bool = False
    receipt_written: bool = False
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


class V3RunReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    receipt_version: Literal[RECEIPT_VERSION]
    runner_version: Literal[RUNNER_VERSION]
    status: Literal["source_behavior_qualified", "execution_failed"]
    model: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_design_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_design_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_compatibility_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v3_compatibility_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_relation_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_relation_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_relation_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runner_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_contract: V3SchemaContract = Field(alias="schema")
    generation_parameters: V3GenerationParameters
    logical_request_count: int = Field(ge=0, le=REQUEST_CAP)
    transport_attempt_count: int = Field(ge=0, le=REQUEST_CAP)
    completed_case_count: int = Field(ge=0, le=REQUEST_CAP)
    parser_invalid_count: int = Field(ge=0, le=REQUEST_CAP)
    projection_case_counts: dict[str, int]
    seed_case_counts: dict[str, int]
    scenario_case_counts: dict[str, int]
    token_usage: V3TokenUsage
    latency_ms: float = Field(ge=0.0)
    safe_case_audit_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    failure_audit: V3FailureAudit | None = None
    bindings_unchanged_before_receipt: bool
    safety: V3RunnerSafety

    @model_validator(mode="after")
    def state_consistent(self) -> "V3RunReceipt":
        if self.completed_case_count == 0 and self.safe_case_audit_sha256 is not None:
            raise ValueError("source_behavior_v3_empty_audit_hash_present")
        if self.completed_case_count > 0 and self.safe_case_audit_sha256 is None:
            raise ValueError("source_behavior_v3_completed_audit_hash_missing")
        if self.status == "source_behavior_qualified":
            if (
                self.failure_audit is not None
                or self.completed_case_count != REQUEST_CAP
                or self.logical_request_count != REQUEST_CAP
                or self.transport_attempt_count != REQUEST_CAP
                or self.parser_invalid_count != 0
                or self.safe_case_audit_sha256 is None
                or not self.bindings_unchanged_before_receipt
                or self.projection_case_counts != {
                    "independent_roots": 12,
                    "same_root": 12,
                    "source_free": 12,
                }
                or self.seed_case_counts != {
                    "20261201": 12,
                    "20261202": 12,
                    "20261203": 12,
                }
                or self.scenario_case_counts != {
                    "source-behavior-v3-s01": 9,
                    "source-behavior-v3-s02": 9,
                    "source-behavior-v3-s03": 9,
                    "source-behavior-v3-s04": 9,
                }
            ):
                raise ValueError("source_behavior_v3_success_receipt_inconsistent")
        elif self.failure_audit is None:
            raise ValueError("source_behavior_v3_failure_receipt_missing_audit")
        return self


class V3RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["network_disabled", "blocked", "completed", "failed"]
    qualification_status: Literal[
        "not_run", "execution_blocked", "execution_failed", "source_behavior_qualified"
    ]
    error_code: str | None = Field(default=None, pattern=r"^[a-z0-9_]+$")
    model: str | None = None
    logical_request_count: int = Field(default=0, ge=0, le=REQUEST_CAP)
    transport_attempt_count: int = Field(default=0, ge=0, le=REQUEST_CAP)
    completed_case_count: int = Field(default=0, ge=0, le=REQUEST_CAP)
    parser_invalid_count: int = Field(default=0, ge=0, le=REQUEST_CAP)
    output_root: str | None = None
    safe_case_audit_path: str | None = None
    receipt_path: str | None = None
    claim_cleanup_status: Literal[
        "not_applicable", "released", "retained_after_attempt", "release_failed"
    ] = "not_applicable"
    safety: V3RunnerSafety = Field(default_factory=V3RunnerSafety)


class _FrozenBindings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config: V3RunnerConfig
    config_sha256: str
    approval_sha256: str
    compatibility_receipt_sha256: str
    source_relation_receipt_sha256: str
    compatibility_receipt: CompatibilityReceiptV3


class _CountingTransport:
    def __init__(self, transport: OpenAICompatibleTransport, cap: int) -> None:
        self.transport = transport
        self.cap = cap
        self.calls = 0

    def post(self, url, headers, payload, timeout_seconds):  # type: ignore[no-untyped-def]
        if self.calls >= self.cap:
            raise SourceBehaviorV3RunnerError("source_behavior_v3_transport_cap_exceeded")
        self.calls += 1
        return self.transport.post(url, headers, payload, timeout_seconds)


class _SeededProvider:
    def __init__(
        self,
        base_config: ProviderConfig,
        *,
        transport: OpenAICompatibleTransport,
        environment: Mapping[str, str] | None,
    ) -> None:
        self.base_config = base_config
        self.transport = transport
        self.environment = environment
        self.providers: dict[int, OpenAICompatibleProvider] = {}
        self.constructed_count = 0

    def complete(self, request):  # type: ignore[no-untyped-def]
        provider = self.providers.get(request.seed)
        if provider is None:
            provider = OpenAICompatibleProvider(
                self.base_config.model_copy(update={"seed": request.seed}),
                transport=self.transport,
                environment=self.environment,
            )
            self.providers[request.seed] = provider
            self.constructed_count += 1
        return provider.complete(request)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def _safe_path(path: Path) -> str:
    try:
        return str(path.relative_to(_ROOT))
    except ValueError:
        return str(path)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def _read_model(path: str | Path, model_type: type[BaseModel], error_code: str):
    try:
        raw = _resolve(path).read_bytes()
        return model_type.model_validate(tomllib.loads(raw.decode("utf-8")))
    except Exception as exc:
        raise SourceBehaviorV3RunnerError(error_code) from exc


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_field")
        result[key] = value
    return result


def load_runner_config(path: str | Path = DEFAULT_CONFIG) -> V3RunnerConfig:
    config = _read_model(path, V3RunnerConfig, "source_behavior_v3_runner_config_invalid")
    assert isinstance(config, V3RunnerConfig)
    try:
        design_config, _ = load_design_config(config.v3_design_config_path)
        if _sha(config.v3_design_config_path) != config.v3_design_config_sha256:
            raise ValueError("design config hash")
        if _sha(config.v3_design_approval_path) != config.v3_design_approval_sha256:
            raise ValueError("design approval hash")
        if _sha(config.v3_compatibility_receipt_path) != config.v3_compatibility_receipt_sha256:
            raise ValueError("compatibility receipt hash")
        if _sha(config.v3_compatibility_module_path) != config.v3_compatibility_module_sha256:
            raise ValueError("compatibility module hash")
        if _sha(config.source_relation_config_path) != config.source_relation_config_sha256:
            raise ValueError("source relation config hash")
        if _sha(config.source_relation_approval_path) != config.source_relation_approval_sha256:
            raise ValueError("source relation approval hash")
        if _sha(config.source_relation_receipt_path) != config.source_relation_receipt_sha256:
            raise ValueError("source relation receipt hash")
        if _sha(config.protocol_path) != config.protocol_sha256:
            raise ValueError("protocol hash")
        if _sha(config.corpus_path) != config.corpus_sha256:
            raise ValueError("corpus hash")
        expected = {
            "protocol_version": PROTOCOL_VERSION,
            "template_version": TEMPLATE_VERSION,
            "response_schema_name": SCHEMA_NAME,
            "response_schema_version": "conformity_source_behavior_response.v3",
            "response_schema_sha256": RESPONSE_SCHEMA_SHA256,
            "protocol_sha256": design_config.protocol_sha256,
            "corpus_sha256": design_config.corpus_sha256,
            "scenario_count": design_config.scenario_count,
            "projection_count": len(design_config.projections),
            "seed_count": len(design_config.seeds),
            "case_count": REQUEST_CAP,
            "logical_request_cap": REQUEST_CAP,
            "completion_reservation_cap": COMPLETION_RESERVATION_CAP,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "max_retries": MAX_RETRIES,
            "timeout_seconds": int(TIMEOUT_SECONDS),
            "reasoning_effort": REASONING_EFFORT.value,
            "output_root": DEFAULT_OUTPUT_ROOT,
            "attempt_claim_path": DEFAULT_ATTEMPT_CLAIM,
            "runner_path": DEFAULT_RUNNER,
            "runner_sha256": sha256_file(DEFAULT_RUNNER),
            "v3_design_config_path": DESIGN_CONFIG,
            "v3_design_approval_path": DESIGN_APPROVAL,
            "v3_compatibility_receipt_path": COMPATIBILITY_RECEIPT,
            "v3_compatibility_module_path": COMPATIBILITY_MODULE,
            "source_relation_config_path": SOURCE_RELATION_CONFIG,
            "source_relation_approval_path": SOURCE_RELATION_APPROVAL,
            "source_relation_receipt_path": (
                f"{SOURCE_RELATION_OUTPUT}/qualification_attempt_receipt.json"
            ),
        }
        if {key: getattr(config, key) for key in expected} != expected:
            raise ValueError("runner static binding")
    except Exception as exc:
        raise SourceBehaviorV3RunnerError("source_behavior_v3_runner_config_binding_mismatch") from exc
    return config


def load_runner_approval(
    path: str | Path = DEFAULT_APPROVAL,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    require_accepted: bool = False,
) -> V3RunnerApproval:
    config = load_runner_config(config_path)
    approval = _read_model(path, V3RunnerApproval, "source_behavior_v3_runner_approval_invalid")
    assert isinstance(approval, V3RunnerApproval)
    expected = {
        "config_sha256": _sha(config_path),
        "v3_design_config_sha256": config.v3_design_config_sha256,
        "v3_design_approval_sha256": config.v3_design_approval_sha256,
        "v3_compatibility_receipt_sha256": config.v3_compatibility_receipt_sha256,
        "source_relation_receipt_sha256": config.source_relation_receipt_sha256,
        "protocol_sha256": config.protocol_sha256,
        "corpus_sha256": config.corpus_sha256,
        "response_schema_sha256": config.response_schema_sha256,
        "runner_sha256": config.runner_sha256,
        "required_model": config.required_model,
    }
    if {key: getattr(approval, key) for key in expected} != expected:
        raise SourceBehaviorV3RunnerError("source_behavior_v3_runner_approval_binding_mismatch")
    if require_accepted and approval.acceptance_status != "accepted":
        raise SourceBehaviorV3RunnerError("source_behavior_v3_runner_approval_not_accepted")
    return approval


def _freeze_bindings(
    *,
    config_path: str | Path,
    approval_path: str | Path,
    require_accepted: bool,
) -> _FrozenBindings:
    config = load_runner_config(config_path)
    approval = load_runner_approval(
        approval_path, config_path=config_path, require_accepted=require_accepted
    )
    try:
        design_approval = load_design_approval(config.v3_design_approval_path)
        if design_approval.acceptance_status != "accepted":
            raise ValueError("design approval")
        compatibility_receipt = load_receipt_v3(
            config.v3_compatibility_receipt_path,
            approval_path=config.v3_design_approval_path,
        )
        source_receipt = load_source_relation_receipt(
            config.source_relation_receipt_path,
            config_path=SOURCE_RELATION_CONFIG,
            approval_path=SOURCE_RELATION_APPROVAL,
        )
        if source_receipt.status != "source_structure_qualified":
            raise ValueError("source structure qualification")
        if compatibility_receipt.model != config.required_model:
            raise ValueError("model")
        if compatibility_receipt.protocol_sha256 != config.protocol_sha256:
            raise ValueError("protocol")
        if compatibility_receipt.schema_contract.schema_sha256 != config.response_schema_sha256:
            raise ValueError("schema")
        if source_receipt.model != config.required_model:
            raise ValueError("source model")
        if compatibility_receipt.generation_parameters.max_tokens != MAX_TOKENS:
            raise ValueError("generation")
        if compatibility_receipt.generation_parameters.temperature != TEMPERATURE:
            raise ValueError("generation")
        if compatibility_receipt.generation_parameters.max_retries != MAX_RETRIES:
            raise ValueError("generation")
        if compatibility_receipt.generation_parameters.timeout_seconds != TIMEOUT_SECONDS:
            raise ValueError("generation")
    except Exception as exc:
        raise SourceBehaviorV3RunnerError("source_behavior_v3_runner_parent_binding_mismatch") from exc
    return _FrozenBindings(
        config=config,
        config_sha256=_sha(config_path),
        approval_sha256=_sha(approval_path),
        compatibility_receipt_sha256=_sha(config.v3_compatibility_receipt_path),
        source_relation_receipt_sha256=_sha(config.source_relation_receipt_path),
        compatibility_receipt=compatibility_receipt,
    )


def _bindings_current(
    bindings: _FrozenBindings,
    *,
    config_path: str | Path,
    approval_path: str | Path,
) -> bool:
    try:
        return _freeze_bindings(
            config_path=config_path,
            approval_path=approval_path,
            require_accepted=True,
        ) == bindings
    except Exception:
        return False


def _http_status_class(value: object) -> Literal["2xx", "3xx", "4xx", "5xx"] | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if 200 <= value < 300:
        return "2xx"
    if 300 <= value < 400:
        return "3xx"
    if 400 <= value < 500:
        return "4xx"
    if 500 <= value < 600:
        return "5xx"
    return None


def _safe_finish_reason(value: object) -> Literal["stop", "length", "content_filter", "tool_calls", "other"]:
    if value in {"stop", "length", "content_filter", "tool_calls"}:
        return value  # type: ignore[return-value]
    return "other"


def _error_category(error: LLMProviderError) -> str:
    mapping = {
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
        ProviderErrorCode.PROVIDER_FAILURE.value: "provider_failure",
    }
    return mapping.get(error.code.value, "provider_failure")


def _parser_category(error: ValueError) -> str:
    allowed = {
        "malformed_json", "top_level_type", "invalid_schema", "duplicate_content_id",
        "unavailable_content_id", "judgment_action_mismatch", "share_content_id_unexpected",
        "share_content_id_invalid",
    }
    return str(error) if str(error) in allowed else "invalid_structure_response"


def _usage(response: object) -> V3TokenUsage:
    return V3TokenUsage(
        prompt_tokens=getattr(response, "prompt_tokens", None),
        completion_tokens=getattr(response, "completion_tokens", None),
        total_tokens=getattr(response, "total_tokens", None),
    )


def _provider_config(environment: Mapping[str, str] | None, config: V3RunnerConfig) -> ProviderConfig:
    source = os.environ if environment is None else environment
    base_url = source.get("EVICON_LLM_BASE_URL") or None
    model = source.get("EVICON_LLM_MODEL") or None
    if not base_url:
        raise SourceBehaviorV3RunnerError("missing_base_url")
    if not model:
        raise SourceBehaviorV3RunnerError("missing_model")
    if model != config.required_model:
        raise SourceBehaviorV3RunnerError("source_behavior_v3_runner_model_mismatch")
    return ProviderConfig(
        base_url=base_url,
        model_name=model,
        allow_network=True,
        timeout_seconds=TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        seed=SEEDS[0],
        reasoning_effort=REASONING_EFFORT,
        response_format=ResponseFormatMode.JSON_SCHEMA,
        response_schema_name=SCHEMA_NAME,
        response_schema=RESPONSE_SCHEMA,
    )


def _acquire_claim(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise SourceBehaviorV3RunnerError(
            "source_behavior_v3_runner_attempt_claim_exists"
        ) from exc
    except OSError as exc:
        raise SourceBehaviorV3RunnerError(
            "source_behavior_v3_runner_claim_acquire_failed"
        ) from exc
    else:
        os.close(descriptor)


def _release_claim(path: Path) -> bool:
    try:
        path.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def _atomic_create(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise SourceBehaviorV3RunnerError(
                "source_behavior_v3_runner_output_exists"
            ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_artifacts(
    *,
    target: Path,
    bindings: _FrozenBindings,
    audits: Sequence[V3CaseAudit],
    failure: V3FailureAudit | None,
    transport_attempts: int,
    parser_invalid_count: int,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    latency_ms: float,
    bindings_current: bool,
    safety: V3RunnerSafety,
) -> tuple[Path, Path]:
    target.mkdir(parents=True, exist_ok=False)
    audit_path = target / SAFE_AUDIT_NAME
    serialized_audits = "".join(
        json.dumps(item.model_dump(mode="json"), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
        for item in audits
    )
    if audits:
        _atomic_create(audit_path, serialized_audits)
        audit_hash = hashlib.sha256(serialized_audits.encode("utf-8")).hexdigest()
    else:
        audit_hash = None
    projection_counts = dict(sorted(Counter(item.projection for item in audits).items()))
    seed_counts = dict(sorted(Counter(str(item.seed) for item in audits).items()))
    scenario_counts = dict(sorted(Counter(item.scenario_id for item in audits).items()))
    receipt = V3RunReceipt(
        receipt_version=RECEIPT_VERSION,
        runner_version=RUNNER_VERSION,
        status="source_behavior_qualified" if failure is None and bindings_current else "execution_failed",
        model=bindings.config.required_model,
        config_sha256=bindings.config_sha256,
        approval_sha256=bindings.approval_sha256,
        v3_design_config_sha256=bindings.config.v3_design_config_sha256,
        v3_design_approval_sha256=bindings.config.v3_design_approval_sha256,
        v3_compatibility_receipt_sha256=bindings.compatibility_receipt_sha256,
        v3_compatibility_module_sha256=bindings.config.v3_compatibility_module_sha256,
        source_relation_config_sha256=bindings.config.source_relation_config_sha256,
        source_relation_approval_sha256=bindings.config.source_relation_approval_sha256,
        source_relation_receipt_sha256=bindings.source_relation_receipt_sha256,
        protocol_sha256=bindings.config.protocol_sha256,
        corpus_sha256=bindings.config.corpus_sha256,
        runner_sha256=bindings.config.runner_sha256,
        schema=V3SchemaContract(schema_sha256=bindings.config.response_schema_sha256),
        generation_parameters=V3GenerationParameters(),
        logical_request_count=len(audits) + int(failure is not None and failure.case_id is not None),
        transport_attempt_count=transport_attempts,
        completed_case_count=len(audits),
        parser_invalid_count=parser_invalid_count,
        projection_case_counts=projection_counts,
        seed_case_counts=seed_counts,
        scenario_case_counts=scenario_counts,
        token_usage=V3TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
        latency_ms=latency_ms,
        safe_case_audit_sha256=audit_hash,
        failure_audit=failure,
        bindings_unchanged_before_receipt=bindings_current,
        safety=safety.model_copy(
            update={
                "safe_case_audits_written": bool(audits),
                "receipt_written": True,
            }
        ),
    )
    receipt_path = target / RECEIPT_NAME
    _atomic_create(
        receipt_path,
        json.dumps(receipt.model_dump(mode="json", by_alias=True), ensure_ascii=True, sort_keys=True, indent=2)
        + "\n",
    )
    return audit_path, receipt_path


def _blocked(code: str, *, safety: V3RunnerSafety | None = None) -> V3RunResult:
    return V3RunResult(
        status="blocked",
        qualification_status="execution_blocked",
        error_code=code,
        safety=safety or V3RunnerSafety(),
    )


def safe_preflight(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    approval_path: str | Path = DEFAULT_APPROVAL,
) -> dict[str, object]:
    """Read bindings only; this function never reads the provider environment."""

    try:
        config = load_runner_config(config_path)
        approval = load_runner_approval(approval_path, config_path=config_path)
        compatibility = load_receipt_v3(
            config.v3_compatibility_receipt_path,
            approval_path=config.v3_design_approval_path,
        )
        source = load_source_relation_receipt(
            config.source_relation_receipt_path,
            config_path=config.source_relation_config_path,
            approval_path=config.source_relation_approval_path,
        )
    except Exception as exc:
        return {
            "status": "blocked",
            "blocking_reasons": [getattr(exc, "code", "source_behavior_v3_runner_preflight_failed")],
            "network": "disabled",
            "api_key_read": False,
            "provider_constructed": False,
            "results_written": False,
        }
    reasons = []
    if approval.acceptance_status != "accepted":
        reasons.append("runner_execution_approval_required")
    if compatibility.status != "completed":
        reasons.append("v3_compatibility_receipt_unavailable")
    if source.status != "source_structure_qualified":
        reasons.append("source_structure_qualification_unavailable")
    return {
        "status": "source_behavior_v3_runner_preflight_ready",
        "approval_status": approval.acceptance_status,
        "compatibility_receipt_available": compatibility.status == "completed",
        "source_structure_qualification_available": source.status == "source_structure_qualified",
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


def _execute_v3(
    *,
    allow_network: bool = False,
    confirm_run: bool = False,
    confirm_request_cap: int | None = None,
    confirm_completion_reservation_cap: int | None = None,
    environment: Mapping[str, str] | None = None,
    transport: OpenAICompatibleTransport | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    approval_path: str | Path = DEFAULT_APPROVAL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    attempt_claim_path: str | Path = DEFAULT_ATTEMPT_CLAIM,
) -> V3RunResult:
    if not allow_network:
        return V3RunResult(status="network_disabled", qualification_status="not_run", error_code="network_disabled")
    if not confirm_run:
        return _blocked("source_behavior_v3_runner_run_confirmation_required")
    if confirm_request_cap != REQUEST_CAP:
        return _blocked("source_behavior_v3_runner_request_cap_confirmation_mismatch")
    if confirm_completion_reservation_cap != COMPLETION_RESERVATION_CAP:
        return _blocked("source_behavior_v3_runner_reservation_confirmation_mismatch")
    target = _resolve(output_root)
    claim = _resolve(attempt_claim_path)
    if target.exists():
        return _blocked("source_behavior_v3_runner_output_exists")
    if claim.exists():
        return _blocked("source_behavior_v3_runner_attempt_claim_exists")
    try:
        bindings = _freeze_bindings(
            config_path=config_path, approval_path=approval_path, require_accepted=True
        )
    except SourceBehaviorV3RunnerError as exc:
        return _blocked(exc.code)
    try:
        _acquire_claim(claim)
    except SourceBehaviorV3RunnerError as exc:
        return _blocked(exc.code)
    safety = V3RunnerSafety(
        network_enabled=True,
        compatibility_receipt_loaded=True,
        source_structure_qualification_loaded=True,
    )
    try:
        provider_config = _provider_config(environment, bindings.config)
    except SourceBehaviorV3RunnerError as exc:
        released = _release_claim(claim)
        return V3RunResult(
            status="failed", qualification_status="execution_failed", error_code=exc.code,
            model=bindings.config.required_model,
            claim_cleanup_status="released" if released else "release_failed",
            safety=safety,
        )
    counted = _CountingTransport(transport or UrllibTransport(), REQUEST_CAP)
    key_source = os.environ if environment is None else environment
    # Check the prerequisite before constructing a provider. A missing key is
    # a pre-request failure and must not leave a misleading provider-created flag.
    if not str(key_source.get("EVICON_LLM_API_KEY") or "").strip():
        released = _release_claim(claim)
        return V3RunResult(
            status="failed", qualification_status="execution_failed", error_code="missing_api_key",
            model=bindings.config.required_model,
            claim_cleanup_status="released" if released else "release_failed",
            safety=safety.model_copy(update={"api_key_read": True}),
        )
    provider = _SeededProvider(provider_config, transport=counted, environment=environment)
    safety = safety.model_copy(update={"provider_constructed": True, "api_key_read": True})
    if not _bindings_current(bindings, config_path=config_path, approval_path=approval_path):
        released = _release_claim(claim)
        return V3RunResult(
            status="blocked", qualification_status="execution_blocked",
            error_code="source_behavior_v3_runner_binding_changed_before_request",
            model=bindings.config.required_model,
            claim_cleanup_status="released" if released else "release_failed",
            safety=safety,
        )
    safety = safety.model_copy(update={"qualification_executed": True})
    try:
        design_cases = tuple(
            case.model_copy(update={"model_name": bindings.config.required_model})
            for case in build_cases(DESIGN_CONFIG)
        )
    except Exception:
        released = _release_claim(claim)
        return V3RunResult(
            status="failed", qualification_status="execution_failed", error_code="internal_case_build_failure",
            model=bindings.config.required_model,
            claim_cleanup_status="released" if released else "release_failed",
            safety=safety,
        )
    audits: list[V3CaseAudit] = []
    failure: V3FailureAudit | None = None
    prompt_tokens = completion_tokens = total_tokens = 0
    latency_ms = 0.0
    parser_invalid_count = 0
    for context in design_cases:
        if not _bindings_current(bindings, config_path=config_path, approval_path=approval_path):
            failure = V3FailureAudit(
                stage="binding",
                error_category="source_behavior_v3_runner_binding_changed_before_request",
                transport_attempt_count=counted.calls,
            )
            break
        request = render_behavior_request_v3(context)
        try:
            response = provider.complete(request)
        except LLMProviderError as exc:
            failure = V3FailureAudit(
                case_id=context.case_id,
                stage="transport",
                error_category=_error_category(exc),
                http_status_class=_http_status_class(exc.status_code),
                transport_attempt_count=counted.calls,
            )
            break
        except Exception:
            failure = V3FailureAudit(
                case_id=context.case_id,
                stage="transport",
                error_category="provider_failure",
                transport_attempt_count=counted.calls,
            )
            break
        finish_reason = _safe_finish_reason(response.finish_reason)
        metadata = response.provider_metadata if isinstance(response.provider_metadata, dict) else {}
        status_class = _http_status_class(metadata.get("status_code")) or "2xx"
        try:
            usage = _usage(response)
        except Exception:
            failure = V3FailureAudit(
                case_id=context.case_id,
                stage="response",
                error_category="invalid_token_usage",
                finish_reason=finish_reason,
                http_status_class=status_class,
                latency_ms=response.latency_ms,
                transport_attempt_count=counted.calls,
            )
            break
        if response.model_name != bindings.config.required_model:
            failure = V3FailureAudit(
                case_id=context.case_id,
                stage="response",
                error_category="provider_model_mismatch",
                finish_reason=finish_reason,
                http_status_class=status_class,
                parser_valid=True,
                token_usage=usage,
                latency_ms=response.latency_ms,
                transport_attempt_count=counted.calls,
            )
            break
        if finish_reason != "stop":
            failure = V3FailureAudit(
                case_id=context.case_id,
                stage="response",
                error_category="finish_reason_invalid",
                finish_reason=finish_reason,
                http_status_class=status_class,
                parser_valid=None,
                token_usage=usage,
                latency_ms=response.latency_ms,
                transport_attempt_count=counted.calls,
            )
            break
        if status_class != "2xx" or not usage.complete:
            failure = V3FailureAudit(
                case_id=context.case_id,
                stage="response",
                error_category="invalid_token_usage" if not usage.complete else "invalid_provider_response",
                finish_reason=finish_reason,
                http_status_class=status_class,
                parser_valid=None,
                token_usage=usage,
                latency_ms=response.latency_ms,
                transport_attempt_count=counted.calls,
            )
            break
        try:
            parsed = parse_behavior_response_v3(response.content, context)
        except ValueError as exc:
            parser_invalid_count += 1
            failure = V3FailureAudit(
                case_id=context.case_id,
                stage="response",
                error_category=_parser_category(exc),
                finish_reason=finish_reason,
                http_status_class=status_class,
                parser_valid=False,
                token_usage=usage,
                latency_ms=response.latency_ms,
                transport_attempt_count=counted.calls,
            )
            break
        structural = safe_case_audit_v3(context, parsed)
        audits.append(
            V3CaseAudit(
                **structural.model_dump(mode="python"),
                finish_reason="stop",
                http_status_class="2xx",
                token_usage=usage,
                latency_ms=response.latency_ms,
            )
        )
        prompt_tokens += usage.prompt_tokens or 0
        completion_tokens += usage.completion_tokens or 0
        total_tokens += usage.total_tokens or 0
        latency_ms += response.latency_ms
    expected = {(case.scenario_id, case.projection.value, case.seed) for case in design_cases}
    observed = {(audit.scenario_id, audit.projection, audit.seed) for audit in audits}
    current = _bindings_current(bindings, config_path=config_path, approval_path=approval_path)
    if failure is None and not current:
        failure = V3FailureAudit(
            stage="binding",
            error_category="source_behavior_v3_runner_binding_changed_before_receipt",
            transport_attempt_count=counted.calls,
        )
    if failure is None and (
        len(audits) != REQUEST_CAP
        or len({audit.case_id for audit in audits}) != REQUEST_CAP
        or observed != expected
    ):
        failure = V3FailureAudit(
            stage="coverage",
            error_category="source_behavior_v3_runner_coverage_incomplete",
            transport_attempt_count=counted.calls,
        )
    # Once a request has been attempted, retain the claim even if the provider failed.
    if counted.calls == 0:
        released = _release_claim(claim)
        return V3RunResult(
            status="failed", qualification_status="execution_failed",
            error_code=failure.error_category if failure else "source_behavior_v3_runner_failed",
            model=bindings.config.required_model,
            claim_cleanup_status="released" if released else "release_failed",
            safety=safety,
        )
    try:
        audit_path, receipt_path = _write_artifacts(
            target=target,
            bindings=bindings,
            audits=audits,
            failure=failure,
            transport_attempts=counted.calls,
            parser_invalid_count=parser_invalid_count,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            bindings_current=current,
            safety=safety,
        )
    except Exception:
        return V3RunResult(
            status="failed", qualification_status="execution_failed",
            error_code="source_behavior_v3_runner_artifact_write_failed",
            model=bindings.config.required_model,
            logical_request_count=len(audits) + int(failure is not None and failure.case_id is not None),
            transport_attempt_count=counted.calls,
            completed_case_count=len(audits),
            parser_invalid_count=parser_invalid_count,
            output_root=_safe_path(target),
            claim_cleanup_status="retained_after_attempt",
            safety=safety,
        )
    if failure is not None or not current:
        return V3RunResult(
            status="failed", qualification_status="execution_failed",
            error_code=failure.error_category if failure else "source_behavior_v3_runner_failed",
            model=bindings.config.required_model,
            logical_request_count=len(audits) + int(failure is not None and failure.case_id is not None),
            transport_attempt_count=counted.calls,
            completed_case_count=len(audits),
            parser_invalid_count=parser_invalid_count,
            output_root=_safe_path(target),
            safe_case_audit_path=_safe_path(audit_path) if audits else None,
            receipt_path=_safe_path(receipt_path),
            claim_cleanup_status="retained_after_attempt",
            safety=safety.model_copy(update={"safe_case_audits_written": bool(audits), "receipt_written": True}),
        )
    released = _release_claim(claim)
    return V3RunResult(
        status="completed", qualification_status="source_behavior_qualified",
        model=bindings.config.required_model,
        logical_request_count=REQUEST_CAP,
        transport_attempt_count=counted.calls,
        completed_case_count=len(audits),
        parser_invalid_count=0,
        output_root=_safe_path(target),
        safe_case_audit_path=_safe_path(audit_path),
        receipt_path=_safe_path(receipt_path),
        claim_cleanup_status="released" if released else "release_failed",
        safety=safety.model_copy(update={"safe_case_audits_written": True, "receipt_written": True}),
    )


def execute_qualification_v3(
    *,
    allow_network: bool = False,
    confirm_run: bool = False,
    confirm_request_cap: int | None = None,
    confirm_completion_reservation_cap: int | None = None,
    environment: Mapping[str, str] | None = None,
    transport: OpenAICompatibleTransport | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    approval_path: str | Path = DEFAULT_APPROVAL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    attempt_claim_path: str | Path = DEFAULT_ATTEMPT_CLAIM,
) -> V3RunResult:
    return _execute_v3(
        allow_network=allow_network,
        confirm_run=confirm_run,
        confirm_request_cap=confirm_request_cap,
        confirm_completion_reservation_cap=confirm_completion_reservation_cap,
        environment=environment,
        transport=transport,
        config_path=config_path,
        approval_path=approval_path,
        output_root=output_root,
        attempt_claim_path=attempt_claim_path,
    )


def load_execution_receipt(
    path: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    approval_path: str | Path = DEFAULT_APPROVAL,
) -> V3RunReceipt:
    target = _resolve(path)
    try:
        receipt = V3RunReceipt.model_validate(
            json.loads(
                target.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_pairs,
            )
        )
        audit_path = target.with_name(SAFE_AUDIT_NAME)
        if receipt.safe_case_audit_sha256 is None:
            if audit_path.exists():
                raise ValueError("unexpected audit")
        else:
            raw = audit_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != receipt.safe_case_audit_sha256:
                raise ValueError("audit hash")
            audits = [
                V3CaseAudit.model_validate(
                    json.loads(line, object_pairs_hook=_unique_pairs)
                )
                for line in raw.decode().splitlines()
                if line
            ]
            if len(audits) != receipt.completed_case_count:
                raise ValueError("audit count")
        bindings = _freeze_bindings(
            config_path=config_path, approval_path=approval_path, require_accepted=False
        )
        if (
            receipt.config_sha256 != bindings.config_sha256
            or receipt.approval_sha256 != bindings.approval_sha256
            or receipt.v3_compatibility_receipt_sha256 != bindings.compatibility_receipt_sha256
            or receipt.source_relation_receipt_sha256 != bindings.source_relation_receipt_sha256
            or receipt.runner_sha256 != bindings.config.runner_sha256
            or receipt.model != bindings.config.required_model
            or receipt.schema_contract.schema_name != SCHEMA_NAME
            or receipt.schema_contract.schema_version != "conformity_source_behavior_response.v3"
            or receipt.schema_contract.schema_sha256 != bindings.config.response_schema_sha256
            or receipt.generation_parameters != V3GenerationParameters()
        ):
            raise ValueError("receipt binding")
        if receipt.status == "source_behavior_qualified":
            expected = tuple(
                case.model_copy(update={"model_name": bindings.config.required_model})
                for case in build_cases(DESIGN_CONFIG)
            )
            expected_coordinates = {
                (case.scenario_id, case.projection.value, case.seed) for case in expected
            }
            observed_coordinates = {
                (audit.scenario_id, audit.projection, audit.seed) for audit in audits
            }
            if observed_coordinates != expected_coordinates:
                raise ValueError("audit coverage")
        return receipt
    except Exception as exc:
        raise SourceBehaviorV3RunnerError("source_behavior_v3_runner_receipt_invalid") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded source-behavior qualification v3")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--confirm-request-cap", type=int)
    parser.add_argument("--confirm-completion-reservation-cap", type=int)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--approval", default=DEFAULT_APPROVAL)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--attempt-claim", default=DEFAULT_ATTEMPT_CLAIM)
    args = parser.parse_args(argv)
    result = execute_qualification_v3(
        allow_network=args.allow_network,
        confirm_run=args.confirm_run,
        confirm_request_cap=args.confirm_request_cap,
        confirm_completion_reservation_cap=args.confirm_completion_reservation_cap,
        config_path=args.config,
        approval_path=args.approval,
        output_root=args.output_root,
        attempt_claim_path=args.attempt_claim,
    )
    print(safe_json(result.model_dump(mode="json")))
    return 0 if result.status in {"network_disabled", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPLETION_RESERVATION_CAP",
    "DEFAULT_APPROVAL",
    "DEFAULT_ATTEMPT_CLAIM",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_RUNNER",
    "MAX_RETRIES",
    "MAX_TOKENS",
    "RECEIPT_NAME",
    "RECEIPT_VERSION",
    "REQUEST_CAP",
    "RUNNER_VERSION",
    "SEEDS",
    "V3CaseAudit",
    "V3FailureAudit",
    "V3GenerationParameters",
    "V3RunReceipt",
    "V3RunResult",
    "V3RunnerApproval",
    "V3RunnerConfig",
    "V3RunnerSafety",
    "SourceBehaviorV3RunnerError",
    "execute_qualification_v3",
    "load_execution_receipt",
    "load_runner_approval",
    "load_runner_config",
    "safe_preflight",
]
