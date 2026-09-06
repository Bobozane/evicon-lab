"""One-shot compatibility gate for the source-relation Provider subset v2.

The canonical source-relation response model and parser remain v1.  This
module changes only the JSON Schema sent over the wire after the original v1
request was rejected by the configured gateway.  It never executes the
24-case qualification or a behavior experiment.

The default path is completely offline: no environment lookup, Provider
construction, network call, or output creation occurs until ``--allow-network``
is supplied and the new approval is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_identification import safe_json, sha256_file
from .conformity_source_relation_comprehension_calibration_v1 import (
    DEFAULT_CONFIG as CALIBRATION_CONFIG,
    PROTOCOL_VERSION,
    RESPONSE_SCHEMA_SHA256 as CANONICAL_SCHEMA_SHA256,
    SCHEMA_NAME as CANONICAL_SCHEMA_NAME,
    TEMPLATE_VERSION,
    EvidenceState,
    MaterialType,
    PresentationCounterbalance,
    PublicRootAssignment,
    SourceRelation,
    SourceRelationCalibrationContext,
    VisibleMaterial,
    parse_response,
    render_request,
)
from .conformity_source_relation_comprehension_provider_subset_v2 import (
    PROVIDER_RESPONSE_SCHEMA,
    PROVIDER_RESPONSE_SCHEMA_SHA256,
    PROVIDER_SCHEMA_NAME,
    PROVIDER_SCHEMA_VERSION,
    REMOVED_PROVIDER_KEYWORDS,
)
from .conformity_source_relation_comprehension_provider_subset_v2_approval import (
    DEFAULT_APPROVAL,
    DEFAULT_CONFIG,
    DEFAULT_MODULE,
    PROVIDER_SCHEMA_MODULE,
    SourceRelationProviderSubsetApprovalError,
    SourceRelationProviderSubsetApprovalV2,
    SourceRelationProviderSubsetConfigV2,
    load_compatibility_approval,
    load_compatibility_config,
)
from .llm_contract import LLMProvider, LLMProviderError
from .openai_provider import (
    OpenAICompatibleProvider,
    OpenAICompatibleTransport,
    ProviderConfig,
    ReasoningEffort,
    ResponseFormatMode,
    TransportResponse,
    UrllibTransport,
)


_ROOT = Path(__file__).resolve().parents[2]

COMPATIBILITY_MODULE_VERSION = (
    "conformity_source_relation_comprehension_compatibility_provider_subset.v2"
)
RECEIPT_VERSION = (
    "conformity_source_relation_comprehension_compatibility_provider_subset_receipt.v2"
)
SCHEMA_VERSION = PROVIDER_SCHEMA_VERSION
SCHEMA_NAME = PROVIDER_SCHEMA_NAME
MAX_TOKENS = 128
TEMPERATURE = 0.0
SEED = 20261401
MAX_RETRIES = 0
TIMEOUT_SECONDS = 5.0
REASONING_EFFORT = ReasoningEffort.NONE
DEFAULT_RECEIPT = (
    "outputs/study-locks/"
    "conformity_source_relation_comprehension_compatibility_provider_subset.v2.json"
)
DEFAULT_ATTEMPT_CLAIM = (
    "outputs/study-locks/"
    "conformity_source_relation_comprehension_compatibility_provider_subset.v2.claim"
)


class CompatibilitySchemaV2(BaseModel):
    """The amended Provider-facing schema identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    response_format: Literal["json_schema"] = "json_schema"
    schema_name: Literal[PROVIDER_SCHEMA_NAME] = SCHEMA_NAME
    schema_version: Literal[PROVIDER_SCHEMA_VERSION] = SCHEMA_VERSION
    schema_sha256: str = Field(
        default=PROVIDER_RESPONSE_SCHEMA_SHA256,
        pattern=r"^[0-9a-f]{64}$",
    )


class CompatibilityTokenUsageV2(BaseModel):
    """Token usage is complete or absent; no provider metadata is retained."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def complete_or_absent(self) -> "CompatibilityTokenUsageV2":
        values = (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        if any(value is None for value in values) and not all(
            value is None for value in values
        ):
            raise ValueError("token_usage_must_be_complete_or_absent")
        if all(value is not None for value in values):
            assert self.prompt_tokens is not None
            assert self.completion_tokens is not None
            if self.total_tokens != self.prompt_tokens + self.completion_tokens:
                raise ValueError("token_usage_total_mismatch")
        return self

    @property
    def available(self) -> bool:
        return self.total_tokens is not None


class CompatibilitySafetyV2(BaseModel):
    """Allowlisted safety facts for one invocation result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    one_shot: Literal[True] = True
    network_enabled: bool = False
    provider_constructed: bool = False
    api_key_read: bool = False
    minimal_public_synthetic_content_only: Literal[True] = True
    canonical_parser_used: Literal[True] = True
    provider_subset_schema_used: Literal[True] = True
    parent_v1_failure_preserved: Literal[True] = True
    historical_pilot_material_used: Literal[False] = False
    wvs_material_used: Literal[False] = False
    evaluator_private_truth_exposed: Literal[False] = False
    prompt_saved: Literal[False] = False
    full_response_saved: Literal[False] = False
    api_key_saved: Literal[False] = False
    headers_saved: Literal[False] = False
    provider_metadata_saved: Literal[False] = False
    request_ledger_written: Literal[False] = False
    results_written: Literal[False] = False
    comprehension_calibration_executed: Literal[False] = False
    behavior_protocol_executed: Literal[False] = False
    compatibility_receipt_written: bool = False
    not_paper_result: Literal[True] = True
    no_causal_conclusion: Literal[True] = True


TransportCategory = Literal[
    "network_disabled",
    "compatibility_receipt_exists",
    "compatibility_attempt_claim_exists",
    "compatibility_approval_invalid",
    "compatibility_approval_not_accepted",
    "configuration_invalid",
    "configuration_changed",
    "missing_api_key",
    "missing_base_url",
    "missing_model",
    "timeout",
    "connection_failure",
    "dns_or_url_failure",
    "authentication_failed",
    "http_client_error",
    "http_server_error",
    "rate_limited",
    "malformed_provider_response",
    "invalid_provider_response",
    "invalid_token_usage",
    "response_format_unsupported",
    "provider_failure",
    "receipt_write_failed",
]


class CompatibilityResultV2(BaseModel):
    """Redacted result of at most one amended compatibility request."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    status: Literal[
        "network_disabled", "blocked", "completed", "invalid_response", "provider_error"
    ]
    model: str | None = None
    finish_reason: Literal[
        "stop", "length", "content_filter", "tool_calls", "other"
    ] | None = None
    parser_valid: bool | None = None
    http_status_class: Literal["2xx", "3xx", "4xx", "5xx"] | None = None
    transport_category: TransportCategory | None = None
    token_usage: CompatibilityTokenUsageV2 = Field(
        default_factory=CompatibilityTokenUsageV2
    )
    latency_ms: float | None = Field(default=None, ge=0.0)
    schema_contract: CompatibilitySchemaV2 = Field(
        default_factory=CompatibilitySchemaV2,
        alias="schema",
    )
    attempt_count: int = Field(default=0, ge=0, le=1)
    safety: CompatibilitySafetyV2 = Field(default_factory=CompatibilitySafetyV2)


class CompatibilityGenerationParametersV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tokens: Literal[128] = MAX_TOKENS
    temperature: Literal[0.0] = TEMPERATURE
    seed: Literal[20261401] = SEED
    max_retries: Literal[0] = MAX_RETRIES
    timeout_seconds: Literal[5.0] = TIMEOUT_SECONDS
    reasoning_effort: Literal["none"] = REASONING_EFFORT.value


class CompatibilityExecutionBindingsV2(BaseModel):
    """Static hashes captured before the single permitted request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    compatibility_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_schema_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_compatibility_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_compatibility_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_compatibility_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_attempt_claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_failure_category: Literal["response_format_unsupported"]
    protocol_version: Literal[PROTOCOL_VERSION]
    template_version: Literal[TEMPLATE_VERSION]
    provider_schema_name: Literal[PROVIDER_SCHEMA_NAME]
    provider_schema_version: Literal[PROVIDER_SCHEMA_VERSION]


class CompatibilityReceiptSafetyV2(BaseModel):
    """Safety facts that remain true after a successful receipt write."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    compatibility_only: Literal[True] = True
    provider_subset_amendment: Literal[True] = True
    one_shot: Literal[True] = True
    network_used: Literal[True] = True
    minimal_public_synthetic_content_only: Literal[True] = True
    canonical_parser_used: Literal[True] = True
    local_parser_remains_strict: Literal[True] = True
    parent_v1_failure_preserved: Literal[True] = True
    historical_pilot_material_used: Literal[False] = False
    wvs_material_used: Literal[False] = False
    evaluator_private_truth_exposed: Literal[False] = False
    prompt_saved: Literal[False] = False
    full_response_saved: Literal[False] = False
    api_key_saved: Literal[False] = False
    headers_saved: Literal[False] = False
    provider_metadata_saved: Literal[False] = False
    request_ledger_written: Literal[False] = False
    results_written: Literal[False] = False
    comprehension_calibration_executed: Literal[False] = False
    behavior_protocol_executed: Literal[False] = False
    not_paper_result: Literal[True] = True
    no_causal_conclusion: Literal[True] = True


class CompatibilityReceiptV2(BaseModel):
    """Immutable, redacted receipt for a completed subset compatibility check."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    receipt_version: Literal[RECEIPT_VERSION]
    compatibility_module_version: Literal[COMPATIBILITY_MODULE_VERSION]
    status: Literal["completed"]
    model: str
    finish_reason: Literal["stop"]
    parser_valid: Literal[True]
    http_status_class: Literal["2xx"]
    token_usage: CompatibilityTokenUsageV2
    latency_ms: float = Field(ge=0.0)
    attempt_count: Literal[1]
    compatibility_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_schema_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_compatibility_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_compatibility_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_compatibility_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_attempt_claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_failure_category: Literal["response_format_unsupported"]
    protocol_version: Literal[PROTOCOL_VERSION]
    template_version: Literal[TEMPLATE_VERSION]
    schema_contract: CompatibilitySchemaV2 = Field(alias="schema")
    removed_provider_keywords: tuple[str, ...]
    generation_parameters: CompatibilityGenerationParametersV2
    safety: CompatibilityReceiptSafetyV2

    @model_validator(mode="after")
    def successful_usage_is_present(self) -> "CompatibilityReceiptV2":
        if not self.token_usage.available:
            raise ValueError("provider_subset_receipt_requires_token_usage")
        if self.removed_provider_keywords != tuple(REMOVED_PROVIDER_KEYWORDS):
            raise ValueError("provider_subset_receipt_keyword_binding_mismatch")
        if self.schema_contract.schema_sha256 != PROVIDER_RESPONSE_SCHEMA_SHA256:
            raise ValueError("provider_subset_receipt_schema_binding_mismatch")
        if self.canonical_schema_sha256 != CANONICAL_SCHEMA_SHA256:
            raise ValueError("provider_subset_receipt_canonical_schema_mismatch")
        return self


class _CountingTransport:
    """Hard local cap that protects the one-request compatibility boundary."""

    def __init__(self, transport: OpenAICompatibleTransport) -> None:
        self.transport = transport
        self.calls = 0

    def post(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> TransportResponse:
        if self.calls >= 1:
            raise RuntimeError("source_relation_provider_subset_transport_call_cap_exceeded")
        self.calls += 1
        return self.transport.post(url, headers, payload, timeout_seconds)


_ERROR_CATEGORY_MAP: dict[str, TransportCategory] = {
    "missing_api_key": "missing_api_key",
    "missing_base_url": "missing_base_url",
    "missing_model": "missing_model",
    "timeout": "timeout",
    "connection_failure": "connection_failure",
    "dns_or_url_failure": "dns_or_url_failure",
    "authentication_failed": "authentication_failed",
    "http_client_error": "http_client_error",
    "http_server_error": "http_server_error",
    "rate_limited": "rate_limited",
    "malformed_response": "malformed_provider_response",
    "invalid_response": "invalid_provider_response",
    "usage_mismatch": "invalid_token_usage",
    "response_format_unsupported": "response_format_unsupported",
    "provider_failure": "provider_failure",
}


def _safety(
    *,
    network_enabled: bool = False,
    provider_constructed: bool = False,
    api_key_read: bool = False,
    compatibility_receipt_written: bool = False,
) -> CompatibilitySafetyV2:
    return CompatibilitySafetyV2(
        network_enabled=network_enabled,
        provider_constructed=provider_constructed,
        api_key_read=api_key_read,
        compatibility_receipt_written=compatibility_receipt_written,
    )


def _http_status_class(
    value: object,
) -> Literal["2xx", "3xx", "4xx", "5xx"] | None:
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


def _safe_finish_reason(
    value: str,
) -> Literal["stop", "length", "content_filter", "tool_calls", "other"]:
    if value in {"stop", "length", "content_filter", "tool_calls"}:
        return value  # type: ignore[return-value]
    return "other"


def _attempt_count(provider: object, *, invoked: bool) -> int:
    transport = getattr(provider, "_transport", None)
    calls = getattr(transport, "calls", None)
    if isinstance(calls, int) and not isinstance(calls, bool):
        return min(calls, 1)
    if isinstance(calls, list):
        return min(len(calls), 1)
    return 1 if invoked else 0


def minimal_public_context_v2(
    model_name: str = "source-relation-provider-subset-compatibility-model",
) -> SourceRelationCalibrationContext:
    """Fresh public synthetic material used only for the one-shot check."""

    return SourceRelationCalibrationContext(
        case_id="source-relation-v1-c99",
        scenario_id="source-relation-v1-s01",
        source_relation=SourceRelation.DISTINCT_SHOWN_ROOTS,
        evidence_state=EvidenceState.PRESENT,
        presentation_counterbalance=PresentationCounterbalance.FORWARD,
        visible_material=(
            VisibleMaterial(
                content_id="message-a",
                material_type=MaterialType.SOCIAL_REPORT,
                public_summary=(
                    "A fictional Birchline transit bulletin says an extra evening shuttle may operate on Thursday."
                ),
            ),
            VisibleMaterial(
                content_id="message-b",
                material_type=MaterialType.SOCIAL_REPORT,
                public_summary=(
                    "A fictional Birchline community notice reports a possible additional Thursday evening shuttle."
                ),
            ),
            VisibleMaterial(
                content_id="evidence-card-01",
                material_type=MaterialType.DIRECT_EVIDENCE_CARD,
                public_summary=(
                    "A fictional Birchline public schedule record lists an extra Thursday evening shuttle for review."
                ),
            ),
        ),
        public_root_assignments=(
            PublicRootAssignment(content_id="message-a", source_root_id="lineage-k7"),
            PublicRootAssignment(content_id="message-b", source_root_id="lineage-p4"),
        ),
        model_name=model_name,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        seed=SEED,
    )


def check_provider_v2(
    provider: LLMProvider,
    *,
    context: SourceRelationCalibrationContext | None = None,
) -> CompatibilityResultV2:
    """Send exactly one amended request and parse with the canonical parser."""

    context = context or minimal_public_context_v2()
    request = render_request(context)
    invoked = False
    try:
        invoked = True
        response = provider.complete(request)
    except LLMProviderError as error:
        category = _ERROR_CATEGORY_MAP.get(error.code.value, "provider_failure")
        return CompatibilityResultV2(
            status="provider_error",
            model=request.model_name,
            http_status_class=_http_status_class(error.status_code),
            transport_category=category,
            attempt_count=_attempt_count(provider, invoked=invoked),
            safety=_safety(
                network_enabled=True,
                provider_constructed=True,
                api_key_read=True,
            ),
        )
    except Exception:
        return CompatibilityResultV2(
            status="provider_error",
            model=request.model_name,
            transport_category="provider_failure",
            attempt_count=_attempt_count(provider, invoked=invoked),
            safety=_safety(
                network_enabled=True,
                provider_constructed=True,
                api_key_read=True,
            ),
        )

    try:
        parse_response(response.content, context)
        parser_valid = True
    except ValueError:
        parser_valid = False
    finish_reason = _safe_finish_reason(response.finish_reason)
    metadata = (
        response.provider_metadata
        if isinstance(response.provider_metadata, dict)
        else {}
    )
    status_class = _http_status_class(metadata.get("status_code")) or "2xx"
    try:
        usage = CompatibilityTokenUsageV2(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        )
    except Exception:
        return CompatibilityResultV2(
            status="invalid_response",
            model=request.model_name,
            finish_reason=finish_reason,
            parser_valid=parser_valid,
            http_status_class=status_class,
            transport_category="invalid_token_usage",
            latency_ms=response.latency_ms,
            attempt_count=_attempt_count(provider, invoked=invoked),
            safety=_safety(
                network_enabled=True,
                provider_constructed=True,
                api_key_read=True,
            ),
        )

    completed = parser_valid and finish_reason == "stop" and usage.available
    return CompatibilityResultV2(
        status="completed" if completed else "invalid_response",
        model=request.model_name,
        finish_reason=finish_reason,
        parser_valid=parser_valid,
        http_status_class=status_class,
        transport_category=(
            None
            if completed
            else (
                "invalid_provider_response"
                if not parser_valid or finish_reason != "stop"
                else "invalid_token_usage"
            )
        ),
        token_usage=usage,
        latency_ms=response.latency_ms,
        attempt_count=_attempt_count(provider, invoked=invoked),
        safety=_safety(
            network_enabled=True,
            provider_constructed=True,
            api_key_read=True,
        ),
    )


def _approval_failure_result(error: Exception) -> CompatibilityResultV2:
    code = getattr(error, "code", "")
    if code == "source_relation_provider_subset_approval_not_accepted":
        category: TransportCategory = "compatibility_approval_not_accepted"
    elif "config" in code or "binding" in code:
        category = "configuration_invalid"
    else:
        category = "compatibility_approval_invalid"
    return CompatibilityResultV2(
        status="blocked",
        transport_category=category,
        safety=_safety(),
    )


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_toml(path: str | Path, model_type: type[BaseModel], code: str) -> BaseModel:
    target = _resolve(path)
    try:
        return model_type.model_validate(tomllib.loads(target.read_text(encoding="utf-8")))
    except Exception as exc:
        raise SourceRelationProviderSubsetApprovalError(code) from exc


def _freeze_execution_bindings(
    approval_path: str | Path,
    *,
    require_accepted: bool = True,
) -> CompatibilityExecutionBindingsV2:
    """Snapshot all amended and parent bytes that govern one request."""

    config = load_compatibility_config()
    approval = load_compatibility_approval(approval_path)
    if require_accepted and approval.acceptance_status != "accepted":
        raise SourceRelationProviderSubsetApprovalError(
            "source_relation_provider_subset_approval_not_accepted"
        )
    config_raw = _resolve(DEFAULT_CONFIG).read_bytes()
    approval_raw = _resolve(approval_path).read_bytes()
    return CompatibilityExecutionBindingsV2(
        compatibility_config_sha256=_sha256_bytes(config_raw),
        calibration_config_sha256=config.calibration_config_sha256,
        calibration_protocol_sha256=config.calibration_protocol_sha256,
        compatibility_approval_sha256=_sha256_bytes(approval_raw),
        compatibility_module_sha256=config.compatibility_module_sha256,
        provider_schema_module_sha256=config.provider_schema_module_sha256,
        canonical_schema_sha256=config.canonical_schema_sha256,
        provider_schema_sha256=config.provider_schema_sha256,
        parent_compatibility_config_sha256=config.parent_compatibility_config_sha256,
        parent_compatibility_approval_sha256=config.parent_compatibility_approval_sha256,
        parent_compatibility_module_sha256=config.parent_compatibility_module_sha256,
        parent_attempt_claim_sha256=config.parent_attempt_claim_sha256,
        parent_failure_category=config.parent_failure_category,
        protocol_version=config.calibration_protocol_version,
        template_version=config.template_version,
        provider_schema_name=config.provider_schema_name,
        provider_schema_version=config.provider_schema_version,
    )


def _bindings_still_current(
    bindings: CompatibilityExecutionBindingsV2,
    *,
    approval_path: str | Path,
) -> bool:
    try:
        return _freeze_execution_bindings(approval_path) == bindings
    except Exception:
        return False


def _nonsecret_provider_config(
    environment: Mapping[str, str] | None,
) -> ProviderConfig:
    """Read only endpoint/model before Provider construction can touch a key."""

    source = os.environ if environment is None else environment
    base_url = source.get("EVICON_LLM_BASE_URL") or None
    if not base_url:
        raise ValueError("missing_base_url")
    model_name = source.get("EVICON_LLM_MODEL") or None
    if not model_name:
        raise ValueError("missing_model")
    return ProviderConfig(
        base_url=base_url,
        model_name=model_name,
        allow_network=True,
        timeout_seconds=TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        seed=SEED,
        reasoning_effort=REASONING_EFFORT,
        response_format=ResponseFormatMode.JSON_SCHEMA,
        response_schema_name=PROVIDER_SCHEMA_NAME,
        response_schema=PROVIDER_RESPONSE_SCHEMA,
    )


def _prepare_authorized_provider(
    *,
    environment: Mapping[str, str] | None,
    transport: OpenAICompatibleTransport | None,
) -> tuple[OpenAICompatibleProvider, ProviderConfig] | CompatibilityResultV2:
    try:
        provider_config = _nonsecret_provider_config(environment)
    except ValueError as error:
        category = str(error)
        if category not in {"missing_base_url", "missing_model"}:
            category = "provider_failure"
        return CompatibilityResultV2(
            status="provider_error",
            transport_category=category,  # type: ignore[arg-type]
            safety=_safety(network_enabled=True),
        )
    except Exception:
        return CompatibilityResultV2(
            status="provider_error",
            transport_category="provider_failure",
            safety=_safety(network_enabled=True),
        )
    try:
        counted_transport = _CountingTransport(transport or UrllibTransport())
        provider = OpenAICompatibleProvider(
            provider_config,
            transport=counted_transport,
            environment=environment,
        )
    except Exception:
        return CompatibilityResultV2(
            status="provider_error",
            transport_category="provider_failure",
            safety=_safety(network_enabled=True),
        )
    if not getattr(provider, "_api_key", None):
        return CompatibilityResultV2(
            status="provider_error",
            transport_category="missing_api_key",
            safety=_safety(
                network_enabled=True,
                provider_constructed=True,
                api_key_read=True,
            ),
        )
    return provider, provider_config


def _resolve_receipt_target(path: str | Path) -> Path:
    candidate = Path(path)
    target = candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()
    default = (_ROOT / DEFAULT_RECEIPT).resolve()
    try:
        target.relative_to(_ROOT)
        inside_repository = True
    except ValueError:
        inside_repository = False
    if inside_repository and target != default:
        raise ValueError("source_relation_provider_subset_receipt_path_invalid")
    return target


def _claim_target_for_receipt(receipt_target: Path) -> Path:
    default_receipt = (_ROOT / DEFAULT_RECEIPT).resolve()
    if receipt_target == default_receipt:
        return (_ROOT / DEFAULT_ATTEMPT_CLAIM).resolve()
    return receipt_target.with_name(f".{receipt_target.name}.attempt-claim")


def _acquire_attempt_claim(claim_target: Path) -> None:
    claim_target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(claim_target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError("compatibility_attempt_claim_exists") from exc
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _release_attempt_claim(claim_target: Path) -> None:
    try:
        claim_target.unlink()
    except FileNotFoundError:
        pass


def _receipt_from_result(
    result: CompatibilityResultV2,
    *,
    bindings: CompatibilityExecutionBindingsV2,
) -> CompatibilityReceiptV2:
    if (
        result.status != "completed"
        or result.parser_valid is not True
        or result.finish_reason != "stop"
        or result.http_status_class != "2xx"
        or result.attempt_count != 1
        or result.model is None
        or result.latency_ms is None
        or not result.token_usage.available
    ):
        raise ValueError("source_relation_provider_subset_result_not_completed")
    return CompatibilityReceiptV2(
        receipt_version=RECEIPT_VERSION,
        compatibility_module_version=COMPATIBILITY_MODULE_VERSION,
        status="completed",
        model=result.model,
        finish_reason="stop",
        parser_valid=True,
        http_status_class="2xx",
        token_usage=result.token_usage,
        latency_ms=result.latency_ms,
        attempt_count=1,
        compatibility_config_sha256=bindings.compatibility_config_sha256,
        calibration_config_sha256=bindings.calibration_config_sha256,
        calibration_protocol_sha256=bindings.calibration_protocol_sha256,
        compatibility_approval_sha256=bindings.compatibility_approval_sha256,
        compatibility_module_sha256=bindings.compatibility_module_sha256,
        provider_schema_module_sha256=bindings.provider_schema_module_sha256,
        canonical_schema_sha256=bindings.canonical_schema_sha256,
        provider_schema_sha256=bindings.provider_schema_sha256,
        parent_compatibility_config_sha256=bindings.parent_compatibility_config_sha256,
        parent_compatibility_approval_sha256=bindings.parent_compatibility_approval_sha256,
        parent_compatibility_module_sha256=bindings.parent_compatibility_module_sha256,
        parent_attempt_claim_sha256=bindings.parent_attempt_claim_sha256,
        parent_failure_category=bindings.parent_failure_category,
        protocol_version=bindings.protocol_version,
        template_version=bindings.template_version,
        schema=CompatibilitySchemaV2(),
        removed_provider_keywords=tuple(REMOVED_PROVIDER_KEYWORDS),
        generation_parameters=CompatibilityGenerationParametersV2(),
        safety=CompatibilityReceiptSafetyV2(),
    )


def _write_immutable_text(target: Path, serialized: str) -> None:
    """Atomically create a receipt and never replace an existing file."""

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o600)
        try:
            os.link(temporary_path, target)
        except FileExistsError as exc:
            raise ValueError("compatibility_receipt_exists") from exc
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _register_receipt(
    result: CompatibilityResultV2,
    *,
    bindings: CompatibilityExecutionBindingsV2,
    approval_path: str | Path,
    receipt_path: str | Path,
) -> tuple[str, Path]:
    if not _bindings_still_current(bindings, approval_path=approval_path):
        raise ValueError("source_relation_provider_subset_configuration_changed")
    receipt = _receipt_from_result(result, bindings=bindings)
    target = _resolve_receipt_target(receipt_path)
    serialized = (
        json.dumps(
            receipt.model_dump(mode="json", by_alias=True),
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    _write_immutable_text(target, serialized)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), target


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate_receipt_field")
        payload[key] = value
    return payload


def load_receipt_v2(
    path: str | Path = DEFAULT_RECEIPT,
    *,
    approval_path: str | Path = DEFAULT_APPROVAL,
) -> CompatibilityReceiptV2:
    target = _resolve_receipt_target(path)
    try:
        payload = json.loads(
            target.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_pairs,
        )
        receipt = CompatibilityReceiptV2.model_validate(payload)
    except Exception as exc:
        raise ValueError("source_relation_provider_subset_receipt_invalid") from exc
    try:
        current = _freeze_execution_bindings(approval_path, require_accepted=False)
    except Exception as exc:
        raise ValueError("source_relation_provider_subset_receipt_binding_mismatch") from exc
    expected = {
        "compatibility_config_sha256": current.compatibility_config_sha256,
        "calibration_config_sha256": current.calibration_config_sha256,
        "calibration_protocol_sha256": current.calibration_protocol_sha256,
        "compatibility_approval_sha256": current.compatibility_approval_sha256,
        "compatibility_module_sha256": current.compatibility_module_sha256,
        "provider_schema_module_sha256": current.provider_schema_module_sha256,
        "canonical_schema_sha256": current.canonical_schema_sha256,
        "provider_schema_sha256": current.provider_schema_sha256,
        "parent_compatibility_config_sha256": current.parent_compatibility_config_sha256,
        "parent_compatibility_approval_sha256": current.parent_compatibility_approval_sha256,
        "parent_compatibility_module_sha256": current.parent_compatibility_module_sha256,
        "parent_attempt_claim_sha256": current.parent_attempt_claim_sha256,
        "parent_failure_category": current.parent_failure_category,
        "protocol_version": current.protocol_version,
        "template_version": current.template_version,
    }
    if any(getattr(receipt, key) != value for key, value in expected.items()):
        raise ValueError("source_relation_provider_subset_receipt_binding_mismatch")
    return receipt


def execute_compatibility_provider_subset_v2(
    *,
    allow_network: bool = False,
    environment: Mapping[str, str] | None = None,
    transport: OpenAICompatibleTransport | None = None,
    approval_path: str | Path = DEFAULT_APPROVAL,
    receipt_path: str | Path = DEFAULT_RECEIPT,
) -> CompatibilityResultV2:
    """Run the amended one-shot gate and write only a valid redacted receipt."""

    if not allow_network:
        return CompatibilityResultV2(
            status="network_disabled",
            transport_category="network_disabled",
        )
    try:
        target = _resolve_receipt_target(receipt_path)
    except ValueError:
        return CompatibilityResultV2(
            status="blocked",
            transport_category="receipt_write_failed",
        )
    if target.exists():
        return CompatibilityResultV2(
            status="blocked",
            transport_category="compatibility_receipt_exists",
        )
    claim_target = _claim_target_for_receipt(target)
    if claim_target.exists():
        return CompatibilityResultV2(
            status="blocked",
            transport_category="compatibility_attempt_claim_exists",
        )
    try:
        bindings = _freeze_execution_bindings(approval_path)
    except Exception as error:
        result = _approval_failure_result(error)
        if getattr(error, "code", "") == "source_relation_provider_subset_approval_not_accepted":
            return result
        return result
    try:
        _acquire_attempt_claim(claim_target)
    except ValueError:
        return CompatibilityResultV2(
            status="blocked",
            transport_category="compatibility_attempt_claim_exists",
            safety=_safety(),
        )
    except OSError:
        return CompatibilityResultV2(
            status="blocked",
            transport_category="receipt_write_failed",
            safety=_safety(),
        )

    prepared = _prepare_authorized_provider(environment=environment, transport=transport)
    if isinstance(prepared, CompatibilityResultV2):
        _release_attempt_claim(claim_target)
        return prepared
    provider, provider_config = prepared
    if not _bindings_still_current(bindings, approval_path=approval_path):
        _release_attempt_claim(claim_target)
        return CompatibilityResultV2(
            status="blocked",
            transport_category="configuration_changed",
            safety=_safety(),
        )
    result = check_provider_v2(
        provider,
        context=minimal_public_context_v2(
            provider_config.model_name or "source-relation-provider-subset-compatibility-model"
        ),
    )
    if not _bindings_still_current(bindings, approval_path=approval_path):
        return result.model_copy(
            update={"status": "blocked", "transport_category": "configuration_changed"}
        )
    if result.status != "completed":
        return result
    try:
        _register_receipt(
            result,
            bindings=bindings,
            approval_path=approval_path,
            receipt_path=target,
        )
    except ValueError as error:
        category: TransportCategory = (
            "compatibility_receipt_exists"
            if str(error) == "compatibility_receipt_exists"
            else (
                "configuration_changed"
                if str(error) == "source_relation_provider_subset_configuration_changed"
                else "receipt_write_failed"
            )
        )
        return result.model_copy(update={"status": "blocked", "transport_category": category})
    except Exception:
        return result.model_copy(
            update={"status": "blocked", "transport_category": "receipt_write_failed"}
        )
    _release_attempt_claim(claim_target)
    return result.model_copy(
        update={
            "safety": result.safety.model_copy(
                update={"compatibility_receipt_written": True}
            )
        }
    )


# Short aliases make the command and test API easier to discover.
execute_compatibility_v2 = execute_compatibility_provider_subset_v2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the one-shot source-relation Provider-subset compatibility gate"
    )
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--approval-path", default=DEFAULT_APPROVAL)
    parser.add_argument("--receipt-path", default=DEFAULT_RECEIPT)
    arguments = parser.parse_args(argv)
    result = execute_compatibility_provider_subset_v2(
        allow_network=arguments.allow_network,
        approval_path=arguments.approval_path,
        receipt_path=arguments.receipt_path,
    )
    print(safe_json(result.model_dump(mode="json", by_alias=True)))
    return 0 if result.status in {"network_disabled", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_SCHEMA_SHA256",
    "COMPATIBILITY_MODULE_VERSION",
    "DEFAULT_APPROVAL",
    "DEFAULT_ATTEMPT_CLAIM",
    "DEFAULT_CONFIG",
    "DEFAULT_RECEIPT",
    "MAX_RETRIES",
    "MAX_TOKENS",
    "PROVIDER_RESPONSE_SCHEMA",
    "PROVIDER_RESPONSE_SCHEMA_SHA256",
    "PROVIDER_SCHEMA_NAME",
    "PROVIDER_SCHEMA_VERSION",
    "RECEIPT_VERSION",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SEED",
    "TEMPERATURE",
    "TIMEOUT_SECONDS",
    "CompatibilityExecutionBindingsV2",
    "CompatibilityGenerationParametersV2",
    "CompatibilityReceiptV2",
    "CompatibilityResultV2",
    "CompatibilitySafetyV2",
    "CompatibilitySchemaV2",
    "CompatibilityTokenUsageV2",
    "check_provider_v2",
    "execute_compatibility_provider_subset_v2",
    "execute_compatibility_v2",
    "load_receipt_v2",
    "minimal_public_context_v2",
]
