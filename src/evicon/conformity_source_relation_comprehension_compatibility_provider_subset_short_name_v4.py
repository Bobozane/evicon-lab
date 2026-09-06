"""One-shot short-schema-name compatibility amendment for source relations.

The v1, v2, and v3 compatibility attempts remain immutable consumed parents.
This v4 amendment changes exactly one on-the-wire value relative to v3:
``response_format.json_schema.name``.  The response-schema body, public
synthetic context, canonical parser, model settings, and fifteen-second
timeout are retained.  It is offline by default and never runs calibration,
qualification, or a behavior experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_identification import safe_json
from .conformity_source_relation_comprehension_calibration_v1 import (
    PROTOCOL_VERSION,
    RESPONSE_SCHEMA_SHA256 as CANONICAL_SCHEMA_SHA256,
    TEMPLATE_VERSION,
    SourceRelationCalibrationContext,
    parse_response,
    render_request,
)
from .conformity_source_relation_comprehension_compatibility_provider_subset_v2 import (
    CompatibilityTokenUsageV2,
    minimal_public_context_v2,
)
from .conformity_source_relation_comprehension_provider_subset_short_name_v4_approval import (
    DEFAULT_APPROVAL,
    DEFAULT_CONFIG,
    DEFAULT_MODULE,
    MAX_PROVIDER_SCHEMA_NAME_LENGTH,
    PROVIDER_SCHEMA_MODULE,
    SHORT_PROVIDER_SCHEMA_NAME,
    SourceRelationProviderSubsetShortNameApprovalError,
    load_compatibility_approval,
    load_compatibility_config,
)
from .conformity_source_relation_comprehension_provider_subset_v2 import (
    PROVIDER_RESPONSE_SCHEMA,
    PROVIDER_RESPONSE_SCHEMA_SHA256,
    PROVIDER_SCHEMA_NAME as V3_PROVIDER_SCHEMA_NAME,
    PROVIDER_SCHEMA_VERSION,
    REMOVED_PROVIDER_KEYWORDS,
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
    "conformity_source_relation_comprehension_compatibility_"
    "provider_subset_short_name.v4"
)
RECEIPT_VERSION = (
    "conformity_source_relation_comprehension_compatibility_"
    "provider_subset_short_name_receipt.v4"
)
MAX_TOKENS = 128
TEMPERATURE = 0.0
SEED = 20261401
MAX_RETRIES = 0
TIMEOUT_SECONDS = 15.0
REASONING_EFFORT = ReasoningEffort.NONE
DEFAULT_RECEIPT = (
    "outputs/study-locks/"
    "conformity_source_relation_comprehension_compatibility_"
    "provider_subset_short_name.v4.json"
)
DEFAULT_ATTEMPT_CLAIM = (
    "outputs/study-locks/"
    "conformity_source_relation_comprehension_compatibility_"
    "provider_subset_short_name.v4.claim"
)


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


class CompatibilitySchemaV4(BaseModel):
    """Wire schema identity with only its provider name shortened."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    response_format: Literal["json_schema"] = "json_schema"
    schema_name: Literal[SHORT_PROVIDER_SCHEMA_NAME] = SHORT_PROVIDER_SCHEMA_NAME
    schema_version: Literal[PROVIDER_SCHEMA_VERSION] = PROVIDER_SCHEMA_VERSION
    schema_sha256: str = Field(
        default=PROVIDER_RESPONSE_SCHEMA_SHA256,
        pattern=r"^[0-9a-f]{64}$",
    )


class CompatibilitySafetyV4(BaseModel):
    """Allowlisted facts only; raw provider material is deliberately absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    one_shot: Literal[True] = True
    network_enabled: bool = False
    provider_constructed: bool = False
    api_key_read: bool = False
    minimal_public_synthetic_content_only: Literal[True] = True
    canonical_parser_used: Literal[True] = True
    provider_subset_schema_used: Literal[True] = True
    short_schema_name_amendment: Literal[True] = True
    only_v3_payload_difference_is_schema_name: Literal[True] = True
    parent_v1_v2_v3_preserved: Literal[True] = True
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


class CompatibilityResultV4(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, serialize_by_alias=True
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
    schema_contract: CompatibilitySchemaV4 = Field(
        default_factory=CompatibilitySchemaV4, alias="schema"
    )
    attempt_count: int = Field(default=0, ge=0, le=1)
    safety: CompatibilitySafetyV4 = Field(default_factory=CompatibilitySafetyV4)


class CompatibilityGenerationParametersV4(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tokens: Literal[128] = MAX_TOKENS
    temperature: Literal[0.0] = TEMPERATURE
    seed: Literal[20261401] = SEED
    max_retries: Literal[0] = MAX_RETRIES
    timeout_seconds: Literal[15.0] = TIMEOUT_SECONDS
    reasoning_effort: Literal["none"] = REASONING_EFFORT.value


class CompatibilityExecutionBindingsV4(BaseModel):
    """Hashes and immutable parent facts captured before the request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    compatibility_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_schema_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
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
    v1_attempt_status: Literal["provider_error"]
    v1_failure_category: Literal["response_format_unsupported"]
    v2_attempt_status: Literal["provider_error"]
    v2_failure_category: Literal["timeout"]
    v3_attempt_status: Literal["provider_error"]
    v3_failure_category: Literal["http_client_error"]
    protocol_version: Literal[PROTOCOL_VERSION]
    template_version: Literal[TEMPLATE_VERSION]
    v3_provider_schema_name: Literal[V3_PROVIDER_SCHEMA_NAME]
    v3_provider_schema_name_length: Literal[71]
    provider_schema_name: Literal[SHORT_PROVIDER_SCHEMA_NAME]
    provider_schema_name_length: Literal[42]
    provider_schema_name_max_length: Literal[MAX_PROVIDER_SCHEMA_NAME_LENGTH]
    provider_schema_version: Literal[PROVIDER_SCHEMA_VERSION]


class CompatibilityReceiptSafetyV4(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    compatibility_only: Literal[True] = True
    one_field_schema_name_amendment: Literal[True] = True
    one_shot: Literal[True] = True
    network_used: Literal[True] = True
    minimal_public_synthetic_content_only: Literal[True] = True
    canonical_parser_used: Literal[True] = True
    parent_v1_v2_v3_preserved: Literal[True] = True
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


class CompatibilityReceiptV4(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, serialize_by_alias=True
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
    execution_bindings: CompatibilityExecutionBindingsV4
    schema_contract: CompatibilitySchemaV4 = Field(alias="schema")
    removed_provider_keywords: tuple[str, ...]
    generation_parameters: CompatibilityGenerationParametersV4
    safety: CompatibilityReceiptSafetyV4

    @model_validator(mode="after")
    def valid_completed_receipt(self) -> "CompatibilityReceiptV4":
        if not self.token_usage.available:
            raise ValueError("short_name_v4_receipt_requires_token_usage")
        if self.removed_provider_keywords != tuple(REMOVED_PROVIDER_KEYWORDS):
            raise ValueError("short_name_v4_keyword_binding_mismatch")
        if self.schema_contract.schema_sha256 != PROVIDER_RESPONSE_SCHEMA_SHA256:
            raise ValueError("short_name_v4_schema_binding_mismatch")
        if (
            self.execution_bindings.provider_schema_name_length
            > self.execution_bindings.provider_schema_name_max_length
        ):
            raise ValueError("short_name_v4_name_length_mismatch")
        if self.execution_bindings.canonical_schema_sha256 != CANONICAL_SCHEMA_SHA256:
            raise ValueError("short_name_v4_canonical_schema_mismatch")
        return self


class _CountingTransport:
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
            raise RuntimeError("source_relation_short_name_v4_transport_call_cap_exceeded")
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
) -> CompatibilitySafetyV4:
    return CompatibilitySafetyV4(
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


def check_provider_v4(
    provider: LLMProvider,
    *,
    context: SourceRelationCalibrationContext | None = None,
) -> CompatibilityResultV4:
    """Issue exactly one request and reduce its outcome to redacted fields."""

    context = context or minimal_public_context_v2()
    request = render_request(context)
    invoked = False
    try:
        invoked = True
        response = provider.complete(request)
    except LLMProviderError as error:
        return CompatibilityResultV4(
            status="provider_error",
            model=request.model_name,
            http_status_class=_http_status_class(error.status_code),
            transport_category=_ERROR_CATEGORY_MAP.get(
                error.code.value, "provider_failure"
            ),
            attempt_count=_attempt_count(provider, invoked=invoked),
            safety=_safety(
                network_enabled=True, provider_constructed=True, api_key_read=True
            ),
        )
    except Exception:
        return CompatibilityResultV4(
            status="provider_error",
            model=request.model_name,
            transport_category="provider_failure",
            attempt_count=_attempt_count(provider, invoked=invoked),
            safety=_safety(
                network_enabled=True, provider_constructed=True, api_key_read=True
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
        return CompatibilityResultV4(
            status="invalid_response",
            model=request.model_name,
            finish_reason=finish_reason,
            parser_valid=parser_valid,
            http_status_class=status_class,
            transport_category="invalid_token_usage",
            latency_ms=response.latency_ms,
            attempt_count=_attempt_count(provider, invoked=invoked),
            safety=_safety(
                network_enabled=True, provider_constructed=True, api_key_read=True
            ),
        )
    completed = parser_valid and finish_reason == "stop" and usage.available
    return CompatibilityResultV4(
        status="completed" if completed else "invalid_response",
        model=request.model_name,
        finish_reason=finish_reason,
        parser_valid=parser_valid,
        http_status_class=status_class,
        transport_category=None if completed else "invalid_provider_response",
        token_usage=usage,
        latency_ms=response.latency_ms,
        attempt_count=_attempt_count(provider, invoked=invoked),
        safety=_safety(
            network_enabled=True, provider_constructed=True, api_key_read=True
        ),
    )


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def _freeze_execution_bindings(
    approval_path: str | Path,
    *,
    require_accepted: bool = True,
) -> CompatibilityExecutionBindingsV4:
    config = load_compatibility_config()
    approval = load_compatibility_approval(approval_path)
    if require_accepted and approval.acceptance_status != "accepted":
        raise SourceRelationProviderSubsetShortNameApprovalError(
            "source_relation_short_name_v4_approval_not_accepted"
        )
    return CompatibilityExecutionBindingsV4(
        compatibility_config_sha256=_sha256(DEFAULT_CONFIG),
        compatibility_approval_sha256=_sha256(approval_path),
        compatibility_module_sha256=config.compatibility_module_sha256,
        provider_schema_module_sha256=config.provider_schema_module_sha256,
        canonical_schema_sha256=CANONICAL_SCHEMA_SHA256,
        provider_schema_sha256=config.provider_schema_sha256,
        v1_config_sha256=config.v1_config_sha256,
        v1_approval_sha256=config.v1_approval_sha256,
        v1_module_sha256=config.v1_module_sha256,
        v1_attempt_claim_sha256=config.v1_attempt_claim_sha256,
        v2_config_sha256=config.v2_config_sha256,
        v2_approval_sha256=config.v2_approval_sha256,
        v2_module_sha256=config.v2_module_sha256,
        v2_attempt_claim_sha256=config.v2_attempt_claim_sha256,
        v3_config_sha256=config.v3_config_sha256,
        v3_approval_sha256=config.v3_approval_sha256,
        v3_module_sha256=config.v3_module_sha256,
        v3_attempt_claim_sha256=config.v3_attempt_claim_sha256,
        v1_attempt_status=config.v1_attempt_status,
        v1_failure_category=config.v1_failure_category,
        v2_attempt_status=config.v2_attempt_status,
        v2_failure_category=config.v2_failure_category,
        v3_attempt_status=config.v3_attempt_status,
        v3_failure_category=config.v3_failure_category,
        protocol_version=PROTOCOL_VERSION,
        template_version=TEMPLATE_VERSION,
        v3_provider_schema_name=config.v3_provider_schema_name,
        v3_provider_schema_name_length=config.v3_provider_schema_name_length,
        provider_schema_name=config.provider_schema_name,
        provider_schema_name_length=config.provider_schema_name_length,
        provider_schema_name_max_length=config.provider_schema_name_max_length,
        provider_schema_version=config.provider_schema_version,
    )


def _bindings_still_current(
    bindings: CompatibilityExecutionBindingsV4,
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
        response_schema_name=SHORT_PROVIDER_SCHEMA_NAME,
        response_schema=PROVIDER_RESPONSE_SCHEMA,
    )


def _prepare_provider(
    *,
    environment: Mapping[str, str] | None,
    transport: OpenAICompatibleTransport | None,
) -> tuple[OpenAICompatibleProvider, ProviderConfig] | CompatibilityResultV4:
    try:
        config = _nonsecret_provider_config(environment)
    except ValueError as error:
        category = str(error)
        return CompatibilityResultV4(
            status="provider_error",
            transport_category=(
                category
                if category in {"missing_base_url", "missing_model"}
                else "provider_failure"
            ),  # type: ignore[arg-type]
            safety=_safety(network_enabled=True),
        )
    except Exception:
        return CompatibilityResultV4(
            status="provider_error",
            transport_category="provider_failure",
            safety=_safety(network_enabled=True),
        )
    try:
        provider = OpenAICompatibleProvider(
            config,
            transport=_CountingTransport(transport or UrllibTransport()),
            environment=environment,
        )
    except Exception:
        return CompatibilityResultV4(
            status="provider_error",
            transport_category="provider_failure",
            safety=_safety(network_enabled=True),
        )
    if not getattr(provider, "_api_key", None):
        return CompatibilityResultV4(
            status="provider_error",
            transport_category="missing_api_key",
            safety=_safety(
                network_enabled=True, provider_constructed=True, api_key_read=True
            ),
        )
    return provider, config


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
        raise ValueError("source_relation_short_name_v4_receipt_path_invalid")
    return target


def _claim_target_for_receipt(receipt_target: Path) -> Path:
    if receipt_target == (_ROOT / DEFAULT_RECEIPT).resolve():
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
    result: CompatibilityResultV4,
    *,
    bindings: CompatibilityExecutionBindingsV4,
) -> CompatibilityReceiptV4:
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
        raise ValueError("source_relation_short_name_v4_result_not_completed")
    return CompatibilityReceiptV4(
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
        execution_bindings=bindings,
        schema=CompatibilitySchemaV4(),
        removed_provider_keywords=tuple(REMOVED_PROVIDER_KEYWORDS),
        generation_parameters=CompatibilityGenerationParametersV4(),
        safety=CompatibilityReceiptSafetyV4(),
    )


def _write_immutable_text(target: Path, serialized: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
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


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate_receipt_field")
        payload[key] = value
    return payload


def load_receipt_v4(
    path: str | Path = DEFAULT_RECEIPT,
    *,
    approval_path: str | Path = DEFAULT_APPROVAL,
) -> CompatibilityReceiptV4:
    target = _resolve(path)
    try:
        receipt = CompatibilityReceiptV4.model_validate(
            json.loads(
                target.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs
            )
        )
    except Exception as exc:
        raise ValueError("source_relation_short_name_v4_receipt_invalid") from exc
    try:
        current = _freeze_execution_bindings(approval_path, require_accepted=False)
    except Exception as exc:
        raise ValueError("source_relation_short_name_v4_receipt_binding_mismatch") from exc
    if receipt.execution_bindings != current:
        raise ValueError("source_relation_short_name_v4_receipt_binding_mismatch")
    return receipt


def execute_compatibility_provider_subset_short_name_v4(
    *,
    allow_network: bool = False,
    environment: Mapping[str, str] | None = None,
    transport: OpenAICompatibleTransport | None = None,
    approval_path: str | Path = DEFAULT_APPROVAL,
    receipt_path: str | Path = DEFAULT_RECEIPT,
) -> CompatibilityResultV4:
    """Run the one allowed v4 request after separate acceptance and opt-in."""

    if not allow_network:
        return CompatibilityResultV4(
            status="network_disabled", transport_category="network_disabled"
        )
    try:
        target = _resolve_receipt_target(receipt_path)
    except ValueError:
        return CompatibilityResultV4(
            status="blocked", transport_category="receipt_write_failed"
        )
    if target.exists():
        return CompatibilityResultV4(
            status="blocked", transport_category="compatibility_receipt_exists"
        )
    claim_target = _claim_target_for_receipt(target)
    if claim_target.exists():
        return CompatibilityResultV4(
            status="blocked", transport_category="compatibility_attempt_claim_exists"
        )
    try:
        bindings = _freeze_execution_bindings(approval_path)
    except Exception as error:
        code = getattr(error, "code", "")
        if code == "source_relation_short_name_v4_approval_not_accepted":
            category: TransportCategory = "compatibility_approval_not_accepted"
        elif "config" in code or "binding" in code:
            category = "configuration_invalid"
        else:
            category = "compatibility_approval_invalid"
        return CompatibilityResultV4(status="blocked", transport_category=category)
    try:
        _acquire_attempt_claim(claim_target)
    except ValueError:
        return CompatibilityResultV4(
            status="blocked", transport_category="compatibility_attempt_claim_exists"
        )
    except OSError:
        return CompatibilityResultV4(
            status="blocked", transport_category="receipt_write_failed"
        )

    prepared = _prepare_provider(environment=environment, transport=transport)
    if isinstance(prepared, CompatibilityResultV4):
        _release_attempt_claim(claim_target)
        return prepared
    provider, provider_config = prepared
    if not _bindings_still_current(bindings, approval_path=approval_path):
        _release_attempt_claim(claim_target)
        return CompatibilityResultV4(
            status="blocked", transport_category="configuration_changed"
        )
    result = check_provider_v4(
        provider,
        context=minimal_public_context_v2(
            provider_config.model_name or "source-relation-short-name-v4-model"
        ),
    )
    if not _bindings_still_current(bindings, approval_path=approval_path):
        return result.model_copy(
            update={"status": "blocked", "transport_category": "configuration_changed"}
        )
    if result.status != "completed":
        return result
    try:
        receipt = _receipt_from_result(result, bindings=bindings)
        serialized = (
            json.dumps(
                receipt.model_dump(mode="json", by_alias=True),
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        if not _bindings_still_current(bindings, approval_path=approval_path):
            raise ValueError("configuration_changed")
        _write_immutable_text(target, serialized)
    except ValueError as error:
        category = (
            "compatibility_receipt_exists"
            if str(error) == "compatibility_receipt_exists"
            else (
                "configuration_changed"
                if str(error) == "configuration_changed"
                else "receipt_write_failed"
            )
        )
        return result.model_copy(
            update={"status": "blocked", "transport_category": category}
        )
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


execute_compatibility_v4 = execute_compatibility_provider_subset_short_name_v4


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the one-shot source-relation short-name amendment v4"
    )
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--approval-path", default=DEFAULT_APPROVAL)
    parser.add_argument("--receipt-path", default=DEFAULT_RECEIPT)
    arguments = parser.parse_args(argv)
    result = execute_compatibility_provider_subset_short_name_v4(
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
    "PROVIDER_SCHEMA_VERSION",
    "RECEIPT_VERSION",
    "REASONING_EFFORT",
    "SEED",
    "SHORT_PROVIDER_SCHEMA_NAME",
    "TEMPERATURE",
    "TIMEOUT_SECONDS",
    "CompatibilityExecutionBindingsV4",
    "CompatibilityGenerationParametersV4",
    "CompatibilityReceiptV4",
    "CompatibilityResultV4",
    "CompatibilitySafetyV4",
    "CompatibilitySchemaV4",
    "check_provider_v4",
    "execute_compatibility_provider_subset_short_name_v4",
    "execute_compatibility_v4",
    "load_receipt_v4",
]
