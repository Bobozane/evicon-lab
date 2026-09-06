"""One-shot Provider compatibility gate for source-relation comprehension v1.

The default path is inert: it does not read environment variables, construct a
Provider, send a request, or create output.  The explicitly enabled path is
limited to one request and may write only a redacted compatibility receipt.
It never runs the 24-case comprehension calibration or a behavior experiment.
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

from .conformity_identification import safe_json
from .conformity_source_relation_comprehension_compatibility_approval_v1 import (
    DEFAULT_COMPATIBILITY_APPROVAL,
    DEFAULT_COMPATIBILITY_CONFIG,
    DEFAULT_COMPATIBILITY_MODULE,
    SourceRelationCompatibilityApprovalError,
    SourceRelationCompatibilityApprovalV1,
    SourceRelationCompatibilityConfigV1,
)
from .conformity_source_relation_comprehension_calibration_v1 import (
    DEFAULT_CONFIG,
    RESPONSE_SCHEMA,
    RESPONSE_SCHEMA_SHA256,
    SCHEMA_NAME,
    EvidenceState,
    MaterialType,
    PresentationCounterbalance,
    PublicRootAssignment,
    SourceRelationCalibrationConfig,
    SourceRelation,
    SourceRelationCalibrationContext,
    VisibleMaterial,
    parse_response,
    render_request,
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

COMPATIBILITY_MODULE_VERSION = "conformity_source_relation_comprehension_compatibility.v1"
RECEIPT_VERSION = "conformity_source_relation_comprehension_compatibility_receipt.v1"
SCHEMA_VERSION = "conformity_source_relation_comprehension_response.v1"
MAX_TOKENS = 128
TEMPERATURE = 0.0
SEED = 20261401
MAX_RETRIES = 0
TIMEOUT_SECONDS = 5.0
REASONING_EFFORT = ReasoningEffort.NONE
DEFAULT_RECEIPT = (
    "outputs/study-locks/"
    "conformity_source_relation_comprehension_compatibility.v1.json"
)
DEFAULT_ATTEMPT_CLAIM = (
    "outputs/study-locks/"
    "conformity_source_relation_comprehension_compatibility.v1.claim"
)
DEFAULT_MODULE = DEFAULT_COMPATIBILITY_MODULE


class CompatibilitySchemaV1(BaseModel):
    """The exact response-format contract sent to the Provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    response_format: Literal["json_schema"] = "json_schema"
    schema_name: Literal[
        "conformity_source_relation_comprehension_response_v1"
    ] = SCHEMA_NAME
    schema_version: Literal[
        "conformity_source_relation_comprehension_response.v1"
    ] = SCHEMA_VERSION
    schema_sha256: str = Field(
        default=RESPONSE_SCHEMA_SHA256,
        pattern=r"^[0-9a-f]{64}$",
    )


class CompatibilityTokenUsageV1(BaseModel):
    """Usage is either complete or absent; a receipt requires it to be complete."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def complete_or_absent(self) -> "CompatibilityTokenUsageV1":
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


class CompatibilitySafetyV1(BaseModel):
    """Allowlisted safety facts for an invocation result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    one_shot: Literal[True] = True
    network_enabled: bool = False
    provider_constructed: bool = False
    api_key_read: bool = False
    minimal_public_synthetic_content_only: Literal[True] = True
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


class CompatibilityResultV1(BaseModel):
    """Redacted outcome of at most one compatibility request."""

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
    token_usage: CompatibilityTokenUsageV1 = Field(
        default_factory=CompatibilityTokenUsageV1
    )
    latency_ms: float | None = Field(default=None, ge=0.0)
    schema_contract: CompatibilitySchemaV1 = Field(
        default_factory=CompatibilitySchemaV1,
        alias="schema",
    )
    attempt_count: int = Field(default=0, ge=0, le=1)
    safety: CompatibilitySafetyV1 = Field(default_factory=CompatibilitySafetyV1)


class CompatibilityGenerationParametersV1(BaseModel):
    """Fixed generation bounds of the one-shot compatibility request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tokens: Literal[128] = MAX_TOKENS
    temperature: Literal[0.0] = TEMPERATURE
    seed: Literal[20261401] = SEED
    max_retries: Literal[0] = MAX_RETRIES
    timeout_seconds: Literal[5.0] = TIMEOUT_SECONDS
    reasoning_effort: Literal["none"] = REASONING_EFFORT.value


class CompatibilityExecutionBindingsV1(BaseModel):
    """Immutable static bindings captured before the one permitted request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    compatibility_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_version: Literal[
        "conformity_source_relation_comprehension_calibration.v1"
    ]
    template_version: Literal["conformity_source_relation_comprehension_turn.v1"]
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CompatibilityReceiptSafetyV1(BaseModel):
    """Safety facts that remain true after a successful receipt write."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    compatibility_only: Literal[True] = True
    one_shot: Literal[True] = True
    network_used: Literal[True] = True
    minimal_public_synthetic_content_only: Literal[True] = True
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


class CompatibilityReceiptV1(BaseModel):
    """Binding-only receipt; it deliberately excludes request and response text."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    receipt_version: Literal[
        "conformity_source_relation_comprehension_compatibility_receipt.v1"
    ]
    compatibility_module_version: Literal[
        "conformity_source_relation_comprehension_compatibility.v1"
    ]
    status: Literal["completed"]
    model: str
    finish_reason: Literal["stop"]
    parser_valid: Literal[True]
    http_status_class: Literal["2xx"]
    token_usage: CompatibilityTokenUsageV1
    latency_ms: float = Field(ge=0.0)
    attempt_count: Literal[1]
    compatibility_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_version: Literal[
        "conformity_source_relation_comprehension_calibration.v1"
    ]
    template_version: Literal["conformity_source_relation_comprehension_turn.v1"]
    schema_contract: CompatibilitySchemaV1 = Field(alias="schema")
    generation_parameters: CompatibilityGenerationParametersV1
    safety: CompatibilityReceiptSafetyV1

    @model_validator(mode="after")
    def successful_usage_is_present(self) -> "CompatibilityReceiptV1":
        if not self.token_usage.available:
            raise ValueError("compatibility_receipt_requires_token_usage")
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
            raise RuntimeError("source_relation_compatibility_transport_call_cap_exceeded")
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
) -> CompatibilitySafetyV1:
    return CompatibilitySafetyV1(
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
        return value
    return "other"


def _attempt_count(provider: object, *, invoked: bool) -> int:
    transport = getattr(provider, "_transport", None)
    calls = getattr(transport, "calls", None)
    if isinstance(calls, int) and not isinstance(calls, bool):
        return min(calls, 1)
    if isinstance(calls, list):
        return min(len(calls), 1)
    return 1 if invoked else 0


def minimal_public_context_v1(
    model_name: str = "source-relation-compatibility-model",
) -> SourceRelationCalibrationContext:
    """Use fresh, minimal public text rather than any calibration case or output."""

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
                    "A fictional Maple Transit bulletin says an extra evening shuttle may operate on Thursday."
                ),
            ),
            VisibleMaterial(
                content_id="message-b",
                material_type=MaterialType.SOCIAL_REPORT,
                public_summary=(
                    "A fictional Maple Transit community notice reports a possible additional Thursday evening shuttle."
                ),
            ),
            VisibleMaterial(
                content_id="evidence-card-01",
                material_type=MaterialType.DIRECT_EVIDENCE_CARD,
                public_summary=(
                    "A fictional Maple Transit public schedule record lists an extra Thursday evening shuttle for review."
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


def _check_provider_v1(
    provider: LLMProvider,
    *,
    context: SourceRelationCalibrationContext | None = None,
) -> CompatibilityResultV1:
    """Send and parse one structural request after authorization is verified."""

    context = context or minimal_public_context_v1()
    request = render_request(context)
    invoked = False
    try:
        invoked = True
        response = provider.complete(request)
    except LLMProviderError as error:
        category = _ERROR_CATEGORY_MAP.get(error.code.value, "provider_failure")
        return CompatibilityResultV1(
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
        return CompatibilityResultV1(
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
    try:
        usage = CompatibilityTokenUsageV1(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        )
    except Exception:
        return CompatibilityResultV1(
            status="invalid_response",
            # Provider-supplied model labels are untrusted response material.
            # The redacted result/receipt may contain only the local request model.
            model=request.model_name,
            finish_reason=finish_reason,
            parser_valid=parser_valid,
            http_status_class=_http_status_class(metadata.get("status_code")) or "2xx",
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
    return CompatibilityResultV1(
        status="completed" if completed else "invalid_response",
        model=request.model_name,
        finish_reason=finish_reason,
        parser_valid=parser_valid,
        http_status_class=_http_status_class(metadata.get("status_code")) or "2xx",
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


def _approval_failure_result(error: Exception) -> CompatibilityResultV1:
    code = getattr(error, "code", "")
    category: TransportCategory
    if code == "source_relation_compatibility_approval_not_accepted":
        category = "compatibility_approval_not_accepted"
    elif "config" in code or "binding" in code:
        category = "configuration_invalid"
    else:
        category = "compatibility_approval_invalid"
    return CompatibilityResultV1(
        status="blocked",
        transport_category=category,
        safety=_safety(),
    )


def _resolve_static_path_v1(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def _read_binding_bytes_v1(path: str | Path, error_code: str) -> tuple[Path, bytes]:
    target = _resolve_static_path_v1(path)
    try:
        return target, target.read_bytes()
    except Exception as exc:
        raise SourceRelationCompatibilityApprovalError(error_code) from exc


def _parse_toml_binding_v1(
    raw: bytes,
    model_type: type[BaseModel],
    error_code: str,
) -> BaseModel:
    try:
        return model_type.model_validate(tomllib.loads(raw.decode("utf-8")))
    except Exception as exc:
        raise SourceRelationCompatibilityApprovalError(error_code) from exc


def _sha256_bytes_v1(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _freeze_execution_bindings_v1(
    approval_path: str | Path,
    *,
    require_accepted: bool = True,
) -> CompatibilityExecutionBindingsV1:
    """Snapshot exactly the bytes that govern one compatibility request."""

    _, calibration_raw = _read_binding_bytes_v1(
        DEFAULT_CONFIG, "source_relation_compatibility_calibration_config_invalid"
    )
    calibration = _parse_toml_binding_v1(
        calibration_raw,
        SourceRelationCalibrationConfig,
        "source_relation_compatibility_calibration_config_invalid",
    )
    assert isinstance(calibration, SourceRelationCalibrationConfig)
    _, protocol_raw = _read_binding_bytes_v1(
        calibration.protocol_path,
        "source_relation_compatibility_calibration_protocol_invalid",
    )
    if _sha256_bytes_v1(protocol_raw) != calibration.protocol_sha256:
        raise SourceRelationCompatibilityApprovalError(
            "source_relation_compatibility_calibration_protocol_hash_mismatch"
        )

    _, module_raw = _read_binding_bytes_v1(
        DEFAULT_MODULE,
        "source_relation_compatibility_module_invalid",
    )
    _, compatibility_config_raw = _read_binding_bytes_v1(
        DEFAULT_COMPATIBILITY_CONFIG,
        "source_relation_compatibility_config_invalid",
    )
    compatibility_config = _parse_toml_binding_v1(
        compatibility_config_raw,
        SourceRelationCompatibilityConfigV1,
        "source_relation_compatibility_config_invalid",
    )
    assert isinstance(compatibility_config, SourceRelationCompatibilityConfigV1)
    expected_config = {
        "calibration_config_sha256": _sha256_bytes_v1(calibration_raw),
        "calibration_protocol_sha256": calibration.protocol_sha256,
        "calibration_protocol_version": calibration.protocol_version,
        "template_version": calibration.template_version,
        "response_schema_name": SCHEMA_NAME,
        "response_schema_sha256": RESPONSE_SCHEMA_SHA256,
        "compatibility_module_path": DEFAULT_MODULE,
        "compatibility_module_sha256": _sha256_bytes_v1(module_raw),
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "max_retries": MAX_RETRIES,
        "timeout_seconds": TIMEOUT_SECONDS,
        "reasoning_effort": REASONING_EFFORT.value,
    }
    if {
        name: getattr(compatibility_config, name) for name in expected_config
    } != expected_config:
        raise SourceRelationCompatibilityApprovalError(
            "source_relation_compatibility_config_binding_mismatch"
        )

    _, approval_raw = _read_binding_bytes_v1(
        approval_path,
        "source_relation_compatibility_approval_invalid",
    )
    approval = _parse_toml_binding_v1(
        approval_raw,
        SourceRelationCompatibilityApprovalV1,
        "source_relation_compatibility_approval_invalid",
    )
    assert isinstance(approval, SourceRelationCompatibilityApprovalV1)
    expected_approval = {
        "compatibility_config_sha256": _sha256_bytes_v1(compatibility_config_raw),
        "calibration_config_sha256": compatibility_config.calibration_config_sha256,
        "calibration_protocol_sha256": compatibility_config.calibration_protocol_sha256,
        "response_schema_sha256": compatibility_config.response_schema_sha256,
        "compatibility_module_sha256": compatibility_config.compatibility_module_sha256,
    }
    if {name: getattr(approval, name) for name in expected_approval} != expected_approval:
        raise SourceRelationCompatibilityApprovalError(
            "source_relation_compatibility_approval_binding_mismatch"
        )
    if require_accepted and approval.acceptance_status != "accepted":
        raise SourceRelationCompatibilityApprovalError(
            "source_relation_compatibility_approval_not_accepted"
        )
    return CompatibilityExecutionBindingsV1(
        compatibility_config_sha256=_sha256_bytes_v1(compatibility_config_raw),
        calibration_config_sha256=_sha256_bytes_v1(calibration_raw),
        calibration_protocol_sha256=calibration.protocol_sha256,
        compatibility_approval_sha256=_sha256_bytes_v1(approval_raw),
        compatibility_module_sha256=_sha256_bytes_v1(module_raw),
        protocol_version=calibration.protocol_version,
        template_version=calibration.template_version,
        response_schema_sha256=compatibility_config.response_schema_sha256,
    )


def _bindings_still_current_v1(
    bindings: CompatibilityExecutionBindingsV1,
    *,
    approval_path: str | Path,
) -> bool:
    try:
        return _freeze_execution_bindings_v1(approval_path) == bindings
    except Exception:
        return False


def _nonsecret_provider_config_v1(
    environment: Mapping[str, str] | None,
) -> ProviderConfig:
    """Read base URL and model before Provider construction can touch a key."""

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
        response_schema_name=SCHEMA_NAME,
        response_schema=RESPONSE_SCHEMA,
    )


def _prepare_authorized_provider_v1(
    *,
    environment: Mapping[str, str] | None,
    transport: OpenAICompatibleTransport | None,
) -> tuple[OpenAICompatibleProvider, ProviderConfig] | CompatibilityResultV1:
    """Validate non-secret settings before acquiring the one-shot claim."""

    try:
        provider_config = _nonsecret_provider_config_v1(environment)
    except ValueError as error:
        category = str(error)
        if category not in {"missing_base_url", "missing_model"}:
            category = "provider_failure"
        return CompatibilityResultV1(
            status="provider_error",
            transport_category=category,  # type: ignore[arg-type]
            safety=_safety(network_enabled=True),
        )
    except Exception:
        return CompatibilityResultV1(
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
        return CompatibilityResultV1(
            status="provider_error",
            transport_category="provider_failure",
            safety=_safety(network_enabled=True),
        )
    if not getattr(provider, "_api_key", None):
        return CompatibilityResultV1(
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
        raise ValueError("source_relation_compatibility_receipt_path_invalid")
    return target


def _claim_target_for_receipt_v1(receipt_target: Path) -> Path:
    default_receipt = (_ROOT / DEFAULT_RECEIPT).resolve()
    if receipt_target == default_receipt:
        return (_ROOT / DEFAULT_ATTEMPT_CLAIM).resolve()
    return receipt_target.with_name(f".{receipt_target.name}.attempt-claim")


def _acquire_attempt_claim_v1(claim_target: Path) -> None:
    """Claim the sole real request without exposing any request material."""

    claim_target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            claim_target,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise ValueError("compatibility_attempt_claim_exists") from exc
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _release_attempt_claim_v1(claim_target: Path) -> None:
    try:
        claim_target.unlink()
    except FileNotFoundError:
        pass


def _receipt_from_result_v1(
    result: CompatibilityResultV1,
    *,
    bindings: CompatibilityExecutionBindingsV1,
) -> CompatibilityReceiptV1:
    """Build a redacted receipt from bindings frozen before the request."""

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
        raise ValueError("source_relation_compatibility_result_not_completed")
    return CompatibilityReceiptV1(
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
        protocol_version=bindings.protocol_version,
        template_version=bindings.template_version,
        schema=CompatibilitySchemaV1(),
        generation_parameters=CompatibilityGenerationParametersV1(),
        safety=CompatibilityReceiptSafetyV1(),
    )


def _write_immutable_text_v1(target: Path, serialized: str) -> None:
    """Atomically create, but never replace, a completed receipt file."""

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


def _register_receipt_v1(
    result: CompatibilityResultV1,
    *,
    bindings: CompatibilityExecutionBindingsV1,
    approval_path: str | Path,
    receipt_path: str | Path,
) -> tuple[str, Path]:
    """Write the one immutable receipt only while its pre-request bindings hold."""

    if not _bindings_still_current_v1(bindings, approval_path=approval_path):
        raise ValueError("source_relation_compatibility_configuration_changed")
    receipt = _receipt_from_result_v1(result, bindings=bindings)
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
    _write_immutable_text_v1(target, serialized)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), target


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate_receipt_field")
        payload[key] = value
    return payload


def load_receipt_v1(
    path: str | Path = DEFAULT_RECEIPT,
    *,
    approval_path: str | Path = DEFAULT_COMPATIBILITY_APPROVAL,
) -> CompatibilityReceiptV1:
    """Load a receipt only when all static bindings still match."""

    target = _resolve_receipt_target(path)
    try:
        payload = json.loads(
            target.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_pairs,
        )
        receipt = CompatibilityReceiptV1.model_validate(payload)
    except Exception as exc:
        raise ValueError("source_relation_compatibility_receipt_invalid") from exc
    try:
        current = _freeze_execution_bindings_v1(
            approval_path,
            require_accepted=False,
        )
    except Exception as exc:
        raise ValueError("source_relation_compatibility_receipt_binding_mismatch") from exc
    if (
        receipt.compatibility_config_sha256 != current.compatibility_config_sha256
        or receipt.calibration_config_sha256 != current.calibration_config_sha256
        or receipt.calibration_protocol_sha256 != current.calibration_protocol_sha256
        or receipt.compatibility_approval_sha256 != current.compatibility_approval_sha256
        or receipt.compatibility_module_sha256 != current.compatibility_module_sha256
        or receipt.protocol_version != current.protocol_version
        or receipt.template_version != current.template_version
        or receipt.schema_contract.schema_sha256 != current.response_schema_sha256
    ):
        raise ValueError("source_relation_compatibility_receipt_binding_mismatch")
    return receipt


def execute_compatibility_v1(
    *,
    allow_network: bool = False,
    environment: Mapping[str, str] | None = None,
    transport: OpenAICompatibleTransport | None = None,
    approval_path: str | Path = DEFAULT_COMPATIBILITY_APPROVAL,
    receipt_path: str | Path = DEFAULT_RECEIPT,
) -> CompatibilityResultV1:
    """Execute the one-shot gate and write a receipt only for a valid success."""

    if not allow_network:
        return CompatibilityResultV1(
            status="network_disabled",
            transport_category="network_disabled",
        )
    try:
        target = _resolve_receipt_target(receipt_path)
    except ValueError:
        return CompatibilityResultV1(
            status="blocked",
            transport_category="receipt_write_failed",
        )
    if target.exists():
        return CompatibilityResultV1(
            status="blocked",
            transport_category="compatibility_receipt_exists",
        )
    claim_target = _claim_target_for_receipt_v1(target)
    if claim_target.exists():
        return CompatibilityResultV1(
            status="blocked",
            transport_category="compatibility_attempt_claim_exists",
        )
    try:
        bindings = _freeze_execution_bindings_v1(approval_path)
    except Exception as error:
        return _approval_failure_result(error)
    try:
        _acquire_attempt_claim_v1(claim_target)
    except ValueError:
        return CompatibilityResultV1(
            status="blocked",
            transport_category="compatibility_attempt_claim_exists",
            safety=_safety(),
        )
    except OSError:
        return CompatibilityResultV1(
            status="blocked",
            transport_category="receipt_write_failed",
            safety=_safety(),
        )
    prepared = _prepare_authorized_provider_v1(
        environment=environment,
        transport=transport,
    )
    if isinstance(prepared, CompatibilityResultV1):
        _release_attempt_claim_v1(claim_target)
        return prepared
    if not _bindings_still_current_v1(bindings, approval_path=approval_path):
        _release_attempt_claim_v1(claim_target)
        return CompatibilityResultV1(
            status="blocked",
            transport_category="configuration_changed",
            safety=_safety(),
        )
    provider, provider_config = prepared
    result = _check_provider_v1(
        provider,
        context=minimal_public_context_v1(
            provider_config.model_name or "source-relation-compatibility-model"
        ),
    )
    if not _bindings_still_current_v1(bindings, approval_path=approval_path):
        return result.model_copy(
            update={"status": "blocked", "transport_category": "configuration_changed"}
        )
    if result.status != "completed":
        return result
    try:
        _register_receipt_v1(
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
                if str(error) == "source_relation_compatibility_configuration_changed"
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
    _release_attempt_claim_v1(claim_target)
    return result.model_copy(
        update={
            "safety": result.safety.model_copy(
                update={"compatibility_receipt_written": True}
            )
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the one-shot source-relation comprehension Provider compatibility gate"
    )
    parser.add_argument("--allow-network", action="store_true")
    arguments = parser.parse_args(argv)
    result = execute_compatibility_v1(allow_network=arguments.allow_network)
    print(safe_json(result.model_dump(mode="json", by_alias=True)))
    return 0 if result.status in {"network_disabled", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPATIBILITY_MODULE_VERSION",
    "DEFAULT_MODULE",
    "DEFAULT_ATTEMPT_CLAIM",
    "DEFAULT_RECEIPT",
    "MAX_RETRIES",
    "MAX_TOKENS",
    "REASONING_EFFORT",
    "RECEIPT_VERSION",
    "RESPONSE_SCHEMA",
    "RESPONSE_SCHEMA_SHA256",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SEED",
    "TEMPERATURE",
    "TIMEOUT_SECONDS",
    "CompatibilityGenerationParametersV1",
    "CompatibilityExecutionBindingsV1",
    "CompatibilityReceiptV1",
    "CompatibilityResultV1",
    "CompatibilitySafetyV1",
    "CompatibilitySchemaV1",
    "CompatibilityTokenUsageV1",
    "execute_compatibility_v1",
    "load_receipt_v1",
    "minimal_public_context_v1",
]
