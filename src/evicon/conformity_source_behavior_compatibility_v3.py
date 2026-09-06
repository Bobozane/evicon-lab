"""One-shot Provider compatibility gate for source-behavior qualification v3.

The default command is inert and offline. The explicitly enabled path permits
one HTTP attempt against a Provider and writes a redacted receipt only after a
valid v3 response. It never executes the 36-case qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_identification import safe_json, sha256_file
from .conformity_source_behavior_qualification_v3 import (
    DEFAULT_APPROVAL,
    DEFAULT_CONFIG,
    PROTOCOL_VERSION,
    RESPONSE_SCHEMA,
    RESPONSE_SCHEMA_SHA256,
    SCHEMA_NAME,
    TEMPLATE_VERSION,
    BehaviorQualificationContextV3,
    ContentRole,
    VisibleContent,
    load_approval,
    load_config,
    parse_behavior_response_v3,
    render_behavior_request_v3,
)
from .conformity_source_manipulation import RootAssignment, SourceProjection
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

COMPATIBILITY_MODULE_VERSION = "conformity_source_behavior_compatibility.v3"
RECEIPT_VERSION = "conformity_source_behavior_compatibility_receipt.v3"
SCHEMA_VERSION = "conformity_source_behavior_response.v3"
MAX_TOKENS = 256
TEMPERATURE = 0.2
SEED = 20261201
MAX_RETRIES = 0
TIMEOUT_SECONDS = 5.0
REASONING_EFFORT = ReasoningEffort.NONE
DEFAULT_RECEIPT = (
    "outputs/study-locks/"
    "conformity_source_behavior_compatibility.v3.json"
)
DEFAULT_MODULE = "src/evicon/conformity_source_behavior_compatibility_v3.py"


class CompatibilitySchemaV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    response_format: Literal["json_schema"] = "json_schema"
    schema_name: Literal["conformity_source_behavior_response_v3"] = SCHEMA_NAME
    schema_version: Literal["conformity_source_behavior_response.v3"] = SCHEMA_VERSION
    schema_sha256: str = Field(default=RESPONSE_SCHEMA_SHA256, pattern=r"^[0-9a-f]{64}$")


class CompatibilityTokenUsageV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def complete_or_absent(self) -> "CompatibilityTokenUsageV3":
        values = (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        if any(value is None for value in values) and not all(value is None for value in values):
            raise ValueError("token usage must be complete or absent")
        if all(value is not None for value in values):
            assert self.prompt_tokens is not None
            assert self.completion_tokens is not None
            if self.total_tokens != self.prompt_tokens + self.completion_tokens:
                raise ValueError("token usage total mismatch")
        return self

    @property
    def available(self) -> bool:
        return self.total_tokens is not None


class CompatibilitySafetyV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    one_shot: Literal[True] = True
    network_enabled: bool = False
    provider_constructed: bool = False
    api_key_read: bool = False
    minimal_public_synthetic_content_only: Literal[True] = True
    v2_artifacts_used: Literal[False] = False
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
    qualification_executed: Literal[False] = False
    compatibility_receipt_written: bool = False
    not_paper_result: Literal[True] = True
    no_causal_conclusion: Literal[True] = True


TransportCategory = Literal[
    "network_disabled", "compatibility_receipt_exists", "approval_invalid",
    "approval_not_accepted", "configuration_invalid", "missing_api_key",
    "missing_base_url", "missing_model", "timeout", "connection_failure",
    "dns_or_url_failure", "authentication_failed", "http_client_error",
    "http_server_error", "rate_limited", "malformed_provider_response",
    "invalid_provider_response", "invalid_token_usage", "response_format_unsupported",
    "provider_failure", "receipt_write_failed",
]


class CompatibilityResultV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True, serialize_by_alias=True)

    status: Literal["network_disabled", "blocked", "completed", "invalid_response", "provider_error"]
    model: str | None = None
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls", "other"] | None = None
    parser_valid: bool | None = None
    http_status_class: Literal["2xx", "3xx", "4xx", "5xx"] | None = None
    transport_category: TransportCategory | None = None
    token_usage: CompatibilityTokenUsageV3 = Field(default_factory=CompatibilityTokenUsageV3)
    latency_ms: float | None = Field(default=None, ge=0.0)
    schema_contract: CompatibilitySchemaV3 = Field(default_factory=CompatibilitySchemaV3, alias="schema")
    attempt_count: int = Field(default=0, ge=0, le=1)
    safety: CompatibilitySafetyV3 = Field(default_factory=CompatibilitySafetyV3)


class CompatibilityGenerationParametersV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tokens: Literal[256] = MAX_TOKENS
    temperature: Literal[0.2] = TEMPERATURE
    seed: Literal[20261201] = SEED
    max_retries: Literal[0] = MAX_RETRIES
    timeout_seconds: Literal[5.0] = TIMEOUT_SECONDS
    reasoning_effort: Literal["none"] = REASONING_EFFORT.value


class CompatibilityReceiptSafetyV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    compatibility_only: Literal[True] = True
    one_shot: Literal[True] = True
    network_used: Literal[True] = True
    minimal_public_synthetic_content_only: Literal[True] = True
    v2_artifacts_used: Literal[False] = False
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
    qualification_executed: Literal[False] = False
    not_paper_result: Literal[True] = True
    no_causal_conclusion: Literal[True] = True


class CompatibilityReceiptV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True, serialize_by_alias=True)

    receipt_version: Literal["conformity_source_behavior_compatibility_receipt.v3"]
    compatibility_module_version: Literal["conformity_source_behavior_compatibility.v3"]
    status: Literal["completed"]
    model: str
    finish_reason: Literal["stop"]
    parser_valid: Literal[True]
    http_status_class: Literal["2xx"]
    token_usage: CompatibilityTokenUsageV3
    latency_ms: float = Field(ge=0.0)
    attempt_count: Literal[1]
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_version: Literal["conformity_source_behavior_qualification.v3"]
    template_version: Literal["conformity_source_behavior_turn.v3"]
    schema_contract: CompatibilitySchemaV3 = Field(alias="schema")
    generation_parameters: CompatibilityGenerationParametersV3
    safety: CompatibilityReceiptSafetyV3

    @model_validator(mode="after")
    def successful_usage_is_present(self) -> "CompatibilityReceiptV3":
        if not self.token_usage.available:
            raise ValueError("v3 compatibility receipt requires token usage")
        return self


class _CountingTransport:
    def __init__(self, transport: OpenAICompatibleTransport) -> None:
        self.transport = transport
        self.calls = 0

    def post(self, url: str, headers: Mapping[str, str], payload: dict[str, object], timeout_seconds: float) -> TransportResponse:
        if self.calls >= 1:
            raise RuntimeError("v3 compatibility transport call cap exceeded")
        self.calls += 1
        return self.transport.post(url, headers, payload, timeout_seconds)


_ERROR_CATEGORY_MAP: dict[str, TransportCategory] = {
    "missing_api_key": "missing_api_key", "missing_base_url": "missing_base_url",
    "missing_model": "missing_model", "timeout": "timeout",
    "connection_failure": "connection_failure", "dns_or_url_failure": "dns_or_url_failure",
    "authentication_failed": "authentication_failed", "http_client_error": "http_client_error",
    "http_server_error": "http_server_error", "rate_limited": "rate_limited",
    "malformed_response": "malformed_provider_response", "invalid_response": "invalid_provider_response",
    "usage_mismatch": "invalid_token_usage", "response_format_unsupported": "response_format_unsupported",
    "provider_failure": "provider_failure",
}


def _safety(*, network_enabled: bool = False, provider_constructed: bool = False, api_key_read: bool = False, compatibility_receipt_written: bool = False) -> CompatibilitySafetyV3:
    return CompatibilitySafetyV3(
        network_enabled=network_enabled,
        provider_constructed=provider_constructed,
        api_key_read=api_key_read,
        compatibility_receipt_written=compatibility_receipt_written,
    )


def _http_status_class(value: object) -> Literal["2xx", "3xx", "4xx", "5xx"] | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if 200 <= value < 300: return "2xx"
    if 300 <= value < 400: return "3xx"
    if 400 <= value < 500: return "4xx"
    if 500 <= value < 600: return "5xx"
    return None


def _safe_finish_reason(value: str) -> Literal["stop", "length", "content_filter", "tool_calls", "other"]:
    return value if value in {"stop", "length", "content_filter", "tool_calls"} else "other"  # type: ignore[return-value]


def _attempt_count(provider: object, *, invoked: bool) -> int:
    transport = getattr(provider, "_transport", None)
    calls = getattr(transport, "calls", None)
    if isinstance(calls, int) and not isinstance(calls, bool): return min(calls, 1)
    if isinstance(calls, list): return min(len(calls), 1)
    return 1 if invoked else 0


def minimal_public_context_v3(model_name: str = "source-behavior-v3-compatibility-model") -> BehaviorQualificationContextV3:
    return BehaviorQualificationContextV3(
        case_id="source-behavior-v3-q001",
        scenario_id="source-behavior-v3-s01",
        projection=SourceProjection.SOURCE_FREE,
        seed=SEED,
        order_position=1,
        target_claim="The fictional Linden Hall reading room will open one hour earlier next month.",
        decision_task="Decide whether this claim is reliable enough for a reversible visitor planning notice.",
        visible_content=(
            VisibleContent(content_id="content-01", role=ContentRole.FOCAL, public_summary="A fictional visitor update says the Linden Hall reading room may open one hour earlier next month."),
            VisibleContent(content_id="content-02", role=ContentRole.FOCAL, public_summary="A fictional community notice repeats the proposed earlier opening time for Linden Hall."),
            VisibleContent(content_id="content-03", role=ContentRole.FIXED_CONTEXT, public_summary="A fictional staffing note says the proposed earlier shift is still awaiting confirmation."),
            VisibleContent(content_id="content-04", role=ContentRole.FIXED_CONTEXT, public_summary="A fictional published schedule retains the current opening time and marks the change as pending."),
        ),
        public_root_assignments=(
            RootAssignment(content_id="content-01", source_root_id=None),
            RootAssignment(content_id="content-02", source_root_id=None),
            RootAssignment(content_id="content-03", source_root_id="root-03"),
            RootAssignment(content_id="content-04", source_root_id="root-04"),
        ),
        model_name=model_name,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )


def check_provider_v3(provider: LLMProvider, *, context: BehaviorQualificationContextV3 | None = None) -> CompatibilityResultV3:
    context = context or minimal_public_context_v3()
    request = render_behavior_request_v3(context)
    invoked = False
    try:
        invoked = True
        response = provider.complete(request)
    except LLMProviderError as error:
        category = _ERROR_CATEGORY_MAP.get(error.code.value, "provider_failure")
        return CompatibilityResultV3(
            status="provider_error", model=request.model_name,
            http_status_class=_http_status_class(error.status_code), transport_category=category,
            attempt_count=_attempt_count(provider, invoked=invoked),
            safety=_safety(network_enabled=True, provider_constructed=True, api_key_read=True),
        )
    except Exception:
        return CompatibilityResultV3(
            status="provider_error", model=request.model_name, transport_category="provider_failure",
            attempt_count=_attempt_count(provider, invoked=invoked),
            safety=_safety(network_enabled=True, provider_constructed=True, api_key_read=True),
        )
    try:
        parse_behavior_response_v3(response.content, context)
        parser_valid = True
    except ValueError:
        parser_valid = False
    finish_reason = _safe_finish_reason(response.finish_reason)
    metadata = response.provider_metadata if isinstance(response.provider_metadata, dict) else {}
    try:
        usage = CompatibilityTokenUsageV3(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        )
    except Exception:
        return CompatibilityResultV3(
            status="invalid_response",
            model=response.model_name,
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
    return CompatibilityResultV3(
        status="completed" if completed else "invalid_response",
        model=response.model_name,
        finish_reason=finish_reason,
        parser_valid=parser_valid,
        http_status_class=_http_status_class(metadata.get("status_code")) or "2xx",
        transport_category=None if completed else (
            "invalid_provider_response" if not parser_valid or finish_reason != "stop" else "invalid_token_usage"
        ),
        token_usage=usage,
        latency_ms=response.latency_ms,
        attempt_count=_attempt_count(provider, invoked=invoked),
        safety=_safety(network_enabled=True, provider_constructed=True, api_key_read=True),
    )


def run_compatibility_check_v3(*, allow_network: bool = False, environment: Mapping[str, str] | None = None, transport: OpenAICompatibleTransport | None = None, approval_path: str | Path = DEFAULT_APPROVAL) -> CompatibilityResultV3:
    if not allow_network:
        return CompatibilityResultV3(status="network_disabled", transport_category="network_disabled")
    try:
        approval = load_approval(approval_path)
    except Exception:
        return CompatibilityResultV3(status="blocked", transport_category="approval_invalid", safety=_safety())
    if approval.acceptance_status != "accepted":
        return CompatibilityResultV3(status="blocked", transport_category="approval_not_accepted", safety=_safety())
    try:
        config, _ = load_config(DEFAULT_CONFIG)
        if (config.max_tokens != MAX_TOKENS or config.temperature != TEMPERATURE or config.max_retries != MAX_RETRIES or config.timeout_seconds != int(TIMEOUT_SECONDS) or config.template_version != TEMPLATE_VERSION or config.response_schema_name != SCHEMA_NAME):
            raise ValueError("v3 compatibility configuration drift")
    except Exception:
        return CompatibilityResultV3(status="blocked", transport_category="configuration_invalid", safety=_safety())
    try:
        raw = ProviderConfig.from_env(allow_network=True, environment=environment)
        provider_config = raw.model_copy(update={
            "allow_network": True, "timeout_seconds": TIMEOUT_SECONDS, "max_retries": MAX_RETRIES,
            "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "seed": SEED,
            "reasoning_effort": REASONING_EFFORT, "response_format": ResponseFormatMode.JSON_SCHEMA,
            "response_schema_name": SCHEMA_NAME, "response_schema": RESPONSE_SCHEMA,
        })
        counted = _CountingTransport(transport or UrllibTransport())
        provider = OpenAICompatibleProvider(provider_config, transport=counted, environment=environment)
        return check_provider_v3(provider, context=minimal_public_context_v3(provider_config.model_name or "source-behavior-v3-compatibility-model"))
    except Exception:
        return CompatibilityResultV3(status="provider_error", transport_category="provider_failure", safety=_safety(network_enabled=True))


def _resolve_receipt_target(path: str | Path) -> Path:
    target = Path(path).resolve() if Path(path).is_absolute() else (_ROOT / path).resolve()
    default = (_ROOT / DEFAULT_RECEIPT).resolve()
    try:
        target.relative_to(_ROOT)
        inside_repository = True
    except ValueError:
        inside_repository = False
    if inside_repository and target != default:
        raise ValueError("compatibility_receipt_path_invalid")
    return target


def receipt_from_result_v3(result: CompatibilityResultV3, *, approval_path: str | Path = DEFAULT_APPROVAL) -> CompatibilityReceiptV3:
    approval = load_approval(approval_path)
    config, _ = load_config(DEFAULT_CONFIG)
    if approval.acceptance_status != "accepted": raise ValueError("source_behavior_v3_approval_not_accepted")
    if (result.status != "completed" or result.parser_valid is not True or result.finish_reason != "stop" or result.http_status_class != "2xx" or result.attempt_count != 1 or result.model is None or result.latency_ms is None or not result.token_usage.available):
        raise ValueError("source_behavior_v3_result_not_completed")
    return CompatibilityReceiptV3(
        receipt_version=RECEIPT_VERSION, compatibility_module_version=COMPATIBILITY_MODULE_VERSION,
        status="completed", model=result.model, finish_reason="stop", parser_valid=True,
        http_status_class="2xx", token_usage=result.token_usage, latency_ms=result.latency_ms,
        attempt_count=1, config_sha256=sha256_file(DEFAULT_CONFIG), corpus_sha256=config.corpus_sha256,
        protocol_sha256=config.protocol_sha256, approval_sha256=sha256_file(approval_path),
        compatibility_module_sha256=sha256_file(DEFAULT_MODULE),
        protocol_version=PROTOCOL_VERSION, template_version=TEMPLATE_VERSION,
        schema=CompatibilitySchemaV3(), generation_parameters=CompatibilityGenerationParametersV3(),
        safety=CompatibilityReceiptSafetyV3(),
    )


def register_receipt_v3(result: CompatibilityResultV3, *, approval_path: str | Path = DEFAULT_APPROVAL, receipt_path: str | Path = DEFAULT_RECEIPT) -> tuple[str, Path]:
    receipt = receipt_from_result_v3(result, approval_path=approval_path)
    target = _resolve_receipt_target(receipt_path)
    if target.exists(): raise ValueError("compatibility_receipt_exists")
    serialized = json.dumps(receipt.model_dump(mode="json"), ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(serialized, encoding="utf-8")
    return hashlib.sha256(serialized.encode()).hexdigest(), target


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate receipt field")
        value[key] = item
    return value


def load_receipt_v3(path: str | Path = DEFAULT_RECEIPT, *, approval_path: str | Path = DEFAULT_APPROVAL) -> CompatibilityReceiptV3:
    target = _resolve_receipt_target(path)
    try:
        payload = json.loads(
            target.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_pairs,
        )
        receipt = CompatibilityReceiptV3.model_validate(payload)
    except Exception as exc:
        raise ValueError("compatibility_v3_receipt_invalid") from exc
    config, _ = load_config(DEFAULT_CONFIG)
    if (
        receipt.config_sha256 != sha256_file(DEFAULT_CONFIG)
        or receipt.corpus_sha256 != config.corpus_sha256
        or receipt.protocol_sha256 != config.protocol_sha256
        or receipt.approval_sha256 != sha256_file(approval_path)
        or receipt.compatibility_module_sha256 != sha256_file(DEFAULT_MODULE)
        or receipt.schema_contract.schema_sha256 != RESPONSE_SCHEMA_SHA256
    ):
        raise ValueError("compatibility_v3_receipt_binding_mismatch")
    return receipt


def execute_compatibility_v3(*, allow_network: bool = False, environment: Mapping[str, str] | None = None, transport: OpenAICompatibleTransport | None = None, approval_path: str | Path = DEFAULT_APPROVAL, receipt_path: str | Path = DEFAULT_RECEIPT) -> CompatibilityResultV3:
    if not allow_network:
        return run_compatibility_check_v3(allow_network=False)
    try:
        target = _resolve_receipt_target(receipt_path)
    except ValueError:
        return CompatibilityResultV3(status="blocked", transport_category="receipt_write_failed")
    if target.exists():
        return CompatibilityResultV3(status="blocked", transport_category="compatibility_receipt_exists")
    result = run_compatibility_check_v3(allow_network=True, environment=environment, transport=transport, approval_path=approval_path)
    if result.status != "completed": return result
    try:
        register_receipt_v3(result, approval_path=approval_path, receipt_path=target)
    except Exception:
        return result.model_copy(update={"status": "blocked", "transport_category": "receipt_write_failed"})
    return result.model_copy(update={"safety": result.safety.model_copy(update={"compatibility_receipt_written": True})})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the one-shot source-behavior v3 Provider compatibility gate")
    parser.add_argument("--allow-network", action="store_true")
    arguments = parser.parse_args(argv)
    result = execute_compatibility_v3(allow_network=arguments.allow_network)
    print(safe_json(result.model_dump(mode="json")))
    return 0 if result.status in {"network_disabled", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPATIBILITY_MODULE_VERSION", "DEFAULT_MODULE", "DEFAULT_RECEIPT", "MAX_RETRIES", "MAX_TOKENS",
    "RECEIPT_VERSION", "REASONING_EFFORT", "RESPONSE_SCHEMA", "RESPONSE_SCHEMA_SHA256",
    "SCHEMA_VERSION", "SEED", "TEMPERATURE", "TIMEOUT_SECONDS",
    "CompatibilityGenerationParametersV3", "CompatibilityReceiptV3", "CompatibilityResultV3",
    "CompatibilitySafetyV3", "CompatibilitySchemaV3", "CompatibilityTokenUsageV3",
    "check_provider_v3", "execute_compatibility_v3", "load_receipt_v3",
    "minimal_public_context_v3", "receipt_from_result_v3", "register_receipt_v3",
    "run_compatibility_check_v3",
]
