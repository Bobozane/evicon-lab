"""One-shot, redacted H-D.2.1 Provider compatibility check.

This module is intentionally separate from every pilot runner.  The default
path constructs no Provider and performs no transport call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field

from .cascade_agent_prompts import CascadeAgentPromptContext, CascadeAgentRuntimeConfig
from .cascade_agent_protocol_v21 import (
    V21_SCHEMA_NAME,
    V21_TEMPLATE_VERSION,
    load_hd21_config,
    parse_cascade_agent_response_v21,
    render_cascade_agent_turn_v21,
)
from .cascade_agent_runtime import CascadeAgentRuntimeStatus
from .cascade_outcomes import ClaimStance
from .llm_contract import LLMProvider, LLMProviderError
from .models import EvidenceCard
from .openai_provider import (
    OpenAICompatibleProvider,
    OpenAICompatibleTransport,
    ProviderConfig,
    ResponseFormatMode,
    TransportResponse,
    UrllibTransport,
)
from .provenance_cascade import ProvenanceNode, SourceCategory, VerificationStatus
from .provenance_cascade_exposure import (
    ControllerClaimView,
    ControllerPublicView,
    ControllerSourceRootView,
    VisibleRootRelation,
)

COMPATIBILITY_MODULE_VERSION = "cascade_agent_protocol_v21_compatibility.v1"
COMPATIBILITY_MODEL_FALLBACK = "compatibility-check-model"
COMPATIBILITY_TIMEOUT_SECONDS = 5.0
COMPATIBILITY_MAX_RETRIES = 0
HD21_AGENT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "provenance_cascade"
    / "pilot"
    / "provenance_cascade_pilot_hd21.v1.toml"
)


def _load_hd21_generation_parameters() -> tuple[float, int, int, str | None]:
    """Read and validate only the frozen H-D.2.1 public generation contract."""

    try:
        config, _ = load_hd21_config(HD21_AGENT_CONFIG_PATH)
    except Exception as exc:
        raise RuntimeError("h-d2-1_generation_config_invalid") from exc
    if (
        config.agent_max_tokens != 512
        or config.seeds[0] != 20260911
        or config.response_format != "json_schema"
        or config.response_schema_name != V21_SCHEMA_NAME
    ):
        raise RuntimeError("h-d2-1_generation_config_invalid")
    return 0.2, config.agent_max_tokens, config.seeds[0], None


(
    COMPATIBILITY_TEMPERATURE,
    COMPATIBILITY_MAX_TOKENS,
    COMPATIBILITY_SEED,
    COMPATIBILITY_REASONING_EFFORT,
) = _load_hd21_generation_parameters()

_ERROR_CATEGORY_MAP = {
    "timeout": "timeout",
    "connection_failure": "connection_failure",
    "dns_or_url_failure": "dns_or_url_failure",
    "authentication_failed": "authentication_failed",
    "http_client_error": "http_client_error",
    "http_server_error": "http_server_error",
    "rate_limited": "rate_limited",
    "malformed_response": "malformed_provider_response",
    "invalid_response": "invalid_provider_response",
    "response_format_unsupported": "response_format_unsupported",
    "provider_failure": "provider_failure",
}


def _error_category(error: LLMProviderError) -> str:
    return _ERROR_CATEGORY_MAP.get(error.code.value, "provider_failure")


def _status_class_for_category(category: str | None) -> str | None:
    if category in {"authentication_failed", "http_client_error", "rate_limited", "response_format_unsupported"}:
        return "4xx"
    if category == "http_server_error":
        return "5xx"
    return None


def _safe_http_status_class(value: object) -> str | None:
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


class CompatibilityResult(BaseModel):
    """Only safe fields permitted in the CLI result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    provider_error_code: str | None = None
    transport_category: str | None = None
    http_status_class: str | None = None
    model: str | None = None
    finish_reason: str | None = None
    parser_valid: bool | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    response_format: str = ResponseFormatMode.JSON_SCHEMA.value
    schema_name: str = V21_SCHEMA_NAME
    attempt_count: int = 0
    safety: dict[str, bool] = Field(
        default_factory=lambda: {
            "one_shot": True,
            "network_enabled": False,
            "private_truth_exposed": False,
            "pilot_material_used": False,
            "results_written": False,
            "request_ledger_written": False,
            "secrets_logged": False,
        },
    )


def build_minimal_public_context(
    *,
    model_name: str = COMPATIBILITY_MODEL_FALLBACK,
    temperature: float = COMPATIBILITY_TEMPERATURE,
    max_tokens: int = COMPATIBILITY_MAX_TOKENS,
    seed: int = COMPATIBILITY_SEED,
) -> CascadeAgentPromptContext:
    """Create a tiny synthetic public context; no pilot/WVS material is read."""

    view = ControllerPublicView(
        scenario_id="compatibility-check-synthetic",
        agent_id="compatibility-agent-01",
        round_id=0,
        claims=(
            ControllerClaimView(
                claim_id="compatibility-claim-01",
                verification_status=VerificationStatus.UNVERIFIED,
                evidence_card_ids=("compatibility-evidence-01",),
            ),
        ),
        evidence_cards=(
            EvidenceCard(
                evidence_id="compatibility-evidence-01",
                claim="A bounded synthetic evidence summary.",
                source="synthetic-public-source",
                supports=["compatibility-claim-01"],
                introduced_round=0,
                visible_to=["compatibility-agent-01"],
                reliability=0.5,
            ),
        ),
        provenance_nodes=(
            ProvenanceNode(
                node_id="compatibility-node-01",
                content_id="compatibility-content-01",
                scenario_id="compatibility-check-synthetic",
                claim_id="compatibility-claim-01",
                source_root_id="compatibility-root-01",
                round_id=0,
            ),
        ),
        source_roots=(
            ControllerSourceRootView(
                source_root_id="compatibility-root-01",
                public_source_category=SourceCategory.PRIMARY_RECORD,
            ),
        ),
        root_relations=(
            VisibleRootRelation(
                provenance_node_id="compatibility-node-01",
                source_root_ids=("compatibility-root-01",),
            ),
        ),
    )
    return CascadeAgentPromptContext.from_public_view(
        view,
        claim_id="compatibility-claim-01",
        directive=None,
        runtime_config=CascadeAgentRuntimeConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        ),
    )


def _attempt_count(provider: object, *, invoked: bool) -> int:
    transport = getattr(provider, "_transport", None)
    calls = getattr(transport, "calls", None)
    if isinstance(calls, int):
        return calls
    if isinstance(calls, list):
        return len(calls)
    return 1 if invoked else 0


def check_provider(provider: LLMProvider, *, context: CascadeAgentPromptContext | None = None) -> CompatibilityResult:
    """Perform exactly one rendered request against an injected Provider."""

    context = context or build_minimal_public_context()
    request = render_cascade_agent_turn_v21(context)
    invoked = False
    try:
        invoked = True
        response = provider.complete(request)
    except LLMProviderError as error:
        attempts = _attempt_count(provider, invoked=invoked)
        category = _error_category(error)
        return CompatibilityResult(
            status="provider_error",
            provider_error_code=category,
            transport_category=category,
            http_status_class=_safe_http_status_class(getattr(error, "status_code", None)) or _status_class_for_category(category),
            model=request.model_name,

            attempt_count=attempts,

        )
    except Exception:
        attempts = _attempt_count(provider, invoked=invoked)
        return CompatibilityResult(
            status="provider_error",
            provider_error_code="provider_failure",
            transport_category="provider_failure",
            model=request.model_name,

            attempt_count=attempts,

        )

    parsed = parse_cascade_agent_response_v21(response.content, context)
    attempts = _attempt_count(provider, invoked=invoked)
    metadata = response.provider_metadata if isinstance(response.provider_metadata, dict) else {}
    return CompatibilityResult(
        status="completed" if parsed.valid else "invalid_response",
        provider_error_code=None,
        transport_category=None,
        http_status_class=_safe_http_status_class(metadata.get("status_code")),
        model=response.model_name,
        finish_reason=response.finish_reason,
        parser_valid=parsed.valid,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens,
        latency_ms=response.latency_ms,

        attempt_count=attempts,

    )


class _CountingTransport:
    """Count real attempts without retaining request payloads or responses."""

    def __init__(self, transport: OpenAICompatibleTransport) -> None:
        self.transport = transport
        self.calls = 0

    def post(self, url: str, headers: Mapping[str, str], payload: dict[str, object], timeout_seconds: float) -> TransportResponse:
        self.calls += 1
        return self.transport.post(url, headers, payload, timeout_seconds)


def run_compatibility_check(
    *,
    allow_network: bool = False,
    environment: Mapping[str, str] | None = None,
    transport: OpenAICompatibleTransport | None = None,
) -> CompatibilityResult:
    """Run the CLI policy; network is impossible unless explicitly enabled."""

    if not allow_network:
        return CompatibilityResult(status="network_disabled")

    temperature, max_tokens, seed, reasoning_effort = _load_hd21_generation_parameters()
    raw = ProviderConfig.from_env(allow_network=True, environment=environment)
    config = raw.model_copy(
        update={
            "allow_network": True,
            "timeout_seconds": COMPATIBILITY_TIMEOUT_SECONDS,
            "max_retries": COMPATIBILITY_MAX_RETRIES,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "reasoning_effort": reasoning_effort,
            "response_format": ResponseFormatMode.JSON_SCHEMA,
            "response_schema_name": V21_SCHEMA_NAME,
        },
    )
    wrapped = _CountingTransport(transport or UrllibTransport())
    provider = OpenAICompatibleProvider(config, transport=wrapped, environment=environment)
    context = build_minimal_public_context(
        model_name=config.model_name or COMPATIBILITY_MODEL_FALLBACK,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )
    result = check_provider(provider, context=context)
    return result.model_copy(update={
        "attempt_count": wrapped.calls,
        "model": config.model_name,
        "safety": {**result.safety, "network_enabled": True},
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot offline-by-default H-D.2.1 Provider compatibility check")
    parser.add_argument("--allow-network", action="store_true", help="explicitly allow one compatibility HTTP request")
    args = parser.parse_args()
    result = run_compatibility_check(allow_network=args.allow_network)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
