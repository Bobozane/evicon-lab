"""One-shot, redacted H-D.3 Provider compatibility check.

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
from .cascade_agent_protocol_v2 import (
    V2_SCHEMA_NAME,
    V2_TEMPLATE_VERSION,
    parse_cascade_agent_response_v2,
    render_cascade_agent_turn_v2,
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

COMPATIBILITY_MODULE_VERSION = "cascade_agent_protocol_v2_compatibility.v1"
COMPATIBILITY_MODEL_FALLBACK = "compatibility-check-model"
COMPATIBILITY_TIMEOUT_SECONDS = 5.0
COMPATIBILITY_MAX_RETRIES = 0
HD2_AGENT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "provenance_cascade"
    / "pilot"
    / "provenance_cascade_pilot_hd2.v1.toml"
)


def _load_hd2_generation_parameters() -> tuple[float, int, int, str | None]:
    """Read H-D.2 generation settings without reading secrets.

    The H-D.2 v1 sidecar owns the preregistered seed schedule and strict
    response contract.  Its generation defaults remain inherited from the
    hash-registered parent provider block, so this check reads both files but
    never mutates either one.  An absent ``reasoning_effort`` is intentional
    and causes the optional payload field to be omitted.
    """

    try:
        raw_text = HD2_AGENT_CONFIG_PATH.read_text(encoding="utf-8")
        raw = tomllib.loads(raw_text)
        seeds = raw["seeds"]
        parent_config_path = HD2_AGENT_CONFIG_PATH.parent / str(raw["parent_config_path"])
        parent_text = parent_config_path.read_text(encoding="utf-8")
        expected_parent_hash = str(raw["parent_config_sha256"])
        actual_parent_hash = hashlib.sha256(parent_text.encode("utf-8")).hexdigest()
        if actual_parent_hash != expected_parent_hash:
            raise RuntimeError("h-d2_generation_config_invalid")
        settings = raw
        if "agent_temperature" not in settings or "agent_max_tokens" not in settings:
            parent = tomllib.loads(parent_text)
            settings = parent["provider"]
        temperature = float(settings["agent_temperature"])
        max_tokens = int(settings["agent_max_tokens"])
        seed = int(seeds[0])
        reasoning_effort = settings.get("reasoning_effort")
    except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
        raise RuntimeError("h-d2_generation_config_invalid") from exc
    if reasoning_effort is not None and not isinstance(reasoning_effort, str):
        raise RuntimeError("h-d2_generation_config_invalid")
    if max_tokens <= 0 or not 0 <= temperature <= 2:
        raise RuntimeError("h-d2_generation_config_invalid")
    return temperature, max_tokens, seed, reasoning_effort


(
    COMPATIBILITY_TEMPERATURE,
    COMPATIBILITY_MAX_TOKENS,
    COMPATIBILITY_SEED,
    COMPATIBILITY_REASONING_EFFORT,
) = _load_hd2_generation_parameters()

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
    response_format_requested: str = ResponseFormatMode.JSON_SCHEMA.value
    model: str | None = None
    finish_reason: str | None = None
    parser_valid: bool | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    response_format: str = ResponseFormatMode.JSON_SCHEMA.value
    schema_name: str = V2_SCHEMA_NAME
    request_attempt_count: int = 0
    attempt_count: int = 0
    network: str = "disabled"
    safety: dict[str, bool] = Field(
        default_factory=lambda: {
            "one_shot": True,
            "private_truth_exposed": False,
            "results_written": False,
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
    request = render_cascade_agent_turn_v2(context)
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
            request_attempt_count=attempts,
            attempt_count=attempts,
            network="enabled",
        )
    except Exception:
        attempts = _attempt_count(provider, invoked=invoked)
        return CompatibilityResult(
            status="provider_error",
            provider_error_code="provider_failure",
            transport_category="provider_failure",
            model=request.model_name,
            request_attempt_count=attempts,
            attempt_count=attempts,
            network="enabled",
        )

    parsed = parse_cascade_agent_response_v2(response.content, context)
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
        request_attempt_count=attempts,
        attempt_count=attempts,
        network="enabled",
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
        return CompatibilityResult(status="network_disabled", network="disabled", response_format_requested=ResponseFormatMode.JSON_SCHEMA.value)

    temperature, max_tokens, seed, reasoning_effort = _load_hd2_generation_parameters()
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
            "response_schema_name": V2_SCHEMA_NAME,
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
    return result.model_copy(update={"request_attempt_count": wrapped.calls, "attempt_count": wrapped.calls, "model": config.model_name})


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot offline-by-default H-D.3 Provider compatibility check")
    parser.add_argument("--allow-network", action="store_true", help="explicitly allow one compatibility HTTP request")
    args = parser.parse_args()
    result = run_compatibility_check(allow_network=args.allow_network)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
