"""One-shot, offline-by-default Provider compatibility check for H-G.1."""
from __future__ import annotations

import argparse
import json
import socket
import tomllib
from pathlib import Path
from typing import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .cascade_agent_prompts import CascadeAgentPromptContext, CascadeAgentRuntimeConfig
from .cascade_agent_protocol_hg1 import (
    HG1_PROTOCOL_VERSION,
    HG1_SCHEMA_NAME,
    HG1_TEMPLATE_VERSION,
    build_identifiable_prompt_context,
    render_cascade_agent_turn_hg1,
)
from .cascade_agent_protocol_v21 import parse_cascade_agent_response_v21
from .llm_contract import LLMProvider, LLMProviderError, LLMResponse
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
from .provenance_cascade_hg1 import HG1PilotConfig, sha256_file

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_RELATIVE = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg1.v1.toml"
DEFAULT_PROTOCOL_RELATIVE = "src/evicon/cascade_agent_protocol_hg1.py"
COMPATIBILITY_MODULE_VERSION = "provenance_cascade_hg1_compatibility.v1"
COMPATIBILITY_TIMEOUT_SECONDS = 5.0
COMPATIBILITY_MAX_RETRIES = 0
COMPATIBILITY_MAX_TOKENS = 512
COMPATIBILITY_TEMPERATURE = 0.2
COMPATIBILITY_SEED = 20261001
COMPATIBILITY_MODEL_FALLBACK = "hg1-compatibility-check-model"

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


class HG1CompatibilityResult(BaseModel):
    """Safe compatibility output; request and response content are excluded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    provider_error_code: str | None = None
    transport_category: str | None = None
    http_status_class: str | None = None
    model: str | None = None
    finish_reason: str | None = None
    parser_valid: bool | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    response_format: str = "json_schema"
    schema_name: str = HG1_SCHEMA_NAME
    attempt_count: int = Field(default=0, ge=0)
    network: str = "disabled"
    safety: dict[str, bool] = Field(default_factory=lambda: {
        "one_shot": True,
        "pilot_material_used": False,
        "private_truth_exposed": False,
        "request_ledger_written": False,
        "results_written": False,
        "secrets_logged": False,
    })


class _CountingTransport:
    """Count attempts without retaining URL, headers, payload, or response."""

    def __init__(self, transport: OpenAICompatibleTransport) -> None:
        self.transport = transport
        self.calls = 0

    def post(self, url: str, headers: Mapping[str, str], payload: dict[str, object], timeout_seconds: float) -> TransportResponse:
        self.calls += 1
        return self.transport.post(url, headers, payload, timeout_seconds)


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (_ROOT / path).resolve()


def _load_generation_contract(config_path: str | Path = DEFAULT_CONFIG_RELATIVE) -> HG1PilotConfig:
    """Read only the public H-G.1 config; evaluator-private files are not opened."""
    path = _rooted(config_path)
    try:
        config = HG1PilotConfig.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise ValueError("hg1_compatibility_config_invalid") from exc
    protocol_path = _rooted(config.protocol_path)
    if sha256_file(protocol_path) != config.protocol_sha256:
        raise ValueError("hg1_compatibility_protocol_hash_mismatch")
    if (
        config.protocol_version != HG1_PROTOCOL_VERSION
        or config.template_version != HG1_TEMPLATE_VERSION
        or config.response_schema_name != HG1_SCHEMA_NAME
        or config.response_format != "json_schema"
        or config.agent_max_tokens != COMPATIBILITY_MAX_TOKENS
        or config.seeds[0] != COMPATIBILITY_SEED
    ):
        raise ValueError("hg1_compatibility_contract_mismatch")
    return config


def build_minimal_public_context(
    *,
    model_name: str = COMPATIBILITY_MODEL_FALLBACK,
    temperature: float = COMPATIBILITY_TEMPERATURE,
    max_tokens: int = COMPATIBILITY_MAX_TOKENS,
    seed: int = COMPATIBILITY_SEED,
) -> CascadeAgentPromptContext:
    """Build a synthetic context unrelated to Pilot scenarios or private truth."""
    view = ControllerPublicView(
        scenario_id="hg1-compatibility-synthetic",
        agent_id="hg1-compatibility-agent-01",
        round_id=0,
        claims=(
            ControllerClaimView(
                claim_id="hg1-compatibility-claim-01",
                verification_status=VerificationStatus.UNVERIFIED,
                evidence_card_ids=("hg1-compatibility-evidence-01",),
                public_summary="A synthetic notice proposes a temporary library schedule change.",
            ),
        ),
        evidence_cards=(
            EvidenceCard(
                evidence_id="hg1-compatibility-evidence-01",
                claim="A synthetic public schedule lists the currently announced opening hours.",
                source="synthetic-public-record",
                supports=["hg1-compatibility-claim-01"],
                introduced_round=0,
                visible_to=["hg1-compatibility-agent-01"],
                reliability=0.5,
            ),
        ),
        provenance_nodes=(
            ProvenanceNode(
                node_id="hg1-compatibility-node-01",
                content_id="hg1-compatibility-content-01",
                scenario_id="hg1-compatibility-synthetic",
                claim_id="hg1-compatibility-claim-01",
                source_root_id="hg1-compatibility-root-01",
                round_id=0,
                public_statement="The synthetic notice says the schedule may change next week.",
            ),
        ),
        source_roots=(
            ControllerSourceRootView(
                source_root_id="hg1-compatibility-root-01",
                public_source_category=SourceCategory.PRIMARY_RECORD,
            ),
        ),
        root_relations=(
            VisibleRootRelation(
                provenance_node_id="hg1-compatibility-node-01",
                source_root_ids=("hg1-compatibility-root-01",),
            ),
        ),
    )
    return build_identifiable_prompt_context(
        view,
        claim_id="hg1-compatibility-claim-01",
        runtime_config=CascadeAgentRuntimeConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        ),
    )


def _error_category(error: LLMProviderError) -> str:
    return _ERROR_CATEGORY_MAP.get(error.code.value, "provider_failure")


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


def _status_class_for_category(category: str) -> str | None:
    if category in {"authentication_failed", "http_client_error", "rate_limited", "response_format_unsupported"}:
        return "4xx"
    if category == "http_server_error":
        return "5xx"
    return None


def _attempt_count(provider: object, invoked: bool) -> int:
    transport = getattr(provider, "_transport", None)
    calls = getattr(transport, "calls", None)
    if isinstance(calls, int):
        return calls
    if isinstance(calls, list):
        return len(calls)
    return 1 if invoked else 0


def check_provider(provider: LLMProvider, *, context: CascadeAgentPromptContext | None = None) -> HG1CompatibilityResult:
    """Render one H-G.1 request, call the provider once, and strictly parse it."""
    context = context or build_minimal_public_context()
    request = render_cascade_agent_turn_hg1(context)
    invoked = False
    try:
        invoked = True
        response = provider.complete(request)
    except LLMProviderError as error:
        category = _error_category(error)
        attempts = _attempt_count(provider, invoked)
        return HG1CompatibilityResult(
            status="provider_error",
            provider_error_code=category,
            transport_category=category,
            http_status_class=_safe_http_status_class(getattr(error, "status_code", None)) or _status_class_for_category(category),
            model=request.model_name,
            attempt_count=attempts,
            network="enabled",
        )
    except Exception:
        attempts = _attempt_count(provider, invoked)
        return HG1CompatibilityResult(
            status="provider_error",
            provider_error_code="provider_failure",
            transport_category="provider_failure",
            model=request.model_name,
            attempt_count=attempts,
            network="enabled",
        )
    parsed = parse_cascade_agent_response_v21(response.content, context)
    attempts = _attempt_count(provider, invoked)
    metadata = response.provider_metadata if isinstance(response.provider_metadata, dict) else {}
    return HG1CompatibilityResult(
        status="completed" if parsed.valid else "invalid_response",
        model=response.model_name,
        finish_reason=response.finish_reason,
        parser_valid=parsed.valid,
        http_status_class=_safe_http_status_class(metadata.get("status_code")),
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens,
        latency_ms=response.latency_ms,
        attempt_count=attempts,
        network="enabled",
    )


def run_compatibility_check(
    *,
    allow_network: bool = False,
    environment: Mapping[str, str] | None = None,
    transport: OpenAICompatibleTransport | None = None,
    config_path: str | Path = DEFAULT_CONFIG_RELATIVE,
) -> HG1CompatibilityResult:
    """Network access is impossible unless explicitly opted in."""
    if not allow_network:
        return HG1CompatibilityResult(status="network_disabled")
    _load_generation_contract(config_path)
    raw = ProviderConfig.from_env(allow_network=True, environment=environment)
    provider_config = raw.model_copy(update={
        "allow_network": True,
        "timeout_seconds": COMPATIBILITY_TIMEOUT_SECONDS,
        "max_retries": COMPATIBILITY_MAX_RETRIES,
        "max_tokens": COMPATIBILITY_MAX_TOKENS,
        "temperature": COMPATIBILITY_TEMPERATURE,
        "seed": COMPATIBILITY_SEED,
        "reasoning_effort": None,
        "response_format": ResponseFormatMode.JSON_SCHEMA,
        "response_schema_name": HG1_SCHEMA_NAME,
    })
    wrapped = _CountingTransport(transport or UrllibTransport())
    provider = OpenAICompatibleProvider(provider_config, transport=wrapped, environment=environment)
    context = build_minimal_public_context(model_name=provider_config.model_name or COMPATIBILITY_MODEL_FALLBACK)
    result = check_provider(provider, context=context)
    return result.model_copy(update={
        "attempt_count": wrapped.calls,
        "model": provider_config.model_name,
        "safety": {**result.safety, "one_shot": True},
    })


class _FakeProvider:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.calls = 0

    def complete(self, request) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content=self.content,
            finish_reason=self.finish_reason,
            prompt_tokens=10,
            completion_tokens=12,
            total_tokens=22,
            latency_ms=1.0,
            provider_metadata={"status_code": 200},
        )


def run_fake_compatibility() -> dict[str, object]:
    valid_body = json.dumps({
        "stance": "uncertain",
        "content_ids_used": ["hg1-compatibility-content-01"],
        "evidence_ids_used": ["hg1-compatibility-evidence-01"],
        "share_content_id": "hg1-compatibility-content-01",
    }, sort_keys=True, separators=(",", ":"))
    valid_provider = _FakeProvider(valid_body)
    length_provider = _FakeProvider('{"stance":"uncertain"', finish_reason="length")
    valid = check_provider(valid_provider)
    length = check_provider(length_provider)
    return {
        "status": "passed" if valid.status == "completed" and length.status == "invalid_response" else "failed",
        "valid_parser": valid.parser_valid,
        "length_parser": length.parser_valid,
        "length_finish_reason": length.finish_reason,
        "provider_call_count": valid_provider.calls + length_provider.calls,
        "max_tokens": COMPATIBILITY_MAX_TOKENS,
        "temperature": COMPATIBILITY_TEMPERATURE,
        "seed": COMPATIBILITY_SEED,
        "max_retries": COMPATIBILITY_MAX_RETRIES,
        "timeout_seconds": COMPATIBILITY_TIMEOUT_SECONDS,
        "response_format": "json_schema",
        "schema_name": HG1_SCHEMA_NAME,
        "network": "disabled",
        "results_written": False,
        "private_truth_exposed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-shot offline-by-default H-G.1 Provider compatibility check")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--fake-smoke", action="store_true")
    parser.add_argument("--config", default=DEFAULT_CONFIG_RELATIVE)
    args = parser.parse_args(argv)
    if args.fake_smoke:
        payload: dict[str, object] = run_fake_compatibility()
    else:
        payload = run_compatibility_check(allow_network=args.allow_network, config_path=args.config).model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if payload["status"] in {"completed", "passed", "network_disabled"} else 1


__all__ = [
    "COMPATIBILITY_MAX_RETRIES",
    "COMPATIBILITY_MAX_TOKENS",
    "COMPATIBILITY_MODULE_VERSION",
    "COMPATIBILITY_SEED",
    "COMPATIBILITY_TEMPERATURE",
    "COMPATIBILITY_TIMEOUT_SECONDS",
    "HG1CompatibilityResult",
    "build_minimal_public_context",
    "check_provider",
    "run_compatibility_check",
    "run_fake_compatibility",
]

if __name__ == "__main__":
    raise SystemExit(main())
