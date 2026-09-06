"""One-shot, offline-by-default Provider compatibility check for H-G.2."""
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .cascade_agent_protocol_hg2 import (
    HG2DecisionTask,
    HG2PromptContext,
    HG2_RESPONSE_JSON_SCHEMA,
    HG2_SCHEMA_NAME,
    PublicDecisionRole,
    parse_hg2_response,
    render_hg2_turn,
)
from .llm_contract import LLMProvider, LLMProviderError, LLMResponse
from .openai_provider import (
    OpenAICompatibleProvider,
    OpenAICompatibleTransport,
    ProviderConfig,
    ResponseFormatMode,
    TransportResponse,
    UrllibTransport,
)
from .provenance_cascade_hg11_compatibility import build_minimal_public_context
from .provenance_cascade_hg2 import HG2CalibrationConfig, sha256_file

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg2_calibration.v1.toml"
COMPATIBILITY_MODULE_VERSION = "provenance_cascade_hg2_compatibility.v1"
COMPATIBILITY_TIMEOUT_SECONDS = 5.0
COMPATIBILITY_MAX_RETRIES = 0
COMPATIBILITY_MAX_TOKENS = 1024
COMPATIBILITY_TEMPERATURE = 0.2
COMPATIBILITY_SEED = 20261021

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


class HG2CompatibilityResult(BaseModel):
    """Content-free compatibility result suitable for stdout or a safe receipt."""

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
    schema_name: str = HG2_SCHEMA_NAME
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
    def __init__(self, transport: OpenAICompatibleTransport) -> None:
        self.transport = transport
        self.calls = 0

    def post(self, url: str, headers: Mapping[str, str], payload: dict[str, object], timeout_seconds: float) -> TransportResponse:
        self.calls += 1
        return self.transport.post(url, headers, payload, timeout_seconds)


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def _load_contract(config_path: str | Path = DEFAULT_CONFIG) -> HG2CalibrationConfig:
    try:
        config = HG2CalibrationConfig.model_validate(tomllib.loads(_path(config_path).read_text(encoding="utf-8")))
    except Exception as exc:
        raise ValueError("hg2_compatibility_config_invalid") from exc
    if sha256_file(config.protocol_path) != config.protocol_sha256:
        raise ValueError("hg2_compatibility_protocol_hash_mismatch")
    if (
        config.agent_max_tokens != COMPATIBILITY_MAX_TOKENS
        or config.temperature != COMPATIBILITY_TEMPERATURE
        or config.seed != COMPATIBILITY_SEED
        or config.response_format != "json_schema"
        or config.response_schema_name != HG2_SCHEMA_NAME
    ):
        raise ValueError("hg2_compatibility_contract_mismatch")
    return config


def build_minimal_hg2_context(*, model_name: str = "hg2-compatibility-model") -> HG2PromptContext:
    public = build_minimal_public_context(
        model_name=model_name,
        temperature=COMPATIBILITY_TEMPERATURE,
        max_tokens=COMPATIBILITY_MAX_TOKENS,
        seed=COMPATIBILITY_SEED,
    )
    return HG2PromptContext(
        public_context=public,
        role=PublicDecisionRole.VERIFICATION_FIRST,
        decision_task=HG2DecisionTask(
            task_id="synthetic-library-notice-compatibility-v1",
            public_decision_summary="Choose whether to share a synthetic library notice, support a reversible notice, or defer.",
            reversible_action="Support a provisional notice that can be updated after the public schedule is checked.",
            defer_action="Defer until the visible public schedule is checked.",
        ),
    )


def _safe_http_status_class(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    for lower, upper, label in ((200, 300, "2xx"), (300, 400, "3xx"), (400, 500, "4xx"), (500, 600, "5xx")):
        if lower <= value < upper:
            return label
    return None


def _status_class_for_category(category: str) -> str | None:
    if category in {"authentication_failed", "http_client_error", "rate_limited", "response_format_unsupported"}:
        return "4xx"
    return "5xx" if category == "http_server_error" else None


def _attempt_count(provider: object, invoked: bool) -> int:
    transport = getattr(provider, "_transport", None)
    calls = getattr(transport, "calls", None)
    return calls if isinstance(calls, int) else (1 if invoked else 0)


def check_provider(provider: LLMProvider, *, context: HG2PromptContext | None = None) -> HG2CompatibilityResult:
    context = context or build_minimal_hg2_context()
    request = render_hg2_turn(context)
    invoked = False
    try:
        invoked = True
        response = provider.complete(request)
    except LLMProviderError as error:
        category = _ERROR_CATEGORY_MAP.get(error.code.value, "provider_failure")
        return HG2CompatibilityResult(
            status="provider_error",
            provider_error_code=category,
            transport_category=category,
            http_status_class=_safe_http_status_class(getattr(error, "status_code", None)) or _status_class_for_category(category),
            model=request.model_name,
            attempt_count=_attempt_count(provider, invoked),
            network="enabled",
        )
    except Exception:
        return HG2CompatibilityResult(
            status="provider_error",
            provider_error_code="provider_failure",
            transport_category="provider_failure",
            model=request.model_name,
            attempt_count=_attempt_count(provider, invoked),
            network="enabled",
        )
    parsed, _ = parse_hg2_response(response.content, context)
    metadata = response.provider_metadata if isinstance(response.provider_metadata, dict) else {}
    return HG2CompatibilityResult(
        status="completed" if parsed is not None else "invalid_response",
        model=response.model_name,
        finish_reason=response.finish_reason,
        parser_valid=parsed is not None,
        http_status_class=_safe_http_status_class(metadata.get("status_code")),
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens,
        latency_ms=response.latency_ms,
        attempt_count=_attempt_count(provider, invoked),
        network="enabled",
    )


def run_compatibility_check(
    *,
    allow_network: bool = False,
    environment: Mapping[str, str] | None = None,
    transport: OpenAICompatibleTransport | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
) -> HG2CompatibilityResult:
    if not allow_network:
        return HG2CompatibilityResult(status="network_disabled")
    _load_contract(config_path)
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
        "response_schema_name": HG2_SCHEMA_NAME,
        "response_schema": HG2_RESPONSE_JSON_SCHEMA,
    })
    wrapped = _CountingTransport(transport or UrllibTransport())
    provider = OpenAICompatibleProvider(provider_config, transport=wrapped, environment=environment)
    result = check_provider(provider, context=build_minimal_hg2_context(model_name=provider_config.model_name or "hg2-compatibility-model"))
    return result.model_copy(update={"attempt_count": wrapped.calls, "model": provider_config.model_name})


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
            completion_tokens=15,
            total_tokens=25,
            latency_ms=1.0,
            provider_metadata={"status_code": 200},
        )


def run_fake_compatibility() -> dict[str, object]:
    valid = _FakeProvider(json.dumps({
        "epistemic_stance": "uncertain",
        "behavioral_decision": "share_with_caveat",
        "content_ids_used": ["hg11-compatibility-content-01"],
        "evidence_ids_used": ["hg11-compatibility-evidence-01"],
        "share_content_id": "hg11-compatibility-content-01",
    }, sort_keys=True, separators=(",", ":")))
    truncated = _FakeProvider('{"epistemic_stance":"uncertain"', finish_reason="length")
    valid_result = check_provider(valid)
    truncated_result = check_provider(truncated)
    return {
        "status": "passed" if valid_result.status == "completed" and truncated_result.status == "invalid_response" else "failed",
        "valid_parser": valid_result.parser_valid,
        "length_parser": truncated_result.parser_valid,
        "length_finish_reason": truncated_result.finish_reason,
        "provider_call_count": valid.calls + truncated.calls,
        "max_tokens": COMPATIBILITY_MAX_TOKENS,
        "temperature": COMPATIBILITY_TEMPERATURE,
        "seed": COMPATIBILITY_SEED,
        "max_retries": COMPATIBILITY_MAX_RETRIES,
        "timeout_seconds": COMPATIBILITY_TIMEOUT_SECONDS,
        "response_format": "json_schema",
        "schema_name": HG2_SCHEMA_NAME,
        "network": "disabled",
        "results_written": False,
        "private_truth_exposed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-shot H-G.2 Provider compatibility check")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--fake-smoke", action="store_true")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    if args.fake_smoke:
        payload: dict[str, object] = run_fake_compatibility()
    else:
        from .provenance_cascade_hg2_governance import DEFAULT_COMPATIBILITY_RECEIPT
        if args.allow_network and _path(DEFAULT_COMPATIBILITY_RECEIPT).exists():
            payload = {"status": "blocked", "provider_error_code": "compatibility_receipt_exists", "network": "disabled", "receipt_registered": True}
        else:
            result = run_compatibility_check(allow_network=args.allow_network, config_path=args.config)
            payload = result.model_dump(mode="json")
            if result.status == "completed":
                from .provenance_cascade_hg2_compatibility_receipt import register_compatibility_result
                try:
                    receipt_sha, approval_sha = register_compatibility_result(result)
                    payload.update({"receipt_registered": True, "receipt_sha256": receipt_sha, "approval_sha256": approval_sha})
                except Exception as exc:
                    payload = {"status": "blocked", "provider_error_code": getattr(exc, "code", "compatibility_registration_failed"), "network": "enabled", "receipt_registered": False}
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if payload.get("status") in {"completed", "passed", "network_disabled"} else 1


__all__ = [
    "COMPATIBILITY_MAX_RETRIES", "COMPATIBILITY_MAX_TOKENS", "COMPATIBILITY_MODULE_VERSION",
    "COMPATIBILITY_SEED", "COMPATIBILITY_TEMPERATURE", "COMPATIBILITY_TIMEOUT_SECONDS",
    "HG2CompatibilityResult", "build_minimal_hg2_context", "check_provider",
    "run_compatibility_check", "run_fake_compatibility",
]

if __name__ == "__main__":
    raise SystemExit(main())
