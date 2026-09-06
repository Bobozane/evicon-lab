"""One-shot, offline-by-default compatibility check for H-G.2.1."""
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Mapping, Sequence

from .cascade_agent_protocol_hg2 import HG2PromptContext, parse_hg2_response
from .cascade_agent_protocol_hg21 import HG21_RESPONSE_JSON_SCHEMA, HG21_SCHEMA_NAME, render_hg21_turn
from .llm_contract import LLMProvider, LLMProviderError, LLMResponse
from .openai_provider import (
    OpenAICompatibleProvider,
    OpenAICompatibleTransport,
    ProviderConfig,
    ResponseFormatMode,
    UrllibTransport,
)
from .provenance_cascade_hg2_compatibility import (
    HG2CompatibilityResult as HG21CompatibilityResult,
    _CountingTransport,
    _ERROR_CATEGORY_MAP,
    _attempt_count,
    _safe_http_status_class,
    _status_class_for_category,
    build_minimal_hg2_context,
)
from .provenance_cascade_hg21 import DEFAULT_APPROVAL, DEFAULT_COMPATIBILITY_RECEIPT, DEFAULT_CONFIG, HG21Approval, load_config

COMPATIBILITY_MODULE_VERSION = "provenance_cascade_hg21_compatibility.v1"
COMPATIBILITY_TIMEOUT_SECONDS = 5.0
COMPATIBILITY_MAX_RETRIES = 0
COMPATIBILITY_MAX_TOKENS = 1024
COMPATIBILITY_TEMPERATURE = 0.2
COMPATIBILITY_SEED = 20261021


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    root = Path(__file__).resolve().parents[2]
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def check_provider(provider: LLMProvider, *, context: HG2PromptContext | None = None) -> HG21CompatibilityResult:
    context = context or build_minimal_hg2_context()
    request = render_hg21_turn(context)
    invoked = False
    try:
        invoked = True
        response = provider.complete(request)
    except LLMProviderError as error:
        category = _ERROR_CATEGORY_MAP.get(error.code.value, "provider_failure")
        return HG21CompatibilityResult(
            status="provider_error", provider_error_code=category, transport_category=category,
            http_status_class=_safe_http_status_class(getattr(error, "status_code", None)) or _status_class_for_category(category),
            model=request.model_name, schema_name=HG21_SCHEMA_NAME,
            attempt_count=_attempt_count(provider, invoked), network="enabled",
        )
    except Exception:
        return HG21CompatibilityResult(
            status="provider_error", provider_error_code="provider_failure", transport_category="provider_failure",
            model=request.model_name, schema_name=HG21_SCHEMA_NAME,
            attempt_count=_attempt_count(provider, invoked), network="enabled",
        )
    parsed, _ = parse_hg2_response(response.content, context)
    metadata = response.provider_metadata if isinstance(response.provider_metadata, dict) else {}
    return HG21CompatibilityResult(
        status="completed" if parsed is not None else "invalid_response",
        model=response.model_name, finish_reason=response.finish_reason, parser_valid=parsed is not None,
        http_status_class=_safe_http_status_class(metadata.get("status_code")),
        prompt_tokens=response.prompt_tokens, completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens, latency_ms=response.latency_ms,
        schema_name=HG21_SCHEMA_NAME, attempt_count=_attempt_count(provider, invoked), network="enabled",
    )


def run_compatibility_check(
    *, allow_network: bool = False, environment: Mapping[str, str] | None = None,
    transport: OpenAICompatibleTransport | None = None, config_path: str | Path = DEFAULT_CONFIG,
    approval_path: str | Path = DEFAULT_APPROVAL,
) -> HG21CompatibilityResult:
    if not allow_network:
        return HG21CompatibilityResult(status="network_disabled", schema_name=HG21_SCHEMA_NAME)
    config, _ = load_config(config_path)
    try:
        approval = HG21Approval.model_validate(tomllib.loads(_path(approval_path).read_text(encoding="utf-8")))
    except Exception:
        return HG21CompatibilityResult(status="blocked", provider_error_code="exact_hash_approval_invalid", schema_name=HG21_SCHEMA_NAME)
    if approval.acceptance_status != "accepted":
        return HG21CompatibilityResult(status="blocked", provider_error_code="exact_hash_human_approval_required", schema_name=HG21_SCHEMA_NAME)
    raw = ProviderConfig.from_env(allow_network=True, environment=environment)
    provider_config = raw.model_copy(update={
        "allow_network": True, "timeout_seconds": COMPATIBILITY_TIMEOUT_SECONDS,
        "max_retries": COMPATIBILITY_MAX_RETRIES, "max_tokens": config.agent_max_tokens,
        "temperature": config.temperature, "seed": config.seed, "reasoning_effort": None,
        "response_format": ResponseFormatMode.JSON_SCHEMA, "response_schema_name": HG21_SCHEMA_NAME,
        "response_schema": HG21_RESPONSE_JSON_SCHEMA,
    })
    wrapped = _CountingTransport(transport or UrllibTransport())
    provider = OpenAICompatibleProvider(provider_config, transport=wrapped, environment=environment)
    result = check_provider(provider, context=build_minimal_hg2_context(model_name=provider_config.model_name or "hg21-compatibility-model"))
    return result.model_copy(update={"attempt_count": wrapped.calls, "model": provider_config.model_name, "schema_name": HG21_SCHEMA_NAME})


class _FakeProvider:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.content, self.finish_reason, self.calls = content, finish_reason, 0

    def complete(self, request) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            request_id=request.request_id, model_name=request.model_name, content=self.content,
            finish_reason=self.finish_reason, prompt_tokens=12, completion_tokens=16,
            total_tokens=28, latency_ms=1.0, provider_metadata={"status_code": 200},
        )


def run_fake_smoke() -> dict[str, object]:
    valid = _FakeProvider(json.dumps({
        "epistemic_stance": "uncertain", "behavioral_decision": "share_with_caveat",
        "content_ids_used": ["hg11-compatibility-content-01"],
        "evidence_ids_used": ["hg11-compatibility-evidence-01"],
        "share_content_id": "hg11-compatibility-content-01",
    }))
    duplicate = _FakeProvider(json.dumps({
        "epistemic_stance": "uncertain", "behavioral_decision": "share_with_caveat",
        "content_ids_used": ["hg11-compatibility-content-01", "hg11-compatibility-content-01"],
        "evidence_ids_used": [], "share_content_id": "hg11-compatibility-content-01",
    }))
    good, rejected = check_provider(valid), check_provider(duplicate)
    return {
        "status": "passed" if good.status == "completed" and rejected.status == "invalid_response" else "failed",
        "valid_parser": good.parser_valid, "duplicate_parser": rejected.parser_valid,
        "provider_call_count": valid.calls + duplicate.calls, "max_tokens": COMPATIBILITY_MAX_TOKENS,
        "temperature": COMPATIBILITY_TEMPERATURE, "seed": COMPATIBILITY_SEED,
        "max_retries": COMPATIBILITY_MAX_RETRIES, "timeout_seconds": COMPATIBILITY_TIMEOUT_SECONDS,
        "response_format": "json_schema", "schema_name": HG21_SCHEMA_NAME,
        "network": "disabled", "results_written": False, "private_truth_exposed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-shot H-G.2.1 Provider compatibility check")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--fake-smoke", action="store_true")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    if args.fake_smoke:
        payload: dict[str, object] = run_fake_smoke()
    elif args.allow_network and _path(DEFAULT_COMPATIBILITY_RECEIPT).exists():
        payload = {"status": "blocked", "provider_error_code": "compatibility_receipt_exists", "network": "disabled", "receipt_registered": True}
    else:
        result = run_compatibility_check(allow_network=args.allow_network, config_path=args.config)
        payload = result.model_dump(mode="json")
        if result.status == "completed":
            from .provenance_cascade_hg21_compatibility_receipt import register_compatibility_result
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
    "HG21CompatibilityResult", "check_provider", "run_compatibility_check", "run_fake_smoke",
]

if __name__ == "__main__":
    raise SystemExit(main())
