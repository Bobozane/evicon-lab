"""One-shot, offline-by-default Provider compatibility check for H-G.2.3.1."""
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Mapping, Sequence

from .cascade_agent_protocol_hg2 import HG2PromptContext
from .cascade_agent_protocol_hg232 import (
    HG232_RESPONSE_JSON_SCHEMA, HG232_SCHEMA_NAME, parse_hg23_response,
    render_hg232_turn,
)
from .llm_contract import LLMProvider, LLMProviderError, LLMResponse
from .openai_provider import (
    OpenAICompatibleProvider, OpenAICompatibleTransport, ProviderConfig,
    ResponseFormatMode, UrllibTransport,
)
from .provenance_cascade_hg2_compatibility import (
    HG2CompatibilityResult as HG232CompatibilityResult,
    _CountingTransport, _ERROR_CATEGORY_MAP, _attempt_count,
    _safe_http_status_class, _status_class_for_category,
    build_minimal_hg2_context,
)
from .provenance_cascade_hg232 import (
    DEFAULT_APPROVAL, DEFAULT_CONFIG, HG232Approval, load_config,
)

COMPATIBILITY_MODULE_VERSION = "provenance_cascade_hg232_compatibility.v1"
COMPATIBILITY_TIMEOUT_SECONDS = 5.0
COMPATIBILITY_MAX_RETRIES = 0
COMPATIBILITY_MAX_TOKENS = 2048
COMPATIBILITY_TEMPERATURE = 0.2
COMPATIBILITY_SEED = 20261031
REQUIRED_MODEL_NAME = "gpt-5.6-luna"


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    root = Path(__file__).resolve().parents[2]
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def build_minimal_hg232_context(*, model_name: str = "hg232-compatibility-model") -> HG2PromptContext:
    context = build_minimal_hg2_context(model_name=model_name)
    runtime = context.public_context.runtime_config.model_copy(update={
        "model_name": model_name,
        "temperature": COMPATIBILITY_TEMPERATURE,
        "max_tokens": COMPATIBILITY_MAX_TOKENS,
        "seed": COMPATIBILITY_SEED,
    })
    return context.model_copy(update={
        "public_context": context.public_context.model_copy(update={"runtime_config": runtime}),
    })


def check_provider(provider: LLMProvider, *, context: HG2PromptContext | None = None) -> HG232CompatibilityResult:
    context = context or build_minimal_hg232_context()
    request = render_hg232_turn(context)
    invoked = False
    try:
        invoked = True
        response = provider.complete(request)
    except LLMProviderError as error:
        category = _ERROR_CATEGORY_MAP.get(error.code.value, "provider_failure")
        return HG232CompatibilityResult(
            status="provider_error", provider_error_code=category,
            transport_category=category,
            http_status_class=_safe_http_status_class(getattr(error, "status_code", None)) or _status_class_for_category(category),
            model=request.model_name, schema_name=HG232_SCHEMA_NAME,
            attempt_count=_attempt_count(provider, invoked), network="enabled",
        )
    except Exception:
        return HG232CompatibilityResult(
            status="provider_error", provider_error_code="provider_failure",
            transport_category="provider_failure", model=request.model_name,
            schema_name=HG232_SCHEMA_NAME, attempt_count=_attempt_count(provider, invoked),
            network="enabled",
        )
    parsed, _ = parse_hg23_response(response.content, context)
    metadata = response.provider_metadata if isinstance(response.provider_metadata, dict) else {}
    return HG232CompatibilityResult(
        status="completed" if parsed is not None else "invalid_response",
        model=response.model_name, finish_reason=response.finish_reason,
        parser_valid=parsed is not None,
        http_status_class=_safe_http_status_class(metadata.get("status_code")),
        prompt_tokens=response.prompt_tokens, completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens, latency_ms=response.latency_ms,
        schema_name=HG232_SCHEMA_NAME, attempt_count=_attempt_count(provider, invoked),
        network="enabled",
    )


def run_compatibility_check(
    *,
    allow_network: bool = False,
    environment: Mapping[str, str] | None = None,
    transport: OpenAICompatibleTransport | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    approval_path: str | Path = DEFAULT_APPROVAL,
) -> HG232CompatibilityResult:
    if not allow_network:
        return HG232CompatibilityResult(status="network_disabled", schema_name=HG232_SCHEMA_NAME)
    config, _ = load_config(config_path)
    try:
        approval = HG232Approval.model_validate(tomllib.loads(_path(approval_path).read_text(encoding="utf-8")))
    except Exception:
        return HG232CompatibilityResult(status="blocked", provider_error_code="exact_hash_approval_invalid", schema_name=HG232_SCHEMA_NAME)
    if approval.acceptance_status != "accepted":
        return HG232CompatibilityResult(status="blocked", provider_error_code="exact_hash_human_approval_required", schema_name=HG232_SCHEMA_NAME)
    raw = ProviderConfig.from_env(allow_network=True, environment=environment)
    if raw.model_name != REQUIRED_MODEL_NAME:
        return HG232CompatibilityResult(status="blocked", provider_error_code="provider_model_mismatch", schema_name=HG232_SCHEMA_NAME)
    provider_config = raw.model_copy(update={
        "allow_network": True, "timeout_seconds": COMPATIBILITY_TIMEOUT_SECONDS,
        "max_retries": COMPATIBILITY_MAX_RETRIES, "max_tokens": config.agent_max_tokens,
        "temperature": config.temperature, "seed": config.seed, "reasoning_effort": None,
        "response_format": ResponseFormatMode.JSON_SCHEMA,
        "response_schema_name": HG232_SCHEMA_NAME,
        "response_schema": HG232_RESPONSE_JSON_SCHEMA,
    })
    wrapped = _CountingTransport(transport or UrllibTransport())
    provider = OpenAICompatibleProvider(provider_config, transport=wrapped, environment=environment)
    result = check_provider(provider, context=build_minimal_hg232_context(model_name=REQUIRED_MODEL_NAME))
    return result.model_copy(update={
        "attempt_count": wrapped.calls, "model": REQUIRED_MODEL_NAME,
        "schema_name": HG232_SCHEMA_NAME,
    })


class _FakeProvider:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0

    def complete(self, request) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            request_id=request.request_id, model_name=request.model_name,
            content=json.dumps(self.payload), finish_reason="stop",
            prompt_tokens=12, completion_tokens=18, total_tokens=30,
            latency_ms=1.0, provider_metadata={"status_code": 200},
        )


def run_fake_smoke() -> dict[str, object]:
    context = build_minimal_hg232_context()
    content_id = context.public_context.contents[0].content_id
    valid = _FakeProvider({
        "epistemic_stance": "uncertain", "adoption_decision": "adopt_claim",
        "sharing_decision": "share_with_caveat", "content_ids_used": [content_id],
        "evidence_ids_used": [], "share_content_id": content_id,
    })
    conflict = _FakeProvider({
        "epistemic_stance": "uncertain", "adoption_decision": "adopt_claim",
        "sharing_decision": "do_not_share", "content_ids_used": [content_id],
        "evidence_ids_used": [], "share_content_id": content_id,
    })
    good, rejected = check_provider(valid, context=context), check_provider(conflict, context=context)
    return {
        "status": "passed" if good.status == "completed" and rejected.status == "invalid_response" else "failed",
        "valid_parser": good.parser_valid, "conflict_parser": rejected.parser_valid,
        "provider_call_count": valid.calls + conflict.calls,
        "max_tokens": COMPATIBILITY_MAX_TOKENS, "temperature": COMPATIBILITY_TEMPERATURE,
        "seed": COMPATIBILITY_SEED, "max_retries": COMPATIBILITY_MAX_RETRIES,
        "timeout_seconds": COMPATIBILITY_TIMEOUT_SECONDS,
        "response_format": "json_schema", "schema_name": HG232_SCHEMA_NAME,
        "network": "disabled", "results_written": False,
        "private_truth_exposed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-shot H-G.2.3.1 Provider compatibility check")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--fake-smoke", action="store_true")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    receipt_path = _path("outputs/study-locks/provenance_cascade_hg232_compatibility_receipt.json")
    if args.fake_smoke:
        payload: dict[str, object] = run_fake_smoke()
    elif args.allow_network and receipt_path.exists():
        payload = {"status": "blocked", "provider_error_code": "compatibility_receipt_exists", "network": "disabled"}
    else:
        result = run_compatibility_check(allow_network=args.allow_network, config_path=args.config)
        payload = result.model_dump(mode="json")
        if result.status == "completed":
            from .provenance_cascade_hg232_compatibility_receipt import register_compatibility_result
            try:
                receipt_sha, approval_sha = register_compatibility_result(result)
                payload.update({"receipt_registered": True, "receipt_sha256": receipt_sha, "approval_sha256": approval_sha})
            except Exception as exc:
                payload = {
                    "status": "blocked",
                    "provider_error_code": getattr(exc, "code", "compatibility_registration_failed"),
                    "network": "enabled", "receipt_registered": False,
                }
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if payload.get("status") in {"completed", "passed", "network_disabled"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

