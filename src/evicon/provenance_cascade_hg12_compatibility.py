"""One-shot, offline-by-default Provider compatibility check for H-G.1.2."""
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Mapping, Sequence

from .openai_provider import (
    OpenAICompatibleProvider, OpenAICompatibleTransport, ProviderConfig,
    ResponseFormatMode, UrllibTransport,
)
from .provenance_cascade_hg11_compatibility import (
    HG11CompatibilityResult as HG12CompatibilityResult,
    _CountingTransport,
    build_minimal_public_context,
    check_provider,
)
from .provenance_cascade_hg12_design import HG12DesignConfig
from .provenance_cascade_hg12_governance import (
    DEFAULT_COMPATIBILITY_RECEIPT, DEFAULT_CONFIG, HG12GovernanceError,
    register_compatibility_result, sha256_file,
)

_ROOT = Path(__file__).resolve().parents[2]
MAX_TOKENS = 1024
TEMPERATURE = 0.2
SEED = 20261011
MAX_RETRIES = 0
TIMEOUT_SECONDS = 5.0


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def _validate_contract(config_path: str | Path = DEFAULT_CONFIG) -> HG12DesignConfig:
    path = _path(config_path)
    try:
        config = HG12DesignConfig.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise HG12GovernanceError("compatibility_config_invalid") from exc
    if sha256_file(config.protocol_path) != config.protocol_sha256:
        raise HG12GovernanceError("compatibility_protocol_hash_mismatch")
    if (
        config.agent_max_tokens != MAX_TOKENS
        or config.seeds[0] != SEED
        or config.response_format != "json_schema"
        or config.response_schema_name != "cascade_agent_response_v2_1"
    ):
        raise HG12GovernanceError("compatibility_contract_mismatch")
    return config


def run_compatibility_check(
    *, allow_network: bool = False,
    environment: Mapping[str, str] | None = None,
    transport: OpenAICompatibleTransport | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
) -> HG12CompatibilityResult:
    if not allow_network:
        return HG12CompatibilityResult(status="network_disabled")
    _validate_contract(config_path)
    raw = ProviderConfig.from_env(allow_network=True, environment=environment)
    provider_config = raw.model_copy(update={
        "allow_network": True, "timeout_seconds": TIMEOUT_SECONDS,
        "max_retries": MAX_RETRIES, "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE, "seed": SEED, "reasoning_effort": None,
        "response_format": ResponseFormatMode.JSON_SCHEMA,
        "response_schema_name": "cascade_agent_response_v2_1",
    })
    wrapped = _CountingTransport(transport or UrllibTransport())
    provider = OpenAICompatibleProvider(provider_config, transport=wrapped, environment=environment)
    context = build_minimal_public_context(
        model_name=provider_config.model_name or "hg12-compatibility-check-model",
        temperature=TEMPERATURE, max_tokens=MAX_TOKENS, seed=SEED,
    )
    result = check_provider(provider, context=context)
    return result.model_copy(update={"attempt_count": wrapped.calls, "model": provider_config.model_name})


def run_fake_smoke() -> dict[str, object]:
    from .llm_contract import LLMResponse

    class FakeProvider:
        def __init__(self, content: str, finish_reason: str) -> None:
            self.content = content
            self.finish_reason = finish_reason
            self.calls = 0

        def complete(self, request):
            self.calls += 1
            return LLMResponse(
                request_id=request.request_id, model_name=request.model_name,
                content=self.content, finish_reason=self.finish_reason,
                prompt_tokens=10, completion_tokens=12, total_tokens=22,
                latency_ms=1.0, provider_metadata={"status_code": 200},
            )

    valid_provider = FakeProvider(json.dumps({
        "stance": "uncertain",
        "content_ids_used": ["hg11-compatibility-content-01"],
        "evidence_ids_used": ["hg11-compatibility-evidence-01"],
        "share_content_id": "hg11-compatibility-content-01",
    }), "stop")
    length_provider = FakeProvider('{"stance":"uncertain"', "length")
    context = build_minimal_public_context(
        max_tokens=MAX_TOKENS, temperature=TEMPERATURE, seed=SEED,
    )
    valid = check_provider(valid_provider, context=context)
    length = check_provider(length_provider, context=context)
    return {
        "status": "passed" if valid.status == "completed" and length.status == "invalid_response" else "failed",
        "valid_parser": valid.parser_valid, "length_parser": length.parser_valid,
        "length_finish_reason": length.finish_reason,
        "provider_call_count": valid_provider.calls + length_provider.calls,
        "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
        "seed": SEED, "max_retries": MAX_RETRIES,
        "timeout_seconds": TIMEOUT_SECONDS, "response_format": "json_schema",
        "schema_name": "cascade_agent_response_v2_1", "network": "disabled",
        "results_written": False, "private_truth_exposed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-shot H-G.1.2 Provider compatibility and safe receipt registration")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--fake-smoke", action="store_true")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    if args.fake_smoke:
        payload = run_fake_smoke()
    elif args.allow_network and _path(DEFAULT_COMPATIBILITY_RECEIPT).exists():
        payload = {
            "status": "blocked", "provider_error_code": "compatibility_receipt_exists",
            "network": "disabled", "receipt_registered": True,
        }
    else:
        result = run_compatibility_check(allow_network=args.allow_network, config_path=args.config)
        payload = result.model_dump(mode="json")
        if result.status == "completed":
            try:
                receipt_sha, approval_sha = register_compatibility_result(result)
                payload.update({"receipt_registered": True, "receipt_sha256": receipt_sha, "approval_sha256": approval_sha})
            except HG12GovernanceError as exc:
                payload = {"status": "blocked", "provider_error_code": exc.code, "network": "enabled", "receipt_registered": False}
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if payload.get("status") in {"completed", "passed", "network_disabled"} else 1


__all__ = [
    "MAX_RETRIES", "MAX_TOKENS", "SEED", "TEMPERATURE", "TIMEOUT_SECONDS",
    "run_compatibility_check", "run_fake_smoke",
]

if __name__ == "__main__":
    raise SystemExit(main())
