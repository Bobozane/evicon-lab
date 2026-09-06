"""One-shot Provider compatibility gate for identification protocol v1.1."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .conformity_identification import IdentificationStage, safe_json
from .conformity_identification_protocol import (
    IdentificationPromptContext,
    parse_identification_response,
    render_identification_turn,
)
from .conformity_identification_qualification import (
    DEFAULT_QUALIFICATION_APPROVAL,
    approval_sha256,
    load_qualification_approval,
)
from .llm_contract import LLMProvider, LLMProviderError, LLMResponse
from .openai_provider import (
    OpenAICompatibleProvider,
    OpenAICompatibleTransport,
    ProviderConfig,
    ResponseFormatMode,
    UrllibTransport,
)

_ROOT = Path(__file__).resolve().parents[2]
MODULE_VERSION = "conformity_identification_compatibility.v1"
SCHEMA_NAME = "conformity_identification_response_v1"
MAX_TOKENS = 512
TEMPERATURE = 0.2
SEED = 20261101
MAX_RETRIES = 0
TIMEOUT_SECONDS = 5.0
DEFAULT_RECEIPT = "outputs/study-locks/conformity_identification_compatibility.v1.json"

RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "ordinal_judgment": {
            "type": "string",
            "enum": ["strongly_reject", "lean_reject", "uncertain", "lean_adopt", "strongly_adopt"],
        },
        "public_action": {"type": "string", "enum": ["adopt", "reject", "withhold"]},
        "share_decision": {
            "type": "string",
            "enum": ["share", "share_with_caveat", "do_not_share"],
        },
        "confidence": {"type": "integer", "minimum": 1, "maximum": 7},
        "content_ids_used": {"type": "array", "items": {"type": "string"}},
        "evidence_ids_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "ordinal_judgment", "public_action", "share_decision", "confidence",
        "content_ids_used", "evidence_ids_used",
    ],
}


class CompatibilityResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["network_disabled", "blocked", "completed", "invalid_response", "provider_error"]
    provider_error_code: str | None = None
    model: str | None = None
    finish_reason: str | None = None
    parser_valid: bool | None = None
    http_status_class: str | None = None
    transport_category: str | None = None
    attempt_count: int = Field(default=0, ge=0, le=1)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    response_format: Literal["json_schema"] = "json_schema"
    schema_name: Literal["conformity_identification_response_v1"] = SCHEMA_NAME
    network: Literal["disabled", "enabled"] = "disabled"


class CompatibilityReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["completed"]
    parser_valid: Literal[True]
    finish_reason: Literal["stop"]
    http_status_class: Literal["2xx"]
    model: str
    response_format: Literal["json_schema"]
    schema_name: Literal["conformity_identification_response_v1"]
    attempt_count: Literal[1]
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    study_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    branch_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_module_version: Literal["conformity_identification_compatibility.v1"]
    max_tokens: Literal[512]
    temperature: Literal[0.2]
    seed: Literal[20261101]
    max_retries: Literal[0]
    timeout_seconds: Literal[5.0]
    compatibility_only: Literal[True]
    one_shot: Literal[True]
    network_used: Literal[True]
    pilot_material_used: Literal[False]
    private_truth_exposed: Literal[False]
    request_ledger_written: Literal[False]
    results_written: Literal[False]
    secrets_logged: Literal[False]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]


class _CountingTransport:
    def __init__(self, transport: OpenAICompatibleTransport) -> None:
        self.transport = transport
        self.calls = 0

    def post(self, url, headers, payload, timeout_seconds):
        self.calls += 1
        return self.transport.post(url, headers, payload, timeout_seconds)


def minimal_context(model_name: str = "compatibility-model") -> IdentificationPromptContext:
    return IdentificationPromptContext(
        scenario_id="compatibility-synthetic-scenario",
        agent_id="compatibility-agent",
        seed=SEED,
        stage=IdentificationStage.INITIAL_PRIVATE,
        target_claim_id="compatibility-claim",
        decision_task="Assess a fictional scheduling proposal using only the public summary provided.",
        visible_stimulus_ids=("compatibility-content-01",),
        visible_public_summaries=(
            "A fictional committee proposes moving a routine public meeting to an earlier hour.",
        ),
        visible_source_root_ids=(),
        visible_evidence_ids=(),
        reflection_only=False,
        model_name=model_name,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )


def _status_class(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return f"{value // 100}xx" if 200 <= value < 600 else None


def _provider_error(error: LLMProviderError) -> tuple[str, str | None]:
    category = error.code.value
    allowed = {
        "timeout", "connection_failure", "dns_or_url_failure", "authentication_failed",
        "http_client_error", "http_server_error", "rate_limited",
        "malformed_response", "invalid_response", "response_format_unsupported",
        "provider_failure", "missing_api_key", "missing_base_url", "missing_model",
    }
    return (category if category in allowed else "provider_failure", _status_class(error.status_code))


def check_provider(provider: LLMProvider, *, context: IdentificationPromptContext | None = None) -> CompatibilityResult:
    context = context or minimal_context()
    request = render_identification_turn(context)
    try:
        response = provider.complete(request)
    except LLMProviderError as exc:
        category, status_class = _provider_error(exc)
        return CompatibilityResult(
            status="provider_error", provider_error_code=category,
            transport_category=category, http_status_class=status_class,
            model=request.model_name, attempt_count=1, network="enabled",
        )
    except Exception:
        return CompatibilityResult(
            status="provider_error", provider_error_code="provider_failure",
            transport_category="provider_failure", model=request.model_name,
            attempt_count=1, network="enabled",
        )
    try:
        parse_identification_response(response.content, context)
        valid = True
    except ValueError:
        valid = False
    metadata = response.provider_metadata if isinstance(response.provider_metadata, dict) else {}
    return CompatibilityResult(
        status="completed" if valid and response.finish_reason == "stop" else "invalid_response",
        model=response.model_name, finish_reason=response.finish_reason, parser_valid=valid,
        http_status_class=_status_class(metadata.get("status_code")) or "2xx",
        attempt_count=1, prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens, total_tokens=response.total_tokens,
        latency_ms=response.latency_ms, network="enabled",
    )


def run_compatibility_check(
    *,
    allow_network: bool = False,
    environment: Mapping[str, str] | None = None,
    transport: OpenAICompatibleTransport | None = None,
    approval_path: str | Path = DEFAULT_QUALIFICATION_APPROVAL,
) -> CompatibilityResult:
    if not allow_network:
        return CompatibilityResult(status="network_disabled")
    try:
        approval = load_qualification_approval(approval_path)
    except Exception as exc:
        return CompatibilityResult(
            status="blocked", provider_error_code=getattr(exc, "code", "qualification_approval_invalid")
        )
    if approval.acceptance_status != "accepted":
        return CompatibilityResult(status="blocked", provider_error_code="exact_hash_human_approval_required")
    try:
        base = ProviderConfig.from_env(allow_network=True, environment=environment)
        config = base.model_copy(update={
            "allow_network": True, "timeout_seconds": TIMEOUT_SECONDS,
            "max_retries": MAX_RETRIES, "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS, "seed": SEED,
            "response_format": ResponseFormatMode.JSON_SCHEMA,
            "response_schema_name": SCHEMA_NAME, "response_schema": RESPONSE_SCHEMA,
        })
        wrapped = _CountingTransport(transport or UrllibTransport())
        provider = OpenAICompatibleProvider(config, transport=wrapped, environment=environment)
        result = check_provider(provider, context=minimal_context(config.model_name or "compatibility-model"))
        return result.model_copy(update={"attempt_count": wrapped.calls})
    except Exception:
        return CompatibilityResult(
            status="provider_error", provider_error_code="provider_failure",
            transport_category="provider_failure", attempt_count=0, network="enabled",
        )


def register_receipt(
    result: CompatibilityResult,
    *,
    approval_path: str | Path = DEFAULT_QUALIFICATION_APPROVAL,
    receipt_path: str | Path = DEFAULT_RECEIPT,
) -> tuple[str, Path]:
    approval = load_qualification_approval(approval_path)
    if approval.acceptance_status != "accepted":
        raise ValueError("exact_hash_human_approval_required")
    required = (
        result.status == "completed" and result.parser_valid is True
        and result.finish_reason == "stop" and result.http_status_class == "2xx"
        and result.attempt_count == 1
    )
    if not required or any(getattr(result, name) is None for name in (
        "model", "prompt_tokens", "completion_tokens", "total_tokens", "latency_ms"
    )):
        raise ValueError("compatibility_result_not_completed")
    receipt = CompatibilityReceipt(
        status="completed", parser_valid=True, finish_reason="stop", http_status_class="2xx",
        model=result.model, response_format="json_schema", schema_name=SCHEMA_NAME,
        attempt_count=1, prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens, total_tokens=result.total_tokens,
        latency_ms=result.latency_ms, approval_sha256=approval_sha256(approval_path),
        study_config_sha256=approval.study_config_sha256,
        protocol_sha256=approval.protocol_sha256,
        branch_contract_sha256=approval.branch_contract_sha256,
        compatibility_module_version=MODULE_VERSION, max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE, seed=SEED, max_retries=MAX_RETRIES,
        timeout_seconds=TIMEOUT_SECONDS, compatibility_only=True, one_shot=True,
        network_used=True, pilot_material_used=False, private_truth_exposed=False,
        request_ledger_written=False, results_written=False, secrets_logged=False,
        not_paper_result=True, no_causal_conclusion=True,
    )
    target = Path(receipt_path)
    target = target.resolve() if target.is_absolute() else (_ROOT / target).resolve()
    allowed = (_ROOT / "outputs/study-locks").resolve()
    try:
        target.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("compatibility_receipt_path_invalid") from exc
    if target.exists():
        raise ValueError("compatibility_receipt_exists")
    serialized = receipt.model_dump_json(indent=2)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(serialized, encoding="utf-8")
    return hashlib.sha256(serialized.encode()).hexdigest(), target


class _FakeProvider:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0

    def complete(self, request) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            request_id=request.request_id, model_name=request.model_name,
            content=json.dumps(self.payload), finish_reason="stop",
            prompt_tokens=20, completion_tokens=24, total_tokens=44,
            latency_ms=1.0, provider_metadata={"status_code": 200},
        )


def run_fake_smoke() -> dict[str, object]:
    provider = _FakeProvider({
        "ordinal_judgment": "uncertain", "public_action": "withhold",
        "share_decision": "do_not_share", "confidence": 4,
        "content_ids_used": ["compatibility-content-01"], "evidence_ids_used": [],
    })
    result = check_provider(provider)
    return {
        "status": "compatibility_fake_smoke_passed" if result.status == "completed" else "failed",
        "provider_call_count": provider.calls, "parser_valid": result.parser_valid,
        "response_format": "json_schema", "schema_name": SCHEMA_NAME,
        "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "seed": SEED,
        "network": "disabled", "results_written": False,
        "private_truth_exposed": False, "not_paper_result": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-shot identification Provider compatibility check")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--fake-smoke", action="store_true")
    parser.add_argument("--approval", default=DEFAULT_QUALIFICATION_APPROVAL)
    args = parser.parse_args(argv)
    if args.fake_smoke:
        payload = run_fake_smoke()
    else:
        result = run_compatibility_check(allow_network=args.allow_network, approval_path=args.approval)
        payload = result.model_dump(mode="json")
        if result.status == "completed":
            try:
                digest, target = register_receipt(result, approval_path=args.approval)
                payload.update({"receipt_registered": True, "receipt_sha256": digest, "receipt_path": str(target.relative_to(_ROOT))})
            except Exception as exc:
                payload = {
                    "status": "blocked", "provider_error_code": str(exc),
                    "network": "enabled", "receipt_registered": False,
                }
    print(safe_json(payload))
    return 0 if payload.get("status") in {"network_disabled", "completed", "compatibility_fake_smoke_passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CompatibilityReceipt", "CompatibilityResult", "DEFAULT_RECEIPT", "RESPONSE_SCHEMA",
    "check_provider", "minimal_context", "register_receipt", "run_compatibility_check",
    "run_fake_smoke",
]
