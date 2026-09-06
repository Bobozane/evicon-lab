"""Offline-first compatibility gate for H-D.2.1 strict JSON at 512 tokens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .cascade_agent_protocol_v2_compatibility import build_minimal_public_context
from .cascade_agent_protocol_v21 import (
    V21_PROTOCOL_VERSION,
    V21_SCHEMA_NAME,
    V21_TEMPLATE_VERSION,
    CascadeAgentProtocolV21Runtime,
    load_hd21_config,
)
from .cascade_agent_runtime import CascadeAgentRuntimeStatus
from .cascade_agent_protocol_v21_network_compatibility import (
    CompatibilityResult,
    run_compatibility_check,
)
from .llm_contract import LLMProvider, LLMResponse
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ResponseFormatMode

_ROOT = Path(__file__).resolve().parents[2]
HD21_CONFIG_RELATIVE = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd21.v1.toml"
HD21_PROTOCOL_RELATIVE = "src/evicon/cascade_agent_protocol_v21.py"
HD21_COMPATIBILITY_RECEIPT_RELATIVE = "outputs/study-locks/provenance_cascade_agent_v21_compatibility_receipt.json"
COMPATIBILITY_VERSION = "cascade_agent_protocol_v21_compatibility.v1"


def sha256_file(path: str | Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HD21CompatibilityReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    parser_valid: bool | None = None
    finish_reason: str | None = None
    http_status_class: str | None = None
    model: str | None = None
    response_format: str = "json_schema"
    schema_name: str = V21_SCHEMA_NAME
    attempt_count: int = Field(default=0, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    config_sha256: str
    protocol_sha256: str
    template_sha256: str
    protocol_version: str = V21_PROTOCOL_VERSION
    template_version: str = V21_TEMPLATE_VERSION
    compatibility_module_version: str = COMPATIBILITY_VERSION
    max_tokens: int = 512
    temperature: float = 0.2
    seed: int = 20260911
    max_retries: int = 0
    timeout_seconds: float = 5.0
    compatibility_only: bool = True
    one_shot: bool = True
    network_enabled: bool = False
    pilot_material_used: bool = False
    private_truth_exposed: bool = False
    request_ledger_written: bool = False
    results_written: bool = False
    secrets_logged: bool = False
    not_pilot: bool = True
    not_paper_result: bool = True
    no_causal_conclusion: bool = True

    @property
    def completed(self) -> bool:
        return (
            self.status == "completed"
            and self.parser_valid is True
            and self.finish_reason == "stop"
            and self.http_status_class == "2xx"
            and self.attempt_count == 1
            and self.network_enabled is True
        )


def load_compatibility_receipt(path: str | Path) -> HD21CompatibilityReceipt:
    try:
        return HD21CompatibilityReceipt.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("compatibility_receipt_invalid") from exc


def compatibility_preflight(
    *,
    config_path: str | Path = HD21_CONFIG_RELATIVE,
    protocol_path: str | Path = HD21_PROTOCOL_RELATIVE,
    receipt_path: str | Path = HD21_COMPATIBILITY_RECEIPT_RELATIVE,
) -> dict[str, object]:
    config_file = Path(config_path).resolve()
    protocol_file = Path(protocol_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    config, _ = load_hd21_config(config_file)
    receipt = load_compatibility_receipt(receipt_file)
    reasons: list[str] = []
    if receipt.config_sha256 != sha256_file(config_file): reasons.append("compatibility_config_hash_mismatch")
    if receipt.protocol_sha256 != sha256_file(protocol_file): reasons.append("compatibility_protocol_hash_mismatch")
    if receipt.template_sha256 != sha256_file(protocol_file): reasons.append("compatibility_template_hash_mismatch")
    if receipt.response_format != config.response_format or receipt.schema_name != config.response_schema_name:
        reasons.append("compatibility_response_contract_mismatch")
    if receipt.max_tokens != config.agent_max_tokens or receipt.temperature != 0.2 or receipt.seed != config.seeds[0]:
        reasons.append("compatibility_generation_parameters_mismatch")
    if not receipt.completed:
        reasons.append("compatibility_network_check_pending")
    return {
        "status": "ready" if not reasons else "blocked",
        "blocking_reasons": sorted(set(reasons)),
        "config_sha256": sha256_file(config_file),
        "protocol_sha256": sha256_file(protocol_file),
        "template_sha256": sha256_file(protocol_file),
        "compatibility_receipt_sha256": sha256_file(receipt_file),
        "protocol_version": config.protocol_version,
        "template_version": config.template_version,
        "response_format": config.response_format,
        "schema_name": config.response_schema_name,
        "max_tokens": config.agent_max_tokens,
        "temperature": 0.2,
        "seed": config.seeds[0],
        "max_retries": 0,
        "network": "disabled",
        "provider_constructed": False,
        "transport_called": False,
        "results_written": False,
    }


class HD21FakeProvider:
    def __init__(self, *, content: str | None = None, finish_reason: str = "stop", completion_tokens: int = 16) -> None:
        self.calls = 0
        self.content = content
        self.finish_reason = finish_reason
        self.completion_tokens = completion_tokens

    def complete(self, request) -> LLMResponse:
        self.calls += 1
        payload = json.loads(request.user_prompt)
        contents = [item.get("content_id") for item in payload.get("visible_contents", []) if isinstance(item, dict)]
        evidence = [item.get("evidence_id") for item in payload.get("visible_evidence", []) if isinstance(item, dict)]
        content = self.content or json.dumps({
            "stance": "uncertain",
            "content_ids_used": [value for value in contents[:1] if isinstance(value, str)],
            "evidence_ids_used": [value for value in evidence[:1] if isinstance(value, str)],
            "share_content_id": contents[0] if contents and isinstance(contents[0], str) else None,
        }, sort_keys=True, separators=(",", ":"))
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content=content,
            finish_reason=self.finish_reason,
            prompt_tokens=10,
            completion_tokens=self.completion_tokens,
            total_tokens=10 + self.completion_tokens,
            latency_ms=1.0,
        )


def run_fake_compatibility() -> dict[str, object]:
    context = build_minimal_public_context(
        model_name="hd21-fake-compatibility",
        temperature=0.2,
        max_tokens=512,
        seed=20260911,
    )
    valid_provider = HD21FakeProvider()
    valid = CascadeAgentProtocolV21Runtime().execute(context, valid_provider)
    truncated_provider = HD21FakeProvider(
        content='{"stance":"uncertain"', finish_reason="length", completion_tokens=512,
    )
    truncated = CascadeAgentProtocolV21Runtime().execute(context, truncated_provider)
    return {
        "status": "passed" if valid.status is CascadeAgentRuntimeStatus.COMPLETED and truncated.status is CascadeAgentRuntimeStatus.PARSER_INVALID else "failed",
        "valid_parser": valid.audit.parser_valid,
        "valid_finish_reason": valid.audit.finish_reason,
        "truncation_parser": truncated.audit.parser_valid,
        "truncation_finish_reason": truncated.audit.finish_reason,
        "truncation_completion_tokens": truncated.audit.completion_tokens,
        "max_tokens": 512,
        "response_format": "json_schema",
        "schema_name": V21_SCHEMA_NAME,
        "provider_call_count": valid_provider.calls + truncated_provider.calls,
        "network": "disabled",
        "results_written": False,
        "not_paper_result": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline H-D.2.1 compatibility preflight")
    parser.add_argument("--config", default=HD21_CONFIG_RELATIVE)
    parser.add_argument("--protocol", default=HD21_PROTOCOL_RELATIVE)
    parser.add_argument("--receipt", default=HD21_COMPATIBILITY_RECEIPT_RELATIVE)
    parser.add_argument("--fake-smoke", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    if args.fake_smoke:
        result: dict[str, object] | CompatibilityResult = run_fake_compatibility()
    elif args.allow_network:
        result = run_compatibility_check(allow_network=True)
    else:
        result = compatibility_preflight(
            config_path=args.config, protocol_path=args.protocol, receipt_path=args.receipt,
        )
    payload = result.model_dump(mode="json") if isinstance(result, BaseModel) else result
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if payload["status"] in {"passed", "ready", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
