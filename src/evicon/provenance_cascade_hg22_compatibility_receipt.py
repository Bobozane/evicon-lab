"""Safe one-shot compatibility receipt for H-G.2.2."""
from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cascade_agent_protocol_hg22 import HG22_PROTOCOL_VERSION, HG22_SCHEMA_NAME, HG22_TEMPLATE_VERSION
from .provenance_cascade_hg2 import sha256_file
from .provenance_cascade_hg22 import (
    DEFAULT_AMENDMENT_RECEIPT,
    DEFAULT_APPROVAL,
    DEFAULT_COMPATIBILITY_RECEIPT,
    DEFAULT_CONFIG,
    HG22Approval,
    load_config,
)
from .provenance_cascade_hg22_compatibility import (
    COMPATIBILITY_MAX_RETRIES,
    COMPATIBILITY_MAX_TOKENS,
    COMPATIBILITY_MODULE_VERSION,
    COMPATIBILITY_SEED,
    COMPATIBILITY_TEMPERATURE,
    COMPATIBILITY_TIMEOUT_SECONDS,
    HG22CompatibilityResult,
)

_ROOT = Path(__file__).resolve().parents[2]


class HG22CompatibilityReceiptError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


class HG22CompatibilityReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["completed"]
    parser_valid: Literal[True]
    finish_reason: Literal["stop"]
    http_status_class: Literal["2xx"]
    model: Literal["gpt-5.6-luna"]
    response_format: Literal["json_schema"]
    schema_name: Literal["cascade_agent_epistemic_behavior_share_contract_v2"]
    attempt_count: Literal[1]
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    config_sha256: str
    protocol_sha256: str
    template_version: Literal["cascade_agent_turn.hg2_epistemic_behavior.share_contract.v2"]
    amendment_receipt_sha256: str
    compatibility_module_version: Literal["provenance_cascade_hg22_compatibility.v1"]
    max_tokens: Literal[1024]
    temperature: Literal[0.2]
    seed: Literal[20261021]
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
    not_pilot: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]

    @model_validator(mode="after")
    def token_sum(self) -> "HG22CompatibilityReceipt":
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("token usage mismatch")
        return self


def receipt_from_result(result: HG22CompatibilityResult) -> HG22CompatibilityReceipt:
    data = result.model_dump(mode="python")
    required = {
        "status": "completed",
        "parser_valid": True,
        "finish_reason": "stop",
        "http_status_class": "2xx",
        "response_format": "json_schema",
        "schema_name": HG22_SCHEMA_NAME,
        "attempt_count": 1,
        "network": "enabled",
        "model": "gpt-5.6-luna",
    }
    if any(data.get(key) != value for key, value in required.items()):
        raise HG22CompatibilityReceiptError("compatibility_result_not_completed")
    if any(data.get(key) is None for key in ("prompt_tokens", "completion_tokens", "total_tokens", "latency_ms")):
        raise HG22CompatibilityReceiptError("compatibility_result_incomplete")
    config, config_path = load_config()
    return HG22CompatibilityReceipt(
        status="completed",
        parser_valid=True,
        finish_reason="stop",
        http_status_class="2xx",
        model=config.required_model_name,
        response_format="json_schema",
        schema_name=HG22_SCHEMA_NAME,
        attempt_count=1,
        prompt_tokens=data["prompt_tokens"],
        completion_tokens=data["completion_tokens"],
        total_tokens=data["total_tokens"],
        latency_ms=data["latency_ms"],
        config_sha256=sha256_file(config_path),
        protocol_sha256=config.protocol_sha256,
        template_version=HG22_TEMPLATE_VERSION,
        amendment_receipt_sha256=sha256_file(DEFAULT_AMENDMENT_RECEIPT),
        compatibility_module_version=COMPATIBILITY_MODULE_VERSION,
        max_tokens=COMPATIBILITY_MAX_TOKENS,
        temperature=COMPATIBILITY_TEMPERATURE,
        seed=COMPATIBILITY_SEED,
        max_retries=COMPATIBILITY_MAX_RETRIES,
        timeout_seconds=COMPATIBILITY_TIMEOUT_SECONDS,
        compatibility_only=True,
        one_shot=True,
        network_used=True,
        pilot_material_used=False,
        private_truth_exposed=False,
        request_ledger_written=False,
        results_written=False,
        secrets_logged=False,
        not_pilot=True,
        not_paper_result=True,
        no_causal_conclusion=True,
    )


def register_compatibility_result(
    result: HG22CompatibilityResult,
    *,
    receipt_path: str | Path = DEFAULT_COMPATIBILITY_RECEIPT,
    approval_path: str | Path = DEFAULT_APPROVAL,
) -> tuple[str, str]:
    target = _path(receipt_path)
    try:
        target.relative_to((_ROOT / "outputs/study-locks").resolve())
    except ValueError as exc:
        raise HG22CompatibilityReceiptError("compatibility_receipt_path_invalid") from exc
    if target.exists():
        raise HG22CompatibilityReceiptError("compatibility_receipt_exists")
    receipt = receipt_from_result(result)
    approval_file = _path(approval_path)
    try:
        raw = approval_file.read_text(encoding="utf-8")
        approval = HG22Approval.model_validate(tomllib.loads(raw))
    except Exception as exc:
        raise HG22CompatibilityReceiptError("compatibility_approval_invalid") from exc
    if approval.acceptance_status != "accepted":
        raise HG22CompatibilityReceiptError("compatibility_approval_not_accepted")
    if (
        approval.config_sha256 != receipt.config_sha256
        or approval.protocol_sha256 != receipt.protocol_sha256
        or approval.amendment_receipt_sha256 != receipt.amendment_receipt_sha256
    ):
        raise HG22CompatibilityReceiptError("compatibility_approval_binding_mismatch")
    serialized = receipt.model_dump_json(indent=2)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if approval.compatibility_receipt_sha256 not in {"", digest}:
        raise HG22CompatibilityReceiptError("compatibility_approval_hash_conflict")
    pattern = r'^compatibility_receipt_sha256 = "[0-9a-f]*"$'
    if len(re.findall(pattern, raw, flags=re.MULTILINE)) != 1:
        raise HG22CompatibilityReceiptError("compatibility_approval_field_invalid")
    updated = re.sub(pattern, f'compatibility_receipt_sha256 = "{digest}"', raw, flags=re.MULTILINE)
    HG22Approval.model_validate(tomllib.loads(updated))
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt_tmp = target.with_suffix(".json.tmp")
    approval_tmp = approval_file.with_suffix(".toml.tmp")
    receipt_tmp.write_text(serialized, encoding="utf-8")
    approval_tmp.write_text(updated, encoding="utf-8")
    receipt_tmp.replace(target)
    approval_tmp.replace(approval_file)
    return digest, sha256_file(approval_file)


def validate_receipt(path: str | Path = DEFAULT_COMPATIBILITY_RECEIPT, *, expected_hash: str | None = None) -> HG22CompatibilityReceipt:
    receipt_path = _path(path)
    if not receipt_path.is_file():
        raise HG22CompatibilityReceiptError("provider_compatibility_check_required")
    if expected_hash and sha256_file(receipt_path) != expected_hash:
        raise HG22CompatibilityReceiptError("provider_compatibility_receipt_hash_mismatch")
    try:
        receipt = HG22CompatibilityReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HG22CompatibilityReceiptError("provider_compatibility_receipt_invalid") from exc
    config, config_path = load_config()
    expected = {
        "config_sha256": sha256_file(config_path),
        "protocol_sha256": config.protocol_sha256,
        "amendment_receipt_sha256": sha256_file(DEFAULT_AMENDMENT_RECEIPT),
    }
    if any(getattr(receipt, key) != value for key, value in expected.items()):
        raise HG22CompatibilityReceiptError("provider_compatibility_receipt_binding_mismatch")
    return receipt


__all__ = [
    "HG22CompatibilityReceipt",
    "HG22CompatibilityReceiptError",
    "receipt_from_result",
    "register_compatibility_result",
    "validate_receipt",
]
