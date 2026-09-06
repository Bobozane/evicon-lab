"""Safe registration and validation for the H-G.2 compatibility receipt."""
from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cascade_agent_protocol_hg2 import HG2_PROTOCOL_VERSION, HG2_SCHEMA_NAME, HG2_TEMPLATE_VERSION
from .provenance_cascade_hg2 import sha256_file
from .provenance_cascade_hg2_compatibility import (
    COMPATIBILITY_MAX_RETRIES,
    COMPATIBILITY_MAX_TOKENS,
    COMPATIBILITY_MODULE_VERSION,
    COMPATIBILITY_SEED,
    COMPATIBILITY_TEMPERATURE,
    COMPATIBILITY_TIMEOUT_SECONDS,
    HG2CompatibilityResult,
)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg2_calibration.v1.toml"
DEFAULT_PROTOCOL = "src/evicon/cascade_agent_protocol_hg2.py"
DEFAULT_APPROVAL = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg2_approval_template.toml"
DEFAULT_AMENDMENT_RECEIPT = "outputs/study-locks/provenance_cascade_hg2_epistemic_behavior_amendment_receipt.json"
DEFAULT_RECEIPT = "outputs/study-locks/provenance_cascade_hg2_compatibility_receipt.json"


class HG2CompatibilityReceiptError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


class HG2CompatibilityReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["completed"]
    parser_valid: Literal[True]
    finish_reason: Literal["stop"]
    http_status_class: Literal["2xx"]
    model: str
    response_format: Literal["json_schema"]
    schema_name: Literal["cascade_agent_epistemic_behavior_response_v1"]
    attempt_count: Literal[1]
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    config_sha256: str
    protocol_sha256: str
    template_sha256: str
    amendment_receipt_sha256: str
    protocol_version: Literal["provenance_cascade_agent_protocol.hg2_epistemic_behavior.v1"]
    template_version: Literal["cascade_agent_turn.hg2_epistemic_behavior.v1"]
    compatibility_module_version: Literal["provenance_cascade_hg2_compatibility.v1"]
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
    def consistent(self) -> "HG2CompatibilityReceipt":
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("token_usage_inconsistent")
        for value in (self.config_sha256, self.protocol_sha256, self.template_sha256, self.amendment_receipt_sha256):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("receipt_hash_invalid")
        return self


def receipt_from_result(result: HG2CompatibilityResult) -> HG2CompatibilityReceipt:
    data = result.model_dump(mode="python")
    expected = {
        "status": "completed",
        "parser_valid": True,
        "finish_reason": "stop",
        "http_status_class": "2xx",
        "response_format": "json_schema",
        "schema_name": HG2_SCHEMA_NAME,
        "attempt_count": 1,
        "network": "enabled",
    }
    if any(data.get(key) != value for key, value in expected.items()):
        raise HG2CompatibilityReceiptError("compatibility_result_not_completed")
    for key in ("model", "prompt_tokens", "completion_tokens", "total_tokens", "latency_ms"):
        if data.get(key) is None:
            raise HG2CompatibilityReceiptError("compatibility_result_incomplete")
    return HG2CompatibilityReceipt(
        status="completed",
        parser_valid=True,
        finish_reason="stop",
        http_status_class="2xx",
        model=data["model"],
        response_format="json_schema",
        schema_name=HG2_SCHEMA_NAME,
        attempt_count=1,
        prompt_tokens=data["prompt_tokens"],
        completion_tokens=data["completion_tokens"],
        total_tokens=data["total_tokens"],
        latency_ms=data["latency_ms"],
        config_sha256=sha256_file(_path(DEFAULT_CONFIG)),
        protocol_sha256=sha256_file(_path(DEFAULT_PROTOCOL)),
        template_sha256=sha256_file(_path(DEFAULT_PROTOCOL)),
        amendment_receipt_sha256=sha256_file(_path(DEFAULT_AMENDMENT_RECEIPT)),
        protocol_version=HG2_PROTOCOL_VERSION,
        template_version=HG2_TEMPLATE_VERSION,
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


def _safe_receipt_path(value: str | Path) -> Path:
    target = _path(value)
    try:
        target.relative_to((_ROOT / "outputs/study-locks").resolve())
    except ValueError as exc:
        raise HG2CompatibilityReceiptError("compatibility_receipt_path_invalid") from exc
    return target


def register_compatibility_result(
    result: HG2CompatibilityResult,
    *,
    receipt_path: str | Path = DEFAULT_RECEIPT,
    approval_path: str | Path = DEFAULT_APPROVAL,
) -> tuple[str, str]:
    target = _safe_receipt_path(receipt_path)
    if target.exists():
        raise HG2CompatibilityReceiptError("compatibility_receipt_exists")
    receipt = receipt_from_result(result)
    approval_file = _path(approval_path)
    try:
        raw = approval_file.read_text(encoding="utf-8")
        from .provenance_cascade_hg2_governance import HG2Approval
        approval = HG2Approval.model_validate(tomllib.loads(raw))
    except Exception as exc:
        raise HG2CompatibilityReceiptError("compatibility_approval_invalid") from exc
    if approval.acceptance_status != "accepted":
        raise HG2CompatibilityReceiptError("compatibility_approval_not_accepted")
    if (
        approval.config_sha256 != receipt.config_sha256
        or approval.protocol_sha256 != receipt.protocol_sha256
        or approval.amendment_receipt_sha256 != receipt.amendment_receipt_sha256
    ):
        raise HG2CompatibilityReceiptError("compatibility_approval_binding_mismatch")
    serialized = receipt.model_dump_json(indent=2)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if approval.compatibility_receipt_sha256 not in {"", digest}:
        raise HG2CompatibilityReceiptError("compatibility_approval_hash_conflict")
    pattern = r'^compatibility_receipt_sha256 = "[0-9a-f]*"$'
    if len(re.findall(pattern, raw, flags=re.MULTILINE)) != 1:
        raise HG2CompatibilityReceiptError("compatibility_approval_field_invalid")
    updated = re.sub(pattern, f'compatibility_receipt_sha256 = "{digest}"', raw, flags=re.MULTILINE)
    try:
        HG2Approval.model_validate(tomllib.loads(updated))
    except Exception as exc:
        raise HG2CompatibilityReceiptError("compatibility_approval_update_invalid") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt_tmp = target.with_suffix(".json.tmp")
    approval_tmp = approval_file.with_suffix(".toml.tmp")
    receipt_tmp.write_text(serialized, encoding="utf-8")
    approval_tmp.write_text(updated, encoding="utf-8")
    receipt_tmp.replace(target)
    approval_tmp.replace(approval_file)
    return digest, sha256_file(approval_file)


def validate_receipt(path: str | Path = DEFAULT_RECEIPT, *, expected_hash: str | None = None) -> HG2CompatibilityReceipt:
    receipt_path = _path(path)
    if not receipt_path.is_file():
        raise HG2CompatibilityReceiptError("provider_compatibility_check_required")
    if expected_hash and sha256_file(receipt_path) != expected_hash:
        raise HG2CompatibilityReceiptError("provider_compatibility_receipt_hash_mismatch")
    try:
        receipt = HG2CompatibilityReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HG2CompatibilityReceiptError("provider_compatibility_receipt_invalid") from exc
    expected = {
        "config_sha256": sha256_file(_path(DEFAULT_CONFIG)),
        "protocol_sha256": sha256_file(_path(DEFAULT_PROTOCOL)),
        "template_sha256": sha256_file(_path(DEFAULT_PROTOCOL)),
        "amendment_receipt_sha256": sha256_file(_path(DEFAULT_AMENDMENT_RECEIPT)),
    }
    if any(getattr(receipt, key) != value for key, value in expected.items()):
        raise HG2CompatibilityReceiptError("provider_compatibility_receipt_binding_mismatch")
    return receipt


__all__ = [
    "DEFAULT_RECEIPT", "HG2CompatibilityReceipt", "HG2CompatibilityReceiptError",
    "receipt_from_result", "register_compatibility_result", "validate_receipt",
]
