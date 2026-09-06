"""Safe registration and validation of the completed H-D.2.1 check.

This module records only facts supplied by the completed one-shot check. It
never performs transport, reads credentials, or stores request/response data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cascade_agent_protocol_v21 import V21_PROTOCOL_VERSION, V21_SCHEMA_NAME, V21_TEMPLATE_VERSION
from .cascade_agent_protocol_v21_compatibility import COMPATIBILITY_VERSION

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT_RELATIVE = "outputs/study-locks/provenance_cascade_agent_v21_compatibility_receipt.json"
DEFAULT_CONFIG_RELATIVE = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd21.v1.toml"
DEFAULT_PROTOCOL_RELATIVE = "src/evicon/cascade_agent_protocol_v21.py"


class HD21RegisteredCompatibilityReceiptError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HD21RegisteredCompatibilityReceipt(BaseModel):
    """Strict, content-free receipt for one previously completed request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["completed"]
    parser_valid: Literal[True]
    finish_reason: Literal["stop"]
    http_status_class: Literal["2xx"]
    model: str | None = None
    response_format: Literal["json_schema"]
    schema_name: Literal["cascade_agent_response_v2_1"]
    attempt_count: Literal[1]
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    config_sha256: str
    protocol_sha256: str
    template_sha256: str
    template_version: Literal["cascade_agent_turn.v2_1.strict_json_512"]
    protocol_version: Literal["provenance_cascade_agent_protocol.v2_1"]
    compatibility_module_version: Literal["cascade_agent_protocol_v21_compatibility.v1"]
    max_tokens: Literal[512]
    temperature: Literal[0.2]
    seed: Literal[20260911]
    max_retries: Literal[0]
    timeout_seconds: Literal[5.0]
    compatibility_only: Literal[True]
    one_shot: Literal[True]
    network_enabled: Literal[True]
    pilot_material_used: Literal[False]
    private_truth_exposed: Literal[False]
    request_ledger_written: Literal[False]
    results_written: Literal[False]
    secrets_logged: Literal[False]
    not_pilot: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]

    @model_validator(mode="after")
    def consistent_usage(self) -> "HD21RegisteredCompatibilityReceipt":
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("token_usage_inconsistent")
        for value in (self.config_sha256, self.protocol_sha256, self.template_sha256):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
                raise ValueError("receipt_hash_invalid")
        return self


def observed_receipt(*, config_path: str | Path, protocol_path: str | Path) -> HD21RegisteredCompatibilityReceipt:
    """Build a receipt from the facts explicitly supplied for H-D.2.1."""
    return HD21RegisteredCompatibilityReceipt(
        status="completed",
        parser_valid=True,
        finish_reason="stop",
        http_status_class="2xx",
        model="gpt-5.6-luna",
        response_format="json_schema",
        schema_name=V21_SCHEMA_NAME,
        attempt_count=1,
        prompt_tokens=401,
        completion_tokens=149,
        total_tokens=550,
        latency_ms=4098.600328026805,
        config_sha256=sha256_file(config_path),
        protocol_sha256=sha256_file(protocol_path),
        template_sha256=sha256_file(protocol_path),
        template_version=V21_TEMPLATE_VERSION,
        protocol_version=V21_PROTOCOL_VERSION,
        compatibility_module_version=COMPATIBILITY_VERSION,
        max_tokens=512,
        temperature=0.2,
        seed=20260911,
        max_retries=0,
        timeout_seconds=5.0,
        compatibility_only=True,
        one_shot=True,
        network_enabled=True,
        pilot_material_used=False,
        private_truth_exposed=False,
        request_ledger_written=False,
        results_written=False,
        secrets_logged=False,
        not_pilot=True,
        not_paper_result=True,
        no_causal_conclusion=True,
    )


def _safe_output_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    lock_root = (_ROOT / "outputs" / "study-locks").resolve()
    try:
        candidate.relative_to(lock_root)
    except ValueError as exc:
        raise HD21RegisteredCompatibilityReceiptError("receipt_path_outside_study_locks") from exc
    return candidate


def register_receipt(
    *,
    output_path: str | Path = DEFAULT_RECEIPT_RELATIVE,
    config_path: str | Path = DEFAULT_CONFIG_RELATIVE,
    protocol_path: str | Path = DEFAULT_PROTOCOL_RELATIVE,
) -> tuple[HD21RegisteredCompatibilityReceipt, str]:
    """Replace only the offline pending placeholder with observed facts."""
    target = _safe_output_path(output_path)
    if target.exists():
        try:
            pending = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HD21RegisteredCompatibilityReceiptError("receipt_exists") from exc
        if not (
            isinstance(pending, dict)
            and pending.get("status") == "pending_network_check"
            and pending.get("attempt_count") == 0
            and pending.get("network_used") is False
        ):
            raise HD21RegisteredCompatibilityReceiptError("receipt_exists")
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = (_ROOT / config_file).resolve()
    protocol_file = Path(protocol_path)
    if not protocol_file.is_absolute():
        protocol_file = (_ROOT / protocol_file).resolve()
    try:
        receipt = observed_receipt(config_path=config_file, protocol_path=protocol_file)
    except OSError as exc:
        raise HD21RegisteredCompatibilityReceiptError("receipt_binding_file_missing") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(target)
    return receipt, sha256_file(target)


def validate_receipt(
    path: str | Path,
    *,
    expected_hash: str | None = None,
    config_path: str | Path = DEFAULT_CONFIG_RELATIVE,
    protocol_path: str | Path = DEFAULT_PROTOCOL_RELATIVE,
) -> HD21RegisteredCompatibilityReceipt:
    """Validate a receipt and its immutable config/protocol bindings."""
    receipt_path = Path(path)
    if not receipt_path.is_absolute():
        receipt_path = (_ROOT / receipt_path).resolve()
    if not receipt_path.exists():
        raise HD21RegisteredCompatibilityReceiptError("compatibility_receipt_missing")
    if expected_hash and sha256_file(receipt_path) != expected_hash.lower():
        raise HD21RegisteredCompatibilityReceiptError("compatibility_receipt_hash_mismatch")
    try:
        receipt = HD21RegisteredCompatibilityReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HD21RegisteredCompatibilityReceiptError("compatibility_receipt_invalid") from exc
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = (_ROOT / config_file).resolve()
    protocol_file = Path(protocol_path)
    if not protocol_file.is_absolute():
        protocol_file = (_ROOT / protocol_file).resolve()
    try:
        if receipt.config_sha256 != sha256_file(config_file):
            raise HD21RegisteredCompatibilityReceiptError("compatibility_config_hash_mismatch")
        if receipt.protocol_sha256 != sha256_file(protocol_file):
            raise HD21RegisteredCompatibilityReceiptError("compatibility_protocol_hash_mismatch")
        if receipt.template_sha256 != sha256_file(protocol_file):
            raise HD21RegisteredCompatibilityReceiptError("compatibility_template_hash_mismatch")
    except OSError as exc:
        raise HD21RegisteredCompatibilityReceiptError("compatibility_binding_file_missing") from exc
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Register or validate the safe H-D.2.1 compatibility receipt")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--validate", action="store_true")
    parser.add_argument("--receipt", default=DEFAULT_RECEIPT_RELATIVE)
    parser.add_argument("--config", default=DEFAULT_CONFIG_RELATIVE)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL_RELATIVE)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    try:
        if args.register:
            receipt, digest = register_receipt(output_path=args.receipt, config_path=args.config, protocol_path=args.protocol)
            print(json.dumps({"status": "registered", "receipt_sha256": digest, "receipt_path": str(args.receipt), "safe": True}, sort_keys=True))
        else:
            receipt = validate_receipt(args.receipt, expected_hash=args.expected_sha256, config_path=args.config, protocol_path=args.protocol)
            print(json.dumps({"status": "valid", "receipt_sha256": sha256_file(args.receipt), "schema_name": receipt.schema_name, "response_format": receipt.response_format, "attempt_count": receipt.attempt_count, "safe": True}, sort_keys=True))
    except HD21RegisteredCompatibilityReceiptError as exc:
        print(json.dumps({"status": "blocked", "error_code": exc.code, "safe": True}, sort_keys=True))
        raise SystemExit(1)


__all__ = [
    "COMPATIBILITY_VERSION", "HD21RegisteredCompatibilityReceipt", "HD21RegisteredCompatibilityReceiptError",
    "DEFAULT_CONFIG_RELATIVE", "DEFAULT_PROTOCOL_RELATIVE", "DEFAULT_RECEIPT_RELATIVE",
    "observed_receipt", "register_receipt", "sha256_file", "validate_receipt",
]

if __name__ == "__main__":
    main()
