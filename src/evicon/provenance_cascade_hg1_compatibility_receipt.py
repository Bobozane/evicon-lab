"""Safe registration of a completed H-G.1 one-shot compatibility result."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cascade_agent_protocol_hg1 import HG1_PROTOCOL_VERSION, HG1_SCHEMA_NAME, HG1_TEMPLATE_VERSION
from .provenance_cascade_hg1_compatibility import (
    COMPATIBILITY_MAX_RETRIES,
    COMPATIBILITY_MAX_TOKENS,
    COMPATIBILITY_MODULE_VERSION,
    COMPATIBILITY_SEED,
    COMPATIBILITY_TEMPERATURE,
    COMPATIBILITY_TIMEOUT_SECONDS,
    HG1CompatibilityResult,
)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_RELATIVE = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg1.v1.toml"
DEFAULT_PROTOCOL_RELATIVE = "src/evicon/cascade_agent_protocol_hg1.py"
DEFAULT_APPROVAL_RELATIVE = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg1_approval_template.toml"
DEFAULT_RECEIPT_RELATIVE = "outputs/study-locks/provenance_cascade_hg1_compatibility_receipt.json"


class HG1CompatibilityReceiptError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rooted(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


class HG1RegisteredCompatibilityReceipt(BaseModel):
    """Content-free receipt for one completed H-G.1 compatibility request."""

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
    amendment_sha256: str
    protocol_version: Literal["provenance_cascade_agent_protocol.hg1_identifiable.v1"]
    template_version: Literal["cascade_agent_turn.hg1_public_content.v1"]
    compatibility_module_version: Literal["provenance_cascade_hg1_compatibility.v1"]
    max_tokens: Literal[512]
    temperature: Literal[0.2]
    seed: Literal[20261001]
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
    def consistent(self) -> "HG1RegisteredCompatibilityReceipt":
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("token_usage_inconsistent")
        for value in (self.config_sha256, self.protocol_sha256, self.template_sha256, self.amendment_sha256):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
                raise ValueError("receipt_hash_invalid")
        return self


def receipt_from_result(
    result: HG1CompatibilityResult,
    *,
    config_path: str | Path = DEFAULT_CONFIG_RELATIVE,
    protocol_path: str | Path = DEFAULT_PROTOCOL_RELATIVE,
    amendment_sha256: str,
) -> HG1RegisteredCompatibilityReceipt:
    if not (
        result.status == "completed"
        and result.parser_valid is True
        and result.finish_reason == "stop"
        and result.http_status_class == "2xx"
        and result.attempt_count == 1
        and result.network == "enabled"
        and result.prompt_tokens is not None
        and result.completion_tokens is not None
        and result.total_tokens is not None
    ):
        raise HG1CompatibilityReceiptError("compatibility_result_not_completed")
    config_file = _rooted(config_path)
    protocol_file = _rooted(protocol_path)
    try:
        return HG1RegisteredCompatibilityReceipt(
            status="completed",
            parser_valid=True,
            finish_reason="stop",
            http_status_class="2xx",
            model=result.model,
            response_format="json_schema",
            schema_name=HG1_SCHEMA_NAME,
            attempt_count=1,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            latency_ms=result.latency_ms,
            config_sha256=sha256_file(config_file),
            protocol_sha256=sha256_file(protocol_file),
            template_sha256=sha256_file(protocol_file),
            amendment_sha256=amendment_sha256,
            protocol_version=HG1_PROTOCOL_VERSION,
            template_version=HG1_TEMPLATE_VERSION,
            compatibility_module_version=COMPATIBILITY_MODULE_VERSION,
            max_tokens=COMPATIBILITY_MAX_TOKENS,
            temperature=COMPATIBILITY_TEMPERATURE,
            seed=COMPATIBILITY_SEED,
            max_retries=COMPATIBILITY_MAX_RETRIES,
            timeout_seconds=COMPATIBILITY_TIMEOUT_SECONDS,
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
    except OSError as exc:
        raise HG1CompatibilityReceiptError("receipt_binding_file_missing") from exc


def _safe_receipt_path(path: str | Path) -> Path:
    target = _rooted(path)
    try:
        target.relative_to((_ROOT / "outputs/study-locks").resolve())
    except ValueError as exc:
        raise HG1CompatibilityReceiptError("receipt_path_outside_study_locks") from exc
    return target


def _approval_with_receipt_hash(approval_path: Path, digest: str, receipt: HG1RegisteredCompatibilityReceipt) -> str:
    try:
        raw = approval_path.read_text(encoding="utf-8")
        parsed = tomllib.loads(raw)
    except Exception as exc:
        raise HG1CompatibilityReceiptError("approval_invalid") from exc
    from .provenance_cascade_hg1 import HG1Approval
    try:
        approval = HG1Approval.model_validate(parsed)
    except Exception as exc:
        raise HG1CompatibilityReceiptError("approval_invalid") from exc
    if not approval.ready or approval.acceptance_status != "accepted":
        raise HG1CompatibilityReceiptError("approval_not_accepted")
    if approval.config_sha256 != receipt.config_sha256 or approval.protocol_sha256 != receipt.protocol_sha256:
        raise HG1CompatibilityReceiptError("approval_binding_mismatch")
    if approval.controller_sha256 != sha256_file(_rooted("src/evicon/cascade_controller_hg1.py")):
        raise HG1CompatibilityReceiptError("approval_controller_hash_mismatch")
    if approval.replay_sha256 != sha256_file(_rooted("src/evicon/cascade_hg1_replay.py")):
        raise HG1CompatibilityReceiptError("approval_replay_hash_mismatch")
    if approval.amendment_sha256 != receipt.amendment_sha256:
        raise HG1CompatibilityReceiptError("approval_amendment_hash_mismatch")
    if approval.compatibility_receipt_sha256 not in {"", digest}:
        raise HG1CompatibilityReceiptError("approval_compatibility_hash_conflict")
    pattern = r'^compatibility_receipt_sha256 = "[0-9a-f]*"$'
    if len(re.findall(pattern, raw, flags=re.MULTILINE)) != 1:
        raise HG1CompatibilityReceiptError("approval_compatibility_field_invalid")
    updated = re.sub(pattern, f'compatibility_receipt_sha256 = "{digest}"', raw, flags=re.MULTILINE)
    try:
        HG1Approval.model_validate(tomllib.loads(updated))
    except Exception as exc:
        raise HG1CompatibilityReceiptError("approval_update_invalid") from exc
    return updated


def register_receipt(
    *,
    result_path: str | Path,
    output_path: str | Path = DEFAULT_RECEIPT_RELATIVE,
    approval_path: str | Path = DEFAULT_APPROVAL_RELATIVE,
    config_path: str | Path = DEFAULT_CONFIG_RELATIVE,
    protocol_path: str | Path = DEFAULT_PROTOCOL_RELATIVE,
    amendment_sha256: str,
) -> tuple[HG1RegisteredCompatibilityReceipt, str, str]:
    """Register one safe result and atomically bind its hash into approval."""
    target = _safe_receipt_path(output_path)
    if target.exists():
        raise HG1CompatibilityReceiptError("receipt_exists")
    try:
        result = HG1CompatibilityResult.model_validate_json(_rooted(result_path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise HG1CompatibilityReceiptError("compatibility_result_invalid") from exc
    receipt = receipt_from_result(
        result,
        config_path=config_path,
        protocol_path=protocol_path,
        amendment_sha256=amendment_sha256,
    )
    serialized = receipt.model_dump_json(indent=2)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    approval_file = _rooted(approval_path)
    updated_approval = _approval_with_receipt_hash(approval_file, digest, receipt)
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt_tmp = target.with_suffix(target.suffix + ".tmp")
    approval_tmp = approval_file.with_suffix(approval_file.suffix + ".tmp")
    receipt_tmp.write_text(serialized, encoding="utf-8")
    approval_tmp.write_text(updated_approval, encoding="utf-8")
    receipt_tmp.replace(target)
    approval_tmp.replace(approval_file)
    return receipt, digest, sha256_file(approval_file)


def validate_receipt(
    path: str | Path,
    *,
    expected_hash: str | None = None,
    config_path: str | Path = DEFAULT_CONFIG_RELATIVE,
    protocol_path: str | Path = DEFAULT_PROTOCOL_RELATIVE,
    amendment_sha256: str,
) -> HG1RegisteredCompatibilityReceipt:
    receipt_path = _rooted(path)
    if not receipt_path.is_file():
        raise HG1CompatibilityReceiptError("compatibility_receipt_missing")
    if expected_hash and sha256_file(receipt_path) != expected_hash:
        raise HG1CompatibilityReceiptError("compatibility_receipt_hash_mismatch")
    try:
        receipt = HG1RegisteredCompatibilityReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HG1CompatibilityReceiptError("compatibility_receipt_invalid") from exc
    if receipt.config_sha256 != sha256_file(_rooted(config_path)):
        raise HG1CompatibilityReceiptError("compatibility_config_hash_mismatch")
    if receipt.protocol_sha256 != sha256_file(_rooted(protocol_path)) or receipt.template_sha256 != sha256_file(_rooted(protocol_path)):
        raise HG1CompatibilityReceiptError("compatibility_protocol_hash_mismatch")
    if receipt.amendment_sha256 != amendment_sha256:
        raise HG1CompatibilityReceiptError("compatibility_amendment_hash_mismatch")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register or validate a safe H-G.1 compatibility receipt")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register-result")
    mode.add_argument("--validate", action="store_true")
    parser.add_argument("--receipt", default=DEFAULT_RECEIPT_RELATIVE)
    parser.add_argument("--approval", default=DEFAULT_APPROVAL_RELATIVE)
    parser.add_argument("--config", default=DEFAULT_CONFIG_RELATIVE)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL_RELATIVE)
    parser.add_argument("--amendment-sha256", required=True)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args(argv)
    try:
        if args.register_result:
            _, digest, approval_digest = register_receipt(
                result_path=args.register_result,
                output_path=args.receipt,
                approval_path=args.approval,
                config_path=args.config,
                protocol_path=args.protocol,
                amendment_sha256=args.amendment_sha256,
            )
            payload = {
                "status": "registered",
                "receipt_path": args.receipt,
                "receipt_sha256": digest,
                "approval_sha256": approval_digest,
                "safe": True,
            }
        else:
            receipt = validate_receipt(
                args.receipt,
                expected_hash=args.expected_sha256,
                config_path=args.config,
                protocol_path=args.protocol,
                amendment_sha256=args.amendment_sha256,
            )
            payload = {
                "status": "valid",
                "receipt_sha256": sha256_file(_rooted(args.receipt)),
                "response_format": receipt.response_format,
                "schema_name": receipt.schema_name,
                "attempt_count": receipt.attempt_count,
                "safe": True,
            }
    except HG1CompatibilityReceiptError as exc:
        payload = {"status": "blocked", "error_code": exc.code, "safe": True}
        print(json.dumps(payload, sort_keys=True))
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_APPROVAL_RELATIVE",
    "DEFAULT_CONFIG_RELATIVE",
    "DEFAULT_PROTOCOL_RELATIVE",
    "DEFAULT_RECEIPT_RELATIVE",
    "HG1CompatibilityReceiptError",
    "HG1RegisteredCompatibilityReceipt",
    "receipt_from_result",
    "register_receipt",
    "sha256_file",
    "validate_receipt",
]

if __name__ == "__main__":
    raise SystemExit(main())
