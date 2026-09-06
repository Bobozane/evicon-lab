"""Offline execution-approval contract for source-behavior qualification v2."""
from __future__ import annotations

import json
import tomllib
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_identification import IdentificationError, safe_json, sha256_file
from .conformity_source_behavior_compatibility import (
    DEFAULT_RECEIPT as DEFAULT_COMPATIBILITY_RECEIPT,
    RESPONSE_SCHEMA_SHA256,
    load_receipt as load_compatibility_receipt,
)
from .conformity_source_behavior_qualification import DEFAULT_CONFIG, load_config
from .conformity_source_behavior_qualification_approval import (
    DEFAULT_APPROVAL as DEFAULT_SOURCE_APPROVAL,
    approval_sha256,
    load_approval,
)

_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EXECUTION_APPROVAL = (
    "configs/provenance_cascade/identification/"
    "conformity_source_behavior_execution_approval.v2.toml"
)
DEFAULT_RUNNER = "src/evicon/conformity_source_behavior_runner.py"


class SourceBehaviorExecutionApproval(BaseModel):
    """A sidecar that never itself grants per-command network access."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: Literal["evicon-conformity-source-behavior-execution-v2"]
    approval_version: Literal["conformity_source_behavior_execution_approval.v2"]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    accepted_on: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runner_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_model_name: Literal["gpt-5.6-luna"]
    logical_request_cap: Literal[12]
    completion_reservation_cap: Literal[1536]
    max_tokens: Literal[128]
    temperature: Literal[0.0]
    seed: Literal[20261110]
    max_retries: Literal[0]
    timeout_seconds: Literal[5.0]
    reasoning_effort: Literal["none"]
    output_root: Literal["outputs/conformity-source-behavior-qualification-v2"]
    confirm_compatibility_receipt: bool
    confirm_text_matched_case_set: bool
    confirm_safe_case_audit_only: bool
    confirm_parser_invalid_stops: bool
    confirm_no_overwrite: bool
    confirm_historical_results_excluded: bool
    confirm_not_effect_or_causal_evidence: bool
    network_execution_authorized: Literal[False]

    @model_validator(mode="after")
    def accepted_fields_complete(self) -> "SourceBehaviorExecutionApproval":
        if self.acceptance_status != "accepted":
            return self
        if not self.accepted_by.strip() or not self.accepted_on.strip():
            raise ValueError("accepted execution approval requires reviewer and date")
        try:
            date.fromisoformat(self.accepted_on)
        except ValueError as exc:
            raise ValueError("accepted_on must be an ISO date") from exc
        confirmations = (
            self.confirm_compatibility_receipt,
            self.confirm_text_matched_case_set,
            self.confirm_safe_case_audit_only,
            self.confirm_parser_invalid_stops,
            self.confirm_no_overwrite,
            self.confirm_historical_results_excluded,
            self.confirm_not_effect_or_causal_evidence,
        )
        if not all(confirmations):
            raise ValueError("accepted execution approval is incomplete")
        return self


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def execution_approval_sha256(path: str | Path = DEFAULT_EXECUTION_APPROVAL) -> str:
    return sha256_file(_resolve(path))


def load_execution_approval(
    path: str | Path = DEFAULT_EXECUTION_APPROVAL,
) -> SourceBehaviorExecutionApproval:
    try:
        value = SourceBehaviorExecutionApproval.model_validate(
            tomllib.loads(_resolve(path).read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise IdentificationError("source_behavior_execution_approval_invalid") from exc

    config = load_config(DEFAULT_CONFIG)
    source_approval = load_approval(DEFAULT_SOURCE_APPROVAL)
    compatibility = load_compatibility_receipt(DEFAULT_COMPATIBILITY_RECEIPT)
    expected = {
        "config_sha256": sha256_file(DEFAULT_CONFIG),
        "protocol_sha256": config.protocol_sha256,
        "source_approval_sha256": approval_sha256(DEFAULT_SOURCE_APPROVAL),
        "compatibility_receipt_sha256": sha256_file(DEFAULT_COMPATIBILITY_RECEIPT),
        "runner_sha256": sha256_file(DEFAULT_RUNNER),
        "response_schema_sha256": RESPONSE_SCHEMA_SHA256,
        "required_model_name": compatibility.model,
        "logical_request_cap": config.logical_request_cap,
        "completion_reservation_cap": config.completion_reservation_cap,
        "max_tokens": config.max_tokens,
        "temperature": compatibility.generation_parameters.temperature,
        "seed": compatibility.generation_parameters.seed,
        "max_retries": config.max_retries,
        "timeout_seconds": float(config.timeout_seconds),
        "reasoning_effort": compatibility.generation_parameters.reasoning_effort,
        "output_root": config.output_root,
    }
    observed = {name: getattr(value, name) for name in expected}
    if observed != expected:
        raise IdentificationError("source_behavior_execution_approval_binding_mismatch")
    if source_approval.acceptance_status != "accepted" or compatibility.status != "completed":
        raise IdentificationError("source_behavior_execution_prerequisite_invalid")
    return value


def safe_preflight(
    path: str | Path = DEFAULT_EXECUTION_APPROVAL,
) -> dict[str, object]:
    try:
        approval = load_execution_approval(path)
    except Exception as exc:
        return {
            "status": "blocked",
            "blocking_reasons": [
                getattr(exc, "code", "source_behavior_execution_preflight_failed")
            ],
            "ready_for_network": False,
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "outputs_written": False,
            "behavior_qualification_executed": False,
        }
    if approval.acceptance_status == "accepted":
        reasons = ["behavior_qualification_network_authorization_required"]
        ready = True
    else:
        reasons = ["behavior_qualification_execution_approval_required"]
        ready = False
    return {
        "status": "source_behavior_execution_gate_offline_ready",
        "approval_status": approval.acceptance_status,
        "execution_approval_sha256": execution_approval_sha256(path),
        "compatibility_receipt_sha256": approval.compatibility_receipt_sha256,
        "runner_sha256": approval.runner_sha256,
        "required_model_name": approval.required_model_name,
        "logical_request_cap": approval.logical_request_cap,
        "completion_reservation_cap": approval.completion_reservation_cap,
        "blocking_reasons": reasons,
        "ready_for_network": ready,
        "network": "disabled",
        "provider_constructed": False,
        "api_key_read": False,
        "outputs_written": False,
        "behavior_qualification_executed": False,
        "private_truth_exposed": False,
        "behavior_effect_estimated": False,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def main() -> int:
    print(safe_json(safe_preflight()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_EXECUTION_APPROVAL",
    "DEFAULT_RUNNER",
    "SourceBehaviorExecutionApproval",
    "execution_approval_sha256",
    "load_execution_approval",
    "safe_preflight",
]
