"""Strict contracts for the isolated H-G.2.3.2 protocol-stability probe."""
from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .provenance_cascade_hg2 import sha256_file

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROBE_CONFIG = "configs/provenance_cascade/pilot/provenance_cascade_hg232_protocol_stability_probe.v1.toml"
DEFAULT_PROBE_APPROVAL = "configs/provenance_cascade/pilot/provenance_cascade_hg232_protocol_stability_probe_approval.toml"
DEFAULT_PROBE_RECEIPT = "results/provenance-cascade-hg232-protocol-stability-probe-v1/protocol_stability_probe_receipt.json"


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


class HG232ProbeError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HG232ProbeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    probe_id: Literal["provenance-cascade-hg232-protocol-stability-probe-v1"]
    status: Literal["pending_execution_approval"]
    development_only: Literal[True]
    probe_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    parent_config_path: str
    parent_config_sha256: str
    protocol_path: str
    protocol_sha256: str
    compatibility_receipt_path: str
    compatibility_receipt_sha256: str
    calibration_runner_path: str
    calibration_runner_sha256: str
    response_audit_path: str
    response_audit_sha256: str
    scenario_id: Literal["cascade-hg1-true-minority-correction"]
    condition: Literal["generic_dissent"]
    seed: Literal[20261031]
    run_id: Literal["hg232-stability-true-minority-correction-20261031-generic-dissent"]
    run_namespace: Literal["hg232-stability"]
    output_root: Literal["results/provenance-cascade-hg232-protocol-stability-probe-v1"]
    agent_count: Literal[6]
    round_count: Literal[3]
    logical_request_cap: Literal[18]
    max_tokens: Literal[2048]
    completion_reservation_cap: Literal[36864]
    temperature: Literal[0.2]
    max_retries: Literal[0]
    timeout_seconds: Literal[15.0]
    response_format: Literal["json_schema"]
    schema_name: Literal["cascade_agent_epistemic_adoption_sharing_v1_provider_subset_2048"]
    parser_invalid_stops: Literal[True]
    finish_reason_length_stops: Literal[True]
    token_limit_reached_stops: Literal[True]
    semantic_recovery_enabled: Literal[False]

    @model_validator(mode="after")
    def caps(self) -> "HG232ProbeConfig":
        if self.completion_reservation_cap != self.logical_request_cap * self.max_tokens:
            raise ValueError("probe completion reservation mismatch")
        return self


class HG232ProbeApproval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    approval_id: Literal["provenance-cascade-hg232-protocol-stability-probe-execution-v1"]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    config_sha256: str
    protocol_sha256: str
    compatibility_receipt_sha256: str
    calibration_runner_sha256: str
    probe_runner_sha256: str
    run_count: Literal[1]
    logical_request_cap: Literal[18]
    completion_reservation_cap: Literal[36864]
    confirm_isolated_probe_only: bool
    confirm_no_merge_with_calibration: bool
    confirm_append_only_ledger: bool
    confirm_no_overwrite: bool
    confirm_parser_invalid_stops: bool
    network_execution_authorized: bool

    @model_validator(mode="after")
    def accepted(self) -> "HG232ProbeApproval":
        confirmations = (
            self.confirm_isolated_probe_only,
            self.confirm_no_merge_with_calibration,
            self.confirm_append_only_ledger,
            self.confirm_no_overwrite,
            self.confirm_parser_invalid_stops,
            self.network_execution_authorized,
        )
        if self.acceptance_status == "accepted" and (not self.accepted_by.strip() or not all(confirmations)):
            raise ValueError("probe execution approval incomplete")
        return self


class HG232ProbeReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    receipt_id: Literal["provenance-cascade-hg232-protocol-stability-probe-receipt-v1"]
    status: Literal["passed"]
    probe_config_sha256: str = Field(min_length=64, max_length=64)
    parent_config_sha256: str = Field(min_length=64, max_length=64)
    protocol_sha256: str = Field(min_length=64, max_length=64)
    compatibility_receipt_sha256: str = Field(min_length=64, max_length=64)
    calibration_runner_sha256: str = Field(min_length=64, max_length=64)
    probe_runner_sha256: str = Field(min_length=64, max_length=64)
    approval_sha256: str = Field(min_length=64, max_length=64)
    run_id: str
    scenario_id: str
    condition: str
    seed: int
    logical_request_count: Literal[18]
    provider_call_count: Literal[18]
    transport_attempt_count: Literal[18]
    completion_reservation_cap: Literal[36864]
    max_tokens: Literal[2048]
    response_audit_count: Literal[18]
    finish_reason_stop_count: Literal[18]
    completion_limit_reached_count: Literal[0]
    cascade_replay_status: Literal["passed"]
    application_replay_status: Literal["passed"]
    outcome_replay_status: Literal["passed"]
    sensitive_scan_status: Literal["passed"]
    request_ledger_sha256: str = Field(min_length=64, max_length=64)
    run_record_sha256: str = Field(min_length=64, max_length=64)
    response_audit_sha256: str = Field(min_length=64, max_length=64)
    network: Literal["enabled", "disabled"]
    development_only: Literal[True]
    probe_only: Literal[True]
    merged_with_calibration: Literal[False]
    private_truth_exposed: Literal[False]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]


def load_probe_config(path: str | Path = DEFAULT_PROBE_CONFIG, *, allow_calibration_runner_amendment: bool = False) -> tuple[HG232ProbeConfig, Path]:
    target = _path(path)
    try:
        config = HG232ProbeConfig.model_validate(tomllib.loads(target.read_text(encoding="utf-8")))
    except Exception as exc:
        raise HG232ProbeError("protocol_stability_probe_config_invalid") from exc
    bindings = (
        (config.parent_config_path, config.parent_config_sha256),
        (config.protocol_path, config.protocol_sha256),
        (config.compatibility_receipt_path, config.compatibility_receipt_sha256),
        (config.calibration_runner_path, config.calibration_runner_sha256),
        (config.response_audit_path, config.response_audit_sha256),
    )
    if any(
        (not _path(path_value).is_file() or sha256_file(_path(path_value)) != digest)
        for path_value, digest in bindings
        if not (allow_calibration_runner_amendment and path_value == config.calibration_runner_path)
    ):
        raise HG232ProbeError("protocol_stability_probe_binding_mismatch")
    return config, target


def load_probe_approval(path: str | Path = DEFAULT_PROBE_APPROVAL) -> tuple[HG232ProbeApproval, Path]:
    target = _path(path)
    try:
        return HG232ProbeApproval.model_validate(tomllib.loads(target.read_text(encoding="utf-8"))), target
    except Exception as exc:
        raise HG232ProbeError("protocol_stability_probe_approval_invalid") from exc


def validate_probe_receipt(path: str | Path = DEFAULT_PROBE_RECEIPT, *, expected_hash: str | None = None, allow_calibration_runner_amendment: bool = False) -> HG232ProbeReceipt:
    target = _path(path)
    if not target.is_file():
        raise HG232ProbeError("protocol_stability_probe_required")
    if expected_hash is not None and sha256_file(target) != expected_hash:
        raise HG232ProbeError("protocol_stability_probe_receipt_hash_mismatch")
    try:
        receipt = HG232ProbeReceipt.model_validate_json(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HG232ProbeError("protocol_stability_probe_receipt_invalid") from exc
    config, config_path = load_probe_config(allow_calibration_runner_amendment=allow_calibration_runner_amendment)
    if receipt.network != "enabled":
        raise HG232ProbeError("protocol_stability_probe_receipt_not_real")
    if (
        receipt.probe_config_sha256 != sha256_file(config_path)
        or receipt.parent_config_sha256 != config.parent_config_sha256
        or receipt.protocol_sha256 != config.protocol_sha256
        or receipt.compatibility_receipt_sha256 != config.compatibility_receipt_sha256
        or receipt.calibration_runner_sha256 != config.calibration_runner_sha256
        or receipt.run_id != config.run_id
        or receipt.scenario_id != config.scenario_id
        or receipt.condition != config.condition
        or receipt.seed != config.seed
    ):
        raise HG232ProbeError("protocol_stability_probe_receipt_binding_mismatch")
    return receipt


__all__ = [
    "DEFAULT_PROBE_APPROVAL", "DEFAULT_PROBE_CONFIG", "DEFAULT_PROBE_RECEIPT",
    "HG232ProbeApproval", "HG232ProbeConfig", "HG232ProbeError", "HG232ProbeReceipt",
    "load_probe_approval", "load_probe_config", "validate_probe_receipt",
]
