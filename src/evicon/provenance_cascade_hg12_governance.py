"""Offline approval, receipt, and final preflight gate for H-G.1.2."""
from __future__ import annotations

import hashlib
import json
import re
import tomllib
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cascade_agent_protocol_hg11 import HG11_PROTOCOL_VERSION, HG11_SCHEMA_NAME, HG11_TEMPLATE_VERSION
from .provenance_cascade_hg12_design import HG12DesignConfig

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg12.v1.toml"
DEFAULT_APPROVAL = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg12_approval.toml"
DEFAULT_AMENDMENT_RECEIPT = "outputs/study-locks/provenance_cascade_hg12_observability_amendment_receipt.json"
DEFAULT_COMPATIBILITY_RECEIPT = "outputs/study-locks/provenance_cascade_hg12_compatibility_receipt.json"
COMPATIBILITY_MODULE_VERSION = "provenance_cascade_hg12_compatibility.v1"


class HG12GovernanceError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def sha256_file(value: str | Path) -> str:
    return hashlib.sha256(_path(value).read_bytes()).hexdigest()


def _is_sha(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


class HG12Approval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: Literal["provenance-cascade-hg12-observability-approval-v1"]
    acceptance_status: Literal["accepted"]
    accepted_by: str = Field(min_length=1)
    accepted_on: date
    config_sha256: str
    protocol_sha256: str
    template_sha256: str
    controller_sha256: str
    replay_sha256: str
    amendment_sha256: str
    amendment_receipt_sha256: str
    compatibility_receipt_sha256: str = ""
    response_format: Literal["json_schema"]
    response_schema_name: Literal["cascade_agent_response_v2_1"]
    agent_max_tokens: Literal[1024]
    temperature: Literal[0.2]
    compatibility_seed: Literal[20261011]
    run_count: Literal[48]
    matched_group_count: Literal[12]
    logical_request_cap: Literal[864]
    completion_reservation_cap: Literal[884736]
    max_retries: Literal[1]
    timeout_seconds: Literal[15]
    confirm_observability_amendment: Literal[True]
    confirm_public_inputs_only: Literal[True]
    confirm_parent_results_excluded: Literal[True]
    confirm_scope_and_budget: Literal[True]
    confirm_no_overwrite: Literal[True]
    confirm_append_only_ledger: Literal[True]
    confirm_resume_rules: Literal[True]
    confirm_parser_invalid_stops: Literal[True]
    network_execution_authorized: Literal[False]

    @model_validator(mode="after")
    def hashes_are_safe(self) -> "HG12Approval":
        values = (
            self.config_sha256, self.protocol_sha256, self.template_sha256,
            self.controller_sha256, self.replay_sha256, self.amendment_sha256,
            self.amendment_receipt_sha256,
        )
        if not all(_is_sha(value) for value in values):
            raise ValueError("approval_hash_invalid")
        if self.compatibility_receipt_sha256 and not _is_sha(self.compatibility_receipt_sha256):
            raise ValueError("approval_compatibility_hash_invalid")
        return self


class HG12CompatibilityReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["completed"]
    parser_valid: Literal[True]
    finish_reason: Literal["stop"]
    http_status_class: Literal["2xx"]
    model: str = Field(min_length=1)
    response_format: Literal["json_schema"]
    schema_name: Literal["cascade_agent_response_v2_1"]
    attempt_count: Literal[1]
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    config_sha256: str
    protocol_sha256: str
    template_sha256: str
    controller_sha256: str
    replay_sha256: str
    amendment_sha256: str
    amendment_receipt_sha256: str
    protocol_version: Literal["provenance_cascade_agent_protocol.hg1_1_truncation.v1"]
    template_version: Literal["cascade_agent_turn.hg1_1_public_content.v1"]
    compatibility_module_version: Literal["provenance_cascade_hg12_compatibility.v1"]
    max_tokens: Literal[1024]
    temperature: Literal[0.2]
    seed: Literal[20261011]
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
    def consistent(self) -> "HG12CompatibilityReceipt":
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("token_usage_inconsistent")
        hashes = (
            self.config_sha256, self.protocol_sha256, self.template_sha256,
            self.controller_sha256, self.replay_sha256, self.amendment_sha256,
            self.amendment_receipt_sha256,
        )
        if not all(_is_sha(value) for value in hashes):
            raise ValueError("receipt_hash_invalid")
        return self


def _load_config(path: str | Path = DEFAULT_CONFIG) -> tuple[HG12DesignConfig, Path]:
    config_path = _path(path)
    try:
        config = HG12DesignConfig.model_validate(tomllib.loads(config_path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise HG12GovernanceError("config_invalid") from exc
    return config, config_path


def load_approval(path: str | Path = DEFAULT_APPROVAL) -> HG12Approval:
    try:
        return HG12Approval.model_validate(tomllib.loads(_path(path).read_text(encoding="utf-8")))
    except Exception as exc:
        raise HG12GovernanceError("approval_invalid") from exc


def _validate_public_bindings(config: HG12DesignConfig, config_path: Path, approval: HG12Approval) -> None:
    bindings = (
        (config.protocol_path, config.protocol_sha256, "protocol_hash_mismatch"),
        (config.controller_path, config.controller_sha256, "controller_hash_mismatch"),
        (config.replay_path, config.replay_sha256, "replay_hash_mismatch"),
        (config.amendment_path, config.amendment_sha256, "amendment_hash_mismatch"),
        (config.policy_path, config.policy_sha256, "policy_hash_mismatch"),
    )
    for value, expected, code in bindings:
        candidate = _path(value)
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise HG12GovernanceError(code)
    for material in config.schedule_materials:
        candidate = _path(material.schedule_path)
        if not candidate.is_file() or sha256_file(candidate) != material.schedule_sha256:
            raise HG12GovernanceError("schedule_hash_mismatch")
    amendment_receipt = _path(DEFAULT_AMENDMENT_RECEIPT)
    if not amendment_receipt.is_file() or sha256_file(amendment_receipt) != approval.amendment_receipt_sha256:
        raise HG12GovernanceError("amendment_receipt_hash_mismatch")
    try:
        payload = json.loads(amendment_receipt.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HG12GovernanceError("amendment_receipt_invalid") from exc
    if (
        payload.get("config_sha256") != sha256_file(config_path)
        or payload.get("protocol_sha256") != config.protocol_sha256
        or payload.get("amendment_sha256") != config.amendment_sha256
        or payload.get("logical_request_cap") != config.request_cap
        or payload.get("completion_reservation_cap") != config.completion_reservation_cap
        or payload.get("network") != "disabled"
    ):
        raise HG12GovernanceError("amendment_receipt_binding_mismatch")
    approval_values = (
        approval.config_sha256 == sha256_file(config_path),
        approval.protocol_sha256 == config.protocol_sha256,
        approval.template_sha256 == config.protocol_sha256,
        approval.controller_sha256 == config.controller_sha256,
        approval.replay_sha256 == config.replay_sha256,
        approval.amendment_sha256 == config.amendment_sha256,
        approval.run_count == config.run_count,
        approval.matched_group_count == config.matched_group_count,
        approval.logical_request_cap == config.request_cap,
        approval.completion_reservation_cap == config.completion_reservation_cap,
    )
    if not all(approval_values):
        raise HG12GovernanceError("approval_binding_mismatch")


def validate_receipt(
    path: str | Path = DEFAULT_COMPATIBILITY_RECEIPT,
    *, expected_hash: str | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
) -> HG12CompatibilityReceipt:
    receipt_path = _path(path)
    if not receipt_path.is_file():
        raise HG12GovernanceError("compatibility_check_required")
    if expected_hash and sha256_file(receipt_path) != expected_hash:
        raise HG12GovernanceError("compatibility_receipt_hash_mismatch")
    try:
        receipt = HG12CompatibilityReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HG12GovernanceError("compatibility_receipt_invalid") from exc
    config, resolved_config = _load_config(config_path)
    expected = {
        "config_sha256": sha256_file(resolved_config),
        "protocol_sha256": config.protocol_sha256,
        "template_sha256": config.protocol_sha256,
        "controller_sha256": config.controller_sha256,
        "replay_sha256": config.replay_sha256,
        "amendment_sha256": config.amendment_sha256,
        "amendment_receipt_sha256": sha256_file(DEFAULT_AMENDMENT_RECEIPT),
    }
    if any(getattr(receipt, key) != value for key, value in expected.items()):
        raise HG12GovernanceError("compatibility_receipt_binding_mismatch")
    return receipt


def register_compatibility_result(result: BaseModel) -> tuple[str, str]:
    """Atomically register a successful safe result and bind it to approval."""
    target = _path(DEFAULT_COMPATIBILITY_RECEIPT)
    if target.exists():
        raise HG12GovernanceError("compatibility_receipt_exists")
    config, config_path = _load_config()
    approval_path = _path(DEFAULT_APPROVAL)
    approval = load_approval(approval_path)
    _validate_public_bindings(config, config_path, approval)
    data = result.model_dump(mode="python")
    required = {
        "status": "completed", "parser_valid": True, "finish_reason": "stop",
        "http_status_class": "2xx", "response_format": "json_schema",
        "schema_name": HG11_SCHEMA_NAME, "attempt_count": 1, "network": "enabled",
    }
    if any(data.get(key) != value for key, value in required.items()):
        raise HG12GovernanceError("compatibility_result_not_completed")
    for key in ("model", "prompt_tokens", "completion_tokens", "total_tokens", "latency_ms"):
        if data.get(key) is None:
            raise HG12GovernanceError("compatibility_result_incomplete")
    receipt = HG12CompatibilityReceipt(
        status="completed", parser_valid=True, finish_reason="stop", http_status_class="2xx",
        model=data["model"], response_format="json_schema", schema_name=HG11_SCHEMA_NAME,
        attempt_count=1, prompt_tokens=data["prompt_tokens"], completion_tokens=data["completion_tokens"],
        total_tokens=data["total_tokens"], latency_ms=data["latency_ms"],
        config_sha256=sha256_file(config_path), protocol_sha256=config.protocol_sha256,
        template_sha256=config.protocol_sha256, controller_sha256=config.controller_sha256,
        replay_sha256=config.replay_sha256, amendment_sha256=config.amendment_sha256,
        amendment_receipt_sha256=sha256_file(DEFAULT_AMENDMENT_RECEIPT),
        protocol_version=HG11_PROTOCOL_VERSION, template_version=HG11_TEMPLATE_VERSION,
        compatibility_module_version=COMPATIBILITY_MODULE_VERSION,
        max_tokens=1024, temperature=0.2, seed=20261011, max_retries=0, timeout_seconds=5.0,
        compatibility_only=True, one_shot=True, network_enabled=True, pilot_material_used=False,
        private_truth_exposed=False, request_ledger_written=False, results_written=False,
        secrets_logged=False, not_pilot=True, not_paper_result=True, no_causal_conclusion=True,
    )
    serialized = receipt.model_dump_json(indent=2)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    raw = approval_path.read_text(encoding="utf-8")
    if approval.compatibility_receipt_sha256 not in {"", digest}:
        raise HG12GovernanceError("approval_compatibility_hash_conflict")
    pattern = r'^compatibility_receipt_sha256 = "[0-9a-f]*"$'
    if len(re.findall(pattern, raw, flags=re.MULTILINE)) != 1:
        raise HG12GovernanceError("approval_compatibility_field_invalid")
    updated = re.sub(pattern, f'compatibility_receipt_sha256 = "{digest}"', raw, flags=re.MULTILINE)
    try:
        HG12Approval.model_validate(tomllib.loads(updated))
    except Exception as exc:
        raise HG12GovernanceError("approval_update_invalid") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt_tmp = target.with_suffix(".json.tmp")
    approval_tmp = approval_path.with_suffix(".toml.tmp")
    receipt_tmp.write_text(serialized, encoding="utf-8")
    approval_tmp.write_text(updated, encoding="utf-8")
    receipt_tmp.replace(target)
    approval_tmp.replace(approval_path)
    return digest, sha256_file(approval_path)


def final_preflight(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    reasons: list[str] = []
    try:
        config, resolved_config = _load_config(config_path)
        approval = load_approval(DEFAULT_APPROVAL)
        _validate_public_bindings(config, resolved_config, approval)
    except HG12GovernanceError as exc:
        return _blocked([exc.code])
    receipt_sha: str | None = None
    if not approval.compatibility_receipt_sha256:
        reasons.append("compatibility_check_required")
    else:
        try:
            validate_receipt(
                DEFAULT_COMPATIBILITY_RECEIPT,
                expected_hash=approval.compatibility_receipt_sha256,
                config_path=resolved_config,
            )
            receipt_sha = sha256_file(DEFAULT_COMPATIBILITY_RECEIPT)
        except HG12GovernanceError as exc:
            reasons.append(exc.code)
    if _path(config.output_root).exists():
        reasons.append("output_root_exists")
    return {
        "status": "ready_for_real_pilot" if not reasons else "blocked",
        "ready_for_real_pilot": not reasons,
        "blocking_reasons": sorted(set(reasons)),
        "study_id": config.study_id,
        "config_sha256": sha256_file(resolved_config),
        "protocol_sha256": config.protocol_sha256,
        "template_sha256": approval.template_sha256,
        "controller_sha256": config.controller_sha256,
        "replay_sha256": config.replay_sha256,
        "amendment_sha256": config.amendment_sha256,
        "amendment_receipt_sha256": approval.amendment_receipt_sha256,
        "approval_sha256": sha256_file(DEFAULT_APPROVAL),
        "compatibility_receipt_sha256": receipt_sha,
        "scenario_count": 4, "condition_count": 4, "seed_count": 3,
        "run_count": config.run_count, "matched_group_count": config.matched_group_count,
        "logical_request_cap": config.request_cap,
        "completion_reservation_cap": config.completion_reservation_cap,
        "response_format": config.response_format, "schema_name": config.response_schema_name,
        "agent_max_tokens": config.agent_max_tokens, "max_retries": config.max_retries,
        "timeout_seconds": config.timeout_seconds, "output_root": config.output_root,
        "network": "disabled", "provider_constructed": False, "api_key_read": False,
        "results_written": False, "private_truth_exposed": False,
        "development_only": True, "pilot_only": True,
        "not_paper_result": True, "no_causal_conclusion": True,
    }


def _blocked(reasons: list[str]) -> dict[str, object]:
    return {
        "status": "blocked", "ready_for_real_pilot": False,
        "blocking_reasons": reasons, "network": "disabled",
        "provider_constructed": False, "api_key_read": False,
        "results_written": False, "private_truth_exposed": False,
    }


__all__ = [
    "COMPATIBILITY_MODULE_VERSION", "DEFAULT_AMENDMENT_RECEIPT", "DEFAULT_APPROVAL",
    "DEFAULT_COMPATIBILITY_RECEIPT", "DEFAULT_CONFIG", "HG12Approval",
    "HG12CompatibilityReceipt", "HG12GovernanceError", "final_preflight",
    "load_approval", "register_compatibility_result", "sha256_file", "validate_receipt",
]
