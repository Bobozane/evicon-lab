"""Offline qualification lock for Conformity Identification Study v1.1."""
from __future__ import annotations

import hashlib
import json
import tomllib
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_identification import IdentificationError, sha256_file
from .conformity_identification_stability import DEFAULT_STABILITY_CONFIG, load_stability_config
from .conformity_identification_v11 import DEFAULT_V11_CONFIG, load_v11_config, validate_branch_plan
from .conformity_source_manipulation import DEFAULT_MANIPULATION_CONFIG, load_manipulation_config

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUALIFICATION_APPROVAL = (
    "configs/provenance_cascade/identification/"
    "conformity_identification_qualification_approval.v1.toml"
)
DEFAULT_ANALYSIS_CONFIG = (
    "configs/provenance_cascade/identification/"
    "conformity_identification_analysis.v1_1.toml"
)
DEFAULT_COMPATIBILITY_RECEIPT = "outputs/study-locks/conformity_identification_compatibility.v1.json"
DEFAULT_MANIPULATION_RECEIPT = "outputs/conformity-source-manipulation-v1/source_manipulation_receipt.json"
DEFAULT_STABILITY_RECEIPT = "outputs/conformity-identification-stability-v1/protocol_stability_receipt.json"


class QualificationApproval(BaseModel):
    """Human approval of exact scientific inputs, never network authorization."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    approval_id: Literal["evicon-conformity-identification-qualification-v1"]
    approval_version: Literal["conformity_identification_qualification_approval.v1"]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    accepted_on: str
    study_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    branch_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manipulation_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manipulation_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stability_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    network_gates_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirm_shared_t0_design: bool
    confirm_five_registered_contrasts: bool
    confirm_source_manipulation_is_separate: bool
    confirm_stability_probe_is_separate: bool
    confirm_no_unregistered_thresholds: bool
    confirm_historical_results_excluded: bool
    network_execution_authorized: Literal[False]

    @model_validator(mode="after")
    def accepted_complete(self) -> "QualificationApproval":
        if self.acceptance_status == "accepted":
            if not self.accepted_by.strip() or not self.accepted_on.strip():
                raise ValueError("accepted qualification approval requires reviewer and date")
            try:
                date.fromisoformat(self.accepted_on)
            except ValueError as exc:
                raise ValueError("accepted_on must be ISO date") from exc
            if not all((
                self.confirm_shared_t0_design,
                self.confirm_five_registered_contrasts,
                self.confirm_source_manipulation_is_separate,
                self.confirm_stability_probe_is_separate,
                self.confirm_no_unregistered_thresholds,
                self.confirm_historical_results_excluded,
            )):
                raise ValueError("accepted qualification approval is incomplete")
        return self


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def load_qualification_approval(
    path: str | Path = DEFAULT_QUALIFICATION_APPROVAL,
) -> QualificationApproval:
    approval_path = _resolve(path)
    try:
        approval = QualificationApproval.model_validate(
            tomllib.loads(approval_path.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise IdentificationError("qualification_approval_invalid") from exc

    study, _ = load_v11_config(DEFAULT_V11_CONFIG)
    validate_branch_plan(study)
    manipulation = load_manipulation_config(DEFAULT_MANIPULATION_CONFIG)
    stability, _ = load_stability_config(DEFAULT_STABILITY_CONFIG)
    expected = {
        "study_config_sha256": sha256_file(DEFAULT_V11_CONFIG),
        "protocol_sha256": study.protocol_sha256,
        "branch_contract_sha256": study.branch_contract_sha256,
        "analysis_config_sha256": sha256_file(DEFAULT_ANALYSIS_CONFIG),
        "manipulation_config_sha256": sha256_file(DEFAULT_MANIPULATION_CONFIG),
        "manipulation_protocol_sha256": manipulation.protocol_sha256,
        "stability_config_sha256": sha256_file(DEFAULT_STABILITY_CONFIG),
        "qualification_contract_sha256": sha256_file("src/evicon/conformity_identification_qualification.py"),
        "compatibility_module_sha256": sha256_file("src/evicon/conformity_identification_compatibility.py"),
        "network_gates_module_sha256": sha256_file("src/evicon/conformity_identification_network_gates.py"),
    }
    for field, digest in expected.items():
        if getattr(approval, field) != digest:
            raise IdentificationError(f"qualification_{field}_mismatch")
    if stability.study_config_sha256 != approval.study_config_sha256:
        raise IdentificationError("qualification_stability_study_binding_mismatch")
    return approval


def approval_sha256(path: str | Path = DEFAULT_QUALIFICATION_APPROVAL) -> str:
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def _receipt(path: str, missing_code: str) -> dict[str, object]:
    target = _resolve(path)
    if not target.is_file():
        raise IdentificationError(missing_code)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise IdentificationError(f"{missing_code}_invalid") from exc
    if not isinstance(value, dict):
        raise IdentificationError(f"{missing_code}_invalid")
    forbidden = {
        "prompt", "system_prompt", "user_prompt", "response", "model_reply",
        "api_key", "authorization", "provider_metadata", "ground_truth_label",
        "source_independence_label",
    }
    if forbidden.intersection(key.lower() for key in value):
        raise IdentificationError(f"{missing_code}_unsafe")
    return value


def validate_qualification_receipts(
    approval: QualificationApproval,
) -> tuple[bool, bool, bool]:
    current_approval_sha = approval_sha256()
    compatibility = _receipt(DEFAULT_COMPATIBILITY_RECEIPT, "provider_compatibility_check_required")
    if not (
        compatibility.get("status") == "completed"
        and compatibility.get("parser_valid") is True
        and compatibility.get("finish_reason") == "stop"
        and compatibility.get("http_status_class") == "2xx"
        and compatibility.get("attempt_count") == 1
        and compatibility.get("study_config_sha256") == approval.study_config_sha256
        and compatibility.get("protocol_sha256") == approval.protocol_sha256
        and compatibility.get("branch_contract_sha256") == approval.branch_contract_sha256
        and compatibility.get("approval_sha256") == current_approval_sha
    ):
        raise IdentificationError("provider_compatibility_receipt_invalid")

    manipulation = _receipt(DEFAULT_MANIPULATION_RECEIPT, "real_source_manipulation_check_required")
    if not (
        manipulation.get("status") == "completed"
        and manipulation.get("logical_request_count") == 12
        and manipulation.get("transport_attempt_count") == 12
        and manipulation.get("config_sha256") == approval.manipulation_config_sha256
        and manipulation.get("approval_sha256") == current_approval_sha
        and manipulation.get("root_count_totals") == {
            "independent_roots": 8, "same_root": 4, "source_free": 0,
        }
    ):
        raise IdentificationError("source_manipulation_receipt_invalid")

    stability = _receipt(DEFAULT_STABILITY_RECEIPT, "protocol_stability_probe_required")
    if not (
        stability.get("status") == "completed"
        and stability.get("logical_request_count") == 30
        and stability.get("transport_attempt_count") == 30
        and stability.get("parser_invalid_count") == 0
        and stability.get("config_sha256") == approval.stability_config_sha256
        and stability.get("approval_sha256") == current_approval_sha
        and stability.get("stage_counts") == {
            "correction_evidence": 6, "final_judgment": 6,
            "initial_private": 6, "self_reflection": 6, "social_exposure": 6,
        }
    ):
        raise IdentificationError("protocol_stability_receipt_invalid")
    return True, True, True


def safe_qualification_preflight(
    approval_path: str | Path = DEFAULT_QUALIFICATION_APPROVAL,
) -> dict[str, object]:
    try:
        approval = load_qualification_approval(approval_path)
        reasons: list[str] = []
        if approval.acceptance_status != "accepted":
            reasons.append("exact_hash_human_approval_required")
        receipt_status = {
            "provider_compatibility": False,
            "source_manipulation": False,
            "protocol_stability": False,
        }
        if approval.acceptance_status == "accepted":
            checks = (
                (DEFAULT_COMPATIBILITY_RECEIPT, "provider_compatibility_check_required", "provider_compatibility"),
                (DEFAULT_MANIPULATION_RECEIPT, "real_source_manipulation_check_required", "source_manipulation"),
                (DEFAULT_STABILITY_RECEIPT, "protocol_stability_probe_required", "protocol_stability"),
            )
            for receipt_path, missing_code, key in checks:
                if not _resolve(receipt_path).is_file():
                    reasons.append(missing_code)
                else:
                    receipt_status[key] = True
            if all(receipt_status.values()):
                try:
                    validate_qualification_receipts(approval)
                except IdentificationError as exc:
                    reasons.append(exc.code)
                    receipt_status = {key: False for key in receipt_status}
        else:
            reasons.extend((
                "provider_compatibility_check_required",
                "real_source_manipulation_check_required",
                "protocol_stability_probe_required",
            ))
        return {
            "status": "qualification_gates_offline_ready",
            "qualification_passed": not reasons,
            "ready_for_behavior_study": False,
            "blocking_reasons": reasons,
            "gate_receipts": receipt_status,
            "approval_status": approval.acceptance_status,
            "approval_sha256": approval_sha256(approval_path),
            "study_config_sha256": approval.study_config_sha256,
            "logical_request_cap": 1800,
            "completion_reservation_cap": 921600,
            "compatibility_request_cap": 1,
            "manipulation_request_cap": 12,
            "stability_request_cap": 30,
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "private_truth_exposed": False,
            "development_only": True,
            "not_paper_result": True,
            "no_causal_conclusion": True,
        }
    except Exception as exc:
        return {
            "status": "blocked",
            "ready_for_behavior_study": False,
            "blocking_reasons": [getattr(exc, "code", "qualification_preflight_failed")],
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "private_truth_exposed": False,
        }


def safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


__all__ = [
    "DEFAULT_COMPATIBILITY_RECEIPT",
    "DEFAULT_MANIPULATION_RECEIPT",
    "DEFAULT_QUALIFICATION_APPROVAL",
    "DEFAULT_STABILITY_RECEIPT",
    "QualificationApproval",
    "approval_sha256",
    "load_qualification_approval",
    "safe_qualification_preflight",
    "validate_qualification_receipts",
]
