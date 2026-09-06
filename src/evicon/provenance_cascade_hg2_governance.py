"""Strict offline governance for the H-G.2 eligibility calibration."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .provenance_cascade_hg2 import HG2CalibrationConfig, sha256_file

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = "outputs/study-locks/provenance_cascade_hg2_epistemic_behavior_amendment_receipt.json"
DEFAULT_APPROVAL = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg2_approval_template.toml"
DEFAULT_COMPATIBILITY_RECEIPT = "outputs/study-locks/provenance_cascade_hg2_compatibility_receipt.json"


class HG2GovernanceError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


class HG2AmendmentReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    receipt_id: Literal["provenance-cascade-hg2-epistemic-behavior-amendment-lock-v1"]
    status: Literal["offline_validated_pending_human_approval"]
    config_sha256: str
    protocol_sha256: str
    provider_sha256: str
    request_ledger_sha256: str
    controller_sha256: str
    replay_sha256: str
    amendment_sha256: str
    parent_receipt_sha256: str
    parent_analysis_sha256: str
    schedule_sha256: dict[str, str]
    run_count: Literal[16]
    matched_group_count: Literal[4]
    logical_request_cap: Literal[288]
    completion_reservation_cap: Literal[294912]
    fake_replay_passed_count: Literal[16]
    round0_substantive_behavior_count: int = Field(gt=0, lt=6)
    correction_transition_eligible_count: int = Field(gt=0)
    harmful_cascade_eligible_count: int = Field(gt=0)
    behavior_observation_sha256: str
    effectiveness_claimed: Literal[False]
    real_eligibility_calibration_required: Literal[True]
    full_pilot_authorized: Literal[False]
    network: Literal["disabled"]
    private_truth_exposed: Literal[False]
    results_written: Literal[False]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]


class HG2Approval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    approval_id: Literal["provenance-cascade-hg2-eligibility-calibration-approval-v1"]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    config_sha256: str
    protocol_sha256: str
    amendment_sha256: str
    amendment_receipt_sha256: str
    compatibility_receipt_sha256: str
    response_format: Literal["json_schema"]
    response_schema_name: Literal["cascade_agent_epistemic_behavior_response_v1"]
    agent_max_tokens: Literal[1024]
    temperature: Literal[0.2]
    seed: Literal[20261021]
    run_count: Literal[16]
    matched_group_count: Literal[4]
    logical_request_cap: Literal[288]
    completion_reservation_cap: Literal[294912]
    confirm_epistemic_behavior_separation: bool
    confirm_public_roles_are_condition_invariant: bool
    confirm_fake_results_are_not_effect_evidence: bool
    confirm_real_calibration_before_full_pilot: bool
    confirm_parent_pilot_excluded: bool
    network_execution_authorized: Literal[False]

    @model_validator(mode="after")
    def accepted_shape(self) -> "HG2Approval":
        confirmations = (
            self.confirm_epistemic_behavior_separation,
            self.confirm_public_roles_are_condition_invariant,
            self.confirm_fake_results_are_not_effect_evidence,
            self.confirm_real_calibration_before_full_pilot,
            self.confirm_parent_pilot_excluded,
        )
        if self.acceptance_status == "accepted" and (not self.accepted_by.strip() or not all(confirmations)):
            raise ValueError("accepted approval is incomplete")
        return self


def validate_hg2_governance(config: HG2CalibrationConfig) -> tuple[HG2AmendmentReceipt, HG2Approval, list[str]]:
    try:
        receipt_path = _path(DEFAULT_RECEIPT)
        receipt = HG2AmendmentReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
        approval = HG2Approval.model_validate(tomllib.loads(_path(DEFAULT_APPROVAL).read_text(encoding="utf-8")))
    except Exception as exc:
        raise HG2GovernanceError("hg2_governance_invalid") from exc
    expected_schedules = {
        "cascade-hg1-false-majority": "554079b127e75a82aa3304d04ed6effedb4115a445c72e6d23528bb579533bd6",
        "cascade-hg1-true-minority-correction": "a5850afa6ad9c25d3314a92604fe9a5c7faca772835094b9d7d981815d11b5aa",
        "cascade-hg1-independent-true-consensus": "d6c6e4a209d158f74d44751cd9a0752f9aa2cbc8d35cde7b63d5e2032a088252",
        "cascade-hg1-unresolved-disagreement": "d58352a11044f58cc81d3d67761723ccd80b94a8f4232247985fb700bc474fa8",
    }
    if (
        receipt.config_sha256 != sha256_file("configs/provenance_cascade/pilot/provenance_cascade_pilot_hg2_calibration.v1.toml")
        or receipt.protocol_sha256 != config.protocol_sha256
        or receipt.provider_sha256 != sha256_file("src/evicon/openai_provider.py")
        or receipt.request_ledger_sha256 != sha256_file("src/evicon/request_ledger.py")
        or receipt.controller_sha256 != sha256_file("src/evicon/cascade_controller_hg1.py")
        or receipt.replay_sha256 != sha256_file("src/evicon/cascade_hg1_replay.py")
        or receipt.amendment_sha256 != config.amendment_sha256
        or receipt.parent_receipt_sha256 != config.parent_receipt_sha256
        or receipt.parent_analysis_sha256 != config.parent_analysis_sha256
        or receipt.schedule_sha256 != expected_schedules
    ):
        raise HG2GovernanceError("hg2_receipt_binding_mismatch")
    receipt_sha = sha256_file(receipt_path)
    approval_receipt_matches = approval.amendment_receipt_sha256 == receipt_sha
    approval_binding = (
        approval.config_sha256 == receipt.config_sha256
        and approval.protocol_sha256 == receipt.protocol_sha256
        and approval.amendment_sha256 == receipt.amendment_sha256
        and approval.logical_request_cap == config.request_cap
        and approval.completion_reservation_cap == config.completion_reservation_cap
    )
    if not approval_binding:
        raise HG2GovernanceError("hg2_approval_binding_mismatch")
    reasons: list[str] = []
    if approval.acceptance_status != "accepted":
        reasons.append("human_approval_required")
    if not approval_receipt_matches:
        reasons.append("approval_receipt_hash_mismatch")
    if not approval.compatibility_receipt_sha256:
        reasons.append("provider_compatibility_check_required")
    else:
        try:
            from .provenance_cascade_hg2_compatibility_receipt import validate_receipt
            validate_receipt(
                DEFAULT_COMPATIBILITY_RECEIPT,
                expected_hash=approval.compatibility_receipt_sha256,
            )
        except Exception as exc:
            reasons.append(getattr(exc, "code", "provider_compatibility_receipt_invalid"))
    reasons.append("one_seed_real_eligibility_calibration_required")
    return receipt, approval, reasons


__all__ = [
    "DEFAULT_APPROVAL", "DEFAULT_COMPATIBILITY_RECEIPT", "DEFAULT_RECEIPT", "HG2AmendmentReceipt", "HG2Approval",
    "HG2GovernanceError", "validate_hg2_governance",
]
