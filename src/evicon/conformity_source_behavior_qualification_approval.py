"""Offline approval gate for the source-behavior qualification protocol."""
from __future__ import annotations

import hashlib
import json
import tomllib
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_source_behavior_qualification import DEFAULT_CONFIG, load_config
from .conformity_identification import IdentificationError, sha256_file

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_APPROVAL = (
    "configs/provenance_cascade/identification/"
    "conformity_source_behavior_qualification_approval.v2.toml"
)


class SourceBehaviorApproval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    approval_id: Literal["evicon-conformity-source-behavior-qualification-v2"]
    approval_version: Literal["conformity_source_behavior_qualification_approval.v2"]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    accepted_on: str
    gate_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirm_text_matched_projections: bool
    confirm_public_behavior_only: bool
    confirm_private_truth_excluded: bool
    confirm_fake_smoke_not_effect_evidence: bool
    confirm_historical_results_excluded: bool
    network_execution_authorized: Literal[False]

    @model_validator(mode="after")
    def accepted_fields_complete(self) -> "SourceBehaviorApproval":
        if self.acceptance_status == "accepted":
            if not self.accepted_by.strip() or not self.accepted_on.strip():
                raise ValueError("accepted approval requires reviewer and date")
            try:
                date.fromisoformat(self.accepted_on)
            except ValueError as exc:
                raise ValueError("accepted_on must be ISO date") from exc
            if not all((self.confirm_text_matched_projections, self.confirm_public_behavior_only,
                        self.confirm_private_truth_excluded, self.confirm_fake_smoke_not_effect_evidence,
                        self.confirm_historical_results_excluded)):
                raise ValueError("accepted approval is incomplete")
        return self


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def approval_sha256(path: str | Path = DEFAULT_APPROVAL) -> str:
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def load_approval(path: str | Path = DEFAULT_APPROVAL) -> SourceBehaviorApproval:
    try:
        value = SourceBehaviorApproval.model_validate(
            tomllib.loads(_resolve(path).read_text(encoding="utf-8")))
    except Exception as exc:
        raise IdentificationError("source_behavior_approval_invalid") from exc
    config = load_config(DEFAULT_CONFIG)
    if value.gate_config_sha256 != sha256_file(DEFAULT_CONFIG):
        raise IdentificationError("source_behavior_approval_config_hash_mismatch")
    if value.protocol_sha256 != config.protocol_sha256:
        raise IdentificationError("source_behavior_approval_protocol_hash_mismatch")
    return value


def safe_preflight(path: str | Path = DEFAULT_APPROVAL) -> dict[str, object]:
    try:
        approval = load_approval(path)
    except IdentificationError as exc:
        return {"status": "blocked", "blocking_reasons": [exc.code], "network": "disabled",
                "provider_constructed": False, "results_written": False, "api_key_read": False}
    config = load_config(DEFAULT_CONFIG)
    reasons = [] if approval.acceptance_status == "accepted" else ["human_approval_required"]
    reasons.append("provider_compatibility_not_requested")
    return {"status": "source_behavior_approval_gate_ready", "approval_status": approval.acceptance_status,
            "approval_sha256": approval_sha256(path), "gate_config_sha256": approval.gate_config_sha256,
            "protocol_sha256": approval.protocol_sha256, "case_count": config.case_count,
            "logical_request_cap": config.logical_request_cap,
            "completion_reservation_cap": config.completion_reservation_cap,
            "blocking_reasons": reasons, "ready_for_behavior_qualification": not reasons,
            "network": "disabled", "provider_constructed": False, "results_written": False,
            "api_key_read": False, "behavior_effect_estimated": False,
            "not_paper_result": True, "no_causal_conclusion": True}


def main() -> int:
    print(json.dumps(safe_preflight(), ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
