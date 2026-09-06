"""Final offline preflight after researcher approval of the H-D design.

This gate validates local hashes and approval scope only.  It intentionally has
no network flag, does not read environment variables, and never constructs a
Provider or creates a results directory.
"""
from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .provenance_cascade_amendment import HDAmendmentReceipt, HDPilotError, load_hd_config, preflight_hd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HDFinalPreflightError(ValueError):
    """Stable, non-sensitive final-preflight error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HDApprovedScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_count: Literal[4]
    condition_count: Literal[4]
    seed_count: Literal[3]
    agent_count: Literal[6]
    round_count: Literal[3]
    matched_group_count: Literal[12]
    run_count: Literal[48]
    logical_request_cap: Literal[864]
    completion_reservation_cap: Literal[221184]


class HDApprovedProviderContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_kind: Literal["openai_compatible"]
    model_env_var: Literal["EVICON_LLM_MODEL"]
    base_url_env_var: Literal["EVICON_LLM_BASE_URL"]
    api_key_env_var: Literal["EVICON_LLM_API_KEY"]
    network_default: Literal["disabled"]
    agent_temperature: Literal[0.2]
    agent_max_tokens: Literal[256]
    max_retries: Literal[1]
    timeout_seconds: Literal[15.0]


class HDApprovedMaterial(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    config_sha256: str = Field(min_length=64, max_length=64)
    public_graph_sha256: str = Field(min_length=64, max_length=64)
    exposure_ledger_sha256: str = Field(min_length=64, max_length=64)

    @field_validator("config_sha256", "public_graph_sha256", "exposure_ledger_sha256")
    @classmethod
    def hashes(cls, value: str) -> str:
        value = value.strip().lower()
        if not _SHA256.fullmatch(value):
            raise ValueError("approved material hash must be SHA-256")
        return value


class HDResearcherApproval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: Literal["provenance-cascade-hd-researcher-approval-2026-08-21"]
    approval_version: Literal["provenance_cascade_hd_approval.v1"]
    status: Literal["approved"]
    approved_by: Literal["researcher_user"]
    approved_on: Literal["2026-08-21"]
    approval_scope: Literal["provenance_cascade_hd_real_pilot_design"]
    amendment_id: Literal["provenance-cascade-hb-trigger-observability-amendment"]
    amendment_version: Literal["provenance_cascade_hb_amendment.v2"]
    amendment_path: str = Field(min_length=1)
    amendment_sha256: str = Field(min_length=64, max_length=64)
    pilot_config_path: str = Field(min_length=1)
    pilot_config_sha256: str = Field(min_length=64, max_length=64)
    technical_receipt_path: str = Field(min_length=1)
    technical_receipt_sha256: str = Field(min_length=64, max_length=64)
    confirm_amendment: Literal[True]
    confirm_scenario_materials: Literal[True]
    confirm_provenance_graphs: Literal[True]
    confirm_exposure_ledgers: Literal[True]
    confirm_request_cap: Literal[True]
    confirm_completion_reservation_cap: Literal[True]
    confirm_provider_contract: Literal[True]
    confirm_retry_timeout: Literal[True]
    confirm_append_only_ledger: Literal[True]
    confirm_resume_contract: Literal[True]
    confirm_no_overwrite: Literal[True]
    confirm_calibration_excluded: Literal[True]
    confirm_wvs_legacy_excluded: Literal[True]
    network_execution_authorized: Literal[False]
    network_authorization_required_separately: Literal[True]
    approved_scope: HDApprovedScope
    provider_contract: HDApprovedProviderContract
    approved_materials: tuple[HDApprovedMaterial, ...] = Field(min_length=4, max_length=4)

    @field_validator("amendment_sha256", "pilot_config_sha256", "technical_receipt_sha256")
    @classmethod
    def hashes(cls, value: str) -> str:
        value = value.strip().lower()
        if not _SHA256.fullmatch(value):
            raise ValueError("approval binding must be SHA-256")
        return value

    @model_validator(mode="after")
    def material_set(self) -> "HDResearcherApproval":
        expected = {
            "cascade-false-majority",
            "cascade-true-minority-correction",
            "cascade-independent-true-consensus",
            "cascade-unresolved-disagreement",
        }
        ids = [item.scenario_id for item in self.approved_materials]
        if set(ids) != expected or len(ids) != len(set(ids)):
            raise ValueError("approval must bind four unique formal scenarios")
        return self


class HDFinalPreflightReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["passed_waiting_for_network_authorization", "blocked"]
    study_id: str
    config_version: str
    amendment_version: str
    approval_id: str
    approval_sha256: str
    approved_by: str
    approved_on: str
    design_approved: bool
    technical_receipt_valid: bool
    scenario_count: int = Field(ge=0)
    condition_count: int = Field(ge=0)
    seed_count: int = Field(ge=0)
    agent_count: int = Field(ge=0)
    round_count: int = Field(ge=0)
    matched_group_count: int = Field(ge=0)
    run_count: int = Field(ge=0)
    logical_request_cap: int = Field(ge=0)
    completion_reservation_cap: int = Field(ge=0)
    provider_kind: str
    max_retries: int = Field(ge=0)
    timeout_seconds: float = Field(ge=0.0)
    append_only_ledger_approved: bool
    resume_contract_approved: bool
    no_overwrite_approved: bool
    output_root: str
    output_paths_available: bool
    network: Literal["disabled"]
    network_execution_authorized: Literal[False]
    ready_for_explicit_network_authorization: bool
    ready_for_execution: Literal[False]
    provider_constructed: Literal[False]
    environment_read: Literal[False]
    credential_value_read: Literal[False]
    result_directory_created: Literal[False]
    evaluator_truth_loaded: Literal[False]
    calibration_fixture_excluded: bool
    wvs_legacy_excluded: bool
    next_required_gate: str
    blocking_reasons: tuple[str, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise HDFinalPreflightError("approval_path_invalid")
    resolved = (base / path).resolve()
    try:
        resolved.relative_to(_REPO_ROOT)
    except ValueError as exc:
        raise HDFinalPreflightError("approval_path_outside_repository") from exc
    return resolved


def load_hd_approval(path: str | Path) -> tuple[HDResearcherApproval, object, object, HDAmendmentReceipt]:
    approval_path = Path(path).resolve()
    try:
        payload = tomllib.loads(approval_path.read_text(encoding="utf-8"))
        approval = HDResearcherApproval.model_validate(payload)
        base = approval_path.parent
        config_path = _resolve(base, approval.pilot_config_path)
        amendment_path = _resolve(base, approval.amendment_path)
        receipt_path = _resolve(base, approval.technical_receipt_path)
        if _sha256(config_path) != approval.pilot_config_sha256:
            raise HDFinalPreflightError("approved_config_hash_mismatch")
        if _sha256(amendment_path) != approval.amendment_sha256:
            raise HDFinalPreflightError("approved_amendment_hash_mismatch")
        if _sha256(receipt_path) != approval.technical_receipt_sha256:
            raise HDFinalPreflightError("approved_receipt_hash_mismatch")
        config, amendment = load_hd_config(config_path)
        technical = preflight_hd(config_path)
        allowed_output_collision = (
            technical.status == "blocked"
            and technical.exposure_replay_status == "passed"
            and set(technical.blocking_reasons) == {"human_approval_required", "output_root_exists"}
        )
        if (
            technical.exposure_replay_status != "passed"
            or (technical.status != "ready_for_human_approval" and not allowed_output_collision)
        ):
            raise HDFinalPreflightError("technical_preflight_not_ready")
        receipt = HDAmendmentReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.new_config_sha256 != approval.pilot_config_sha256
            or receipt.amendment_sha256 != approval.amendment_sha256
            or receipt.smoke_replay_status != "passed"
            or receipt.run_count != 48
            or receipt.matched_group_count != 12
        ):
            raise HDFinalPreflightError("technical_receipt_binding_mismatch")
        if amendment.amendment_id != approval.amendment_id or amendment.amendment_version != approval.amendment_version:
            raise HDFinalPreflightError("approved_amendment_identity_mismatch")
        approved_materials = {item.scenario_id: item for item in approval.approved_materials}
        for material in config.scenario_materials:
            approved = approved_materials.get(material.scenario_id)
            if approved is None or (
                approved.config_sha256 != material.config_sha256
                or approved.public_graph_sha256 != material.public_graph_sha256
                or approved.exposure_ledger_sha256 != material.exposure_ledger_sha256
            ):
                raise HDFinalPreflightError("approved_material_hash_mismatch")
        provider = config.provider
        approved_provider = approval.provider_contract
        if (
            provider.provider_kind != approved_provider.provider_kind
            or provider.model_env_var != approved_provider.model_env_var
            or provider.base_url_env_var != approved_provider.base_url_env_var
            or provider.api_key_env_var != approved_provider.api_key_env_var
            or provider.network_default != approved_provider.network_default
            or provider.agent_temperature != approved_provider.agent_temperature
            or provider.agent_max_tokens != approved_provider.agent_max_tokens
            or provider.max_retries != approved_provider.max_retries
            or provider.timeout_seconds != approved_provider.timeout_seconds
        ):
            raise HDFinalPreflightError("approved_provider_contract_mismatch")
        return approval, config, amendment, receipt
    except HDFinalPreflightError:
        raise
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError, ValueError, HDPilotError) as exc:
        raise HDFinalPreflightError("approval_invalid") from exc


def final_preflight_hd(config_path: str | Path, approval_path: str | Path) -> HDFinalPreflightReport:
    try:
        approval, config, amendment, receipt = load_hd_approval(approval_path)
        if Path(config_path).resolve() != _resolve(Path(approval_path).resolve().parent, approval.pilot_config_path):
            raise HDFinalPreflightError("cli_config_not_approved_config")
        output_value = Path(config.output.results_root)
        output = (_REPO_ROOT / output_value).resolve()
        if output_value.is_absolute() or ".." in output_value.parts or output_value.parts[0] != "results":
            raise HDFinalPreflightError("output_root_invalid")
        available = not output.exists()
        if not available:
            raise HDFinalPreflightError("output_root_exists")
        return HDFinalPreflightReport(
            status="passed_waiting_for_network_authorization",
            study_id=config.study_id,
            config_version=config.config_version,
            amendment_version=amendment.amendment_version,
            approval_id=approval.approval_id,
            approval_sha256=_sha256(Path(approval_path).resolve()),
            approved_by=approval.approved_by,
            approved_on=approval.approved_on,
            design_approved=True,
            technical_receipt_valid=receipt.smoke_replay_status == "passed",
            scenario_count=approval.approved_scope.scenario_count,
            condition_count=approval.approved_scope.condition_count,
            seed_count=approval.approved_scope.seed_count,
            agent_count=approval.approved_scope.agent_count,
            round_count=approval.approved_scope.round_count,
            matched_group_count=approval.approved_scope.matched_group_count,
            run_count=approval.approved_scope.run_count,
            logical_request_cap=approval.approved_scope.logical_request_cap,
            completion_reservation_cap=approval.approved_scope.completion_reservation_cap,
            provider_kind=approval.provider_contract.provider_kind,
            max_retries=approval.provider_contract.max_retries,
            timeout_seconds=approval.provider_contract.timeout_seconds,
            append_only_ledger_approved=approval.confirm_append_only_ledger,
            resume_contract_approved=approval.confirm_resume_contract,
            no_overwrite_approved=approval.confirm_no_overwrite,
            output_root=config.output.results_root,
            output_paths_available=True,
            network="disabled",
            network_execution_authorized=False,
            ready_for_explicit_network_authorization=True,
            ready_for_execution=False,
            provider_constructed=False,
            environment_read=False,
            credential_value_read=False,
            result_directory_created=False,
            evaluator_truth_loaded=False,
            calibration_fixture_excluded=approval.confirm_calibration_excluded,
            wvs_legacy_excluded=approval.confirm_wvs_legacy_excluded,
            next_required_gate="explicit_allow_network_and_run_confirmation",
            blocking_reasons=("explicit_network_authorization_required",),
        )
    except HDFinalPreflightError as exc:
        return HDFinalPreflightReport(
            status="blocked", study_id="unknown", config_version="unknown", amendment_version="unknown",
            approval_id="unknown", approval_sha256="0" * 64, approved_by="unknown", approved_on="unknown",
            design_approved=False, technical_receipt_valid=False, scenario_count=0, condition_count=0,
            seed_count=0, agent_count=0, round_count=0, matched_group_count=0, run_count=0,
            logical_request_cap=0, completion_reservation_cap=0, provider_kind="unknown", max_retries=0,
            timeout_seconds=0.0, append_only_ledger_approved=False, resume_contract_approved=False,
            no_overwrite_approved=False, output_root="results/", output_paths_available=False,
            network="disabled", network_execution_authorized=False, ready_for_explicit_network_authorization=False,
            ready_for_execution=False, provider_constructed=False, environment_read=False,
            credential_value_read=False, result_directory_created=False, evaluator_truth_loaded=False,
            calibration_fixture_excluded=False, wvs_legacy_excluded=False,
            next_required_gate="repair_preflight_inputs", blocking_reasons=(exc.code,),
        )


__all__ = [
    "HDApprovedMaterial", "HDApprovedProviderContract", "HDApprovedScope", "HDFinalPreflightError",
    "HDFinalPreflightReport", "HDResearcherApproval", "final_preflight_hd", "load_hd_approval",
]
