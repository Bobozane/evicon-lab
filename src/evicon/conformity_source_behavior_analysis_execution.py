"""One-shot execution gate for the source-behavior descriptive analysis.

The default path is offline and does not parse case-level outcomes. The
authorized path reads the already-bound safe audit once, aggregates it with
the pure analysis core, and writes only an aggregate summary plus a redacted
execution receipt to a new output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from datetime import date
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_identification import IdentificationError, safe_json
from .conformity_source_behavior_analysis_lock import (
    DEFAULT_ANALYSIS_APPROVAL,
    DEFAULT_ANALYSIS_LOCK,
    analysis_lock_sha256,
    load_analysis_approval,
    load_analysis_lock,
)
from .conformity_source_behavior_descriptive_analysis import (
    SourceBehaviorDescriptiveSummary,
    aggregate_safe_case_audits,
    safe_summary_json,
)
from .conformity_source_behavior_qualification import SafeBehaviorCaseAudit

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXECUTION_APPROVAL = (
    "configs/provenance_cascade/identification/"
    "conformity_source_behavior_analysis_execution_approval.v2.toml"
)
DEFAULT_EXECUTION_MODULE = "src/evicon/conformity_source_behavior_analysis_execution.py"
DEFAULT_OUTPUT_ROOT = "results/analyses/conformity-source-behavior-qualification-v2"
SUMMARY_NAME = "descriptive_summary.json"
RECEIPT_NAME = "analysis_execution_receipt.json"
EXECUTION_VERSION = "conformity_source_behavior_analysis_execution.v2"


class AnalysisExecutionApproval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: Literal["evicon-conformity-source-behavior-analysis-execution-v2"]
    approval_version: Literal["conformity_source_behavior_analysis_execution_approval.v2"]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    accepted_on: str
    analysis_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    safe_case_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_root: Literal["results/analyses/conformity-source-behavior-qualification-v2"]
    confirm_analysis_lock: bool
    confirm_aggregate_only: bool
    confirm_fixed_order_limitation: bool
    confirm_no_case_level_output: bool
    confirm_no_historical_join: bool
    confirm_no_private_truth: bool
    confirm_no_causal_claim: bool
    confirm_no_overwrite: bool
    confirm_single_execution: bool
    analysis_execution_authorized: Literal[False]

    @model_validator(mode="after")
    def accepted_fields_complete(self) -> "AnalysisExecutionApproval":
        if self.acceptance_status != "accepted":
            return self
        if not self.accepted_by.strip() or not self.accepted_on.strip():
            raise ValueError("accepted analysis execution approval requires reviewer and date")
        try:
            date.fromisoformat(self.accepted_on)
        except ValueError as exc:
            raise ValueError("accepted_on must be an ISO date") from exc
        confirmations = (
            self.confirm_analysis_lock,
            self.confirm_aggregate_only,
            self.confirm_fixed_order_limitation,
            self.confirm_no_case_level_output,
            self.confirm_no_historical_join,
            self.confirm_no_private_truth,
            self.confirm_no_causal_claim,
            self.confirm_no_overwrite,
            self.confirm_single_execution,
        )
        if not all(confirmations):
            raise ValueError("accepted analysis execution approval is incomplete")
        return self


class AnalysisExecutionSafety(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    aggregate_only: Literal[True] = True
    case_level_values_saved: Literal[False] = False
    prompts_saved: Literal[False] = False
    full_responses_saved: Literal[False] = False
    api_key_read: Literal[False] = False
    network_used: Literal[False] = False
    provider_constructed: Literal[False] = False
    historical_results_used: Literal[False] = False
    evaluator_private_truth_loaded: Literal[False] = False
    overwrite_attempted: Literal[False] = False
    inferential_tests_run: Literal[False] = False
    causal_effect_estimated: Literal[False] = False
    not_paper_result: Literal[True] = True
    no_causal_conclusion: Literal[True] = True


class AnalysisExecutionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt_version: Literal["conformity_source_behavior_analysis_execution_receipt.v2"]
    execution_version: Literal["conformity_source_behavior_analysis_execution.v2"]
    status: Literal["completed"]
    analysis_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_module_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    safe_case_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary_path: Literal["descriptive_summary.json"]
    model: Literal["gpt-5.6-luna"]
    seed: Literal[20261110]
    scenario_count: Literal[4]
    case_count: Literal[12]
    summary_version: Literal["conformity_source_behavior_descriptive_summary.v2"]
    safety: AnalysisExecutionSafety


class AnalysisExecutionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["blocked", "completed", "failed"]
    error_code: str | None = None
    analysis_execution_authorized: bool = False
    case_outcomes_loaded: bool = False
    descriptive_statistics_computed: bool = False
    output_root: str | None = None
    summary_path: str | None = None
    receipt_path: str | None = None
    case_count: int = Field(default=0, ge=0, le=12)
    safety: AnalysisExecutionSafety = Field(default_factory=AnalysisExecutionSafety)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def _sha(path: str | Path) -> str:
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def execution_approval_sha256(path: str | Path = DEFAULT_EXECUTION_APPROVAL) -> str:
    return _sha(path)


def _safe_path(path: Path) -> str:
    try:
        return str(path.relative_to(_ROOT))
    except ValueError:
        return str(path)


def _load_approval(
    path: str | Path = DEFAULT_EXECUTION_APPROVAL,
) -> AnalysisExecutionApproval:
    plan = load_analysis_lock(DEFAULT_ANALYSIS_LOCK)
    accepted_lock = load_analysis_approval(DEFAULT_ANALYSIS_APPROVAL)
    try:
        approval = AnalysisExecutionApproval.model_validate(
            tomllib.loads(_resolve(path).read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise IdentificationError("source_behavior_analysis_execution_approval_invalid") from exc
    expected = {
        "analysis_lock_sha256": analysis_lock_sha256(DEFAULT_ANALYSIS_LOCK),
        "analysis_approval_sha256": _sha(DEFAULT_ANALYSIS_APPROVAL),
        "execution_module_sha256": _sha(DEFAULT_EXECUTION_MODULE),
        "qualification_receipt_sha256": plan.qualification_receipt_sha256,
        "safe_case_audit_sha256": plan.safe_case_audit_sha256,
        "output_root": plan.output_root,
    }
    observed = {key: getattr(approval, key) for key in expected}
    if observed != expected or accepted_lock.acceptance_status != "accepted":
        raise IdentificationError("source_behavior_analysis_execution_binding_mismatch")
    return approval


def safe_preflight(
    path: str | Path = DEFAULT_EXECUTION_APPROVAL,
) -> dict[str, object]:
    try:
        approval = _load_approval(path)
    except Exception as exc:
        return {
            "status": "blocked",
            "blocking_reasons": [
                getattr(exc, "code", "source_behavior_analysis_execution_preflight_failed")
            ],
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "case_outcomes_loaded": False,
            "descriptive_statistics_computed": False,
            "results_written": False,
        }
    if approval.acceptance_status == "accepted":
        reasons = ["source_behavior_analysis_execution_authorization_required"]
        ready = True
    else:
        reasons = ["source_behavior_analysis_execution_approval_required"]
        ready = False
    return {
        "status": "source_behavior_analysis_execution_gate_offline_ready",
        "approval_status": approval.acceptance_status,
        "execution_approval_sha256": execution_approval_sha256(path),
        "analysis_lock_sha256": approval.analysis_lock_sha256,
        "analysis_approval_sha256": approval.analysis_approval_sha256,
        "execution_module_sha256": approval.execution_module_sha256,
        "qualification_receipt_sha256": approval.qualification_receipt_sha256,
        "safe_case_audit_sha256": approval.safe_case_audit_sha256,
        "blocking_reasons": reasons,
        "ready_for_analysis_execution": ready,
        "network": "disabled",
        "provider_constructed": False,
        "api_key_read": False,
        "case_outcomes_loaded": False,
        "descriptive_statistics_computed": False,
        "results_written": False,
        "private_truth_exposed": False,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def _read_audits(path: str | Path) -> tuple[SafeBehaviorCaseAudit, ...]:
    target = _resolve(path)
    try:
        return tuple(
            SafeBehaviorCaseAudit.model_validate_json(line)
            for line in target.read_text(encoding="utf-8").splitlines()
            if line
        )
    except Exception as exc:
        raise IdentificationError("source_behavior_analysis_safe_audit_invalid") from exc


def _new_output(path: str | Path) -> Path:
    target = _resolve(path)
    if target != _resolve(DEFAULT_OUTPUT_ROOT):
        raise IdentificationError("source_behavior_analysis_output_path_mismatch")
    if target.exists():
        raise IdentificationError("source_behavior_analysis_output_exists")
    return target


def execute_analysis(
    *,
    allow_analysis: bool = False,
    confirm_run: bool = False,
    approval_path: str | Path = DEFAULT_EXECUTION_APPROVAL,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> AnalysisExecutionSummary:
    if not allow_analysis:
        return AnalysisExecutionSummary.model_validate({
            "status": "blocked",
            "error_code": "source_behavior_analysis_execution_network_or_flag_disabled",
        })
    if not confirm_run:
        return AnalysisExecutionSummary.model_validate({
            "status": "blocked",
            "error_code": "source_behavior_analysis_explicit_confirmation_required",
        })
    target: Path | None = None
    try:
        approval = _load_approval(approval_path)
        if approval.acceptance_status != "accepted":
            raise IdentificationError("source_behavior_analysis_execution_approval_required")
        target = _new_output(output_root)
        # Hash validation happens before semantic audit parsing. The lock loader
        # already binds the receipt and audit bytes to the accepted plan.
        plan = load_analysis_lock(DEFAULT_ANALYSIS_LOCK)
        audits = _read_audits(plan.safe_case_audit_path)
        summary = aggregate_safe_case_audits(audits, plan=plan)
        summary_bytes = (safe_summary_json(summary) + "\n").encode("utf-8")
        receipt = AnalysisExecutionReceipt(
            receipt_version="conformity_source_behavior_analysis_execution_receipt.v2",
            execution_version=EXECUTION_VERSION,
            status="completed",
            analysis_lock_sha256=analysis_lock_sha256(DEFAULT_ANALYSIS_LOCK),
            analysis_approval_sha256=_sha(DEFAULT_ANALYSIS_APPROVAL),
            execution_approval_sha256=execution_approval_sha256(approval_path),
            execution_module_sha256=_sha(DEFAULT_EXECUTION_MODULE),
            qualification_receipt_sha256=plan.qualification_receipt_sha256,
            safe_case_audit_sha256=plan.safe_case_audit_sha256,
            summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),
            summary_path=SUMMARY_NAME,
            model=plan.model,
            seed=plan.seed,
            scenario_count=plan.scenario_count,
            case_count=plan.case_count,
            summary_version=summary.summary_version,
            safety=AnalysisExecutionSafety(),
        )
        target.mkdir(parents=True, exist_ok=False)
        (target / SUMMARY_NAME).write_bytes(summary_bytes)
        (target / RECEIPT_NAME).write_text(
            json.dumps(receipt.model_dump(mode="json"), ensure_ascii=True, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return AnalysisExecutionSummary(
            status="completed",
            analysis_execution_authorized=True,
            case_outcomes_loaded=True,
            descriptive_statistics_computed=True,
            output_root=_safe_path(target),
            summary_path=_safe_path(target / SUMMARY_NAME),
            receipt_path=_safe_path(target / RECEIPT_NAME),
            case_count=plan.case_count,
        )
    except Exception as exc:
        return AnalysisExecutionSummary(
            status="failed" if target is not None else "blocked",
            error_code=getattr(exc, "code", "source_behavior_analysis_execution_failed"),
            output_root=_safe_path(target) if target is not None and target.exists() else None,
            case_outcomes_loaded=False,
            descriptive_statistics_computed=False,
        )


def load_execution_receipt(
    path: str | Path,
    *,
    approval_path: str | Path = DEFAULT_EXECUTION_APPROVAL,
) -> AnalysisExecutionReceipt:
    receipt_path = _resolve(path)
    try:
        receipt = AnalysisExecutionReceipt.model_validate_json(
            receipt_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise IdentificationError("source_behavior_analysis_execution_receipt_invalid") from exc
    plan = load_analysis_lock(DEFAULT_ANALYSIS_LOCK)
    approval = _load_approval(approval_path)
    summary_path = receipt_path.parent / SUMMARY_NAME
    if (
        receipt.analysis_lock_sha256 != analysis_lock_sha256(DEFAULT_ANALYSIS_LOCK)
        or receipt.analysis_approval_sha256 != _sha(DEFAULT_ANALYSIS_APPROVAL)
        or receipt.execution_approval_sha256 != execution_approval_sha256(approval_path)
        or receipt.execution_module_sha256 != _sha(DEFAULT_EXECUTION_MODULE)
        or receipt.qualification_receipt_sha256 != plan.qualification_receipt_sha256
        or receipt.safe_case_audit_sha256 != plan.safe_case_audit_sha256
        or receipt.summary_sha256 != _sha(summary_path)
        or receipt.summary_version != "conformity_source_behavior_descriptive_summary.v2"
        or approval.acceptance_status != "accepted"
    ):
        raise IdentificationError("source_behavior_analysis_execution_receipt_binding_mismatch")
    try:
        SourceBehaviorDescriptiveSummary.model_validate_json(
            summary_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise IdentificationError("source_behavior_analysis_summary_invalid") from exc
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the one-shot source-behavior descriptive analysis"
    )
    parser.add_argument("--allow-analysis", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--approval", default=DEFAULT_EXECUTION_APPROVAL)
    arguments = parser.parse_args(argv)
    if not arguments.allow_analysis:
        print(safe_json(safe_preflight(arguments.approval)))
        return 0
    result = execute_analysis(
        allow_analysis=True,
        confirm_run=arguments.confirm_run,
        approval_path=arguments.approval,
    )
    print(safe_json(result.model_dump(mode="json")))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AnalysisExecutionApproval",
    "AnalysisExecutionReceipt",
    "AnalysisExecutionSafety",
    "AnalysisExecutionSummary",
    "DEFAULT_EXECUTION_APPROVAL",
    "DEFAULT_EXECUTION_MODULE",
    "DEFAULT_OUTPUT_ROOT",
    "RECEIPT_NAME",
    "SUMMARY_NAME",
    "execute_analysis",
    "execution_approval_sha256",
    "load_execution_receipt",
    "safe_preflight",
]
