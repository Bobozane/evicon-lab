"""Read-only audit for the source-root manipulation qualification gate.

This module deliberately audits transport and protocol completion only.  The
gate asks a model to report public source-root relations already present in
its visible context; it does not observe adoption, sharing, or a causal
behavioral response.  Its report must therefore never be used as a behavior
effect estimate.
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .conformity_identification import IdentificationError, safe_json
from .conformity_source_manipulation import DEFAULT_MANIPULATION_CONFIG, SourceProjection, load_manipulation_config
from .conformity_source_manipulation_resume_v2 import _request_for
from .conformity_source_manipulation_resume_v3 import DEFAULT_AMENDMENT, DEFAULT_LEDGER, DEFAULT_RECEIPT
from .conformity_source_manipulation_smoke import build_cases
from .request_ledger import RequestLedger, RequestLedgerStatus, request_fingerprint_facts

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS_CONFIG = "configs/provenance_cascade/identification/conformity_source_manipulation_analysis.v1.toml"


class SourceManipulationAnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis_id: Literal["evicon-conformity-source-manipulation-analysis-v1"]
    analysis_version: Literal["conformity_source_manipulation_analysis.v1"]
    status: Literal["evaluator_only"]
    development_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    behavior_effect_estimated: Literal[False]
    results_joined_to_behavior_study: Literal[False]
    gate_config_path: str
    gate_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resume_amendment_path: str
    resume_amendment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_path: str
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ledger_path: str
    ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_case_count: Literal[12]
    expected_logical_request_count: Literal[12]
    expected_transport_attempt_count: Literal[15]
    expected_http_retry_count: Literal[1]
    expected_local_configuration_failure_count: Literal[2]
    output_root: Literal["results/analyses/conformity-source-manipulation-v1"]


class SourceManipulationResumeReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["completed"]
    gate_id: Literal["evicon-conformity-source-manipulation-v1"]
    technical_amendment_id: Literal["evicon-conformity-source-manipulation-timeout-resume-v3"]
    amendment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_request_count: Literal[12]
    transport_attempt_count: Literal[15]
    http_transport_retry_count: Literal[1]
    local_configuration_failure_count: Literal[2]
    completed_fingerprint_replay_count: Literal[0]
    root_count_totals: dict[str, int]
    behavior_effect_estimated: Literal[False]
    joined_to_behavior_study: Literal[False]
    network_used: bool
    private_truth_exposed: Literal[False]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def _sha(path: str | Path) -> str:
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def _load_toml(path: str | Path) -> dict[str, object]:
    try:
        return tomllib.loads(_resolve(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise IdentificationError("source_manipulation_analysis_config_invalid") from exc


def load_analysis_config(path: str | Path = DEFAULT_ANALYSIS_CONFIG) -> SourceManipulationAnalysisConfig:
    try:
        config = SourceManipulationAnalysisConfig.model_validate(_load_toml(path))
    except IdentificationError:
        raise
    except Exception as exc:
        raise IdentificationError("source_manipulation_analysis_config_invalid") from exc
    for actual_path, expected, code in (
        (config.gate_config_path, config.gate_config_sha256, "source_manipulation_analysis_gate_hash_mismatch"),
        (config.resume_amendment_path, config.resume_amendment_sha256, "source_manipulation_analysis_amendment_hash_mismatch"),
        (config.receipt_path, config.receipt_sha256, "source_manipulation_analysis_receipt_hash_mismatch"),
        (config.ledger_path, config.ledger_sha256, "source_manipulation_analysis_ledger_hash_mismatch"),
    ):
        if _sha(actual_path) != expected:
            raise IdentificationError(code)
    return config


def _load_receipt(path: str | Path) -> SourceManipulationResumeReceipt:
    try:
        return SourceManipulationResumeReceipt.model_validate_json(_resolve(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise IdentificationError("source_manipulation_analysis_receipt_invalid") from exc


def analyze_source_manipulation_gate(path: str | Path = DEFAULT_ANALYSIS_CONFIG) -> dict[str, object]:
    """Audit only safe receipts and ledger coordinates; never reconstruct text."""
    config = load_analysis_config(path)
    gate = load_manipulation_config(config.gate_config_path)
    receipt = _load_receipt(config.receipt_path)
    if receipt.config_sha256 != config.gate_config_sha256 or receipt.amendment_sha256 != config.resume_amendment_sha256:
        raise IdentificationError("source_manipulation_analysis_receipt_binding_mismatch")
    if receipt.behavior_effect_estimated or receipt.joined_to_behavior_study:
        raise IdentificationError("source_manipulation_analysis_behavior_boundary_violation")

    cases = build_cases(config.gate_config_path)
    expected = {
        request_fingerprint_facts(_request_for(case))["fingerprint"]: case
        for case in cases
    }
    if len(expected) != config.expected_case_count:
        raise IdentificationError("source_manipulation_analysis_case_set_mismatch")
    entries = RequestLedger(_resolve(config.ledger_path)).entries()
    started = [item for item in entries if item.status is RequestLedgerStatus.STARTED]
    if len(started) != config.expected_transport_attempt_count:
        raise IdentificationError("source_manipulation_analysis_transport_count_mismatch")
    started_fingerprints = {item.fingerprint for item in started}
    if started_fingerprints != set(expected):
        raise IdentificationError("source_manipulation_analysis_ledger_case_binding_mismatch")
    terminal: dict[str, object] = {}
    for item in entries:
        if item.status in {RequestLedgerStatus.COMPLETED, RequestLedgerStatus.FAILED}:
            previous = terminal.get(item.fingerprint)
            if previous is None or item.attempt_count > previous.attempt_count:  # type: ignore[attr-defined]
                terminal[item.fingerprint] = item
    if len(terminal) != config.expected_logical_request_count or any(
        item.status is not RequestLedgerStatus.COMPLETED for item in terminal.values()  # type: ignore[attr-defined]
    ):
        raise IdentificationError("source_manipulation_analysis_incomplete_logical_requests")
    if any(item.fingerprint not in expected for item in terminal.values()):  # type: ignore[attr-defined]
        raise IdentificationError("source_manipulation_analysis_unknown_terminal_request")

    attempts_by_fingerprint = Counter(item.fingerprint for item in started)
    retried = [fingerprint for fingerprint, count in attempts_by_fingerprint.items() if count > 1]
    if retried != [receipt.amendment_sha256] and len(retried) != 1:
        # The receipt intentionally does not disclose a request fingerprint; the
        # ledger still has exactly one retry coordinate and no completed replay.
        raise IdentificationError("source_manipulation_analysis_retry_scope_mismatch")
    failures = [item for item in entries if item.status is RequestLedgerStatus.FAILED]
    failure_codes = Counter(item.error_code for item in failures)
    if failure_codes != Counter({"timeout": 1, "missing_base_url": 2}):
        raise IdentificationError("source_manipulation_analysis_failure_history_mismatch")
    if receipt.http_transport_retry_count != config.expected_http_retry_count or receipt.local_configuration_failure_count != config.expected_local_configuration_failure_count:
        raise IdentificationError("source_manipulation_analysis_receipt_count_mismatch")

    projection_case_counts = Counter(case.projection.value for case in cases)
    expected_root_counts = Counter()
    for case in cases:
        expected_root_counts[case.projection.value] += len({item.source_root_id for item in case.public_root_assignments if item.source_root_id is not None})
    # v3's root-count receipt covers only the six calls made by that resume, not
    # the first six historical successes.  Treat it as a recovery audit, not a
    # 12-case response score.
    return {
        "status": "completed_with_semantic_audit_limit",
        "analysis_id": config.analysis_id,
        "gate_id": gate.gate_id,
        "case_count": len(cases),
        "projection_case_counts": dict(sorted(projection_case_counts.items())),
        "expected_visible_root_totals": dict(sorted(expected_root_counts.items())),
        "receipt_resume_root_totals": dict(sorted(receipt.root_count_totals.items())),
        "logical_request_count": len(started_fingerprints),
        "transport_attempt_count": len(started),
        "completed_logical_request_count": len(terminal),
        "failed_latest_logical_request_count": 0,
        "historical_failure_counts": dict(sorted(failure_codes.items())),
        "http_transport_retry_count": receipt.http_transport_retry_count,
        "completed_fingerprint_replay_count": receipt.completed_fingerprint_replay_count,
        "case_level_parse_outcomes_persisted": False,
        "structural_response_score": "not_observable_from_safe_artifacts",
        "behavior_effect_estimated": False,
        "joined_to_behavior_study": False,
        "next_stage_requirement": "behavioral_source_manipulation_with_safe_case_level_outcomes",
        "input_sha256": {
            "gate_config": config.gate_config_sha256,
            "resume_amendment": config.resume_amendment_sha256,
            "receipt": config.receipt_sha256,
            "ledger": config.ledger_sha256,
        },
        "development_only": True,
        "not_paper_result": True,
        "no_causal_conclusion": True,
        "private_truth_exposed": False,
        "network": "disabled",
    }


def write_analysis_report(result: dict[str, object], path: str | Path) -> Path:
    destination = _resolve(path)
    if destination.exists():
        raise IdentificationError("source_manipulation_analysis_output_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2), encoding="utf-8")
    return destination


def safe_analysis_preflight(path: str | Path = DEFAULT_ANALYSIS_CONFIG) -> dict[str, object]:
    try:
        result = analyze_source_manipulation_gate(path)
        return {
            "status": result["status"], "analysis_id": result["analysis_id"],
            "case_count": result["case_count"], "logical_request_count": result["logical_request_count"],
            "transport_attempt_count": result["transport_attempt_count"],
            "structural_response_score": result["structural_response_score"],
            "behavior_effect_estimated": False, "network": "disabled", "provider_constructed": False,
            "api_key_read": False, "results_written": False, "private_truth_exposed": False,
            "not_paper_result": True, "no_causal_conclusion": True,
        }
    except Exception as exc:
        return {"status": "blocked", "blocking_reasons": [getattr(exc, "code", "source_manipulation_analysis_failed")],
                "network": "disabled", "provider_constructed": False, "api_key_read": False,
                "results_written": False, "private_truth_exposed": False}


__all__ = ["DEFAULT_ANALYSIS_CONFIG", "SourceManipulationAnalysisConfig", "analyze_source_manipulation_gate", "load_analysis_config", "safe_analysis_preflight", "write_analysis_report"]
