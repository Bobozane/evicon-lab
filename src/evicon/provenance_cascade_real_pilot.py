"""Explicitly authorized, resumable H-D provenance-cascade pilot executor.

This is a batch sidecar around :class:`CascadeRealAgentRunner`. It binds the
approved H-D design to one fixed output root and keeps request ledgers
append-only. It never stores prompts, raw provider responses, credentials, or
evaluator-private labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tomllib
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .cascade_protocol import CascadeScenarioLoader
from .cascade_real_agent_runner import CascadeRealAgentRunError, CascadeRealAgentRunRecord, CascadeRealAgentRunner
from .llm_contract import LLMProvider
from .openai_provider import OpenAICompatibleProvider, ProviderConfig
from .provenance_cascade_amendment import HDPilotError, ProvenanceCascadeHDConfig, load_hd_config
from .provenance_cascade_real_preflight import HDFinalPreflightError, load_hd_approval
from .request_ledger import RequestLedger, RequestLedgerStatus

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd.v2.toml"
_DEFAULT_APPROVAL = _REPO_ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd_approval_2026-08-21.toml"
_DEFAULT_AUTHORIZATION = _REPO_ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd_execution_authorization_2026-08-21.toml"
_DEFAULT_PARSER_RECOVERY = _REPO_ROOT / "configs/provenance_cascade/pilot/amendments/provenance_cascade_pilot_hd_parser_recovery_2026-08-21.toml"
_SHA256 = set("0123456789abcdef")
_FORBIDDEN_KEYS = {
    "system_prompt", "user_prompt", "api_key", "authorization_header", "provider_metadata",
    "ground_truth_label", "source_independence_label", "evaluator_truth", "raw_response",
}


class HDRealPilotError(ValueError):
    """Stable, content-free execution error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HDExecutionScope(BaseModel):
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


class HDExecutionSafety(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    development_only: Literal[True]
    pilot_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    evaluator_truth_isolated: Literal[True]
    calibration_fixture_excluded: Literal[True]
    wvs_assets_excluded: Literal[True]
    legacy_002_003_excluded: Literal[True]


class HDNetworkExecutionAuthorization(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    authorization_id: Literal["provenance-cascade-hd-network-execution-authorization-2026-08-21"]
    authorization_version: Literal["provenance_cascade_hd_execution_authorization.v1"]
    status: Literal["authorized"]
    authorized_by: Literal["researcher_user"]
    authorized_on: Literal["2026-08-21"]
    authorization_scope: Literal["provenance_cascade_hd_real_pilot_execution"]
    pilot_config_path: str = Field(min_length=1)
    pilot_config_sha256: str = Field(min_length=64, max_length=64)
    design_approval_path: str = Field(min_length=1)
    design_approval_sha256: str = Field(min_length=64, max_length=64)
    amendment_path: str = Field(min_length=1)
    amendment_sha256: str = Field(min_length=64, max_length=64)
    technical_receipt_path: str = Field(min_length=1)
    technical_receipt_sha256: str = Field(min_length=64, max_length=64)
    network_execution_authorized: Literal[True]
    confirm_run_scope: Literal[True]
    confirm_request_cap: Literal[True]
    confirm_completion_reservation_cap: Literal[True]
    confirm_provider_contract: Literal[True]
    confirm_finite_retry: Literal[True]
    confirm_append_only_ledger: Literal[True]
    confirm_resume_only_recovery: Literal[True]
    confirm_no_overwrite: Literal[True]
    confirm_development_only: Literal[True]
    confirm_no_causal_conclusion: Literal[True]
    authorized_scope: HDExecutionScope
    safety: HDExecutionSafety

    @field_validator("pilot_config_sha256", "design_approval_sha256", "amendment_sha256", "technical_receipt_sha256")
    @classmethod
    def validate_sha(cls, value: str) -> str:
        value = value.strip().lower()
        if len(value) != 64 or any(char not in _SHA256 for char in value):
            raise ValueError("execution authorization requires SHA-256 bindings")
        return value


class HDParserRecoveryAmendment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    amendment_id: Literal["provenance-cascade-hd-parser-invalid-recovery-001"]
    amendment_version: Literal["provenance_cascade_hd_parser_recovery.v1"]
    status: Literal["approved"]
    approved_by: Literal["researcher_user"]
    approved_on: Literal["2026-08-21"]
    scope: Literal["single_agent_parser_invalid_semantic_recovery"]
    run_id: Literal["hd-cascade-false-majority-20260901-no_intervention"]
    scenario_id: Literal["cascade-false-majority"]
    condition: Literal["no_intervention"]
    seed: Literal[20260901]
    agent_id: Literal["network-agent-01"]
    round_id: Literal[0]
    phase: Literal["agent_turn"]
    fingerprint: Literal["34573b1741c14676eda0acd0aa75fa3495930ef6affc707b48e9cb46cc8896dc"]
    original_error_code: Literal["invalid_schema"]
    original_batch_record_sha256: str = Field(min_length=64, max_length=64)
    original_request_ledger_sha256: str = Field(min_length=64, max_length=64)
    pilot_config_sha256: str = Field(min_length=64, max_length=64)
    execution_authorization_sha256: str = Field(min_length=64, max_length=64)
    recovery_kind: Literal["parser_invalid_agent.v1"]
    allowed_additional_transport_attempts: Literal[1]
    resume_required: Literal[True]
    checkpoint_coordinate_must_be_missing: Literal[True]
    logical_request_cap_unchanged: Literal[True]
    prompt_unchanged: Literal[True]
    generation_parameters_unchanged: Literal[True]
    seed_unchanged: Literal[True]
    condition_order_unchanged: Literal[True]
    metric_definitions_unchanged: Literal[True]
    development_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]


class HDPilotRunStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class HDPilotBatchStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class HDPilotRunState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    matched_group_id: str
    scenario_id: str
    condition: str
    seed: int
    status: HDPilotRunStatus = HDPilotRunStatus.PLANNED
    logical_request_count: int = Field(default=0, ge=0)
    completion_reserved_token_count: int = Field(default=0, ge=0)
    actual_prompt_token_count: int | None = Field(default=None, ge=0)
    actual_completion_token_count: int | None = Field(default=None, ge=0)
    actual_total_token_count: int | None = Field(default=None, ge=0)
    replay_status: str | None = None
    directive_applied_count: int = Field(default=0, ge=0)
    error_code: str | None = None
    run_record_sha256: str | None = None


class HDPilotBatchRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_id: Literal["provenance-cascade-pilot-hd-v2"]
    status: HDPilotBatchStatus
    binding_sha256: str = Field(min_length=64, max_length=64)
    config_sha256: str = Field(min_length=64, max_length=64)
    approval_sha256: str = Field(min_length=64, max_length=64)
    authorization_sha256: str = Field(min_length=64, max_length=64)
    amendment_sha256: str = Field(min_length=64, max_length=64)
    model_name: str
    run_count: Literal[48]
    matched_group_count: Literal[12]
    request_cap: Literal[864]
    completion_reservation_cap: Literal[221184]
    logical_request_count: int = Field(ge=0, le=864)
    completion_reserved_token_count: int = Field(ge=0, le=221184)
    actual_prompt_token_count: int | None = Field(default=None, ge=0)
    actual_completion_token_count: int | None = Field(default=None, ge=0)
    actual_total_token_count: int | None = Field(default=None, ge=0)
    completed_run_count: int = Field(ge=0, le=48)
    failed_run_count: int = Field(ge=0, le=48)
    planned_run_count: int = Field(ge=0, le=48)
    current_run_id: str | None = None
    failure_code: str | None = None
    runs: tuple[HDPilotRunState, ...] = Field(min_length=48, max_length=48)
    created_at: str
    updated_at: str
    development_only: Literal[True] = True
    pilot_only: Literal[True] = True
    not_paper_result: Literal[True] = True
    no_causal_conclusion: Literal[True] = True
    evaluator_truth_loaded: Literal[False] = False
    calibration_fixture_excluded: Literal[True] = True
    wvs_assets_excluded: Literal[True] = True
    legacy_002_003_excluded: Literal[True] = True

    @model_validator(mode="after")
    def counts_match(self) -> "HDPilotBatchRecord":
        completed = sum(item.status is HDPilotRunStatus.COMPLETED for item in self.runs)
        failed = sum(item.status is HDPilotRunStatus.FAILED for item in self.runs)
        planned = len(self.runs) - completed - failed
        if (completed, failed, planned) != (self.completed_run_count, self.failed_run_count, self.planned_run_count):
            raise ValueError("batch run counts do not match run states")
        return self


class HDPilotExecutionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["blocked", "failed", "completed"]
    network: Literal["disabled", "enabled"]
    provider_constructed: bool
    output_root: str
    run_count: int = Field(ge=0)
    completed_run_count: int = Field(ge=0)
    failed_run_count: int = Field(ge=0)
    logical_request_count: int = Field(ge=0)
    request_cap: Literal[864]
    completion_reserved_token_count: int = Field(ge=0)
    completion_reservation_cap: Literal[221184]
    actual_prompt_token_count: int | None = Field(default=None, ge=0)
    actual_completion_token_count: int | None = Field(default=None, ge=0)
    actual_total_token_count: int | None = Field(default=None, ge=0)
    replay_status: str
    directive_applied_count: int = Field(ge=0)
    error_code: str | None = None
    private_truth_exposed: Literal[False] = False
    not_paper_result: Literal[True] = True
    no_causal_conclusion: Literal[True] = True


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise HDRealPilotError("authorization_path_invalid")
    result = (base / path).resolve()
    try:
        result.relative_to(_REPO_ROOT)
    except ValueError as exc:
        raise HDRealPilotError("authorization_path_outside_repository") from exc
    return result


def load_execution_authorization(path: str | Path) -> HDNetworkExecutionAuthorization:
    source = Path(path).resolve()
    try:
        authorization = HDNetworkExecutionAuthorization.model_validate(tomllib.loads(source.read_text(encoding="utf-8")))
        base = source.parent
        bindings = (
            (authorization.pilot_config_path, authorization.pilot_config_sha256),
            (authorization.design_approval_path, authorization.design_approval_sha256),
            (authorization.amendment_path, authorization.amendment_sha256),
            (authorization.technical_receipt_path, authorization.technical_receipt_sha256),
        )
        if any(_sha256(_resolve(base, item_path)) != expected for item_path, expected in bindings):
            raise HDRealPilotError("execution_authorization_hash_mismatch")
        return authorization
    except HDRealPilotError:
        raise
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise HDRealPilotError("execution_authorization_invalid") from exc


def load_parser_recovery_amendment(path: str | Path) -> HDParserRecoveryAmendment:
    try:
        return HDParserRecoveryAmendment.model_validate(
            tomllib.loads(Path(path).read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise HDRealPilotError("parser_recovery_amendment_invalid") from exc


def _prepare_parser_recovery(
    amendment: HDParserRecoveryAmendment, *, record: HDPilotBatchRecord, root: Path,
    batch_path: Path, config_path: Path, authorization_path: Path,
) -> None:
    if (
        record.status is not HDPilotBatchStatus.FAILED
        or record.failure_code != amendment.original_error_code
        or record.current_run_id != amendment.run_id
    ):
        raise HDRealPilotError("parser_recovery_failure_state_mismatch")
    if _sha256(config_path) != amendment.pilot_config_sha256 or _sha256(authorization_path) != amendment.execution_authorization_sha256:
        raise HDRealPilotError("parser_recovery_binding_mismatch")
    run_dir = root / amendment.run_id
    ledger_path = run_dir / "request_ledger.jsonl"
    checkpoint_path = run_dir / "agent_checkpoint.json"
    receipt_path = root / "hd_parser_recovery_amendment.json"
    if checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            entries = checkpoint.get("entries", [])
        except (OSError, ValueError, TypeError) as exc:
            raise HDRealPilotError("parser_recovery_checkpoint_invalid") from exc
        if any(
            item.get("fingerprint") == amendment.fingerprint
            or (item.get("agent_id") == amendment.agent_id and item.get("round_id") == amendment.round_id)
            for item in entries if isinstance(item, dict)
        ):
            raise HDRealPilotError("parser_recovery_checkpoint_already_complete")
    entries = RequestLedger(ledger_path).entries()
    matching = [item for item in entries if item.fingerprint == amendment.fingerprint]
    if not matching or matching[-1].status is not RequestLedgerStatus.COMPLETED:
        raise HDRealPilotError("parser_recovery_completed_fingerprint_missing")
    terminal = matching[-1]
    if (
        terminal.agent_id != amendment.agent_id or terminal.round_id != amendment.round_id
        or terminal.phase != amendment.phase or terminal.condition != amendment.condition
    ):
        raise HDRealPilotError("parser_recovery_coordinate_mismatch")
    receipt = {
        "status": "approved_single_attempt", "amendment_id": amendment.amendment_id,
        "amendment_version": amendment.amendment_version, "batch_id": record.batch_id,
        "run_id": amendment.run_id, "scenario_id": amendment.scenario_id,
        "condition": amendment.condition, "seed": amendment.seed, "agent_id": amendment.agent_id,
        "round_id": amendment.round_id, "phase": amendment.phase, "fingerprint": amendment.fingerprint,
        "original_error_code": amendment.original_error_code, "recovery_kind": amendment.recovery_kind,
        "allowed_additional_transport_attempts": 1, "pilot_config_sha256": amendment.pilot_config_sha256,
        "execution_authorization_sha256": amendment.execution_authorization_sha256,
        "prompt_unchanged": True, "generation_parameters_unchanged": True,
        "logical_request_cap_unchanged": True, "development_only": True,
        "not_paper_result": True, "no_causal_conclusion": True,
    }
    if receipt_path.exists():
        try:
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HDRealPilotError("parser_recovery_receipt_invalid") from exc
        if existing != receipt:
            raise HDRealPilotError("parser_recovery_receipt_mismatch")
    else:
        if _sha256(batch_path) != amendment.original_batch_record_sha256:
            raise HDRealPilotError("parser_recovery_batch_hash_mismatch")
        if _sha256(ledger_path) != amendment.original_request_ledger_sha256:
            raise HDRealPilotError("parser_recovery_ledger_hash_mismatch")
        _atomic_json(receipt_path, receipt)


def _safe_payload(value: object) -> None:
    if isinstance(value, dict):
        if _FORBIDDEN_KEYS.intersection(str(key).lower() for key in value):
            raise HDRealPilotError("unsafe_result_field")
        for child in value.values():
            _safe_payload(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _safe_payload(child)


def _atomic_json(path: Path, value: BaseModel | Mapping[str, object]) -> None:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
    _safe_payload(payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _binding(*, config_path: Path, approval_path: Path, authorization_path: Path, config: ProvenanceCascadeHDConfig, model_name: str) -> str:
    return _json_sha({
        "config_sha256": _sha256(config_path),
        "approval_sha256": _sha256(approval_path),
        "authorization_sha256": _sha256(authorization_path),
        "amendment_sha256": config.amendment_sha256,
        "preregistration_sha256": config.preregistration_sha256,
        "policy_sha256": config.policy_sha256,
        "model_name": model_name,
        "provider": config.provider.model_dump(mode="json"),
        "request_cap": config.budget.request_cap,
        "completion_reservation_cap": config.budget.completion_reservation_cap,
    })


def _initial_record(config: ProvenanceCascadeHDConfig, *, binding: str, config_sha: str, approval_sha: str, authorization_sha: str, model_name: str) -> HDPilotBatchRecord:
    now = _now()
    runs = tuple(HDPilotRunState(
        run_id=item.run_id, matched_group_id=item.matched_group_id, scenario_id=item.scenario_id,
        condition=item.condition.value, seed=item.seed,
    ) for item in config.runs)
    return HDPilotBatchRecord(
        batch_id="provenance-cascade-pilot-hd-v2", status=HDPilotBatchStatus.RUNNING,
        binding_sha256=binding, config_sha256=config_sha, approval_sha256=approval_sha,
        authorization_sha256=authorization_sha, amendment_sha256=config.amendment_sha256,
        model_name=model_name, run_count=48, matched_group_count=12, request_cap=864,
        completion_reservation_cap=221184, logical_request_count=0,
        completion_reserved_token_count=0, completed_run_count=0, failed_run_count=0,
        planned_run_count=48, runs=runs, created_at=now, updated_at=now,
    )


def _replace_run(record: HDPilotBatchRecord, state: HDPilotRunState, **updates: object) -> HDPilotBatchRecord:
    replacement = state.model_copy(update=updates)
    runs = tuple(replacement if item.run_id == state.run_id else item for item in record.runs)
    return record.model_copy(update={"runs": runs})


def _ledger_totals(config: ProvenanceCascadeHDConfig, root: Path) -> tuple[int, int, int | None, int | None, int | None]:
    logical = reserved = 0
    prompt_values: list[int] = []
    completion_values: list[int] = []
    total_values: list[int] = []
    usage_unknown = False
    for spec in config.runs:
        path = root / spec.output_relative_path / config.output.ledger_filename
        if not path.exists():
            continue
        summary = RequestLedger(path).summary(
            request_cap=spec.expected_provider_requests,
            completion_reservation_cap=spec.completion_reservation,
        )
        logical += summary.unique_logical_request_count
        reserved += summary.completion_reserved_token_count
        if summary.actual_prompt_token_count is None or summary.actual_completion_token_count is None or summary.actual_total_token_count is None:
            if summary.completed_count:
                usage_unknown = True
        else:
            prompt_values.append(summary.actual_prompt_token_count)
            completion_values.append(summary.actual_completion_token_count)
            total_values.append(summary.actual_total_token_count)
    return logical, reserved, None if usage_unknown else sum(prompt_values), None if usage_unknown else sum(completion_values), None if usage_unknown else sum(total_values)


def _refresh(record: HDPilotBatchRecord, config: ProvenanceCascadeHDConfig, root: Path, *, status: HDPilotBatchStatus | None = None, current_run_id: str | None = None, failure_code: str | None = None) -> HDPilotBatchRecord:
    logical, reserved, prompt, completion, total = _ledger_totals(config, root)
    if logical > 864 or reserved > 221184:
        raise HDRealPilotError("batch_cap_exceeded")
    completed = sum(item.status is HDPilotRunStatus.COMPLETED for item in record.runs)
    failed = sum(item.status is HDPilotRunStatus.FAILED for item in record.runs)
    return record.model_copy(update={
        "status": status or record.status, "current_run_id": current_run_id,
        "failure_code": failure_code, "logical_request_count": logical,
        "completion_reserved_token_count": reserved, "actual_prompt_token_count": prompt,
        "actual_completion_token_count": completion, "actual_total_token_count": total,
        "completed_run_count": completed, "failed_run_count": failed,
        "planned_run_count": 48 - completed - failed, "updated_at": _now(),
    })


def _summary(record: HDPilotBatchRecord, root: Path) -> HDPilotExecutionSummary:
    status = "completed" if record.status is HDPilotBatchStatus.COMPLETED else "failed"
    replay = "passed" if status == "completed" and all(item.replay_status == "passed" for item in record.runs) else "incomplete"
    output_root = str(root.relative_to(_REPO_ROOT)) if root.is_relative_to(_REPO_ROOT) else str(root)
    return HDPilotExecutionSummary(
        status=status, network="enabled", provider_constructed=True, output_root=output_root,
        run_count=48, completed_run_count=record.completed_run_count, failed_run_count=record.failed_run_count,
        logical_request_count=record.logical_request_count, request_cap=864,
        completion_reserved_token_count=record.completion_reserved_token_count,
        completion_reservation_cap=221184, actual_prompt_token_count=record.actual_prompt_token_count,
        actual_completion_token_count=record.actual_completion_token_count,
        actual_total_token_count=record.actual_total_token_count, replay_status=replay,
        directive_applied_count=sum(item.directive_applied_count for item in record.runs), error_code=record.failure_code,
    )


def _blocked(code: str, output_root: str) -> HDPilotExecutionSummary:
    return HDPilotExecutionSummary(
        status="blocked", network="disabled", provider_constructed=False, output_root=output_root,
        run_count=48, completed_run_count=0, failed_run_count=0, logical_request_count=0,
        request_cap=864, completion_reserved_token_count=0, completion_reservation_cap=221184,
        replay_status="not_run", directive_applied_count=0, error_code=code,
    )


def execute_hd_real_pilot(
    *,
    config_path: str | Path = _DEFAULT_CONFIG,
    approval_path: str | Path = _DEFAULT_APPROVAL,
    authorization_path: str | Path = _DEFAULT_AUTHORIZATION,
    allow_network: bool = False,
    confirm_run: bool = False,
    confirm_request_cap: int | None = None,
    confirm_completion_reservation_cap: int | None = None,
    resume: bool = False,
    parser_recovery_path: str | Path | None = None,
    confirm_parser_recovery: bool = False,
    environment: Mapping[str, str] | None = None,
    provider_factory: Callable[[ProviderConfig], LLMProvider] | None = None,
    output_root_for_testing: str | Path | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> HDPilotExecutionSummary:
    config_file = Path(config_path).resolve()
    approval_file = Path(approval_path).resolve()
    authorization_file = Path(authorization_path).resolve()
    try:
        config, _ = load_hd_config(config_file)
        approval, approved_config, _, _ = load_hd_approval(approval_file)
        authorization = load_execution_authorization(authorization_file)
        if config != approved_config or authorization.pilot_config_sha256 != _sha256(config_file):
            raise HDRealPilotError("approved_config_mismatch")
        if authorization.design_approval_sha256 != _sha256(approval_file) or authorization.amendment_sha256 != config.amendment_sha256:
            raise HDRealPilotError("authorization_binding_mismatch")
        if not allow_network or not confirm_run:
            return _blocked("explicit_network_and_run_confirmation_required", config.output.results_root)
        if confirm_request_cap != 864:
            return _blocked("request_cap_confirmation_mismatch", config.output.results_root)
        if confirm_completion_reservation_cap != 221184:
            return _blocked("completion_reservation_cap_confirmation_mismatch", config.output.results_root)
        source = os.environ if environment is None else environment
        if not source.get(config.provider.model_env_var):
            return _blocked("provider_model_missing", config.output.results_root)
        if not source.get(config.provider.base_url_env_var):
            return _blocked("provider_base_url_missing", config.output.results_root)
        if not source.get(config.provider.api_key_env_var):
            return _blocked("provider_api_key_missing", config.output.results_root)
        model_name = str(source[config.provider.model_env_var])
        base_url = str(source[config.provider.base_url_env_var])
        root = Path(output_root_for_testing).resolve() if output_root_for_testing is not None else (_REPO_ROOT / config.output.results_root).resolve()
        if output_root_for_testing is None:
            try:
                root.relative_to(_REPO_ROOT / "results")
            except ValueError as exc:
                raise HDRealPilotError("output_root_invalid") from exc
        batch_path = root / config.output.batch_record_filename
        binding = _binding(config_path=config_file, approval_path=approval_file, authorization_path=authorization_file, config=config, model_name=model_name)
        recovery: HDParserRecoveryAmendment | None = None
        if root.exists() and not resume:
            return _blocked("output_root_exists", config.output.results_root)
        if resume:
            if not batch_path.exists():
                return _blocked("resume_batch_record_missing", config.output.results_root)
            try:
                record = HDPilotBatchRecord.model_validate_json(batch_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise HDRealPilotError("batch_record_invalid") from exc
            if record.binding_sha256 != binding:
                return _blocked("resume_binding_mismatch", config.output.results_root)
            if record.status is HDPilotBatchStatus.COMPLETED:
                return _summary(record, root)
            if confirm_parser_recovery:
                if parser_recovery_path is None:
                    return _blocked("parser_recovery_amendment_required", config.output.results_root)
                recovery = load_parser_recovery_amendment(parser_recovery_path)
                _prepare_parser_recovery(
                    recovery, record=record, root=root, batch_path=batch_path,
                    config_path=config_file, authorization_path=authorization_file,
                )
            elif parser_recovery_path is not None:
                return _blocked("parser_recovery_confirmation_required", config.output.results_root)
        else:
            root.mkdir(parents=True, exist_ok=False)
            record = _initial_record(
                config, binding=binding, config_sha=_sha256(config_file), approval_sha=_sha256(approval_file),
                authorization_sha=_sha256(authorization_file), model_name=model_name,
            )
            _atomic_json(batch_path, record)

        material_by_id = {item.scenario_id: item for item in config.scenario_materials}
        runner = CascadeRealAgentRunner()
        for spec in config.runs:
            state = next(item for item in record.runs if item.run_id == spec.run_id)
            run_dir = root / spec.output_relative_path
            run_record_path = run_dir / "run_record.json"
            if state.status is HDPilotRunStatus.COMPLETED:
                if not run_record_path.exists():
                    raise HDRealPilotError("completed_run_record_missing")
                continue
            if run_record_path.exists():
                existing = CascadeRealAgentRunRecord.model_validate_json(run_record_path.read_text(encoding="utf-8"))
                if existing.run_id != spec.run_id or existing.scenario_id != spec.scenario_id or existing.seed != spec.seed or existing.condition != spec.condition:
                    raise HDRealPilotError("existing_run_record_binding_mismatch")
                state = _replace_run(record, state, status=HDPilotRunStatus.COMPLETED,
                    logical_request_count=existing.logical_request_count,
                    completion_reserved_token_count=spec.completion_reservation,
                    replay_status=existing.replay.status.value if existing.replay else None,
                    directive_applied_count=existing.directive_applied_count,
                    run_record_sha256=_sha256(run_record_path), error_code=None)
                record = _refresh(state, config, root)
                _atomic_json(batch_path, record)
                continue
            if run_dir.exists() and not resume:
                return _blocked("run_output_exists", config.output.results_root)
            run_dir.mkdir(parents=True, exist_ok=True)
            record = _replace_run(record, state, status=HDPilotRunStatus.RUNNING, error_code=None)
            record = _refresh(record, config, root, status=HDPilotBatchStatus.RUNNING, current_run_id=spec.run_id)
            _atomic_json(batch_path, record)
            if progress:
                progress({"event": "run_started", "run_id": spec.run_id, "completed_run_count": record.completed_run_count, "network": "enabled"})
            provider_config = ProviderConfig(
                base_url=base_url, model_name=model_name, timeout_seconds=config.provider.timeout_seconds,
                max_retries=config.provider.max_retries, allow_network=True,
                temperature=config.provider.agent_temperature, max_tokens=config.provider.agent_max_tokens,
                seed=spec.seed,
            )
            provider = provider_factory(provider_config) if provider_factory else OpenAICompatibleProvider(provider_config, environment=source)
            scenario_ref = material_by_id[spec.scenario_id]
            scenario = CascadeScenarioLoader.load(config_file.parent / scenario_ref.config_path)
            try:
                run_record = runner.run_scenario(
                    scenario, spec.seed, spec.condition, provider=provider, run_id=spec.run_id,
                    ledger_path=run_dir / config.output.ledger_filename,
                    checkpoint_path=run_dir / "agent_checkpoint.json", model_name=model_name,
                    agent_temperature=config.provider.agent_temperature, agent_max_tokens=config.provider.agent_max_tokens,
                    request_cap=spec.expected_provider_requests,
                    completion_reservation_cap=spec.completion_reservation, resume=resume,
                    recovery_fingerprints={recovery.fingerprint} if recovery and spec.run_id == recovery.run_id else None,
                    recovery_kind=recovery.recovery_kind if recovery and spec.run_id == recovery.run_id else None,
                    max_recovery_attempts=recovery.allowed_additional_transport_attempts if recovery and spec.run_id == recovery.run_id else 0,
                )
                if run_record.logical_request_count != 18 or run_record.replay is None or run_record.replay.status.value != "passed":
                    raise HDRealPilotError("run_integrity_check_failed")
                safe_record = run_record.model_copy(update={"request_ledger_path": config.output.ledger_filename})
                _atomic_json(run_record_path, safe_record)
                ledger_summary = RequestLedger(run_dir / config.output.ledger_filename).summary(
                    request_cap=18, completion_reservation_cap=4608,
                )
                state = next(item for item in record.runs if item.run_id == spec.run_id)
                record = _replace_run(record, state, status=HDPilotRunStatus.COMPLETED,
                    logical_request_count=ledger_summary.unique_logical_request_count,
                    completion_reserved_token_count=ledger_summary.completion_reserved_token_count,
                    actual_prompt_token_count=ledger_summary.actual_prompt_token_count,
                    actual_completion_token_count=ledger_summary.actual_completion_token_count,
                    actual_total_token_count=ledger_summary.actual_total_token_count,
                    replay_status="passed", directive_applied_count=run_record.directive_applied_count,
                    run_record_sha256=_sha256(run_record_path), error_code=None)
                record = _refresh(record, config, root, status=HDPilotBatchStatus.RUNNING, current_run_id=None)
                _atomic_json(batch_path, record)
                if progress:
                    progress({"event": "run_completed", "run_id": spec.run_id, "completed_run_count": record.completed_run_count, "logical_request_count": record.logical_request_count, "network": "enabled"})
            except (CascadeRealAgentRunError, HDRealPilotError) as exc:
                code = exc.code
                state = next(item for item in record.runs if item.run_id == spec.run_id)
                record = _replace_run(record, state, status=HDPilotRunStatus.FAILED, error_code=code)
                record = _refresh(record, config, root, status=HDPilotBatchStatus.FAILED, current_run_id=spec.run_id, failure_code=code)
                _atomic_json(batch_path, record)
                if progress:
                    progress({"event": "run_failed", "run_id": spec.run_id, "error_code": code, "completed_run_count": record.completed_run_count, "network": "enabled"})
                return _summary(record, root)

        record = _refresh(record, config, root, status=HDPilotBatchStatus.COMPLETED, current_run_id=None, failure_code=None)
        if record.completed_run_count != 48 or record.logical_request_count != 864 or any(item.replay_status != "passed" for item in record.runs):
            raise HDRealPilotError("batch_integrity_check_failed")
        _atomic_json(batch_path, record)
        receipt = {
            "status": "completed_development_pilot", "batch_id": record.batch_id,
            "binding_sha256": record.binding_sha256, "config_sha256": record.config_sha256,
            "approval_sha256": record.approval_sha256, "authorization_sha256": record.authorization_sha256,
            "amendment_sha256": record.amendment_sha256, "model_name": record.model_name,
            "run_count": 48, "matched_group_count": 12, "logical_request_count": record.logical_request_count,
            "request_cap": 864, "completion_reserved_token_count": record.completion_reserved_token_count,
            "completion_reservation_cap": 221184, "actual_prompt_token_count": record.actual_prompt_token_count,
            "actual_completion_token_count": record.actual_completion_token_count,
            "actual_total_token_count": record.actual_total_token_count, "replay_status": "passed",
            "development_only": True, "pilot_only": True, "not_paper_result": True,
            "no_causal_conclusion": True, "evaluator_truth_loaded": False,
        }
        receipt_path = root / "pilot_batch_receipt.json"
        if receipt_path.exists():
            raise HDRealPilotError("pilot_receipt_exists")
        _atomic_json(receipt_path, receipt)
        return _summary(record, root)
    except (HDPilotError, HDFinalPreflightError, HDRealPilotError) as exc:
        return _blocked(exc.code, "results/provenance-cascade-pilot-hd-v2")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the explicitly authorized H-D provenance-cascade pilot.")
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG))
    parser.add_argument("--approval", default=str(_DEFAULT_APPROVAL))
    parser.add_argument("--authorization", default=str(_DEFAULT_AUTHORIZATION))
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--confirm-request-cap", type=int)
    parser.add_argument("--confirm-completion-reservation-cap", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--parser-recovery-amendment")
    parser.add_argument("--confirm-parser-recovery", action="store_true")
    args = parser.parse_args(argv)

    def emit(event: dict[str, object]) -> None:
        print(json.dumps(event, ensure_ascii=True, sort_keys=True), flush=True)

    summary = execute_hd_real_pilot(
        config_path=args.config, approval_path=args.approval, authorization_path=args.authorization,
        allow_network=args.allow_network, confirm_run=args.confirm_run,
        confirm_request_cap=args.confirm_request_cap,
        confirm_completion_reservation_cap=args.confirm_completion_reservation_cap,
        resume=args.resume, parser_recovery_path=args.parser_recovery_amendment,
        confirm_parser_recovery=args.confirm_parser_recovery, progress=emit,
    )
    print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=True, sort_keys=True), flush=True)
    return 0 if summary.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HDPilotBatchRecord", "HDPilotExecutionSummary", "HDPilotRunState", "HDNetworkExecutionAuthorization",
    "execute_hd_real_pilot", "load_execution_authorization", "main",
]
