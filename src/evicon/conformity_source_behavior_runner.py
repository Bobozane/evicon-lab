"""Bounded 12-case runner for source-behavior qualification v2.

This runner persists only the preregistered safe semantic audits and one
redacted completion receipt. It has no resume or overwrite path.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_identification import IdentificationError, safe_json, sha256_file
from .conformity_source_behavior_compatibility import (
    DEFAULT_RECEIPT as DEFAULT_COMPATIBILITY_RECEIPT,
    RESPONSE_SCHEMA,
    RESPONSE_SCHEMA_SHA256,
    CompatibilityGenerationParameters,
    load_receipt as load_compatibility_receipt,
)
from .conformity_source_behavior_execution_approval import (
    DEFAULT_EXECUTION_APPROVAL,
    DEFAULT_RUNNER,
    execution_approval_sha256,
    load_execution_approval,
    safe_preflight,
)
from .conformity_source_behavior_qualification import (
    DEFAULT_CONFIG,
    SCHEMA_NAME,
    SafeBehaviorCaseAudit,
    load_config,
    parse_behavior_response,
    render_behavior_request,
    safe_case_audit,
)
from .conformity_source_behavior_qualification_approval import (
    DEFAULT_APPROVAL as DEFAULT_SOURCE_APPROVAL,
    approval_sha256,
)
from .conformity_source_behavior_qualification_smoke import build_cases
from .llm_contract import LLMProvider, LLMProviderError, LLMResponse
from .openai_provider import (
    OpenAICompatibleProvider,
    ProviderConfig,
    ReasoningEffort,
    ResponseFormatMode,
)

_ROOT = Path(__file__).resolve().parents[2]

RUNNER_VERSION = "conformity_source_behavior_runner.v2"
DEFAULT_OUTPUT_ROOT = "outputs/conformity-source-behavior-qualification-v2"
SAFE_AUDIT_NAME = "safe_case_audits.jsonl"
RECEIPT_NAME = "qualification_receipt.json"


class QualificationTokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def total_is_consistent(self) -> "QualificationTokenUsage":
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("qualification token usage total mismatch")
        return self


class QualificationRunSafety(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    public_behavior_only: Literal[True] = True
    text_matched_projections: Literal[True] = True
    evaluator_private_truth_exposed: Literal[False] = False
    historical_results_used: Literal[False] = False
    prompt_saved: Literal[False] = False
    full_response_saved: Literal[False] = False
    api_key_saved: Literal[False] = False
    headers_saved: Literal[False] = False
    provider_metadata_saved: Literal[False] = False
    request_ledger_written: Literal[False] = False
    behavior_effect_estimated: Literal[False] = False
    not_paper_result: Literal[True] = True
    no_causal_conclusion: Literal[True] = True


class QualificationRunReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt_version: Literal["conformity_source_behavior_qualification_receipt.v2"]
    runner_version: Literal["conformity_source_behavior_runner.v2"]
    status: Literal["completed"]
    model: Literal["gpt-5.6-luna"]
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runner_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    safe_case_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_parameters: CompatibilityGenerationParameters
    logical_request_count: Literal[12]
    transport_attempt_count: Literal[12]
    safe_case_audit_count: Literal[12]
    parser_invalid_count: Literal[0]
    projection_case_counts: dict[str, int]
    token_usage: QualificationTokenUsage
    latency_ms: float = Field(ge=0.0)
    safety: QualificationRunSafety

    @model_validator(mode="after")
    def exact_case_coverage(self) -> "QualificationRunReceipt":
        if self.projection_case_counts != {
            "source_free": 4,
            "same_root": 4,
            "independent_roots": 4,
        }:
            raise ValueError("qualification receipt projection coverage mismatch")
        return self


class QualificationRunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["network_disabled", "blocked", "completed", "failed"]
    error_code: str | None = None
    model: str | None = None
    logical_request_count: int = Field(default=0, ge=0, le=12)
    transport_attempt_count: int = Field(default=0, ge=0, le=12)
    completed_case_count: int = Field(default=0, ge=0, le=12)
    parser_invalid_count: int = Field(default=0, ge=0, le=1)
    safe_case_audit_count: int = Field(default=0, ge=0, le=12)
    output_root: str | None = None
    safe_case_audit_path: str | None = None
    receipt_path: str | None = None
    network: Literal["disabled", "enabled"] = "disabled"
    provider_constructed: bool = False
    api_key_read: bool = False
    outputs_written: bool = False
    safety: QualificationRunSafety = Field(default_factory=QualificationRunSafety)


class _CappedProvider:
    def __init__(self, provider: LLMProvider, *, request_cap: int, reservation_cap: int) -> None:
        self.provider = provider
        self.request_cap = request_cap
        self.reservation_cap = reservation_cap
        self.calls = 0
        self.reserved_tokens = 0

    def complete(self, request) -> LLMResponse:
        if self.calls >= self.request_cap:
            raise IdentificationError("source_behavior_request_cap_reached")
        if self.reserved_tokens + request.max_tokens > self.reservation_cap:
            raise IdentificationError("source_behavior_completion_reservation_cap_reached")
        self.calls += 1
        self.reserved_tokens += request.max_tokens
        return self.provider.complete(request)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def _safe_path(path: Path) -> str:
    try:
        return str(path.relative_to(_ROOT))
    except ValueError:
        return str(path)


def _ensure_new_output(path: str | Path) -> Path:
    target = _resolve(path)
    try:
        target.relative_to(_ROOT)
        inside_repository = True
    except ValueError:
        inside_repository = False
    if inside_repository and target != _resolve(DEFAULT_OUTPUT_ROOT):
        raise IdentificationError("source_behavior_output_path_mismatch")
    if target.exists():
        raise IdentificationError("source_behavior_output_exists")
    return target


def _append_audit(path: Path, audit: SafeBehaviorCaseAudit) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(audit.model_dump_json())
        handle.write("\n")


def _safe_error_code(error: Exception) -> str:
    code = getattr(error, "code", None)
    if hasattr(code, "value"):
        code = code.value
    if isinstance(code, str):
        return code
    if isinstance(error, IdentificationError):
        return str(error)
    return "source_behavior_qualification_failed"


def _read_safe_audits(path: Path) -> tuple[SafeBehaviorCaseAudit, ...]:
    try:
        return tuple(
            SafeBehaviorCaseAudit.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    except Exception as exc:
        raise IdentificationError("source_behavior_safe_audit_invalid") from exc


def _write_receipt(path: Path, receipt: QualificationRunReceipt) -> None:
    if path.exists():
        raise IdentificationError("source_behavior_receipt_exists")
    serialized = json.dumps(
        receipt.model_dump(mode="json"), ensure_ascii=True, sort_keys=True, indent=2
    ) + "\n"
    path.write_text(serialized, encoding="utf-8")


def load_qualification_receipt(
    path: str | Path,
    *,
    execution_approval_path: str | Path = DEFAULT_EXECUTION_APPROVAL,
) -> QualificationRunReceipt:
    receipt_path = _resolve(path)
    try:
        receipt = QualificationRunReceipt.model_validate_json(
            receipt_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise IdentificationError("source_behavior_qualification_receipt_invalid") from exc
    audit_path = receipt_path.parent / SAFE_AUDIT_NAME
    audits = _read_safe_audits(audit_path)
    expected_projection_counts = Counter(item.projection.value for item in audits)
    expected = {
        "config_sha256": sha256_file(DEFAULT_CONFIG),
        "protocol_sha256": load_config(DEFAULT_CONFIG).protocol_sha256,
        "source_approval_sha256": approval_sha256(DEFAULT_SOURCE_APPROVAL),
        "compatibility_receipt_sha256": sha256_file(DEFAULT_COMPATIBILITY_RECEIPT),
        "execution_approval_sha256": execution_approval_sha256(execution_approval_path),
        "runner_sha256": sha256_file(DEFAULT_RUNNER),
        "response_schema_sha256": RESPONSE_SCHEMA_SHA256,
        "safe_case_audit_sha256": sha256_file(audit_path),
        "projection_case_counts": dict(sorted(expected_projection_counts.items())),
    }
    observed = {name: getattr(receipt, name) for name in expected}
    if observed != expected or len(audits) != 12 or len({item.case_id for item in audits}) != 12:
        raise IdentificationError("source_behavior_qualification_receipt_binding_mismatch")
    return receipt


def run_behavior_qualification(
    provider: LLMProvider,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    execution_approval_path: str | Path = DEFAULT_EXECUTION_APPROVAL,
    network: Literal["disabled", "enabled"] = "disabled",
) -> QualificationRunSummary:
    """Run all 12 cases once; any failure leaves only preceding safe audits."""

    target: Path | None = None
    audit_path: Path | None = None
    capped: _CappedProvider | None = None
    completed = 0
    parser_invalid = 0
    try:
        approval = load_execution_approval(execution_approval_path)
        if approval.acceptance_status != "accepted":
            raise IdentificationError("behavior_qualification_execution_approval_required")
        target = _ensure_new_output(output_root)
        target.mkdir(parents=True, exist_ok=False)
        audit_path = target / SAFE_AUDIT_NAME
        capped = _CappedProvider(
            provider,
            request_cap=approval.logical_request_cap,
            reservation_cap=approval.completion_reservation_cap,
        )
        prompt_tokens = completion_tokens = total_tokens = 0
        latency_ms = 0.0
        model_name: str | None = None
        for base_context in build_cases(DEFAULT_CONFIG):
            context = base_context.model_copy(update={"model_name": approval.required_model_name})
            request = render_behavior_request(context)
            response = capped.complete(request)
            if response.model_name != approval.required_model_name:
                raise IdentificationError("source_behavior_provider_model_mismatch")
            if response.finish_reason != "stop":
                parser_invalid = 1
                raise IdentificationError("source_behavior_finish_reason_invalid")
            try:
                parsed = parse_behavior_response(response.content, context)
            except ValueError as exc:
                parser_invalid = 1
                raise IdentificationError("source_behavior_parser_invalid") from exc
            if any(value is None for value in (
                response.prompt_tokens,
                response.completion_tokens,
                response.total_tokens,
            )):
                raise IdentificationError("source_behavior_token_usage_missing")
            assert response.prompt_tokens is not None
            assert response.completion_tokens is not None
            assert response.total_tokens is not None
            _append_audit(audit_path, safe_case_audit(context, parsed))
            completed += 1
            prompt_tokens += response.prompt_tokens
            completion_tokens += response.completion_tokens
            total_tokens += response.total_tokens
            latency_ms += response.latency_ms
            model_name = response.model_name

        audits = _read_safe_audits(audit_path)
        if (
            capped.calls != 12
            or capped.reserved_tokens != 1536
            or len(audits) != 12
            or len({item.case_id for item in audits}) != 12
        ):
            raise IdentificationError("source_behavior_run_completeness_failed")
        projection_counts = Counter(item.projection.value for item in audits)
        if projection_counts != Counter({
            "source_free": 4,
            "same_root": 4,
            "independent_roots": 4,
        }):
            raise IdentificationError("source_behavior_projection_coverage_failed")
        assert model_name is not None
        receipt = QualificationRunReceipt(
            receipt_version="conformity_source_behavior_qualification_receipt.v2",
            runner_version=RUNNER_VERSION,
            status="completed",
            model=model_name,
            config_sha256=sha256_file(DEFAULT_CONFIG),
            protocol_sha256=load_config(DEFAULT_CONFIG).protocol_sha256,
            source_approval_sha256=approval_sha256(DEFAULT_SOURCE_APPROVAL),
            compatibility_receipt_sha256=sha256_file(DEFAULT_COMPATIBILITY_RECEIPT),
            execution_approval_sha256=execution_approval_sha256(execution_approval_path),
            runner_sha256=sha256_file(DEFAULT_RUNNER),
            response_schema_sha256=RESPONSE_SCHEMA_SHA256,
            safe_case_audit_sha256=sha256_file(audit_path),
            generation_parameters=CompatibilityGenerationParameters(),
            logical_request_count=12,
            transport_attempt_count=12,
            safe_case_audit_count=12,
            parser_invalid_count=0,
            projection_case_counts=dict(sorted(projection_counts.items())),
            token_usage=QualificationTokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            latency_ms=latency_ms,
            safety=QualificationRunSafety(),
        )
        receipt_path = target / RECEIPT_NAME
        _write_receipt(receipt_path, receipt)
        return QualificationRunSummary(
            status="completed",
            model=model_name,
            logical_request_count=12,
            transport_attempt_count=12,
            completed_case_count=12,
            safe_case_audit_count=12,
            output_root=_safe_path(target),
            safe_case_audit_path=_safe_path(audit_path),
            receipt_path=_safe_path(receipt_path),
            network=network,
            provider_constructed=True,
            api_key_read=network == "enabled",
            outputs_written=True,
        )
    except Exception as exc:
        calls = capped.calls if capped is not None else 0
        audit_count = len(_read_safe_audits(audit_path)) if audit_path is not None and audit_path.exists() else 0
        return QualificationRunSummary(
            status="failed" if capped is not None else "blocked",
            error_code=_safe_error_code(exc),
            logical_request_count=calls,
            transport_attempt_count=calls,
            completed_case_count=completed,
            parser_invalid_count=parser_invalid,
            safe_case_audit_count=audit_count,
            output_root=_safe_path(target) if target is not None else None,
            safe_case_audit_path=_safe_path(audit_path) if audit_path is not None and audit_path.exists() else None,
            network=network,
            provider_constructed=capped is not None,
            api_key_read=network == "enabled" and capped is not None,
            outputs_written=target is not None and target.exists(),
        )


def _require_provider_environment(
    environment: Mapping[str, str] | None,
    *,
    required_model: str,
) -> Mapping[str, str]:
    source = os.environ if environment is None else environment
    if not str(source.get("EVICON_LLM_BASE_URL") or "").strip():
        raise IdentificationError("missing_base_url")
    if not str(source.get("EVICON_LLM_MODEL") or "").strip():
        raise IdentificationError("missing_model")
    if not str(source.get("EVICON_LLM_API_KEY") or "").strip():
        raise IdentificationError("missing_api_key")
    if source.get("EVICON_LLM_MODEL") != required_model:
        raise IdentificationError("source_behavior_provider_model_mismatch")
    return source


def execute_behavior_qualification(
    *,
    allow_network: bool = False,
    confirm_run: bool = False,
    confirm_request_cap: int | None = None,
    confirm_completion_reservation_cap: int | None = None,
    environment: Mapping[str, str] | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    execution_approval_path: str | Path = DEFAULT_EXECUTION_APPROVAL,
) -> QualificationRunSummary | dict[str, object]:
    if not allow_network:
        return safe_preflight(execution_approval_path)
    if (
        not confirm_run
        or confirm_request_cap != 12
        or confirm_completion_reservation_cap != 1536
    ):
        return QualificationRunSummary(
            status="blocked",
            error_code="behavior_qualification_explicit_cap_confirmation_required",
        )
    try:
        approval = load_execution_approval(execution_approval_path)
        if approval.acceptance_status != "accepted":
            raise IdentificationError("behavior_qualification_execution_approval_required")
        _ensure_new_output(output_root)
        source = _require_provider_environment(
            environment,
            required_model=approval.required_model_name,
        )
        base = ProviderConfig.from_env(allow_network=True, environment=source)
        provider_config = base.model_copy(update={
            "allow_network": True,
            "timeout_seconds": approval.timeout_seconds,
            "max_retries": approval.max_retries,
            "temperature": approval.temperature,
            "max_tokens": approval.max_tokens,
            "seed": approval.seed,
            "reasoning_effort": ReasoningEffort.NONE,
            "response_format": ResponseFormatMode.JSON_SCHEMA,
            "response_schema_name": SCHEMA_NAME,
            "response_schema": RESPONSE_SCHEMA,
        })
        provider = OpenAICompatibleProvider(provider_config, environment=source)
        return run_behavior_qualification(
            provider,
            output_root=output_root,
            execution_approval_path=execution_approval_path,
            network="enabled",
        )
    except Exception as exc:
        return QualificationRunSummary(
            status="blocked",
            error_code=_safe_error_code(exc),
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the bounded source-behavior qualification v2"
    )
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--confirm-request-cap", type=int)
    parser.add_argument("--confirm-completion-reservation-cap", type=int)
    arguments = parser.parse_args(argv)
    result = execute_behavior_qualification(
        allow_network=arguments.allow_network,
        confirm_run=arguments.confirm_run,
        confirm_request_cap=arguments.confirm_request_cap,
        confirm_completion_reservation_cap=arguments.confirm_completion_reservation_cap,
    )
    payload = result.model_dump(mode="json") if isinstance(result, BaseModel) else result
    print(safe_json(payload))
    return 0 if payload.get("status") in {
        "completed", "source_behavior_execution_gate_offline_ready"
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "QualificationRunReceipt",
    "QualificationRunSafety",
    "QualificationRunSummary",
    "QualificationTokenUsage",
    "RECEIPT_NAME",
    "RUNNER_VERSION",
    "SAFE_AUDIT_NAME",
    "execute_behavior_qualification",
    "load_qualification_receipt",
    "run_behavior_qualification",
]
