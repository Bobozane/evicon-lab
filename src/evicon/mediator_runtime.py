"""Isolated runtime boundary for one fixed, already validated mediator plan."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .executor import InterventionPlan
from .llm_contract import LLMProvider, LLMProviderError
from .mediator_prompts import PromptContext, render_action_instruction
from .mediator_response import MediatorResponse, MediatorResponseError, parse_mediator_response
from .models import InterventionAction
from .models._validation import normalized_text


class MediatorRuntimeStatus(str, Enum):
    """Stable outcomes for a single mediator invocation attempt."""

    NOT_INVOKED = "not_invoked"
    COMPLETED = "completed"
    REJECTED = "rejected"
    PROVIDER_ERROR = "provider_error"
    PARSER_INVALID = "parser_invalid"


class MediatorRuntimeErrorCode(str, Enum):
    """Runtime-owned failures that do not expose implementation details."""

    INVALID_PLAN = "invalid_plan"
    CONTEXT_MISMATCH = "context_mismatch"
    REQUEST_RENDER_FAILED = "request_render_failed"
    PARSER_FAILURE = "parser_failure"


class MediatorRequestSettings(BaseModel):
    """Non-secret request settings fixed when a runtime instance is created."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str = Field(default="mediator-runtime-v1", min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, gt=0)
    seed: int = 0

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        return normalized_text(value, "model_name")

    @field_validator("seed", mode="before")
    @classmethod
    def validate_seed(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("seed must be an integer")
        return value


class MediatorAuditSummary(BaseModel):
    """Safe audit view that intentionally excludes prompts and mediator text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: MediatorRuntimeStatus
    action: InterventionAction
    plan_id: str = Field(min_length=1)
    template_version: str | None = None
    model_name: str
    request_id: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    parser_valid: bool | None = None
    error_code: str | None = None


class MediatorRuntimeResult(BaseModel):
    """In-memory outcome of one fixed-plan mediator call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: MediatorRuntimeStatus
    response: MediatorResponse | None = None
    audit: MediatorAuditSummary

    def audit_summary(self) -> MediatorAuditSummary:
        """Return the safe, serializable summary without the mediator message."""
        return self.audit


class MediatorRuntime:
    """Run a fixed plan without selecting policy, applying controls, or writing files."""

    def __init__(self, request_settings: MediatorRequestSettings | None = None) -> None:
        self._request_settings = request_settings or MediatorRequestSettings()

    def execute(
        self,
        plan: InterventionPlan,
        public_context: PromptContext,
        provider: LLMProvider,
    ) -> MediatorRuntimeResult:
        """Render, complete, and parse exactly once for a preselected plan."""
        if plan.action is InterventionAction.NO_OP:
            return self._result(
                plan,
                status=MediatorRuntimeStatus.NOT_INVOKED,
                parser_valid=None,
            )
        if not plan.valid:
            return self._result(
                plan,
                status=MediatorRuntimeStatus.REJECTED,
                parser_valid=None,
                error_code=MediatorRuntimeErrorCode.INVALID_PLAN.value,
            )
        if not _context_matches(plan, public_context):
            return self._result(
                plan,
                status=MediatorRuntimeStatus.REJECTED,
                parser_valid=None,
                error_code=MediatorRuntimeErrorCode.CONTEXT_MISMATCH.value,
            )

        try:
            request = render_action_instruction(
                plan,
                public_context,
                model_name=self._request_settings.model_name,
                temperature=self._request_settings.temperature,
                max_tokens=self._request_settings.max_tokens,
                seed=self._request_settings.seed,
            )
        except (TypeError, ValueError):
            return self._result(
                plan,
                status=MediatorRuntimeStatus.REJECTED,
                parser_valid=None,
                error_code=MediatorRuntimeErrorCode.REQUEST_RENDER_FAILED.value,
            )
        if request is None:
            return self._result(
                plan,
                status=MediatorRuntimeStatus.NOT_INVOKED,
                parser_valid=None,
            )

        template_version = _template_version(request.metadata)
        try:
            response = provider.complete(request)
        except LLMProviderError as error:
            return self._result(
                plan,
                status=MediatorRuntimeStatus.PROVIDER_ERROR,
                template_version=template_version,
                request_id=request.request_id,
                parser_valid=None,
                error_code=error.code.value,
            )
        except Exception:
            return self._result(
                plan,
                status=MediatorRuntimeStatus.PROVIDER_ERROR,
                template_version=template_version,
                request_id=request.request_id,
                parser_valid=None,
                error_code="provider_failure",
            )

        try:
            parsed = parse_mediator_response(response.content, plan, public_context)
        except Exception:
            return self._result(
                plan,
                status=MediatorRuntimeStatus.PARSER_INVALID,
                template_version=template_version,
                request_id=request.request_id,
                model_name=response.model_name,
                finish_reason=response.finish_reason,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                latency_ms=response.latency_ms,
                parser_valid=False,
                error_code=MediatorRuntimeErrorCode.PARSER_FAILURE.value,
            )

        error_code = _parser_error_code(parsed)
        return self._result(
            plan,
            status=(
                MediatorRuntimeStatus.COMPLETED
                if parsed.valid
                else MediatorRuntimeStatus.PARSER_INVALID
            ),
            response=parsed,
            template_version=template_version,
            request_id=request.request_id,
            model_name=response.model_name,
            finish_reason=response.finish_reason,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_ms=response.latency_ms,
            parser_valid=parsed.valid,
            error_code=error_code,
        )

    def _result(
        self,
        plan: InterventionPlan,
        *,
        status: MediatorRuntimeStatus,
        parser_valid: bool | None,
        response: MediatorResponse | None = None,
        template_version: str | None = None,
        request_id: str | None = None,
        model_name: str | None = None,
        finish_reason: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        latency_ms: float | None = None,
        error_code: str | None = None,
    ) -> MediatorRuntimeResult:
        return MediatorRuntimeResult(
            status=status,
            response=response,
            audit=MediatorAuditSummary(
                status=status,
                action=plan.action,
                plan_id=plan.plan_id,
                template_version=template_version,
                model_name=model_name or self._request_settings.model_name,
                request_id=request_id,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                parser_valid=parser_valid,
                error_code=error_code,
            ),
        )


def _context_matches(plan: InterventionPlan, context: PromptContext) -> bool:
    return (
        plan.run_id == context.run_id
        and plan.scenario_id == context.scenario_id
        and plan.round_id == context.round_id
        and plan.action is context.action
        and plan.target_agent_ids == context.target_agent_ids
    )


def _template_version(metadata: object) -> str | None:
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("template_version")
    return value if isinstance(value, str) else None


def _parser_error_code(parsed: MediatorResponse) -> str | None:
    if parsed.valid or not parsed.validation_errors:
        return None
    error: MediatorResponseError = parsed.validation_errors[0]
    return error.value


__all__ = [
    "MediatorAuditSummary",
    "MediatorRequestSettings",
    "MediatorRuntime",
    "MediatorRuntimeErrorCode",
    "MediatorRuntimeResult",
    "MediatorRuntimeStatus",
]
