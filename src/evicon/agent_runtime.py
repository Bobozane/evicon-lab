"""Isolated provider boundary for one public agent turn."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .agent_prompts import AGENT_TURN_TEMPLATE_VERSION, AgentPromptContext, render_agent_turn
from .agent_response import AgentResponse, AgentResponseError, parse_agent_response
from .llm_contract import LLMProvider, LLMProviderError
from .models._validation import normalized_text


class AgentRuntimeStatus(str, Enum):
    """Stable outcomes for a single agent provider call."""

    COMPLETED = "completed"
    RENDER_ERROR = "render_error"
    PROVIDER_ERROR = "provider_error"
    PARSER_INVALID = "parser_invalid"


class AgentRuntimeErrorCode(str, Enum):
    """Runtime-owned failure codes that never include raw exception text."""

    REQUEST_RENDER_FAILED = "request_render_failed"
    PARSER_FAILURE = "parser_failure"


class AgentRequestSettings(BaseModel):
    """Non-secret, reproducible request settings for a runtime instance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str = Field(default="agent-runtime-v1", min_length=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
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


class AgentAuditSummary(BaseModel):
    """Safe audit data that deliberately excludes prompts, messages, and metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: AgentRuntimeStatus
    agent_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    template_version: str | None = None
    model_name: str
    request_id: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    parser_valid: bool | None = None
    evidence_ids_used: list[str] = Field(default_factory=list)
    error_code: str | None = None


class AgentRuntimeResult(BaseModel):
    """In-memory runtime outcome; no method writes a turn or an event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: AgentRuntimeStatus
    response: AgentResponse | None = None
    audit: AgentAuditSummary

    def audit_summary(self) -> AgentAuditSummary:
        """Return a safe summary that intentionally omits the agent message."""
        return self.audit


class AgentRuntime:
    """Render, complete, and parse one public agent context without state effects."""

    def __init__(self, request_settings: AgentRequestSettings | None = None) -> None:
        self._request_settings = request_settings or AgentRequestSettings()

    def execute(
        self,
        public_context: AgentPromptContext,
        provider: LLMProvider,
    ) -> AgentRuntimeResult:
        """Use the fixed render -> provider -> parse sequence exactly once."""
        try:
            request = render_agent_turn(
                public_context,
                model_name=self._request_settings.model_name,
                temperature=self._request_settings.temperature,
                max_tokens=self._request_settings.max_tokens,
                seed=self._request_settings.seed,
            )
        except (TypeError, ValueError):
            return self._result(
                public_context,
                status=AgentRuntimeStatus.RENDER_ERROR,
                parser_valid=None,
                error_code=AgentRuntimeErrorCode.REQUEST_RENDER_FAILED.value,
            )

        try:
            response = provider.complete(request)
        except LLMProviderError as error:
            return self._result(
                public_context,
                status=AgentRuntimeStatus.PROVIDER_ERROR,
                template_version=AGENT_TURN_TEMPLATE_VERSION,
                request_id=request.request_id,
                parser_valid=None,
                error_code=error.code.value,
            )
        except Exception:
            return self._result(
                public_context,
                status=AgentRuntimeStatus.PROVIDER_ERROR,
                template_version=AGENT_TURN_TEMPLATE_VERSION,
                request_id=request.request_id,
                parser_valid=None,
                error_code="provider_failure",
            )

        try:
            parsed = parse_agent_response(response.content, public_context)
        except Exception:
            return self._result(
                public_context,
                status=AgentRuntimeStatus.PARSER_INVALID,
                template_version=AGENT_TURN_TEMPLATE_VERSION,
                request_id=request.request_id,
                model_name=response.model_name,
                finish_reason=response.finish_reason,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                latency_ms=response.latency_ms,
                parser_valid=False,
                error_code=AgentRuntimeErrorCode.PARSER_FAILURE.value,
            )

        return self._result(
            public_context,
            status=(AgentRuntimeStatus.COMPLETED if parsed.valid else AgentRuntimeStatus.PARSER_INVALID),
            response=parsed,
            template_version=AGENT_TURN_TEMPLATE_VERSION,
            request_id=request.request_id,
            model_name=response.model_name,
            finish_reason=response.finish_reason,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_ms=response.latency_ms,
            parser_valid=parsed.valid,
            evidence_ids_used=parsed.evidence_ids_used if parsed.valid else [],
            error_code=_parser_error_code(parsed),
        )

    def _result(
        self,
        public_context: AgentPromptContext,
        *,
        status: AgentRuntimeStatus,
        parser_valid: bool | None,
        response: AgentResponse | None = None,
        template_version: str | None = None,
        request_id: str | None = None,
        model_name: str | None = None,
        finish_reason: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        latency_ms: float | None = None,
        evidence_ids_used: list[str] | None = None,
        error_code: str | None = None,
    ) -> AgentRuntimeResult:
        return AgentRuntimeResult(
            status=status,
            response=response,
            audit=AgentAuditSummary(
                status=status,
                agent_id=public_context.agent_id,
                round_id=public_context.round_id,
                template_version=template_version,
                model_name=model_name or self._request_settings.model_name,
                request_id=request_id,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                parser_valid=parser_valid,
                evidence_ids_used=list(evidence_ids_used or []),
                error_code=error_code,
            ),
        )


def _parser_error_code(parsed: AgentResponse) -> str | None:
    if parsed.valid or not parsed.validation_errors:
        return None
    error: AgentResponseError = parsed.validation_errors[0]
    return error.value


__all__ = [
    "AgentAuditSummary",
    "AgentRequestSettings",
    "AgentRuntime",
    "AgentRuntimeErrorCode",
    "AgentRuntimeResult",
    "AgentRuntimeStatus",
]
