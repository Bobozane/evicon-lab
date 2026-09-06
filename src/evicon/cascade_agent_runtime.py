"""Opt-in, one-call LLM boundary for provenance-cascade Agents.

The runtime is intentionally not connected to any cascade runner.  It only
renders one public context, performs one injected provider call, and parses a
strict public stance response.
"""

from __future__ import annotations

from enum import Enum
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .cascade_agent_prompts import CASCADE_AGENT_TEMPLATE_VERSION, CascadeAgentPromptContext, render_cascade_agent_turn
from .cascade_agent_response import CascadeAgentResponse, parse_cascade_agent_response
from .llm_contract import LLMProvider, LLMProviderError
from .request_ledger import RequestLedgerError


class CascadeAgentRuntimeStatus(str, Enum):
    COMPLETED = "completed"
    RENDER_ERROR = "render_error"
    PROVIDER_ERROR = "provider_error"
    PARSER_INVALID = "parser_invalid"


class CascadeAgentRuntimeAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CascadeAgentRuntimeStatus
    scenario_id: str
    agent_id: str
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
    content_ids_used_count: int = 0
    evidence_ids_used_count: int = 0
    share_requested: bool = False
    error_code: str | None = None


class CascadeAgentRuntimeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CascadeAgentRuntimeStatus
    response: CascadeAgentResponse | None = None
    audit: CascadeAgentRuntimeAudit

    def audit_summary(self) -> CascadeAgentRuntimeAudit:
        return self.audit


class CascadeAgentRuntime:
    def execute(
        self,
        context: CascadeAgentPromptContext,
        provider: LLMProvider,
        *,
        request_metadata: Mapping[str, Any] | None = None,
    ) -> CascadeAgentRuntimeResult:
        try:
            request = render_cascade_agent_turn(context)
            if request_metadata:
                # Only caller-supplied coordinate metadata is merged; prompt and
                # provider fields remain rendered by the existing contract.
                allowed = {
                    key: value for key, value in request_metadata.items()
                    if key in {"protocol", "condition", "phase", "run_id", "seed", "matched_group_id"}
                }
                request = request.model_copy(update={"metadata": {**request.metadata, **allowed}})
        except (TypeError, ValueError):
            return self._result(context, CascadeAgentRuntimeStatus.RENDER_ERROR, error_code="request_render_failed")
        try:
            response = provider.complete(request)
        except LLMProviderError as error:
            # Ledger rejection messages are already stable, content-free
            # coordination states. Preserve them so an opt-in runner can
            # distinguish a completed coordinate from an ordinary malformed
            # provider request without exposing a prompt or response.
            ledger_status_codes = {
                "completed_request_fingerprint_exists",
                "failed_request_requires_resume",
                "incomplete_request_requires_resume",
                "request_cap_reached",
                "completion_reservation_cap_reached",
                "parser_recovery_fingerprint_changed",
                "parser_recovery_attempt_limit_reached",
            }
            error_code = (
                error.message
                if isinstance(error, RequestLedgerError) and error.message in ledger_status_codes
                else error.code.value
            )
            return self._result(
                context, CascadeAgentRuntimeStatus.PROVIDER_ERROR,
                template_version=CASCADE_AGENT_TEMPLATE_VERSION,
                request_id=request.request_id,
                error_code=error_code,
            )
        except Exception:
            return self._result(
                context, CascadeAgentRuntimeStatus.PROVIDER_ERROR,
                template_version=CASCADE_AGENT_TEMPLATE_VERSION,
                request_id=request.request_id,
                error_code="provider_failure",
            )
        try:
            parsed = parse_cascade_agent_response(response.content, context)
        except Exception:
            return self._result(
                context,
                CascadeAgentRuntimeStatus.PARSER_INVALID,
                template_version=CASCADE_AGENT_TEMPLATE_VERSION,
                request_id=request.request_id,
                model_name=response.model_name,
                finish_reason=response.finish_reason,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                latency_ms=response.latency_ms,
                parser_valid=False,
                error_code="parser_failure",
            )
        status = CascadeAgentRuntimeStatus.COMPLETED if parsed.valid else CascadeAgentRuntimeStatus.PARSER_INVALID
        return self._result(
            context,
            status,
            response=parsed,
            template_version=CASCADE_AGENT_TEMPLATE_VERSION,
            request_id=request.request_id,
            model_name=response.model_name,
            finish_reason=response.finish_reason,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_ms=response.latency_ms,
            parser_valid=parsed.valid,
            content_ids_used_count=len(parsed.content_ids_used) if parsed.valid else 0,
            evidence_ids_used_count=len(parsed.evidence_ids_used) if parsed.valid else 0,
            share_requested=parsed.share_content_id is not None if parsed.valid else False,
            error_code=parsed.validation_errors[0].value if parsed.validation_errors else None,
        )

    @staticmethod
    def _result(
        context: CascadeAgentPromptContext,
        status: CascadeAgentRuntimeStatus,
        *,
        response: CascadeAgentResponse | None = None,
        template_version: str | None = None,
        request_id: str | None = None,
        model_name: str | None = None,
        finish_reason: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        latency_ms: float | None = None,
        parser_valid: bool | None = None,
        content_ids_used_count: int = 0,
        evidence_ids_used_count: int = 0,
        share_requested: bool = False,
        error_code: str | None = None,
    ) -> CascadeAgentRuntimeResult:
        return CascadeAgentRuntimeResult(
            status=status,
            response=response,
            audit=CascadeAgentRuntimeAudit(
                status=status,
                scenario_id=context.scenario_id,
                agent_id=context.agent_id,
                round_id=context.round_id,
                template_version=template_version,
                model_name=model_name or context.runtime_config.model_name,
                request_id=request_id,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                parser_valid=parser_valid,
                content_ids_used_count=content_ids_used_count,
                evidence_ids_used_count=evidence_ids_used_count,
                share_requested=share_requested,
                error_code=error_code,
            ),
        )


__all__ = [
    "CascadeAgentRuntime",
    "CascadeAgentRuntimeAudit",
    "CascadeAgentRuntimeResult",
    "CascadeAgentRuntimeStatus",
]
