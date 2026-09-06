"""Private in-memory, single-item value probes conditioned on public trajectory context."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .llm_contract import LLMProvider, LLMProviderError, LLMRequest
from .measurement_context import OfflineMeasurementContext
from .models import ValueProbeItem, ValueProbeResponse
from .models._validation import normalized_text


CONTEXTUAL_PROBE_TEMPLATE_VERSION: Final[str] = "contextual_value_probe.v2"
_SYSTEM_PROMPT: Final[str] = (
    "Return only one JSON object with exactly one field: choice. The choice must exactly match one "
    "supplied response_scale label. Do not output action, target_agent_ids, budget, turn_order, "
    "visibility_override, policy, or any other control field."
)
_CONTROLS = frozenset({"action", "target_agent_ids", "budget", "cooldown", "turn_order", "visibility_override", "policy", "executor", "intervention_plan"})
_PRIVATE = ("hidden_probe", "hidden probe", "hidden_profile", "hidden profile", "valueprofile", "evaluationreport", "evaluation_report", "proberesult", "probe result")


class ContextualProbeStatus(str, Enum):
    COMPLETED = "completed"
    RENDER_ERROR = "render_error"
    PROVIDER_ERROR = "provider_error"
    PARSER_INVALID = "parser_invalid"


class ContextualProbeError(str, Enum):
    REQUEST_RENDER_FAILED = "request_render_failed"
    PARSER_FAILURE = "parser_failure"
    MALFORMED_JSON = "malformed_json"
    INVALID_SCHEMA = "invalid_schema"
    EMPTY_CHOICE = "empty_choice"
    FORBIDDEN_CONTROL_FIELD = "forbidden_control_field"
    INVALID_CHOICE = "invalid_choice"
    PRIVATE_CONTENT = "private_content"


class ContextualProbeRequestSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str = Field(default="contextual-probe-runtime-v1", min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=128, gt=0)
    seed: int = 0

    @field_validator("model_name")
    @classmethod
    def validate_model(cls, value: str) -> str:
        return normalized_text(value, "model_name")

    @field_validator("seed", mode="before")
    @classmethod
    def validate_seed(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError("seed must be an integer")
        return value


class ContextualProbeAuditSummary(BaseModel):
    """Safe audit data: it excludes probe text, selected answer, prompts, and metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ContextualProbeStatus
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
    error_code: str | None = None


class ContextualProbeResult(BaseModel):
    """In-memory private response plus a safe audit summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ContextualProbeStatus
    response: ValueProbeResponse | None = None
    audit: ContextualProbeAuditSummary

    def audit_summary(self) -> ContextualProbeAuditSummary:
        return self.audit


class ContextualProbeRuntime:
    """Run exactly one render -> provider -> parse sequence with no writes or control feedback."""

    def __init__(self, request_settings: ContextualProbeRequestSettings | None = None) -> None:
        self._request_settings = request_settings or ContextualProbeRequestSettings()

    def execute(self, context: OfflineMeasurementContext, probe_item: ValueProbeItem, provider: LLMProvider) -> ContextualProbeResult:
        try:
            request = render_contextual_probe(
                context, probe_item, model_name=self._request_settings.model_name,
                temperature=self._request_settings.temperature, max_tokens=self._request_settings.max_tokens,
                seed=self._request_settings.seed,
            )
        except (TypeError, ValueError):
            return self._result(context, ContextualProbeStatus.RENDER_ERROR, None, error_code=ContextualProbeError.REQUEST_RENDER_FAILED.value)
        try:
            provider_response = provider.complete(request)
        except LLMProviderError as error:
            return self._result(context, ContextualProbeStatus.PROVIDER_ERROR, None, request_id=request.request_id, template_version=CONTEXTUAL_PROBE_TEMPLATE_VERSION, error_code=error.code.value)
        except Exception:
            return self._result(context, ContextualProbeStatus.PROVIDER_ERROR, None, request_id=request.request_id, template_version=CONTEXTUAL_PROBE_TEMPLATE_VERSION, error_code="provider_failure")
        try:
            response, error = parse_contextual_probe_response(provider_response.content, context=context, probe_item=probe_item)
        except Exception:
            return self._result(context, ContextualProbeStatus.PARSER_INVALID, None, request_id=request.request_id, template_version=CONTEXTUAL_PROBE_TEMPLATE_VERSION, model_name=provider_response.model_name, finish_reason=provider_response.finish_reason, prompt_tokens=provider_response.prompt_tokens, completion_tokens=provider_response.completion_tokens, total_tokens=provider_response.total_tokens, latency_ms=provider_response.latency_ms, error_code=ContextualProbeError.PARSER_FAILURE.value)
        return self._result(
            context, ContextualProbeStatus.COMPLETED if response else ContextualProbeStatus.PARSER_INVALID,
            response, request_id=request.request_id, template_version=CONTEXTUAL_PROBE_TEMPLATE_VERSION,
            model_name=provider_response.model_name, finish_reason=provider_response.finish_reason,
            prompt_tokens=provider_response.prompt_tokens, completion_tokens=provider_response.completion_tokens,
            total_tokens=provider_response.total_tokens, latency_ms=provider_response.latency_ms,
            error_code=error.value if error else None,
        )

    def _result(
        self, context: OfflineMeasurementContext, status: ContextualProbeStatus,
        response: ValueProbeResponse | None, *, request_id: str | None = None,
        template_version: str | None = None, model_name: str | None = None,
        finish_reason: str | None = None, prompt_tokens: int | None = None,
        completion_tokens: int | None = None, total_tokens: int | None = None,
        latency_ms: float | None = None, error_code: str | None = None,
    ) -> ContextualProbeResult:
        return ContextualProbeResult(
            status=status, response=response,
            audit=ContextualProbeAuditSummary(
                status=status, agent_id=context.agent_id, round_id=context.round_id,
                template_version=template_version, model_name=model_name or self._request_settings.model_name,
                request_id=request_id, finish_reason=finish_reason, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens, total_tokens=total_tokens, latency_ms=latency_ms,
                parser_valid=response is not None if status is not ContextualProbeStatus.RENDER_ERROR and status is not ContextualProbeStatus.PROVIDER_ERROR else None,
                error_code=error_code,
            ),
        )


def render_contextual_probe(context: OfflineMeasurementContext, probe_item: ValueProbeItem, *, model_name: str = "contextual-probe-runtime-v1", temperature: float = 0.0, max_tokens: int = 128, seed: int = 0) -> LLMRequest:
    """Render only the public context and one private probe item."""
    payload = {
        "template_version": CONTEXTUAL_PROBE_TEMPLATE_VERSION,
        "measurement_context": {
            "agent_id": context.agent_id, "role": context.role, "initial_value_labels": context.initial_value_labels,
            "round_id": context.round_id, "scenario_context": context.scenario_context,
            "own_history": [_turn(turn) for turn in context.own_history],
            "visible_peer_turns": [_turn(turn) for turn in context.visible_peer_turns],
            "visible_evidence_cards": [_card(card) for card in context.visible_evidence_cards],
        },
        "probe_item": {"probe_id": probe_item.probe_id, "text": probe_item.text, "dimension": probe_item.dimension, "response_scale": probe_item.response_scale, "reverse_scored": probe_item.reverse_scored},
    }
    prompt = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    request_id = _request_id(context, probe_item, model_name, temperature, max_tokens, seed)
    return LLMRequest(
        request_id=request_id, model_name=model_name, system_prompt=_SYSTEM_PROMPT, user_prompt=prompt,
        temperature=temperature, max_tokens=max_tokens, seed=seed,
        metadata={
            "contract_version": "contextual_probe_request.v2",
            "template_version": CONTEXTUAL_PROBE_TEMPLATE_VERSION,
            "agent_id": context.agent_id,
            "round_id": context.round_id,
            "protocol": context.protocol.value,
        },
    )


def parse_contextual_probe_response(content: str, *, context: OfflineMeasurementContext, probe_item: ValueProbeItem) -> tuple[ValueProbeResponse | None, ContextualProbeError | None]:
    if _private(content):
        return None, ContextualProbeError.PRIVATE_CONTENT
    try:
        decoded = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None, ContextualProbeError.MALFORMED_JSON
    if not isinstance(decoded, dict):
        return None, ContextualProbeError.INVALID_SCHEMA
    if _CONTROLS.intersection(decoded):
        return None, ContextualProbeError.FORBIDDEN_CONTROL_FIELD
    try:
        raw = _RawProbePayload.model_validate(decoded)
    except ValidationError:
        return None, ContextualProbeError.INVALID_SCHEMA
    choice = raw.choice.strip()
    if not choice:
        return None, ContextualProbeError.EMPTY_CHOICE
    if _private(choice):
        return None, ContextualProbeError.PRIVATE_CONTENT
    try:
        normalized = probe_item.normalized_score_for(choice)
    except ValueError:
        return None, ContextualProbeError.INVALID_CHOICE
    return ValueProbeResponse(agent_id=context.agent_id, probe_id=probe_item.probe_id, round_id=context.round_id, raw_response=choice, normalized_score=normalized), None


class _RawProbePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    choice: str


def _turn(turn: object) -> dict[str, object]:
    return {"turn_id": turn.turn_id, "round_id": turn.round_id, "speaker_id": turn.speaker_id, "message": turn.message}


def _card(card: object) -> dict[str, object]:
    return {"evidence_id": card.evidence_id, "claim": card.claim, "source": card.source, "supports": card.supports, "contradicts": card.contradicts, "reliability": card.reliability}


def _request_id(context: OfflineMeasurementContext, item: ValueProbeItem, model_name: str, temperature: float, max_tokens: int, seed: int) -> str:
    canonical = json.dumps({"context": context.model_dump(mode="json"), "probe": item.model_dump(mode="json"), "model_name": model_name, "temperature": temperature, "max_tokens": max_tokens, "seed": seed}, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return f"contextual-probe-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:20]}"


def _private(value: object) -> bool:
    return isinstance(value, str) and any(token in value.lower() for token in _PRIVATE)


__all__ = ["CONTEXTUAL_PROBE_TEMPLATE_VERSION", "ContextualProbeAuditSummary", "ContextualProbeError", "ContextualProbeRequestSettings", "ContextualProbeResult", "ContextualProbeRuntime", "ContextualProbeStatus", "parse_contextual_probe_response", "render_contextual_probe"]
