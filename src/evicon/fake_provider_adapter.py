"""Deterministic adapter from FakeLLM-style local behavior to ``LLMProvider``."""

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum

from .llm_contract import LLMProviderError, LLMRequest, LLMResponse, ProviderErrorCode
from .models.protocols import InterventionAction


class FakeResponseMode(str, Enum):
    """Explicit test-only response modes; none invokes a model or network."""

    VALID = "valid"
    INVALID_JSON = "invalid_json"
    UNKNOWN_EVIDENCE = "unknown_evidence"
    DECISION_OVERRIDE = "decision_override"


class FakeLLMProviderAdapter:
    """Render deterministic JSON responses for the offline provider contract."""

    def __init__(
        self,
        response_modes: dict[str, FakeResponseMode] | None = None,
        *,
        simulated_latency_ms: float = 1.0,
    ) -> None:
        if not math.isfinite(simulated_latency_ms) or simulated_latency_ms < 0.0:
            raise ValueError("simulated_latency_ms must be a finite non-negative number")
        self._response_modes = dict(response_modes or {})
        self._simulated_latency_ms = float(simulated_latency_ms)

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return local deterministic JSON or a structured provider exception."""
        try:
            action = InterventionAction(_required_string(request, "action"))
            target_agent_ids = _required_string_list(request, "target_agent_ids")
            requested_evidence_ids = _optional_string_list(request, "requested_evidence_ids")
            template_version = _required_string(request, "template_version")
        except ValueError as exc:
            raise LLMProviderError(ProviderErrorCode.INVALID_REQUEST, str(exc)) from exc

        mode = self._response_modes.get(request.request_id, FakeResponseMode.VALID)
        content = _content_for(
            request,
            action=action,
            target_agent_ids=target_agent_ids,
            requested_evidence_ids=requested_evidence_ids,
            mode=mode,
        )
        prompt_tokens = _token_estimate(request.system_prompt) + _token_estimate(request.user_prompt)
        completion_tokens = _token_estimate(content)
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content=content,
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=self._simulated_latency_ms,
            provider_metadata={
                "adapter": "fake_llm_provider_adapter.v1",
                "response_mode": mode.value,
                "template_version": template_version,
                "request_fingerprint": _fingerprint(request, action, target_agent_ids),
            },
        )


FakeProviderAdapter = FakeLLMProviderAdapter


def _content_for(
    request: LLMRequest,
    *,
    action: InterventionAction,
    target_agent_ids: list[str],
    requested_evidence_ids: list[str],
    mode: FakeResponseMode,
) -> str:
    if mode is FakeResponseMode.INVALID_JSON:
        return "{invalid-json"
    evidence_ids = requested_evidence_ids[:1] if action is InterventionAction.REQUEST_EVIDENCE else []
    payload: dict[str, object] = {
        "message": (
            f"local fake mediator response action={action.value} "
            f"targets={','.join(target_agent_ids) or 'none'} seed={request.seed}"
        ),
        "evidence_ids_requested": evidence_ids,
        "dissent_preserved": action
        in {InterventionAction.SOLICIT_DISSENT, InterventionAction.MINORITY_REPORT},
        "factual_claims": [],
    }
    if mode is FakeResponseMode.UNKNOWN_EVIDENCE:
        payload["evidence_ids_requested"] = ["not-visible-evidence"]
    if mode is FakeResponseMode.DECISION_OVERRIDE:
        payload["action"] = InterventionAction.RESTRUCTURE.value
        payload["target_agent_ids"] = ["model-chosen-agent"]
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _required_string(request: LLMRequest, field_name: str) -> str:
    value = request.metadata.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"LLMRequest.metadata.{field_name} must be a non-empty string")
    return value


def _required_string_list(request: LLMRequest, field_name: str) -> list[str]:
    value = request.metadata.get(field_name)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"LLMRequest.metadata.{field_name} must be a list of non-empty strings")
    return list(value)


def _optional_string_list(request: LLMRequest, field_name: str) -> list[str]:
    value = request.metadata.get(field_name, [])
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"LLMRequest.metadata.{field_name} must be a list of non-empty strings")
    return list(value)


def _fingerprint(
    request: LLMRequest,
    action: InterventionAction,
    target_agent_ids: list[str],
) -> str:
    canonical = json.dumps(
        {
            "request_id": request.request_id,
            "seed": request.seed,
            "action": action.value,
            "target_agent_ids": target_agent_ids,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _token_estimate(value: str) -> int:
    return 0 if not value else max(1, (len(value) + 3) // 4)


__all__ = ["FakeLLMProviderAdapter", "FakeProviderAdapter", "FakeResponseMode"]
