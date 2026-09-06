"""Offline tests for the isolated, fixed-plan mediator runtime."""

from __future__ import annotations

import json

from mediator_cases import plan_for, prompt_context_for

from evicon.fake_provider_adapter import FakeLLMProviderAdapter, FakeResponseMode
from evicon.llm_contract import LLMProviderError, LLMRequest, LLMResponse, ProviderErrorCode
from evicon.mediator_prompts import render_action_instruction
from evicon.mediator_runtime import (
    MediatorRequestSettings,
    MediatorRuntime,
    MediatorRuntimeStatus,
)
from evicon.models import InterventionAction


class RecordingProvider:
    """Local test double that records requests without using a transport."""

    def __init__(self, *, content: str | None = None, error: LLMProviderError | None = None) -> None:
        self.content = content
        self.error = error
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content=self.content or _valid_content("stored-only-mediator-message"),
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=12,
            total_tokens=22,
            latency_ms=1.0,
            provider_metadata={"provider_name": "local_test"},
        )


def _settings() -> MediatorRequestSettings:
    return MediatorRequestSettings(model_name="runtime-test-model", max_tokens=128, seed=17)


def _runtime() -> MediatorRuntime:
    return MediatorRuntime(_settings())


def _rendered_request(action: InterventionAction) -> tuple[object, object, LLMRequest]:
    plan = plan_for(action)
    context = prompt_context_for(plan)
    request = render_action_instruction(
        plan,
        context,
        model_name=_settings().model_name,
        temperature=_settings().temperature,
        max_tokens=_settings().max_tokens,
        seed=_settings().seed,
    )
    assert request is not None
    return plan, context, request


def _valid_content(message: str) -> str:
    return json.dumps(
        {
            "message": message,
            "evidence_ids_requested": [],
            "dissent_preserved": True,
            "factual_claims": [],
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def test_runtime_completes_and_retains_valid_message_only_in_memory() -> None:
    plan = plan_for(InterventionAction.SOLICIT_DISSENT)
    context = prompt_context_for(plan)

    result = _runtime().execute(plan, context, FakeLLMProviderAdapter())

    assert result.status is MediatorRuntimeStatus.COMPLETED
    assert result.response is not None
    assert result.response.valid is True
    assert result.response.message.startswith("local fake mediator response")
    assert result.audit.parser_valid is True
    assert result.audit.template_version == "solicit_dissent.v1"


def test_no_op_makes_zero_provider_calls() -> None:
    plan = plan_for(InterventionAction.NO_OP)
    context = prompt_context_for(plan)
    provider = RecordingProvider()

    result = _runtime().execute(plan, context, provider)

    assert result.status is MediatorRuntimeStatus.NOT_INVOKED
    assert result.response is None
    assert result.audit.request_id is None
    assert provider.requests == []


def test_provider_failure_is_redacted_to_a_stable_error_code() -> None:
    plan = plan_for(InterventionAction.SOLICIT_DISSENT)
    context = prompt_context_for(plan)
    provider = RecordingProvider(
        error=LLMProviderError(
            ProviderErrorCode.HTTP_SERVER_ERROR,
            "api_key=must-not-appear prompt=must-not-appear",
        )
    )

    result = _runtime().execute(plan, context, provider)
    audit_json = result.audit_summary().model_dump_json()

    assert result.status is MediatorRuntimeStatus.PROVIDER_ERROR
    assert result.audit.error_code == ProviderErrorCode.HTTP_SERVER_ERROR.value
    assert "api_key" not in audit_json
    assert "must-not-appear" not in audit_json


def test_malformed_json_is_a_structured_parser_failure() -> None:
    plan, context, request = _rendered_request(InterventionAction.SOLICIT_DISSENT)
    provider = FakeLLMProviderAdapter({request.request_id: FakeResponseMode.INVALID_JSON})

    result = _runtime().execute(plan, context, provider)

    assert result.status is MediatorRuntimeStatus.PARSER_INVALID
    assert result.response is not None
    assert result.response.valid is False
    assert result.audit.error_code == "malformed_json"


def test_unavailable_evidence_is_rejected_by_the_existing_parser() -> None:
    plan, context, request = _rendered_request(InterventionAction.REQUEST_EVIDENCE)
    provider = FakeLLMProviderAdapter({request.request_id: FakeResponseMode.UNKNOWN_EVIDENCE})

    result = _runtime().execute(plan, context, provider)

    assert result.status is MediatorRuntimeStatus.PARSER_INVALID
    assert result.audit.error_code == "unavailable_evidence"


def test_context_mismatch_rejects_before_provider_invocation() -> None:
    plan = plan_for(InterventionAction.SOLICIT_DISSENT)
    context = prompt_context_for(plan).model_copy(update={"scenario_id": "other-scenario"})
    provider = RecordingProvider()

    result = _runtime().execute(plan, context, provider)

    assert result.status is MediatorRuntimeStatus.REJECTED
    assert result.audit.error_code == "context_mismatch"
    assert provider.requests == []


def test_audit_summary_excludes_prompts_message_and_private_terms() -> None:
    plan = plan_for(InterventionAction.SOLICIT_DISSENT)
    context = prompt_context_for(plan).model_copy(
        update={"public_dialogue_summary": "unique-public-prompt-material"}
    )
    provider = RecordingProvider(content=_valid_content("unique-mediator-message"))

    result = _runtime().execute(plan, context, provider)
    audit_json = result.audit_summary().model_dump_json()

    assert result.response is not None
    assert result.response.message == "unique-mediator-message"
    assert "unique-public-prompt-material" not in audit_json
    assert "unique-mediator-message" not in audit_json
    assert "hidden_probe" not in audit_json
    assert "hidden_profile" not in audit_json
    assert "evaluation_report" not in audit_json
