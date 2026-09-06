"""FakeLLM provider-adapter coverage without network access."""

from __future__ import annotations

import socket

import pytest

from mediator_cases import plan_for, prompt_context_for

from evicon.fake_provider_adapter import FakeLLMProviderAdapter, FakeResponseMode
from evicon.llm_contract import LLMProviderError, ProviderErrorCode
from evicon.mediator_prompts import render_action_instruction
from evicon.models import InterventionAction


def _request():
    plan = plan_for(InterventionAction.REQUEST_EVIDENCE)
    request = render_action_instruction(plan, prompt_context_for(plan), seed=23)
    assert request is not None
    return request


def test_fake_adapter_satisfies_the_serializable_provider_contract_deterministically() -> None:
    request = _request()
    adapter = FakeLLMProviderAdapter(simulated_latency_ms=2.5)

    first = adapter.complete(request)
    second = adapter.complete(request)

    assert first == second
    assert first.request_id == request.request_id
    assert first.model_name == request.model_name
    assert first.total_tokens == first.prompt_tokens + first.completion_tokens
    assert first.latency_ms == 2.5
    assert "api_key" not in first.model_dump_json().lower()


def test_fake_adapter_supports_invalid_json_for_parser_tests() -> None:
    request = _request()
    adapter = FakeLLMProviderAdapter({request.request_id: FakeResponseMode.INVALID_JSON})

    assert adapter.complete(request).content == "{invalid-json"


def test_fake_adapter_reports_invalid_contract_metadata_structurally() -> None:
    request = _request().model_copy(update={"metadata": {"template_version": "x"}})

    with pytest.raises(LLMProviderError) as error:
        FakeLLMProviderAdapter().complete(request)

    assert error.value.code is ProviderErrorCode.INVALID_REQUEST
    assert error.value.as_dict()["code"] == "invalid_request"


def test_fake_adapter_opens_no_network_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("fake provider adapter attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", reject_connection)

    assert FakeLLMProviderAdapter().complete(_request()).finish_reason == "stop"
