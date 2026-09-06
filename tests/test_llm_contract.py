"""Offline LLM provider data-contract coverage."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evicon.llm_contract import LLMRequest, LLMResponse


def _request() -> LLMRequest:
    return LLMRequest(
        request_id="request-1",
        model_name="fake-mediator-v1",
        system_prompt="public-only system instruction",
        user_prompt="public-only user instruction",
        temperature=0.0,
        max_tokens=128,
        seed=7,
        metadata={"template_version": "test.v1"},
    )


def test_llm_request_round_trips_as_json_and_has_no_credential_field() -> None:
    request = _request()

    assert LLMRequest.model_validate_json(request.model_dump_json()) == request
    with pytest.raises(ValidationError, match="api_key"):
        LLMRequest.model_validate({**request.model_dump(), "api_key": "not-allowed"})
    with pytest.raises(ValidationError, match="prohibited"):
        LLMRequest.model_validate({**request.model_dump(), "metadata": {"api_key": "not-allowed"}})


def test_llm_response_round_trips_as_json_and_validates_usage() -> None:
    response = LLMResponse(
        request_id="request-1",
        model_name="fake-mediator-v1",
        content="{}",
        finish_reason="stop",
        prompt_tokens=3,
        completion_tokens=2,
        total_tokens=5,
        latency_ms=1.0,
        provider_metadata={"provider": "fake"},
    )

    assert LLMResponse.model_validate_json(response.model_dump_json()) == response
    with pytest.raises(ValidationError, match="total_tokens"):
        LLMResponse.model_validate({**response.model_dump(), "total_tokens": 6})
    with pytest.raises(ValidationError, match="prohibited"):
        LLMResponse.model_validate({**response.model_dump(), "provider_metadata": {"secret": "x"}})
