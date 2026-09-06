"""Strict local parsing tests for Agent JSON responses."""

from __future__ import annotations

import json

import pytest

from agent_cases import agent_response_content, public_context

from evicon.agent_response import AgentResponseError, parse_agent_response


def test_valid_agent_response_is_parsed_with_visible_evidence_only() -> None:
    result = parse_agent_response(
        agent_response_content(evidence_ids_used=["evidence-0"]),
        public_context(),
    )

    assert result.valid is True
    assert result.evidence_ids_used == ["evidence-0"]


@pytest.mark.parametrize(
    "content,error",
    [
        ("{not-json", AgentResponseError.MALFORMED_JSON),
        (agent_response_content(message="   "), AgentResponseError.EMPTY_MESSAGE),
        (
            agent_response_content(evidence_ids_used=["not-visible"]),
            AgentResponseError.UNAVAILABLE_EVIDENCE,
        ),
        (
            json.dumps(
                {
                    "message": "Attempting to control execution.",
                    "evidence_ids_used": [],
                    "action": "restructure",
                }
            ),
            AgentResponseError.FORBIDDEN_CONTROL_FIELD,
        ),
    ],
)
def test_invalid_agent_response_has_a_stable_error_code(
    content: str,
    error: AgentResponseError,
) -> None:
    result = parse_agent_response(content, public_context())

    assert result.valid is False
    assert result.validation_errors == [error]


def test_hidden_content_is_rejected() -> None:
    result = parse_agent_response(
        agent_response_content(message="This hidden probe must not be exposed."),
        public_context(),
    )

    assert result.valid is False
    assert result.validation_errors == [AgentResponseError.PRIVATE_CONTENT]
