"""Strict local mediator response parsing coverage."""

from __future__ import annotations

import json
import socket

import pytest

from mediator_cases import plan_for, prompt_context_for

from evicon.fake_provider_adapter import FakeLLMProviderAdapter, FakeResponseMode
from evicon.mediator_prompts import render_action_instruction
from evicon.mediator_response import MediatorResponseError, parse_mediator_response
from evicon.models import InterventionAction


def _rendered_request(action: InterventionAction):
    plan = plan_for(action)
    context = prompt_context_for(plan)
    request = render_action_instruction(plan, context, seed=19)
    assert request is not None
    return plan, context, request


def test_parser_accepts_a_valid_local_json_response() -> None:
    plan, context, request = _rendered_request(InterventionAction.REQUEST_EVIDENCE)
    response = FakeLLMProviderAdapter().complete(request)

    parsed = parse_mediator_response(response.content, plan, context)

    assert parsed.valid is True
    assert parsed.evidence_ids_requested == ["evidence-0"]
    assert parsed.validation_errors == []


@pytest.mark.parametrize("field_name", ["action", "target_agent_ids"])
def test_model_cannot_override_action_or_targets(field_name: str) -> None:
    plan, context, _ = _rendered_request(InterventionAction.SOLICIT_DISSENT)
    payload = {
        "message": "A bounded public response.",
        "evidence_ids_requested": [],
        "dissent_preserved": True,
        "factual_claims": [],
        field_name: "restructure" if field_name == "action" else ["agent-b"],
    }

    parsed = parse_mediator_response(json.dumps(payload), plan, context)

    assert parsed.valid is False
    assert parsed.validation_errors == [MediatorResponseError.FORBIDDEN_DECISION_FIELD]


def test_unknown_evidence_and_malformed_json_are_rejected() -> None:
    plan, context, request = _rendered_request(InterventionAction.REQUEST_EVIDENCE)
    adapter = FakeLLMProviderAdapter({request.request_id: FakeResponseMode.UNKNOWN_EVIDENCE})

    assert parse_mediator_response(adapter.complete(request).content, plan, context).validation_errors == [
        MediatorResponseError.UNAVAILABLE_EVIDENCE
    ]
    assert parse_mediator_response("not json", plan, context).validation_errors == [
        MediatorResponseError.MALFORMED_JSON
    ]


def test_empty_message_and_private_probe_leak_are_rejected() -> None:
    plan, context, _ = _rendered_request(InterventionAction.MINORITY_REPORT)
    empty = json.dumps(
        {
            "message": " ",
            "evidence_ids_requested": [],
            "dissent_preserved": False,
            "factual_claims": [],
        }
    )
    private = json.dumps(
        {
            "message": "hidden probe response",
            "evidence_ids_requested": [],
            "dissent_preserved": False,
            "factual_claims": [],
        }
    )

    assert parse_mediator_response(empty, plan, context).validation_errors == [MediatorResponseError.EMPTY_MESSAGE]
    assert parse_mediator_response(private, plan, context).validation_errors == [MediatorResponseError.PRIVATE_CONTENT]


def test_parser_opens_no_network_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("parser attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", reject_connection)
    plan, context, _ = _rendered_request(InterventionAction.SOLICIT_DISSENT)
    result = parse_mediator_response(
        '{"message":"public response","evidence_ids_requested":[],"dissent_preserved":true,"factual_claims":[]}',
        plan,
        context,
    )

    assert result.valid is True
