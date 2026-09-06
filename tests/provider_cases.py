"""Fake OpenAI-compatible transport helpers; no test makes a real request."""

from __future__ import annotations

import json
from collections.abc import Mapping

from mediator_cases import plan_for, prompt_context_for

from evicon.mediator_prompts import render_action_instruction
from evicon.models import InterventionAction
from evicon.openai_provider import TransportResponse


class FakeTransport:
    def __init__(self, outcomes: list[TransportResponse | BaseException]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def post(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> TransportResponse:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def request():
    plan = plan_for(InterventionAction.REQUEST_EVIDENCE)
    rendered = render_action_instruction(plan, prompt_context_for(plan), model_name="provider-test-model", seed=37)
    assert rendered is not None
    return plan, prompt_context_for(plan), rendered


def response_body(
    *,
    content: str | None = None,
    usage: object = None,
    choices: object | None = None,
) -> str:
    if content is None:
        content = json.dumps(
            {
                "message": "A bounded public response.",
                "evidence_ids_requested": ["evidence-0"],
                "dissent_preserved": False,
                "factual_claims": [],
            },
            ensure_ascii=True,
        )
    if usage is None:
        usage = {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    if choices is None:
        choices = [{"message": {"content": content}, "finish_reason": "stop"}]
    return json.dumps({"model": "provider-test-model", "choices": choices, "usage": usage}, ensure_ascii=True)
