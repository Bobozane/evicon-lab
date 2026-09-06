"""Public exposure validation and deterministic Agent prompt rendering tests."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agent_cases import context_data, public_context

from evicon.agent_prompts import AGENT_TURN_TEMPLATE_VERSION, AgentPromptContext, render_agent_turn
from evicon.models import ProtocolCondition


def test_rendered_prompt_contains_only_exposed_public_history_and_evidence() -> None:
    context = public_context()

    request = render_agent_turn(context, model_name="agent-test-model", seed=41)
    payload = json.loads(request.user_prompt)

    assert request.metadata["template_version"] == AGENT_TURN_TEMPLATE_VERSION
    assert request.metadata["visible_peer_turn_ids"] == ["turn-peer-r0"]
    assert request.metadata["visible_evidence_ids"] == ["evidence-0"]
    assert payload["visible_history"] == [
        {
            "message": "My earlier public view noted a tradeoff.",
            "round_id": 0,
            "speaker_id": "agent-a",
            "turn_id": "turn-own-r0",
        },
        {
            "message": "A peer expressed a public preference.",
            "round_id": 0,
            "speaker_id": "agent-b",
            "turn_id": "turn-peer-r0",
        },
    ]
    assert payload["visible_evidence_cards"][0]["evidence_id"] == "evidence-0"
    assert "peer turns are viewpoints" in request.system_prompt.lower()
    assert "target_agent_ids" in request.system_prompt


def test_unexposed_or_unauthorized_peer_turn_is_rejected() -> None:
    context = public_context()
    data = context_data(context)
    snapshot = data["exposure_snapshot"]
    assert isinstance(snapshot, dict)
    snapshot["visible_peer_turn_ids"] = []

    with pytest.raises(ValidationError, match="visible_peer_turn_ids"):
        AgentPromptContext.model_validate(data)

    data = context_data(context)
    snapshot = data["exposure_snapshot"]
    assert isinstance(snapshot, dict)
    history = snapshot["visible_history"]
    assert isinstance(history, list)
    peer = history[1]
    assert isinstance(peer, dict)
    peer["visible_to"] = ["agent-b"]

    with pytest.raises(ValidationError, match="not authorized"):
        AgentPromptContext.model_validate(data)


def test_unavailable_or_protocol_forbidden_evidence_is_rejected() -> None:
    context = public_context()
    data = context_data(context)
    cards = data["visible_evidence_cards"]
    assert isinstance(cards, list)
    card = cards[0]
    assert isinstance(card, dict)
    card["visible_to"] = ["agent-b"]

    with pytest.raises(ValidationError, match="not authorized"):
        AgentPromptContext.model_validate(data)

    independent = public_context(protocol=ProtocolCondition.INDEPENDENT)
    data = context_data(independent)
    snapshot = data["exposure_snapshot"]
    assert isinstance(snapshot, dict)
    snapshot["visible_evidence_ids"] = ["evidence-0"]
    data["visible_evidence_cards"] = [
        {
            "evidence_id": "evidence-0",
            "claim": "A public fact.",
            "source": "public-source",
            "supports": [],
            "contradicts": [],
            "introduced_round": 0,
            "visible_to": ["agent-a"],
            "reliability": 0.9,
        }
    ]

    with pytest.raises(ValidationError, match="protocol does not permit visible evidence"):
        AgentPromptContext.model_validate(data)


def test_context_rejects_hidden_and_unknown_input_fields() -> None:
    data = context_data(public_context())
    data["hidden_probe"] = "forbidden"

    with pytest.raises(ValidationError, match="hidden_probe"):
        AgentPromptContext.model_validate(data)

    data = context_data(public_context())
    data["scenario_context"] = "This includes a hidden profile."

    with pytest.raises(ValidationError, match="hidden probe"):
        AgentPromptContext.model_validate(data)
