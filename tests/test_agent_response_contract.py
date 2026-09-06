"""Model-level invariants for AgentResponse independent of the parser."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evicon.agent_response import AgentResponse


def test_valid_agent_response_requires_a_non_empty_message() -> None:
    with pytest.raises(ValidationError, match="non-empty message"):
        AgentResponse(valid=True, validation_errors=[])

    response = AgentResponse(message="  public response  ", evidence_ids_used=[], valid=True)
    assert response.message == "public response"
