"""Deterministic, local-only provider contracts for protocol tests."""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .models import DialogueTurn, ProtocolCondition, ValueProbeItem
from .models._validation import identifier_list, normalized_text


class FakeLLMRequest(BaseModel):
    """The complete visibility-bounded input supplied to a local provider."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    protocol: ProtocolCondition
    scenario_context: str = ""
    visible_history: list[DialogueTurn] = Field(default_factory=list)
    visible_peer_turn_ids: list[str] = Field(default_factory=list)
    visible_evidence_ids: list[str] = Field(default_factory=list)
    seed: int

    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, value: str) -> str:
        return normalized_text(value, "agent_id")

    @field_validator("visible_peer_turn_ids", "visible_evidence_ids")
    @classmethod
    def validate_visible_ids(cls, value: list[str], info: object) -> list[str]:
        return identifier_list(value, getattr(info, "field_name", "visible_ids"))


class FakeLLMResponse(BaseModel):
    """A deterministic structured response; no probe response is produced."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    round_id: int
    protocol: ProtocolCondition
    message: str
    visible_peer_turn_ids: list[str]
    visible_evidence_ids: list[str]
    request_fingerprint: str


class FakeProbeRequest(BaseModel):
    """A probe-only request with no dialogue history or peer visibility fields."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    probe: ValueProbeItem
    seed: int

    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, value: str) -> str:
        return normalized_text(value, "agent_id")


class FakeProbeResponse(BaseModel):
    """Strict local response for one probe item, never emitted as a dialogue turn."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    probe_id: str
    round_id: int
    raw_response: str
    normalized_score: float
    request_fingerprint: str


class LocalProvider(Protocol):
    """Minimal provider interface deliberately independent of real LLM SDKs."""

    def complete(self, request: FakeLLMRequest) -> FakeLLMResponse:
        """Produce a response for a fully rendered local request."""


class ProbeProvider(Protocol):
    """Probe-only provider interface with no access to dialogue state."""

    def complete_probe(self, request: FakeProbeRequest) -> FakeProbeResponse:
        """Produce one structured local response for a single probe item."""


class FakeLLM:
    """A deterministic provider that only echoes visibility-safe request facts."""

    def complete(self, request: FakeLLMRequest) -> FakeLLMResponse:
        canonical_request = json.dumps(
            request.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        fingerprint = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()[:16]
        peers = ",".join(request.visible_peer_turn_ids) or "none"
        evidence = ",".join(request.visible_evidence_ids) or "none"
        message = (
            f"agent={request.agent_id} round={request.round_id} "
            f"protocol={request.protocol.value} visible_peers={peers} "
            f"scenario_context={request.scenario_context or 'none'} "
            f"visible_evidence={evidence}"
        )
        return FakeLLMResponse(
            agent_id=request.agent_id,
            round_id=request.round_id,
            protocol=request.protocol,
            message=message,
            visible_peer_turn_ids=request.visible_peer_turn_ids,
            visible_evidence_ids=request.visible_evidence_ids,
            request_fingerprint=fingerprint,
        )

    def complete_probe(self, request: FakeProbeRequest) -> FakeProbeResponse:
        """Return a deterministic scale value derived solely from probe inputs."""
        canonical_request = json.dumps(
            request.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
        scale_index = int(digest[:16], 16) % len(request.probe.response_scale)
        raw_response = request.probe.response_scale[scale_index]
        return FakeProbeResponse(
            agent_id=request.agent_id,
            probe_id=request.probe.probe_id,
            round_id=request.round_id,
            raw_response=raw_response,
            normalized_score=request.probe.normalized_score_for(raw_response),
            request_fingerprint=digest[:16],
        )
