"""Deterministic local provider with optional structured intervention metadata."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import fake_llm_core as _core
from .models import DialogueTurn, ProtocolCondition
from .models._validation import identifier_list, normalized_text
from .models.protocols import InterventionAction

FakeProbeRequest = _core.FakeProbeRequest
FakeProbeResponse = _core.FakeProbeResponse
FakeLLMResponse = _core.FakeLLMResponse
LocalProvider = _core.LocalProvider
ProbeProvider = _core.ProbeProvider


class InterventionMetadata(BaseModel):
    """Structured control facts for FakeLLM; it has no natural-language prompt field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    current_action: InterventionAction
    target_agent: bool = False
    evidence_request: bool = False
    blind_reflection: bool = False
    speaking_priority: bool = False
    final_report_requirement: bool = False


class FakeLLMRequest(BaseModel):
    """Visibility-bounded local request with optional prospective control metadata."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    protocol: ProtocolCondition
    scenario_context: str = ""
    visible_history: list[DialogueTurn] = Field(default_factory=list)
    visible_peer_turn_ids: list[str] = Field(default_factory=list)
    visible_evidence_ids: list[str] = Field(default_factory=list)
    seed: int
    intervention_metadata: InterventionMetadata | None = None

    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, value: str) -> str:
        return normalized_text(value, "agent_id")

    @field_validator("visible_peer_turn_ids", "visible_evidence_ids")
    @classmethod
    def validate_visible_ids(cls, value: list[str], info: object) -> list[str]:
        return identifier_list(value, getattr(info, "field_name", "visible_ids"))


class FakeLLM:
    """Echo only request facts deterministically, without a network or model SDK."""

    def complete(self, request: FakeLLMRequest) -> FakeLLMResponse:
        canonical_request = json.dumps(
            request.model_dump(mode="json", exclude_none=True),
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
        if request.intervention_metadata is not None:
            metadata = request.intervention_metadata
            message += (
                f" current_action={metadata.current_action.value}"
                f" target_agent={str(metadata.target_agent).lower()}"
                f" evidence_request={str(metadata.evidence_request).lower()}"
                f" blind_reflection={str(metadata.blind_reflection).lower()}"
                f" speaking_priority={str(metadata.speaking_priority).lower()}"
                f" final_report_requirement={str(metadata.final_report_requirement).lower()}"
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
        """Delegate isolated probe behavior unchanged from the original FakeLLM."""
        return _core.FakeLLM().complete_probe(request)


__all__ = [
    "FakeLLM",
    "FakeLLMRequest",
    "FakeLLMResponse",
    "FakeProbeRequest",
    "FakeProbeResponse",
    "InterventionMetadata",
    "LocalProvider",
    "ProbeProvider",
]
