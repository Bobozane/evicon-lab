"""Shared-T0 fingerprint wrapper for Conformity Identification Study v1.1."""
from __future__ import annotations

import hashlib
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from .conformity_identification_protocol import (
    IdentificationAgentResponse,
    IdentificationPromptContext,
    parse_identification_response,
    render_identification_turn,
)
from .llm_contract import LLMRequest

PROTOCOL_VERSION: Final[str] = "conformity_identification_protocol.shared_t0.v1"
TEMPLATE_VERSION: Final[str] = "conformity_identification_turn.shared_t0.v1"
SCHEMA_NAME: Final[str] = "conformity_identification_response_v1"


class BranchPromptContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    public_context: IdentificationPromptContext
    matched_group_id: str = Field(min_length=1)
    branch_id: str = Field(min_length=1)
    shared_t0_observation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def render_branch_turn(context: BranchPromptContext) -> LLMRequest:
    base = render_identification_turn(context.public_context)
    digest = hashlib.sha256(
        (
            f"{PROTOCOL_VERSION}|{context.matched_group_id}|{context.branch_id}|"
            f"{context.shared_t0_observation_sha256}|{base.system_prompt}|{base.user_prompt}"
        ).encode()
    ).hexdigest()[:24]
    return base.model_copy(update={
        "request_id": f"ident-v11-{digest}",
        "metadata": {
            "protocol_version": PROTOCOL_VERSION,
            "template_version": TEMPLATE_VERSION,
            "matched_group_id": context.matched_group_id,
            "branch_id": context.branch_id,
            "shared_t0_observation_sha256": context.shared_t0_observation_sha256,
        },
    })


def parse_branch_response(raw: str, context: BranchPromptContext) -> IdentificationAgentResponse:
    return parse_identification_response(raw, context.public_context)


__all__ = [
    "BranchPromptContext", "PROTOCOL_VERSION", "SCHEMA_NAME", "TEMPLATE_VERSION",
    "parse_branch_response", "render_branch_turn",
]
