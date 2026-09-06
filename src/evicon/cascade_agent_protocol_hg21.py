"""H-G.2.1 Provider-subset schema amendment with unchanged prompt semantics."""
from __future__ import annotations

import hashlib
from typing import Any, Final

from .cascade_agent_protocol_hg2 import (
    HG2AgentResponse,
    HG2PromptContext,
    HG2Runtime,
    parse_hg2_response,
    render_hg2_turn,
)
from .llm_contract import LLMRequest

HG21_PROTOCOL_VERSION: Final[str] = "provenance_cascade_agent_protocol.hg2_1_provider_subset.v1"
HG21_TEMPLATE_VERSION: Final[str] = "cascade_agent_turn.hg2_epistemic_behavior.v1"
HG21_SCHEMA_NAME: Final[str] = "cascade_agent_epistemic_behavior_response_v1_provider_subset"
HG21_RESPONSE_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "epistemic_stance": {"type": "string", "enum": ["supported", "unsupported", "uncertain"]},
        "behavioral_decision": {"type": "string", "enum": ["share", "share_with_caveat", "do_not_share", "support_reversible_action", "defer_action"]},
        "content_ids_used": {"type": "array", "items": {"type": "string"}},
        "evidence_ids_used": {"type": "array", "items": {"type": "string"}},
        "share_content_id": {"type": ["string", "null"]},
    },
    "required": ["epistemic_stance", "behavioral_decision", "content_ids_used", "evidence_ids_used", "share_content_id"],
}


def render_hg21_turn(context: HG2PromptContext) -> LLMRequest:
    """Keep both prompts identical while versioning request identity and audit metadata."""
    base = render_hg2_turn(context)
    digest = hashlib.sha256(f"{HG21_PROTOCOL_VERSION}|{base.request_id}".encode("utf-8")).hexdigest()[:24]
    return base.model_copy(update={
        "request_id": f"hg21-{digest}",
        "metadata": {
            **base.metadata,
            "transport_protocol_version": HG21_PROTOCOL_VERSION,
            "transport_schema_name": HG21_SCHEMA_NAME,
            "provider_subset_schema": True,
        },
    })


class HG21Runtime(HG2Runtime):
    """Same strict parser and decision contract with a provider-subset transport schema."""

    def render_request(self, context: HG2PromptContext) -> LLMRequest:
        return render_hg21_turn(context)


__all__ = [
    "HG21_PROTOCOL_VERSION", "HG21_RESPONSE_JSON_SCHEMA", "HG21_SCHEMA_NAME",
    "HG21_TEMPLATE_VERSION", "HG21Runtime", "HG2AgentResponse",
    "parse_hg2_response", "render_hg21_turn",
]
