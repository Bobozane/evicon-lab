"""H-G.2.3.1 provider-subset schema with unchanged H-G.2.3 semantics."""
from __future__ import annotations

import hashlib
from typing import Any, Final

from .cascade_agent_protocol_hg2 import HG2PromptContext
from .cascade_agent_protocol_hg23 import (
    HG23AgentResponse, HG23Runtime, parse_hg23_response, render_hg23_turn,
)
from .llm_contract import LLMRequest

HG231_PROTOCOL_VERSION: Final[str] = "provenance_cascade_agent_protocol.hg2_3_1_provider_subset.v1"
HG231_TEMPLATE_VERSION: Final[str] = "cascade_agent_turn.hg2_3_observable_adoption.v1"
HG231_SCHEMA_NAME: Final[str] = "cascade_agent_epistemic_adoption_sharing_v1_provider_subset"
HG231_RESPONSE_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "epistemic_stance": {"type": "string", "enum": ["supported", "unsupported", "uncertain"]},
        "adoption_decision": {"type": "string", "enum": ["adopt_claim", "reject_claim", "withhold_judgment"]},
        "sharing_decision": {"type": "string", "enum": ["share", "share_with_caveat", "do_not_share"]},
        "content_ids_used": {"type": "array", "items": {"type": "string"}},
        "evidence_ids_used": {"type": "array", "items": {"type": "string"}},
        "share_content_id": {"type": ["string", "null"]},
    },
    "required": [
        "epistemic_stance", "adoption_decision", "sharing_decision",
        "content_ids_used", "evidence_ids_used", "share_content_id",
    ],
}


def render_hg231_turn(context: HG2PromptContext) -> LLMRequest:
    base = render_hg23_turn(context)
    digest = hashlib.sha256(f"{HG231_PROTOCOL_VERSION}|{base.request_id}".encode("utf-8")).hexdigest()[:24]
    return base.model_copy(update={
        "request_id": f"hg231-{digest}",
        "metadata": {
            **base.metadata,
            "transport_protocol_version": HG231_PROTOCOL_VERSION,
            "transport_schema_name": HG231_SCHEMA_NAME,
            "provider_subset_schema": True,
        },
    })


class HG231Runtime(HG23Runtime):
    def render_request(self, context: HG2PromptContext) -> LLMRequest:
        return render_hg231_turn(context)


__all__ = [
    "HG231_PROTOCOL_VERSION", "HG231_RESPONSE_JSON_SCHEMA", "HG231_SCHEMA_NAME",
    "HG231_TEMPLATE_VERSION", "HG231Runtime", "HG23AgentResponse",
    "parse_hg23_response", "render_hg231_turn",
]

