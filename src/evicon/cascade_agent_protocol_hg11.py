"""H-G.1.1 Agent protocol: H-G.1 semantics with a larger output reservation."""
from __future__ import annotations

import hashlib
import json
from typing import Final

from .cascade_agent_protocol_hg1 import build_identifiable_prompt_context
from .cascade_agent_protocol_v21 import CascadeAgentProtocolV21Runtime
from .cascade_agent_prompts import CascadeAgentPromptContext
from .llm_contract import LLMProvider, LLMRequest
from .cascade_agent_protocol_hg1 import render_cascade_agent_turn_hg1

HG11_PROTOCOL_VERSION: Final[str] = "provenance_cascade_agent_protocol.hg1_1_truncation.v1"
HG11_TEMPLATE_VERSION: Final[str] = "cascade_agent_turn.hg1_1_public_content.v1"
HG11_SCHEMA_NAME: Final[str] = "cascade_agent_response_v2_1"


def render_cascade_agent_turn_hg11(context: CascadeAgentPromptContext) -> LLMRequest:
    base = render_cascade_agent_turn_hg1(context)
    payload = json.loads(base.user_prompt)
    payload.update({
        "template_version": HG11_TEMPLATE_VERSION,
        "protocol_version": HG11_PROTOCOL_VERSION,
    })
    user_prompt = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(
        f"{context.scenario_id}|{context.agent_id}|{context.round_id}|{user_prompt}".encode()
    ).hexdigest()[:24]
    return base.model_copy(update={
        "request_id": f"hg11-{digest}",
        "user_prompt": user_prompt,
        "metadata": {
            **base.metadata,
            "contract_version": HG11_PROTOCOL_VERSION,
            "template_version": HG11_TEMPLATE_VERSION,
            "output_reservation_amendment": "max_tokens_1024_no_semantic_change",
        },
    })


class CascadeAgentProtocolHG11Runtime(CascadeAgentProtocolV21Runtime):
    def render_request(self, context: CascadeAgentPromptContext) -> LLMRequest:
        return render_cascade_agent_turn_hg11(context)

    def execute(self, context: CascadeAgentPromptContext, provider: LLMProvider, *, request_metadata=None):
        result = super().execute(context, provider, request_metadata=request_metadata)
        return result.model_copy(update={
            "audit": result.audit.model_copy(update={"template_version": HG11_TEMPLATE_VERSION})
        })


__all__ = [
    "HG11_PROTOCOL_VERSION", "HG11_TEMPLATE_VERSION", "HG11_SCHEMA_NAME",
    "CascadeAgentProtocolHG11Runtime", "build_identifiable_prompt_context",
    "render_cascade_agent_turn_hg11",
]
