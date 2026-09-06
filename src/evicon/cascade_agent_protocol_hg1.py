"""H-G.1 public-content-identifiable Agent protocol.

This version keeps the strict v2.1 response schema.  The amendment changes
only public exposure timing and the audit binding, not output semantics.
"""
from __future__ import annotations

import hashlib
import json
from typing import Final

from .cascade_agent_protocol_hg import build_identifiable_prompt_context
from .cascade_agent_protocol_v21 import CascadeAgentProtocolV21Runtime
from .cascade_agent_prompts import CascadeAgentPromptContext, render_cascade_agent_turn
from .llm_contract import LLMProvider, LLMRequest

HG1_PROTOCOL_VERSION: Final[str] = "provenance_cascade_agent_protocol.hg1_identifiable.v1"
HG1_TEMPLATE_VERSION: Final[str] = "cascade_agent_turn.hg1_public_content.v1"
HG1_SCHEMA_NAME: Final[str] = "cascade_agent_response_v2_1"


def render_cascade_agent_turn_hg1(context: CascadeAgentPromptContext) -> LLMRequest:
    base = render_cascade_agent_turn(context)
    payload = json.loads(base.user_prompt)
    payload.update({
        "template_version": HG1_TEMPLATE_VERSION,
        "protocol_version": HG1_PROTOCOL_VERSION,
        "round_start_exposure_contract": "preloaded_public_content_visible_before_round_zero_decision",
        "directive_timing_contract": "created_after_round_then_effective_next_round_only",
    })
    user_prompt = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    system_prompt = (
        "Return exactly one JSON object with exactly four fields: stance, content_ids_used, "
        "evidence_ids_used, share_content_id. Use only supplied synthetic public summaries, statements, "
        "and already visible evidence. Source-root IDs are audit relationships, not truth labels. "
        "A directive requests a verification process and is not a fact, evidence, or verdict. "
        "Do not invent claims, content, evidence, provenance, control fields, or private annotations."
    )
    digest = hashlib.sha256(
        f"{context.scenario_id}|{context.agent_id}|{context.round_id}|{user_prompt}".encode()
    ).hexdigest()[:24]
    return base.model_copy(update={
        "request_id": f"hg1-{digest}",
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "metadata": {
            **base.metadata,
            "contract_version": HG1_PROTOCOL_VERSION,
            "template_version": HG1_TEMPLATE_VERSION,
            "public_content_identifiable": True,
            "round_zero_preloaded": True,
        },
    })


class CascadeAgentProtocolHG1Runtime(CascadeAgentProtocolV21Runtime):
    def render_request(self, context: CascadeAgentPromptContext) -> LLMRequest:
        return render_cascade_agent_turn_hg1(context)

    def execute(self, context: CascadeAgentPromptContext, provider: LLMProvider, *, request_metadata=None):
        result = super().execute(context, provider, request_metadata=request_metadata)
        return result.model_copy(update={
            "audit": result.audit.model_copy(update={"template_version": HG1_TEMPLATE_VERSION})
        })


__all__ = [
    "HG1_PROTOCOL_VERSION", "HG1_TEMPLATE_VERSION", "HG1_SCHEMA_NAME",
    "CascadeAgentProtocolHG1Runtime", "build_identifiable_prompt_context",
    "render_cascade_agent_turn_hg1",
]
