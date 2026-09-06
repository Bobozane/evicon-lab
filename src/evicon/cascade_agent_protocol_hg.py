"""Versioned public-content-identifiable cascade Agent protocol for H-G."""
from __future__ import annotations

import hashlib
import json
from typing import Final

from .cascade_agent_prompts import (
    CascadeAgentDirective,
    CascadeAgentPromptContext,
    CascadeAgentRuntimeConfig,
    CascadeVisibleClaim,
    CascadeVisibleContent,
    CascadeVisibleEvidence,
    CascadeVisibleRoot,
    render_cascade_agent_turn,
)
from .cascade_agent_protocol_v21 import CascadeAgentProtocolV21Runtime
from .cascade_intervention_application import StructuredCascadeDirective
from .llm_contract import LLMProvider, LLMRequest
from .provenance_cascade_exposure import ControllerPublicView

HG_PROTOCOL_VERSION: Final[str] = "provenance_cascade_agent_protocol.hg_identifiable.v1"
HG_TEMPLATE_VERSION: Final[str] = "cascade_agent_turn.hg_public_content.v1"
HG_SCHEMA_NAME: Final[str] = "cascade_agent_response_v2_1"
_PRIVATE_MARKERS = (
    "ground_truth_label",
    "source_independence_label",
    "evaluator_truth",
    "private_fixture",
    "api_key",
    "authorization",
)


def _assert_safe_text(value: str, code: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(code)
    lowered = cleaned.lower()
    if any(marker in lowered for marker in _PRIVATE_MARKERS):
        raise ValueError("private_marker_in_public_material")
    return cleaned


def build_identifiable_prompt_context(
    view: ControllerPublicView,
    *,
    claim_id: str,
    directive: StructuredCascadeDirective | None = None,
    runtime_config: CascadeAgentRuntimeConfig | None = None,
) -> CascadeAgentPromptContext:
    """Project only actually exposed public text into one Agent context."""
    target = next((item for item in view.claims if item.claim_id == claim_id), None)
    if target is None:
        raise ValueError("target_claim_not_visible")
    relation_map = {item.provenance_node_id: item.source_root_ids for item in view.root_relations}
    claims = tuple(
        CascadeVisibleClaim(
            claim_id=item.claim_id,
            verification_status=item.verification_status,
            visible_evidence_ids=tuple(item.evidence_card_ids),
            public_summary=_assert_safe_text(item.public_summary or "", "claim_public_summary_missing"),
        )
        for item in view.claims
    )
    contents = tuple(
        CascadeVisibleContent(
            content_id=node.content_id,
            provenance_node_id=node.node_id,
            claim_id=node.claim_id,
            source_root_ids=tuple(relation_map.get(node.node_id, ())),
            public_summary=_assert_safe_text(node.public_statement or "", "content_public_statement_missing"),
        )
        for node in view.provenance_nodes
    )
    evidence = tuple(
        CascadeVisibleEvidence(
            evidence_id=card.evidence_id,
            public_summary=_assert_safe_text(card.claim, "evidence_public_summary_missing"),
            supports=tuple(card.supports),
            contradicts=tuple(card.contradicts),
        )
        for card in view.evidence_cards
    )
    directive_model = None
    if directive is not None:
        if directive.scenario_id != view.scenario_id or directive.target_agent_id != view.agent_id:
            raise ValueError("directive_target_mismatch")
        directive_model = CascadeAgentDirective(
            directive_type=directive.directive_type,
            effective_round_id=directive.effective_round_id,
            allowed_content_ids=directive.visible_content_ids,
            allowed_evidence_ids=directive.visible_evidence_ids,
            visible_same_root_repetition_count=directive.visible_same_root_repetition_count,
            visible_root_count=directive.visible_root_count,
            reason_codes=directive.reason_codes,
        )
    return CascadeAgentPromptContext(
        scenario_id=view.scenario_id,
        agent_id=view.agent_id,
        round_id=view.round_id,
        claim_id=claim_id,
        claims=claims,
        contents=contents,
        evidence=evidence,
        roots=tuple(
            CascadeVisibleRoot(
                source_root_id=item.source_root_id,
                public_source_category=item.public_source_category.value,
            )
            for item in view.source_roots
        ),
        directive=directive_model,
        runtime_config=runtime_config or CascadeAgentRuntimeConfig(),
    )


def render_cascade_agent_turn_hg(context: CascadeAgentPromptContext) -> LLMRequest:
    base = render_cascade_agent_turn(context)
    payload = json.loads(base.user_prompt)
    payload["template_version"] = HG_TEMPLATE_VERSION
    payload["protocol_version"] = HG_PROTOCOL_VERSION
    payload["public_content_contract"] = "synthetic_exposed_summaries_only"
    user_prompt = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    system_prompt = (
        "Return exactly one JSON object with exactly four fields: stance, content_ids_used, "
        "evidence_ids_used, share_content_id. Interpret only the supplied synthetic public claim summaries, "
        "public statements, and already visible evidence. IDs are audit references, not claim meaning. "
        "A directive requests a verification process and is not a fact, truth label, source, or evidence. "
        "Do not invent content, evidence, provenance, control fields, or private annotations."
    )
    digest = hashlib.sha256(
        f"{context.scenario_id}|{context.agent_id}|{context.round_id}|{user_prompt}".encode()
    ).hexdigest()[:24]
    metadata = {
        **base.metadata,
        "contract_version": HG_PROTOCOL_VERSION,
        "template_version": HG_TEMPLATE_VERSION,
        "public_content_identifiable": True,
    }
    return base.model_copy(
        update={
            "request_id": f"hg-{digest}",
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "metadata": metadata,
        }
    )


class CascadeAgentProtocolHGRuntime(CascadeAgentProtocolV21Runtime):
    """Strict v2.1 response parsing with the H-G public-text request contract."""

    def render_request(self, context: CascadeAgentPromptContext) -> LLMRequest:
        return render_cascade_agent_turn_hg(context)

    def execute(self, context: CascadeAgentPromptContext, provider: LLMProvider, *, request_metadata=None):
        result = super().execute(context, provider, request_metadata=request_metadata)
        return result.model_copy(
            update={"audit": result.audit.model_copy(update={"template_version": HG_TEMPLATE_VERSION})}
        )


__all__ = [
    "HG_PROTOCOL_VERSION",
    "HG_SCHEMA_NAME",
    "HG_TEMPLATE_VERSION",
    "CascadeAgentProtocolHGRuntime",
    "build_identifiable_prompt_context",
    "render_cascade_agent_turn_hg",
]
