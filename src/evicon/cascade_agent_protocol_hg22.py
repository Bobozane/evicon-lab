"""H-G.2.2 explicit share-semantics protocol amendment."""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Final, Mapping

from .cascade_agent_protocol_hg2 import (
    BehavioralDecision,
    EpistemicStance,
    HG2AgentResponse,
    HG2PromptContext,
    HG2RuntimeResult,
)
from .cascade_agent_runtime import CascadeAgentRuntimeAudit, CascadeAgentRuntimeStatus
from .llm_contract import LLMProvider, LLMProviderError, LLMRequest
from .request_ledger import RequestLedgerError

HG22_PROTOCOL_VERSION: Final[str] = "provenance_cascade_agent_protocol.hg2_2_share_contract.v1"
HG22_TEMPLATE_VERSION: Final[str] = "cascade_agent_turn.hg2_epistemic_behavior.share_contract.v2"
HG22_SCHEMA_NAME: Final[str] = "cascade_agent_epistemic_behavior_share_contract_v2"
HG22_RESPONSE_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "epistemic_stance": {"type": "string", "enum": ["supported", "unsupported", "uncertain"]},
        "behavioral_decision": {
            "type": "string",
            "enum": ["share", "share_with_caveat", "do_not_share", "support_reversible_action", "defer_action"],
        },
        "content_ids_used": {"type": "array", "items": {"type": "string"}},
        "evidence_ids_used": {"type": "array", "items": {"type": "string"}},
        "share_content_id": {"type": ["string", "null"]},
    },
    "required": [
        "epistemic_stance",
        "behavioral_decision",
        "content_ids_used",
        "evidence_ids_used",
        "share_content_id",
    ],
}


class HG22ParserError(str, Enum):
    MALFORMED_JSON = "malformed_json"
    TOP_LEVEL_TYPE = "top_level_type"
    MISSING_FIELD = "missing_field"
    EXTRA_FIELD = "extra_field"
    FIELD_TYPE = "field_type"
    INVALID_EPISTEMIC_STANCE = "invalid_epistemic_stance"
    INVALID_BEHAVIORAL_DECISION = "invalid_behavioral_decision"
    UNAVAILABLE_CONTENT_ID = "unavailable_content_id"
    UNAVAILABLE_EVIDENCE_ID = "unavailable_evidence_id"
    SHARE_FIELD_TYPE = "share_field_type"
    SHARE_BEHAVIOR_INCONSISTENT = "share_behavior_inconsistent"


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate")
        output[key] = value
    return output


def parse_hg22_response(content: str, context: HG2PromptContext) -> tuple[HG2AgentResponse | None, HG22ParserError | None]:
    try:
        data = json.loads(content, object_pairs_hook=_pairs)
    except ValueError as exc:
        return None, HG22ParserError.EXTRA_FIELD if str(exc) == "duplicate" else HG22ParserError.MALFORMED_JSON
    if not isinstance(data, dict):
        return None, HG22ParserError.TOP_LEVEL_TYPE
    fields = {"epistemic_stance", "behavioral_decision", "content_ids_used", "evidence_ids_used", "share_content_id"}
    if set(data) - fields:
        return None, HG22ParserError.EXTRA_FIELD
    if set(data) != fields:
        return None, HG22ParserError.MISSING_FIELD
    if not isinstance(data["epistemic_stance"], str) or not isinstance(data["behavioral_decision"], str):
        return None, HG22ParserError.FIELD_TYPE
    try:
        epistemic = EpistemicStance(data["epistemic_stance"])
    except ValueError:
        return None, HG22ParserError.INVALID_EPISTEMIC_STANCE
    try:
        behavior = BehavioralDecision(data["behavioral_decision"])
    except ValueError:
        return None, HG22ParserError.INVALID_BEHAVIORAL_DECISION
    for field in ("content_ids_used", "evidence_ids_used"):
        value = data[field]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value) or len(value) != len(set(value)):
            return None, HG22ParserError.FIELD_TYPE
    share = data["share_content_id"]
    if share is not None and not isinstance(share, str):
        return None, HG22ParserError.SHARE_FIELD_TYPE
    visible_content = {item.content_id for item in context.public_context.contents}
    visible_evidence = {item.evidence_id for item in context.public_context.evidence}
    content_ids = tuple(data["content_ids_used"])
    evidence_ids = tuple(data["evidence_ids_used"])
    if not set(content_ids).issubset(visible_content):
        return None, HG22ParserError.UNAVAILABLE_CONTENT_ID
    if not set(evidence_ids).issubset(visible_evidence):
        return None, HG22ParserError.UNAVAILABLE_EVIDENCE_ID
    sharing = behavior in {BehavioralDecision.SHARE, BehavioralDecision.SHARE_WITH_CAVEAT}
    if sharing != (share is not None):
        return None, HG22ParserError.SHARE_BEHAVIOR_INCONSISTENT
    if sharing and (share not in content_ids or share not in visible_content):
        return None, HG22ParserError.SHARE_BEHAVIOR_INCONSISTENT
    return HG2AgentResponse(
        epistemic_stance=epistemic,
        behavioral_decision=behavior,
        content_ids_used=content_ids,
        evidence_ids_used=evidence_ids,
        share_content_id=share,
    ), None


def render_hg22_turn(context: HG2PromptContext) -> LLMRequest:
    public = context.public_context
    payload = {
        "template_version": HG22_TEMPLATE_VERSION,
        "protocol_version": HG22_PROTOCOL_VERSION,
        "scenario_id": public.scenario_id,
        "agent_id": public.agent_id,
        "round_id": public.round_id,
        "target_claim_id": public.claim_id,
        "public_role": context.role.value,
        "decision_task": context.decision_task.model_dump(mode="json"),
        "visible_claims": [item.model_dump(mode="json") for item in public.claims],
        "visible_contents": [item.model_dump(mode="json") for item in public.contents],
        "visible_evidence": [item.model_dump(mode="json") for item in public.evidence],
        "visible_roots": [item.model_dump(mode="json") for item in public.roots],
        "directive": public.directive.model_dump(mode="json") if public.directive is not None else None,
        "decision_contract": {
            "epistemic_stance_is_not_behavior": True,
            "uncertainty_may_coexist_with_reversible_action": True,
            "sharing_does_not_assert_truth": True,
            "directive_is_process_not_evidence": True,
            "share_requires_non_null_visible_used_content_id": True,
            "non_share_requires_null_share_content_id": True,
        },
    }
    user_prompt = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    system_prompt = (
        "Return exactly one JSON object with exactly five fields: epistemic_stance, behavioral_decision, "
        "content_ids_used, evidence_ids_used, share_content_id. If behavioral_decision is share or "
        "share_with_caveat, share_content_id MUST be one non-null ID that also appears in content_ids_used and "
        "is visible. If behavioral_decision is do_not_share, support_reversible_action, or defer_action, "
        "share_content_id MUST be null. content_ids_used may list visible content considered even when not "
        "sharing. Separate evidential confidence from behavior. A directive is process, not evidence or truth. "
        "Never invent claims, content, evidence, provenance, control fields, or private annotations."
    )
    digest = hashlib.sha256(
        f"{public.scenario_id}|{public.agent_id}|{public.round_id}|{context.role.value}|{user_prompt}".encode()
    ).hexdigest()[:24]
    settings = public.runtime_config
    return LLMRequest(
        request_id=f"hg22-{digest}",
        model_name=settings.model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        seed=settings.seed,
        metadata={
            "contract_version": HG22_PROTOCOL_VERSION,
            "template_version": HG22_TEMPLATE_VERSION,
            "response_schema_name": HG22_SCHEMA_NAME,
            "epistemic_behavior_separated": True,
            "share_semantics_explicit": True,
            "agent_id": public.agent_id,
            "round_id": public.round_id,
        },
    )


class HG22Runtime:
    """One-call runtime with strict structural and local semantic validation."""

    def __init__(self) -> None:
        self.decisions: list[tuple[HG2PromptContext, HG2AgentResponse]] = []

    def render_request(self, context: HG2PromptContext) -> LLMRequest:
        return render_hg22_turn(context)

    def execute(self, context: HG2PromptContext, provider: LLMProvider, *, request_metadata: Mapping[str, Any] | None = None) -> HG2RuntimeResult:
        public = context.public_context
        try:
            request = self.render_request(context)
            if request_metadata:
                allowed = {
                    key: value for key, value in request_metadata.items()
                    if key in {"protocol", "condition", "phase", "run_id", "seed", "matched_group_id"}
                }
                request = request.model_copy(update={"metadata": {**request.metadata, **allowed}})
        except Exception:
            return self._result(context, CascadeAgentRuntimeStatus.RENDER_ERROR, error="request_render_failed")
        try:
            provider_response = provider.complete(request)
        except LLMProviderError as error:
            code = error.message if isinstance(error, RequestLedgerError) else error.code.value
            return self._result(context, CascadeAgentRuntimeStatus.PROVIDER_ERROR, request_id=request.request_id, error=code)
        except Exception:
            return self._result(context, CascadeAgentRuntimeStatus.PROVIDER_ERROR, request_id=request.request_id, error="provider_failure")
        parsed, error = parse_hg22_response(provider_response.content, context)
        if parsed is None:
            return self._result(
                context,
                CascadeAgentRuntimeStatus.PARSER_INVALID,
                request_id=request.request_id,
                provider_response=provider_response,
                parser_valid=False,
                error=error.value if error else "invalid_schema",
            )
        self.decisions.append((context, parsed))
        return self._result(
            context,
            CascadeAgentRuntimeStatus.COMPLETED,
            response=parsed,
            request_id=request.request_id,
            provider_response=provider_response,
            parser_valid=True,
        )

    @staticmethod
    def _result(
        context: HG2PromptContext,
        status: CascadeAgentRuntimeStatus,
        *,
        response: HG2AgentResponse | None = None,
        request_id: str | None = None,
        provider_response: Any | None = None,
        parser_valid: bool | None = None,
        error: str | None = None,
    ) -> HG2RuntimeResult:
        public = context.public_context
        return HG2RuntimeResult(
            status=status,
            response=response,
            audit=CascadeAgentRuntimeAudit(
                status=status,
                scenario_id=public.scenario_id,
                agent_id=public.agent_id,
                round_id=public.round_id,
                template_version=HG22_TEMPLATE_VERSION,
                model_name=provider_response.model_name if provider_response is not None else public.runtime_config.model_name,
                request_id=request_id,
                finish_reason=provider_response.finish_reason if provider_response is not None else None,
                prompt_tokens=provider_response.prompt_tokens if provider_response is not None else None,
                completion_tokens=provider_response.completion_tokens if provider_response is not None else None,
                total_tokens=provider_response.total_tokens if provider_response is not None else None,
                latency_ms=provider_response.latency_ms if provider_response is not None else None,
                parser_valid=parser_valid,
                content_ids_used_count=len(response.content_ids_used) if response else 0,
                evidence_ids_used_count=len(response.evidence_ids_used) if response else 0,
                share_requested=response.share_content_id is not None if response else False,
                error_code=error,
            ),
        )


__all__ = [
    "HG22ParserError",
    "HG22Runtime",
    "HG22_PROTOCOL_VERSION",
    "HG22_RESPONSE_JSON_SCHEMA",
    "HG22_SCHEMA_NAME",
    "HG22_TEMPLATE_VERSION",
    "parse_hg22_response",
    "render_hg22_turn",
]
