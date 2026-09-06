"""H-G.2.3 observable adoption and sharing protocol.

Epistemic confidence, public claim adoption, and sharing are deliberately
separate.  Public outcome stance is derived from adoption, never from the
model's confidence label.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Final, Mapping

from pydantic import BaseModel, ConfigDict

from .cascade_agent_protocol_hg2 import EpistemicStance, HG2PromptContext
from .cascade_agent_runtime import CascadeAgentRuntimeAudit, CascadeAgentRuntimeStatus
from .cascade_outcomes import ClaimStance
from .llm_contract import LLMProvider, LLMProviderError, LLMRequest
from .request_ledger import RequestLedgerError

HG23_PROTOCOL_VERSION: Final[str] = "provenance_cascade_agent_protocol.hg2_3_observable_adoption.v1"
HG23_TEMPLATE_VERSION: Final[str] = "cascade_agent_turn.hg2_3_observable_adoption.v1"
HG23_SCHEMA_NAME: Final[str] = "cascade_agent_epistemic_adoption_sharing_v1"


class AdoptionDecision(str, Enum):
    ADOPT_CLAIM = "adopt_claim"
    REJECT_CLAIM = "reject_claim"
    WITHHOLD_JUDGMENT = "withhold_judgment"


class SharingDecision(str, Enum):
    SHARE = "share"
    SHARE_WITH_CAVEAT = "share_with_caveat"
    DO_NOT_SHARE = "do_not_share"


HG23_RESPONSE_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "epistemic_stance": {"type": "string", "enum": [item.value for item in EpistemicStance]},
        "adoption_decision": {"type": "string", "enum": [item.value for item in AdoptionDecision]},
        "sharing_decision": {"type": "string", "enum": [item.value for item in SharingDecision]},
        "content_ids_used": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "evidence_ids_used": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "share_content_id": {"type": ["string", "null"]},
    },
    "required": [
        "epistemic_stance", "adoption_decision", "sharing_decision",
        "content_ids_used", "evidence_ids_used", "share_content_id",
    ],
}


class HG23ParserError(str, Enum):
    MALFORMED_JSON = "malformed_json"
    TOP_LEVEL_TYPE = "top_level_type"
    MISSING_FIELD = "missing_field"
    EXTRA_FIELD = "extra_field"
    FIELD_TYPE = "field_type"
    INVALID_EPISTEMIC_STANCE = "invalid_epistemic_stance"
    INVALID_ADOPTION_DECISION = "invalid_adoption_decision"
    INVALID_SHARING_DECISION = "invalid_sharing_decision"
    UNAVAILABLE_CONTENT_ID = "unavailable_content_id"
    UNAVAILABLE_EVIDENCE_ID = "unavailable_evidence_id"
    SHARE_BEHAVIOR_INCONSISTENT = "share_behavior_inconsistent"


class HG23AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    epistemic_stance: EpistemicStance
    adoption_decision: AdoptionDecision
    sharing_decision: SharingDecision
    content_ids_used: tuple[str, ...]
    evidence_ids_used: tuple[str, ...]
    share_content_id: str | None

    @property
    def stance(self) -> ClaimStance:
        return {
            AdoptionDecision.ADOPT_CLAIM: ClaimStance.ENDORSES,
            AdoptionDecision.REJECT_CLAIM: ClaimStance.REJECTS,
            AdoptionDecision.WITHHOLD_JUDGMENT: ClaimStance.UNCERTAIN,
        }[self.adoption_decision]


class HG23RuntimeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    status: CascadeAgentRuntimeStatus
    response: HG23AgentResponse | None = None
    audit: CascadeAgentRuntimeAudit


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate")
        result[key] = value
    return result


def parse_hg23_response(content: str, context: HG2PromptContext) -> tuple[HG23AgentResponse | None, HG23ParserError | None]:
    try:
        data = json.loads(content, object_pairs_hook=_pairs)
    except ValueError as exc:
        return None, HG23ParserError.EXTRA_FIELD if str(exc) == "duplicate" else HG23ParserError.MALFORMED_JSON
    if not isinstance(data, dict):
        return None, HG23ParserError.TOP_LEVEL_TYPE
    fields = {
        "epistemic_stance", "adoption_decision", "sharing_decision",
        "content_ids_used", "evidence_ids_used", "share_content_id",
    }
    if set(data) - fields:
        return None, HG23ParserError.EXTRA_FIELD
    if set(data) != fields:
        return None, HG23ParserError.MISSING_FIELD
    if any(not isinstance(data[field], str) for field in ("epistemic_stance", "adoption_decision", "sharing_decision")):
        return None, HG23ParserError.FIELD_TYPE
    try:
        epistemic = EpistemicStance(data["epistemic_stance"])
    except ValueError:
        return None, HG23ParserError.INVALID_EPISTEMIC_STANCE
    try:
        adoption = AdoptionDecision(data["adoption_decision"])
    except ValueError:
        return None, HG23ParserError.INVALID_ADOPTION_DECISION
    try:
        sharing = SharingDecision(data["sharing_decision"])
    except ValueError:
        return None, HG23ParserError.INVALID_SHARING_DECISION
    for field in ("content_ids_used", "evidence_ids_used"):
        value = data[field]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value) or len(value) != len(set(value)):
            return None, HG23ParserError.FIELD_TYPE
    share = data["share_content_id"]
    if share is not None and not isinstance(share, str):
        return None, HG23ParserError.FIELD_TYPE
    visible_content = {item.content_id for item in context.public_context.contents}
    visible_evidence = {item.evidence_id for item in context.public_context.evidence}
    content_ids = tuple(data["content_ids_used"])
    evidence_ids = tuple(data["evidence_ids_used"])
    if not set(content_ids).issubset(visible_content):
        return None, HG23ParserError.UNAVAILABLE_CONTENT_ID
    if not set(evidence_ids).issubset(visible_evidence):
        return None, HG23ParserError.UNAVAILABLE_EVIDENCE_ID
    is_sharing = sharing in {SharingDecision.SHARE, SharingDecision.SHARE_WITH_CAVEAT}
    if is_sharing != (share is not None):
        return None, HG23ParserError.SHARE_BEHAVIOR_INCONSISTENT
    if is_sharing and (share not in content_ids or share not in visible_content):
        return None, HG23ParserError.SHARE_BEHAVIOR_INCONSISTENT
    return HG23AgentResponse(
        epistemic_stance=epistemic,
        adoption_decision=adoption,
        sharing_decision=sharing,
        content_ids_used=content_ids,
        evidence_ids_used=evidence_ids,
        share_content_id=share,
    ), None


def render_hg23_turn(context: HG2PromptContext) -> LLMRequest:
    public = context.public_context
    payload = {
        "template_version": HG23_TEMPLATE_VERSION,
        "protocol_version": HG23_PROTOCOL_VERSION,
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
            "epistemic_confidence_is_separate_from_public_adoption": True,
            "adopt_claim_means_use_as_current_reversible_public_premise_not_certify_truth": True,
            "sharing_is_separate_from_adoption": True,
            "directive_is_process_not_evidence": True,
        },
    }
    user_prompt = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    system_prompt = (
        "Return exactly one JSON object with exactly six fields: epistemic_stance, adoption_decision, "
        "sharing_decision, content_ids_used, evidence_ids_used, share_content_id. Epistemic stance reports "
        "evidential confidence. adoption_decision reports whether you would publicly use the target claim as "
        "the current premise for the stated reversible decision; adopt_claim does not certify objective truth. "
        "Sharing is a separate decision. A directive is process, not fact or evidence. Use only visible IDs. "
        "Never invent claims, evidence, provenance, control fields, or private annotations."
    )
    digest = hashlib.sha256(
        f"{public.scenario_id}|{public.agent_id}|{public.round_id}|{context.role.value}|{user_prompt}".encode()
    ).hexdigest()[:24]
    settings = public.runtime_config
    return LLMRequest(
        request_id=f"hg23-{digest}", model_name=settings.model_name,
        system_prompt=system_prompt, user_prompt=user_prompt,
        temperature=settings.temperature, max_tokens=settings.max_tokens, seed=settings.seed,
        metadata={
            "contract_version": HG23_PROTOCOL_VERSION,
            "template_version": HG23_TEMPLATE_VERSION,
            "response_schema_name": HG23_SCHEMA_NAME,
            "agent_id": public.agent_id,
            "round_id": public.round_id,
        },
    )


class HG23Runtime:
    def __init__(self) -> None:
        self.decisions: list[tuple[HG2PromptContext, HG23AgentResponse]] = []

    def render_request(self, context: HG2PromptContext) -> LLMRequest:
        return render_hg23_turn(context)

    def execute(self, context: HG2PromptContext, provider: LLMProvider, *, request_metadata: Mapping[str, Any] | None = None) -> HG23RuntimeResult:
        public = context.public_context
        try:
            request = self.render_request(context)
            if request_metadata:
                allowed = {key: value for key, value in request_metadata.items() if key in {"protocol", "condition", "phase", "run_id", "seed", "matched_group_id"}}
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
        parsed, error = parse_hg23_response(provider_response.content, context)
        if parsed is None:
            return self._result(context, CascadeAgentRuntimeStatus.PARSER_INVALID, request_id=request.request_id, provider_response=provider_response, parser_valid=False, error=error.value if error else "invalid_schema")
        self.decisions.append((context, parsed))
        return self._result(context, CascadeAgentRuntimeStatus.COMPLETED, response=parsed, request_id=request.request_id, provider_response=provider_response, parser_valid=True)

    @staticmethod
    def _result(context: HG2PromptContext, status: CascadeAgentRuntimeStatus, *, response: HG23AgentResponse | None = None, request_id: str | None = None, provider_response: Any | None = None, parser_valid: bool | None = None, error: str | None = None) -> HG23RuntimeResult:
        public = context.public_context
        return HG23RuntimeResult(
            status=status,
            response=response,
            audit=CascadeAgentRuntimeAudit(
                status=status, scenario_id=public.scenario_id, agent_id=public.agent_id,
                round_id=public.round_id, template_version=HG23_TEMPLATE_VERSION,
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
    "AdoptionDecision", "SharingDecision", "HG23AgentResponse", "HG23ParserError", "HG23Runtime", "HG23RuntimeResult",
    "HG23_PROTOCOL_VERSION", "HG23_TEMPLATE_VERSION", "HG23_SCHEMA_NAME", "HG23_RESPONSE_JSON_SCHEMA",
    "parse_hg23_response", "render_hg23_turn",
]
