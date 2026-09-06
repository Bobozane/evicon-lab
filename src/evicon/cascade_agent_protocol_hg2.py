"""H-G.2 epistemic/behavior separation protocol for offline calibration."""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Final, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cascade_agent_prompts import CascadeAgentPromptContext, CascadeAgentRuntimeConfig
from .cascade_agent_runtime import CascadeAgentRuntimeAudit, CascadeAgentRuntimeStatus
from .cascade_agent_protocol_hg import build_identifiable_prompt_context
from .cascade_outcomes import ClaimStance
from .llm_contract import LLMProvider, LLMProviderError, LLMRequest
from .request_ledger import RequestLedgerError

HG2_PROTOCOL_VERSION: Final[str] = "provenance_cascade_agent_protocol.hg2_epistemic_behavior.v1"
HG2_TEMPLATE_VERSION: Final[str] = "cascade_agent_turn.hg2_epistemic_behavior.v1"
HG2_SCHEMA_NAME: Final[str] = "cascade_agent_epistemic_behavior_response_v1"
HG2_RESPONSE_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "epistemic_stance": {"type": "string", "enum": ["supported", "unsupported", "uncertain"]},
        "behavioral_decision": {"type": "string", "enum": ["share", "share_with_caveat", "do_not_share", "support_reversible_action", "defer_action"]},
        "content_ids_used": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "evidence_ids_used": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "share_content_id": {"type": ["string", "null"]},
    },
    "required": ["epistemic_stance", "behavioral_decision", "content_ids_used", "evidence_ids_used", "share_content_id"],
}


class EpistemicStance(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"


class BehavioralDecision(str, Enum):
    SHARE = "share"
    SHARE_WITH_CAVEAT = "share_with_caveat"
    DO_NOT_SHARE = "do_not_share"
    SUPPORT_REVERSIBLE_ACTION = "support_reversible_action"
    DEFER_ACTION = "defer_action"


class PublicDecisionRole(str, Enum):
    HARM_AVOIDANCE = "harm_avoidance"
    VERIFICATION_FIRST = "verification_first"
    RAPID_RESPONSE = "rapid_response"
    REVERSIBILITY_FOCUSED = "reversibility_focused"
    PROCEDURAL_FAIRNESS = "procedural_fairness"
    COMMUNICATION_STABILITY = "communication_stability"


class HG2DecisionTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str = Field(min_length=1)
    public_decision_summary: str = Field(min_length=1)
    reversible_action: str = Field(min_length=1)
    defer_action: str = Field(min_length=1)
    allow_share_with_caveat: bool = True


class HG2PromptContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    public_context: CascadeAgentPromptContext
    role: PublicDecisionRole
    decision_task: HG2DecisionTask


class HG2ParserError(str, Enum):
    MALFORMED_JSON = "malformed_json"
    TOP_LEVEL_TYPE = "top_level_type"
    MISSING_FIELD = "missing_field"
    EXTRA_FIELD = "extra_field"
    FIELD_TYPE = "field_type"
    INVALID_EPISTEMIC_STANCE = "invalid_epistemic_stance"
    INVALID_BEHAVIORAL_DECISION = "invalid_behavioral_decision"
    UNAVAILABLE_CONTENT_ID = "unavailable_content_id"
    UNAVAILABLE_EVIDENCE_ID = "unavailable_evidence_id"
    SHARE_FIELD = "share_field"


class HG2AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    epistemic_stance: EpistemicStance
    behavioral_decision: BehavioralDecision
    content_ids_used: tuple[str, ...]
    evidence_ids_used: tuple[str, ...]
    share_content_id: str | None

    @property
    def stance(self) -> ClaimStance:
        return {
            EpistemicStance.SUPPORTED: ClaimStance.ENDORSES,
            EpistemicStance.UNSUPPORTED: ClaimStance.REJECTS,
            EpistemicStance.UNCERTAIN: ClaimStance.UNCERTAIN,
        }[self.epistemic_stance]


class HG2RuntimeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    status: CascadeAgentRuntimeStatus
    response: HG2AgentResponse | None = None
    audit: CascadeAgentRuntimeAudit


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate")
        output[key] = value
    return output


def parse_hg2_response(content: str, context: HG2PromptContext) -> tuple[HG2AgentResponse | None, HG2ParserError | None]:
    try:
        data = json.loads(content, object_pairs_hook=_pairs)
    except ValueError as exc:
        return None, HG2ParserError.EXTRA_FIELD if str(exc) == "duplicate" else HG2ParserError.MALFORMED_JSON
    if not isinstance(data, dict):
        return None, HG2ParserError.TOP_LEVEL_TYPE
    fields = {"epistemic_stance", "behavioral_decision", "content_ids_used", "evidence_ids_used", "share_content_id"}
    if set(data) - fields:
        return None, HG2ParserError.EXTRA_FIELD
    if set(data) != fields:
        return None, HG2ParserError.MISSING_FIELD
    if not isinstance(data["epistemic_stance"], str) or not isinstance(data["behavioral_decision"], str):
        return None, HG2ParserError.FIELD_TYPE
    try:
        epistemic = EpistemicStance(data["epistemic_stance"])
    except ValueError:
        return None, HG2ParserError.INVALID_EPISTEMIC_STANCE
    try:
        behavior = BehavioralDecision(data["behavioral_decision"])
    except ValueError:
        return None, HG2ParserError.INVALID_BEHAVIORAL_DECISION
    for field in ("content_ids_used", "evidence_ids_used"):
        value = data[field]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value) or len(value) != len(set(value)):
            return None, HG2ParserError.FIELD_TYPE
    share = data["share_content_id"]
    if share is not None and not isinstance(share, str):
        return None, HG2ParserError.SHARE_FIELD
    visible_content = {item.content_id for item in context.public_context.contents}
    visible_evidence = {item.evidence_id for item in context.public_context.evidence}
    content_ids = tuple(data["content_ids_used"])
    evidence_ids = tuple(data["evidence_ids_used"])
    if not set(content_ids).issubset(visible_content):
        return None, HG2ParserError.UNAVAILABLE_CONTENT_ID
    if not set(evidence_ids).issubset(visible_evidence):
        return None, HG2ParserError.UNAVAILABLE_EVIDENCE_ID
    sharing = behavior in {BehavioralDecision.SHARE, BehavioralDecision.SHARE_WITH_CAVEAT}
    if sharing and (share is None or share not in content_ids or share not in visible_content):
        return None, HG2ParserError.SHARE_FIELD
    if not sharing and share is not None:
        return None, HG2ParserError.SHARE_FIELD
    return HG2AgentResponse(
        epistemic_stance=epistemic, behavioral_decision=behavior,
        content_ids_used=content_ids, evidence_ids_used=evidence_ids,
        share_content_id=share,
    ), None


def render_hg2_turn(context: HG2PromptContext) -> LLMRequest:
    public = context.public_context
    payload = {
        "template_version": HG2_TEMPLATE_VERSION,
        "protocol_version": HG2_PROTOCOL_VERSION,
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
        },
    }
    user_prompt = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    system_prompt = (
        "Return exactly one JSON object with exactly five fields: epistemic_stance, behavioral_decision, "
        "content_ids_used, evidence_ids_used, share_content_id. Separate evidential confidence from the public "
        "decision. A reversible action or caveated share does not assert that a claim is true. Use only exposed "
        "synthetic public content and evidence. A directive is a process request, not a fact, verdict, source, or "
        "evidence. Never invent claims, content, evidence, provenance, control fields, or private annotations."
    )
    digest = hashlib.sha256(
        f"{public.scenario_id}|{public.agent_id}|{public.round_id}|{context.role.value}|{user_prompt}".encode()
    ).hexdigest()[:24]
    settings = public.runtime_config
    return LLMRequest(
        request_id=f"hg2-{digest}", model_name=settings.model_name,
        system_prompt=system_prompt, user_prompt=user_prompt,
        temperature=settings.temperature, max_tokens=settings.max_tokens,
        seed=settings.seed,
        metadata={
            "contract_version": HG2_PROTOCOL_VERSION,
            "template_version": HG2_TEMPLATE_VERSION,
            "response_schema_name": HG2_SCHEMA_NAME,
            "epistemic_behavior_separated": True,
            "agent_id": public.agent_id,
            "round_id": public.round_id,
        },
    )


class HG2Runtime:
    """One-call runtime; decisions are retained only in this in-memory instance."""

    def __init__(self) -> None:
        self.decisions: list[tuple[HG2PromptContext, HG2AgentResponse]] = []

    def render_request(self, context: HG2PromptContext) -> LLMRequest:
        return render_hg2_turn(context)

    def execute(self, context: HG2PromptContext, provider: LLMProvider, *, request_metadata: Mapping[str, Any] | None = None) -> HG2RuntimeResult:
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
        parsed, error = parse_hg2_response(provider_response.content, context)
        if parsed is None:
            return self._result(
                context, CascadeAgentRuntimeStatus.PARSER_INVALID,
                request_id=request.request_id, provider_response=provider_response,
                parser_valid=False, error=error.value if error else "invalid_schema",
            )
        self.decisions.append((context, parsed))
        return self._result(
            context, CascadeAgentRuntimeStatus.COMPLETED, response=parsed,
            request_id=request.request_id, provider_response=provider_response,
            parser_valid=True,
        )

    @staticmethod
    def _result(context: HG2PromptContext, status: CascadeAgentRuntimeStatus, *, response: HG2AgentResponse | None = None, request_id: str | None = None, provider_response: Any | None = None, parser_valid: bool | None = None, error: str | None = None) -> HG2RuntimeResult:
        public = context.public_context
        return HG2RuntimeResult(
            status=status, response=response,
            audit=CascadeAgentRuntimeAudit(
                status=status, scenario_id=public.scenario_id,
                agent_id=public.agent_id, round_id=public.round_id,
                template_version=HG2_TEMPLATE_VERSION,
                model_name=(provider_response.model_name if provider_response is not None else public.runtime_config.model_name),
                request_id=request_id,
                finish_reason=(provider_response.finish_reason if provider_response is not None else None),
                prompt_tokens=(provider_response.prompt_tokens if provider_response is not None else None),
                completion_tokens=(provider_response.completion_tokens if provider_response is not None else None),
                total_tokens=(provider_response.total_tokens if provider_response is not None else None),
                latency_ms=(provider_response.latency_ms if provider_response is not None else None),
                parser_valid=parser_valid,
                content_ids_used_count=len(response.content_ids_used) if response else 0,
                evidence_ids_used_count=len(response.evidence_ids_used) if response else 0,
                share_requested=response.share_content_id is not None if response else False,
                error_code=error,
            ),
        )


def build_hg2_context(public_context: CascadeAgentPromptContext, *, role: PublicDecisionRole, task: HG2DecisionTask) -> HG2PromptContext:
    return HG2PromptContext(public_context=public_context, role=role, decision_task=task)


__all__ = [
    "BehavioralDecision", "EpistemicStance", "HG2AgentResponse", "HG2DecisionTask",
    "HG2ParserError", "HG2PromptContext", "HG2Runtime", "HG2RuntimeResult",
    "HG2_PROTOCOL_VERSION", "HG2_SCHEMA_NAME", "HG2_TEMPLATE_VERSION",
    "HG2_RESPONSE_JSON_SCHEMA",
    "PublicDecisionRole", "build_hg2_context", "build_identifiable_prompt_context",
    "parse_hg2_response", "render_hg2_turn",
]
