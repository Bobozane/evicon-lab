"""Public-only prompt contract for one provenance-cascade Agent call.

This module is deliberately narrower than ``ControllerPublicView``.  It never
accepts a graph, ledger, evaluator record, or controller configuration.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .cascade_intervention_application import CascadeDirectiveType, StructuredCascadeDirective
from .llm_contract import LLMRequest
from .models import EvidenceCard
from .models._validation import identifier_list, normalized_text
from .provenance_cascade import VerificationStatus
from .provenance_cascade_exposure import ControllerPublicView


CASCADE_AGENT_TEMPLATE_VERSION: Final[str] = "cascade_agent_turn.v1"


class CascadeAgentRuntimeConfig(BaseModel):
    """Non-secret settings fixed outside the prompt context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str = Field(default="cascade-agent-runtime-v1", min_length=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, gt=0)
    seed: int = 0
    template_version: Literal["cascade_agent_turn.v1"] = CASCADE_AGENT_TEMPLATE_VERSION

    @field_validator("model_name", "template_version")
    @classmethod
    def clean_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("seed", mode="before")
    @classmethod
    def validate_seed(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("seed must be an integer")
        return value


class CascadeVisibleClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    verification_status: VerificationStatus
    visible_evidence_ids: tuple[str, ...] = ()
    public_summary: str = Field(min_length=1)

    @field_validator("public_summary")
    @classmethod
    def public_text(cls, value: str) -> str:
        return _public_text(value, "public_summary")

    @field_validator("claim_id")
    @classmethod
    def clean_id(cls, value: str) -> str:
        return normalized_text(value, "claim_id")

    @field_validator("visible_evidence_ids")
    @classmethod
    def clean_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(identifier_list(list(value), "visible_evidence_ids"))


class CascadeVisibleContent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    content_id: str = Field(min_length=1)
    provenance_node_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    source_root_ids: tuple[str, ...] = ()
    public_summary: str = Field(min_length=1)

    @field_validator("public_summary")
    @classmethod
    def public_text(cls, value: str) -> str:
        return _public_text(value, "public_summary")

    @field_validator("content_id", "provenance_node_id", "claim_id")
    @classmethod
    def clean_id(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "id"))

    @field_validator("source_root_ids")
    @classmethod
    def clean_roots(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(identifier_list(list(value), "source_root_ids"))


class CascadeVisibleEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    public_summary: str = Field(min_length=1)
    supports: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()

    @field_validator("evidence_id")
    @classmethod
    def clean_id(cls, value: str) -> str:
        return normalized_text(value, "evidence_id")

    @field_validator("public_summary")
    @classmethod
    def public_text(cls, value: str) -> str:
        return _public_text(value, "public_summary")

    @field_validator("supports", "contradicts")
    @classmethod
    def clean_claims(cls, value: tuple[str, ...], info: object) -> tuple[str, ...]:
        return tuple(identifier_list(list(value), getattr(info, "field_name", "claim_ids")))


class CascadeVisibleRoot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_root_id: str = Field(min_length=1)
    public_source_category: str = Field(min_length=1)

    @field_validator("public_source_category")
    @classmethod
    def public_text(cls, value: str) -> str:
        return _public_text(value, "public_source_category")


class CascadeAgentDirective(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    directive_type: CascadeDirectiveType
    effective_round_id: int = Field(ge=0)
    allowed_content_ids: tuple[str, ...] = ()
    allowed_evidence_ids: tuple[str, ...] = ()
    visible_same_root_repetition_count: int = Field(default=0, ge=0)
    visible_root_count: int = Field(default=0, ge=0)
    reason_codes: tuple[str, ...] = ()

    @field_validator("allowed_content_ids", "allowed_evidence_ids", "reason_codes")
    @classmethod
    def clean_ids(cls, value: tuple[str, ...], info: object) -> tuple[str, ...]:
        return tuple(identifier_list(list(value), getattr(info, "field_name", "ids")))


class CascadeAgentPromptContext(BaseModel):
    """The only public data accepted by ``CascadeAgentRuntime``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    claim_id: str | None = None
    claims: tuple[CascadeVisibleClaim, ...] = ()
    contents: tuple[CascadeVisibleContent, ...] = ()
    evidence: tuple[CascadeVisibleEvidence, ...] = ()
    roots: tuple[CascadeVisibleRoot, ...] = ()
    directive: CascadeAgentDirective | None = None
    runtime_config: CascadeAgentRuntimeConfig = Field(default_factory=CascadeAgentRuntimeConfig)

    @field_validator("scenario_id", "agent_id")
    @classmethod
    def clean_ids(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "id"))

    @field_validator("claim_id")
    @classmethod
    def clean_claim_id(cls, value: str | None) -> str | None:
        return normalized_text(value, "claim_id") if value is not None else None

    @model_validator(mode="after")
    def validate_public_projection(self) -> "CascadeAgentPromptContext":
        claims = {item.claim_id: item for item in self.claims}
        contents = {item.content_id: item for item in self.contents}
        evidence = {item.evidence_id: item for item in self.evidence}
        roots = {item.source_root_id for item in self.roots}
        if len(claims) != len(self.claims) or len(contents) != len(self.contents) or len(evidence) != len(self.evidence):
            raise ValueError("public context IDs must be unique")
        if self.claim_id is not None and self.claim_id not in claims:
            raise ValueError("target claim is not visible")
        if any(item.claim_id not in claims for item in self.contents):
            raise ValueError("content references a non-visible claim")
        if any(root not in roots for item in self.contents for root in item.source_root_ids):
            raise ValueError("content references a non-visible root")
        claim_evidence = {eid for item in claims.values() for eid in item.visible_evidence_ids}
        if not set(evidence).issubset(claim_evidence):
            raise ValueError("evidence is not allowed by a visible claim")
        if self.directive is not None:
            if self.directive.effective_round_id != self.round_id:
                raise ValueError("directive must be effective in the current round")
            if not set(self.directive.allowed_content_ids).issubset(contents):
                raise ValueError("directive references unavailable content")
            if not set(self.directive.allowed_evidence_ids).issubset(evidence):
                raise ValueError("directive references unavailable evidence")
        return self

    @classmethod
    def from_public_view(
        cls,
        view: ControllerPublicView,
        *,
        claim_id: str,
        directive: StructuredCascadeDirective | None = None,
        runtime_config: CascadeAgentRuntimeConfig | None = None,
    ) -> "CascadeAgentPromptContext":
        """Build a context using only one already validated public view."""
        claim = next((item for item in view.claims if item.claim_id == claim_id), None)
        if claim is None:
            raise ValueError("target claim is not visible")
        relation_map = {item.provenance_node_id: item.source_root_ids for item in view.root_relations}
        contents = tuple(
            CascadeVisibleContent(
                content_id=node.content_id,
                provenance_node_id=node.node_id,
                claim_id=node.claim_id,
                source_root_ids=tuple(relation_map.get(node.node_id, ())),
                public_summary=node.content_id,
            )
            for node in view.provenance_nodes
        )
        evidence = tuple(
            CascadeVisibleEvidence(
                evidence_id=card.evidence_id,
                public_summary=card.claim,
                supports=tuple(card.supports),
                contradicts=tuple(card.contradicts),
            )
            for card in view.evidence_cards
        )
        directive_model = None
        if directive is not None:
            if directive.scenario_id != view.scenario_id or directive.target_agent_id != view.agent_id:
                raise ValueError("directive target does not match public view")
            directive_model = CascadeAgentDirective(
                directive_type=directive.directive_type,
                effective_round_id=directive.effective_round_id,
                allowed_content_ids=directive.visible_content_ids,
                allowed_evidence_ids=directive.visible_evidence_ids,
                visible_same_root_repetition_count=directive.visible_same_root_repetition_count,
                visible_root_count=directive.visible_root_count,
                reason_codes=directive.reason_codes,
            )
        return cls(
            scenario_id=view.scenario_id,
            agent_id=view.agent_id,
            round_id=view.round_id,
            claim_id=claim_id,
            claims=tuple(
                CascadeVisibleClaim(
                    claim_id=item.claim_id,
                    verification_status=item.verification_status,
                    visible_evidence_ids=tuple(item.evidence_card_ids),
                    public_summary=item.claim_id,
                )
                for item in view.claims
            ),
            contents=contents,
            evidence=evidence,
            roots=tuple(
                CascadeVisibleRoot(source_root_id=item.source_root_id, public_source_category=item.public_source_category.value)
                for item in view.source_roots
            ),
            directive=directive_model,
            runtime_config=runtime_config or CascadeAgentRuntimeConfig(),
        )


_SYSTEM_PROMPT = (
    "Return exactly one JSON object with only stance, content_ids_used, evidence_ids_used, and share_content_id. "
    "Stance is a public position, not a factual verdict. Use only content and evidence already visible in the supplied context. "
    "A directive is a process constraint, not a fact, evidence item, source root, or truth label. "
    "Do not invent claims, sources, evidence, provenance parents, or controller fields."
)


def render_cascade_agent_turn(context: CascadeAgentPromptContext) -> LLMRequest:
    """Render one deterministic request without selecting or applying policy."""
    payload: dict[str, object] = {
        "template_version": CASCADE_AGENT_TEMPLATE_VERSION,
        "scenario_id": context.scenario_id,
        "agent_id": context.agent_id,
        "round_id": context.round_id,
        "target_claim_id": context.claim_id,
        "visible_claims": [item.model_dump(mode="json") for item in context.claims],
        "visible_contents": [item.model_dump(mode="json") for item in context.contents],
        "visible_evidence": [item.model_dump(mode="json") for item in context.evidence],
        "visible_roots": [item.model_dump(mode="json") for item in context.roots],
    }
    if context.directive is not None:
        payload["directive"] = {
            "type": context.directive.directive_type.value,
            "effective_round_id": context.directive.effective_round_id,
            "allowed_content_ids": list(context.directive.allowed_content_ids),
            "allowed_evidence_ids": list(context.directive.allowed_evidence_ids),
            "visible_same_root_repetition_count": context.directive.visible_same_root_repetition_count,
            "visible_root_count": context.directive.visible_root_count,
            "reason_codes": list(context.directive.reason_codes),
            "semantic": _directive_semantics(context.directive.directive_type),
        }
    else:
        payload["directive"] = None
    user_prompt = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    settings = context.runtime_config
    request_id = _request_id(context, user_prompt)
    return LLMRequest(
        request_id=request_id,
        model_name=settings.model_name,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        seed=settings.seed,
        metadata={
            "contract_version": "cascade_agent_request.v1",
            "template_version": CASCADE_AGENT_TEMPLATE_VERSION,
            "agent_id": context.agent_id,
            "round_id": context.round_id,
            "target_claim_id": context.claim_id,
            "visible_content_count": len(context.contents),
            "visible_evidence_count": len(context.evidence),
        },
    )


def _directive_semantics(directive_type: CascadeDirectiveType) -> str:
    return {
        CascadeDirectiveType.VERIFICATION_REQUEST: "Distinguish repeated public content from an independent source; uncertainty is allowed.",
        CascadeDirectiveType.REASONING_REQUEST: "Compare only the already visible evidence; uncertainty is allowed.",
        CascadeDirectiveType.PRIORITY_EVIDENCE: "Review only the already authorized evidence before stating a position.",
    }[directive_type]


def _request_id(context: CascadeAgentPromptContext, user_prompt: str) -> str:
    canonical = json.dumps(
        {"prompt": user_prompt, "settings": context.runtime_config.model_dump(mode="json")},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"cascade-agent-{hashlib.sha256(canonical.encode()).hexdigest()[:24]}"


__all__ = [
    "CASCADE_AGENT_TEMPLATE_VERSION",
    "CascadeAgentRuntimeConfig",
    "CascadeVisibleClaim",
    "CascadeVisibleContent",
    "CascadeVisibleEvidence",
    "CascadeVisibleRoot",
    "CascadeAgentDirective",
    "CascadeAgentPromptContext",
    "render_cascade_agent_turn",
]


def _public_text(value: str, field_name: str) -> str:
    normalized = normalized_text(value, field_name)
    lowered = normalized.lower()
    if any(token in lowered for token in (
        "ground_truth_label", "source_independence_label", "hidden_probe", "hidden probe",
        "hidden_profile", "hidden profile", "evaluationreport", "private fixture", "api_key", "authorization:",
    )):
        raise ValueError(f"{field_name} contains private or credential material")
    return normalized
