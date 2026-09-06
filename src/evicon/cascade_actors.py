"""Deterministic public-view actors for provenance-cascade outcomes.

No provider, model output, evaluator truth, or private fixture is accepted by
this module. Structured directives are next-round metadata, never evidence.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cascade_intervention_application import CascadeDirectiveType, StructuredCascadeDirective
from .cascade_outcomes import ClaimStance
from .provenance_cascade import VerificationStatus
from .provenance_cascade_exposure import ControllerPublicView
from .models._validation import normalized_text


class ActorProfile(str, Enum):
    REPEAT_ACTOR = "repeat_actor"
    VERIFICATION_RESPONSIVE_ACTOR = "verification_responsive_actor"


class ActorSafetyError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class DeterministicActorSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    profile: ActorProfile = ActorProfile.REPEAT_ACTOR
    preferred_claim_id: str | None = None
    schedule_id: str = "default"

    @model_validator(mode="after")
    def clean(self) -> "DeterministicActorSchedule":
        if self.preferred_claim_id is not None:
            normalized_text(self.preferred_claim_id, "preferred_claim_id")
        normalized_text(self.schedule_id, "schedule_id")
        return self


class PublicActorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    agent_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    claim_id: str = Field(min_length=1)
    stance: ClaimStance
    used_content_ids: tuple[str, ...] = ()
    used_evidence_card_ids: tuple[str, ...] = ()
    provenance_parent_content_id: str | None = None
    decision_sha256: str = Field(min_length=64, max_length=64)
    error_code: str | None = None

    @model_validator(mode="after")
    def shape_and_digest(self) -> "PublicActorDecision":
        if len(set(self.used_content_ids)) != len(self.used_content_ids):
            raise ValueError("duplicate used_content_ids")
        if len(set(self.used_evidence_card_ids)) != len(self.used_evidence_card_ids):
            raise ValueError("duplicate used_evidence_card_ids")
        if self.provenance_parent_content_id is not None and self.provenance_parent_content_id not in self.used_content_ids:
            raise ValueError("provenance parent must be used visible content")
        payload = {
            "agent_id": self.agent_id, "round_id": self.round_id,
            "claim_id": self.claim_id, "stance": self.stance.value,
            "used_content_ids": list(self.used_content_ids),
            "used_evidence_card_ids": list(self.used_evidence_card_ids),
            "provenance_parent_content_id": self.provenance_parent_content_id,
            "error_code": self.error_code,
        }
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if self.decision_sha256 != expected:
            raise ValueError("decision hash mismatch")
        return self

    @classmethod
    def create(cls, **kwargs: Any) -> "PublicActorDecision":
        payload = {key: value for key, value in kwargs.items() if key != "decision_sha256"}
        encoded = {
            "agent_id": payload["agent_id"], "round_id": payload["round_id"],
            "claim_id": payload["claim_id"],
            "stance": payload["stance"].value if isinstance(payload["stance"], ClaimStance) else payload["stance"],
            "used_content_ids": list(payload.get("used_content_ids", ())),
            "used_evidence_card_ids": list(payload.get("used_evidence_card_ids", ())),
            "provenance_parent_content_id": payload.get("provenance_parent_content_id"),
            "error_code": payload.get("error_code"),
        }
        payload["decision_sha256"] = hashlib.sha256(json.dumps(encoded, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return cls(**payload)


class DeterministicActor:
    """A pure actor whose input is one immutable public view."""

    def __init__(self, profile: ActorProfile = ActorProfile.REPEAT_ACTOR) -> None:
        self.profile = profile

    @staticmethod
    def _validate_inputs(agent_id: str, round_id: int, view: ControllerPublicView,
                         directive: StructuredCascadeDirective | None) -> None:
        if view.agent_id != agent_id:
            raise ActorSafetyError("actor_target_mismatch")
        if view.round_id != round_id:
            raise ActorSafetyError("actor_round_mismatch")
        if directive is None:
            return
        if directive.scenario_id != view.scenario_id or directive.target_agent_id != agent_id:
            raise ActorSafetyError("directive_target_mismatch")
        if directive.effective_round_id != round_id:
            raise ActorSafetyError("directive_round_mismatch")
        visible_content = {node.content_id for node in view.provenance_nodes}
        visible_evidence = {card.evidence_id for card in view.evidence_cards}
        if not set(directive.visible_content_ids).issubset(visible_content):
            raise ActorSafetyError("directive_content_not_visible")
        if not set(directive.visible_evidence_ids).issubset(visible_evidence):
            raise ActorSafetyError("directive_evidence_not_visible")

    @staticmethod
    def _stance(status: VerificationStatus) -> ClaimStance:
        return {
            VerificationStatus.SUPPORTED: ClaimStance.ENDORSES,
            VerificationStatus.REFUTED: ClaimStance.REJECTS,
            VerificationStatus.CONTESTED: ClaimStance.UNCERTAIN,
            VerificationStatus.UNVERIFIED: ClaimStance.ENDORSES,
        }[status]

    def act(self, agent_id: str, round_id: int, controller_public_view: ControllerPublicView,
            applied_structured_directive: StructuredCascadeDirective | None,
            deterministic_schedule: DeterministicActorSchedule) -> PublicActorDecision:
        self._validate_inputs(agent_id, round_id, controller_public_view, applied_structured_directive)
        claim_id = deterministic_schedule.preferred_claim_id or (
            controller_public_view.claims[0].claim_id if controller_public_view.claims else ""
        )
        claim = next((item for item in controller_public_view.claims if item.claim_id == claim_id), None)
        if claim is None:
            safe_claim = deterministic_schedule.preferred_claim_id or "no_visible_claim"
            return PublicActorDecision.create(agent_id=agent_id, round_id=round_id, claim_id=safe_claim,
                                              stance=ClaimStance.NO_POSITION, error_code="claim_not_visible")
        nodes = tuple(sorted((node for node in controller_public_view.provenance_nodes if node.claim_id == claim_id),
                             key=lambda node: node.content_id))
        content_ids = tuple(node.content_id for node in nodes)
        visible_evidence = tuple(sorted(card.evidence_id for card in controller_public_view.evidence_cards
                                        if card.evidence_id in set(claim.evidence_card_ids)))
        stance = self._stance(claim.verification_status)
        used_evidence = visible_evidence
        if self.profile is ActorProfile.VERIFICATION_RESPONSIVE_ACTOR and applied_structured_directive is not None:
            directive_type = applied_structured_directive.directive_type
            if directive_type is CascadeDirectiveType.VERIFICATION_REQUEST:
                if claim.verification_status is VerificationStatus.UNVERIFIED:
                    stance = ClaimStance.UNCERTAIN
            elif directive_type is CascadeDirectiveType.REASONING_REQUEST:
                used_evidence = tuple(eid for eid in applied_structured_directive.visible_evidence_ids if eid in visible_evidence)
                if not used_evidence:
                    stance = ClaimStance.UNCERTAIN
            elif directive_type is CascadeDirectiveType.PRIORITY_EVIDENCE:
                used_evidence = tuple(eid for eid in applied_structured_directive.visible_evidence_ids if eid in visible_evidence)
                if not used_evidence:
                    raise ActorSafetyError("priority_evidence_not_visible")
                if claim.verification_status is VerificationStatus.CONTESTED:
                    stance = ClaimStance.UNCERTAIN
        parent = content_ids[0] if content_ids else None
        return PublicActorDecision.create(agent_id=agent_id, round_id=round_id, claim_id=claim_id, stance=stance,
                                          used_content_ids=content_ids, used_evidence_card_ids=used_evidence,
                                          provenance_parent_content_id=parent)


def act(agent_id: str, round_id: int, controller_public_view: ControllerPublicView,
        applied_structured_directive: StructuredCascadeDirective | None,
        deterministic_schedule: DeterministicActorSchedule) -> PublicActorDecision:
    return DeterministicActor(deterministic_schedule.profile).act(agent_id, round_id, controller_public_view,
                                                                   applied_structured_directive, deterministic_schedule)


__all__ = ["ActorProfile", "ActorSafetyError", "DeterministicActorSchedule", "PublicActorDecision", "DeterministicActor", "act"]
