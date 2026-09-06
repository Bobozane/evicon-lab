"""Pure, controller-safe decisions for synthetic provenance-cascade studies.

This module deliberately produces *proposals only*.  It never changes an
exposure ledger, snapshot, provenance graph, actor schedule, or runner state.
Every decision is derived from one current ``ControllerPublicView`` and a
frozen public policy configuration; evaluator truth and global graph access
are outside this interface.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from enum import Enum
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .models._validation import identifier_list, normalized_text
from .provenance_cascade import VerificationStatus
from .provenance_cascade_exposure import ControllerPublicView
from .provenance_cascade_preregistration import CascadeCondition, CascadeScenario


class CascadeAction(str, Enum):
    """Future-safe action semantics; no action is applied in this stage."""

    ABSTAIN = "abstain"
    REQUEST_INDEPENDENT_SOURCE = "request_independent_source"
    REQUEST_EVIDENCE_BASED_REASONING = "request_evidence_based_reasoning"
    EVIDENCE_FIRST_EXPOSURE = "evidence_first_exposure"


class CascadeReasonCode(str, Enum):
    """Stable descriptions of allowed, currently public observations only."""

    VISIBLE_UNVERIFIED_SAME_ROOT_REPETITION = "visible_unverified_same_root_repetition"
    VISIBLE_UNVERIFIED_REPETITION = "visible_unverified_repetition"
    VISIBLE_REFUTED_CLAIM = "visible_refuted_claim"
    VISIBLE_SUPPORTED_CORRECTION = "visible_supported_correction"
    MULTIPLE_VISIBLE_INDEPENDENT_ROOTS = "multiple_visible_independent_roots"
    INSUFFICIENT_PUBLIC_EVIDENCE = "insufficient_public_evidence"
    POLICY_ABSTAIN_PROTECTION = "policy_abstain_protection"


class CascadeProposalWarning(str, Enum):
    """Reserved structured warnings; no free-text model or evaluator data."""

    NO_WARNING = "no_warning"


class CascadeProposalError(str, Enum):
    """Reserved stable errors for invalid proposal construction."""

    PUBLIC_VIEW_INVALID = "public_view_invalid"


class CascadeInterventionProposal(BaseModel):
    """A deterministic, non-executing policy proposal.

    ``used_*`` coordinates must be a subset of one current public view.  They
    are audit references only; they do not authorize a future exposure change.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    target_agent_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    condition_id: CascadeCondition
    action: CascadeAction
    reason_codes: tuple[CascadeReasonCode, ...] = ()
    used_content_ids: tuple[str, ...] = ()
    used_evidence_ids: tuple[str, ...] = ()
    used_source_root_ids: tuple[str, ...] = ()
    valid: Literal[True] = True
    warnings: tuple[CascadeProposalWarning, ...] = ()
    errors: tuple[CascadeProposalError, ...] = ()

    @field_validator("proposal_id", "scenario_id", "target_agent_id", "claim_id")
    @classmethod
    def clean_ids(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "id"))

    @field_validator("used_content_ids", "used_evidence_ids", "used_source_root_ids")
    @classmethod
    def unique_ids(cls, value: tuple[str, ...], info: object) -> tuple[str, ...]:
        return tuple(identifier_list(list(value), getattr(info, "field_name", "ids")))

    @field_validator("reason_codes", "warnings")
    @classmethod
    def unique_codes(cls, value: tuple[Enum, ...], info: object) -> tuple[Enum, ...]:
        if len(value) != len(set(value)):
            raise ValueError(f"{getattr(info, 'field_name', 'codes')} must not repeat")
        return value


class CascadePolicyScenarioRule(BaseModel):
    """Public scenario classification fixed before any cascade run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    scenario_type: CascadeScenario

    @field_validator("scenario_id")
    @classmethod
    def clean_id(cls, value: str) -> str:
        return normalized_text(value, "scenario_id")


class CascadeControllerPolicyConfig(BaseModel):
    """Frozen public policy inputs, not a calibrated effectiveness claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: str = Field(min_length=1)
    policy_version: Literal["provenance_cascade_controller.v1"]
    conditions: tuple[CascadeCondition, ...] = Field(min_length=4, max_length=4)
    scenario_rules: tuple[CascadePolicyScenarioRule, ...] = Field(min_length=4, max_length=4)
    minimum_visible_repetitions: int = Field(ge=2)
    allow_unresolved_independent_source_request: bool = False
    development_only: Literal[True]
    not_paper_result: Literal[True]

    @field_validator("policy_id")
    @classmethod
    def clean_id(cls, value: str) -> str:
        return normalized_text(value, "policy_id")

    @model_validator(mode="after")
    def fixed_policy_shape(self) -> "CascadeControllerPolicyConfig":
        expected_conditions = tuple(CascadeCondition)
        expected_scenarios = set(CascadeScenario)
        if self.conditions != expected_conditions:
            raise ValueError("conditions must contain the four preregistered conditions in order")
        ids = [rule.scenario_id for rule in self.scenario_rules]
        types = [rule.scenario_type for rule in self.scenario_rules]
        if len(ids) != len(set(ids)) or set(types) != expected_scenarios or len(types) != len(set(types)):
            raise ValueError("scenario_rules must map each public cascade scenario exactly once")
        return self

    def scenario_type_for(self, scenario_id: str) -> CascadeScenario:
        for rule in self.scenario_rules:
            if rule.scenario_id == scenario_id:
                return rule.scenario_type
        raise CascadeControllerValidationError("scenario_not_in_policy")


class CascadeControllerConfigError(ValueError):
    """Safe local policy-config error; no raw TOML or secret is surfaced."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class CascadeControllerValidationError(ValueError):
    """Stable proposal/view validation error for a non-executing decision."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class CascadeControllerPolicyLoader:
    """Read only a local, public controller policy file."""

    @staticmethod
    def load(path: str | Path) -> CascadeControllerPolicyConfig:
        try:
            payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
            return CascadeControllerPolicyConfig.model_validate(payload)
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
            raise CascadeControllerConfigError("cascade_controller_policy_invalid") from exc


class CascadeDecisionController(Protocol):
    """Uniform pure-function controller interface for one public view."""

    def propose(
        self,
        view: ControllerPublicView,
        config: CascadeControllerPolicyConfig,
    ) -> CascadeInterventionProposal: ...


class _PublicObservation(BaseModel):
    """Source-blind projection. It intentionally has no root or parent fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    verification_status: VerificationStatus
    visible_content_ids: tuple[str, ...]
    visible_evidence_ids: tuple[str, ...]
    supporting_evidence_ids: tuple[str, ...]


class _ProvenanceObservation(_PublicObservation):
    """Provenance-aware projection limited to already visible root summaries."""

    source_root_ids: tuple[str, ...]


def _validate_common_view_shape(view: ControllerPublicView, config: CascadeControllerPolicyConfig) -> None:
    """Validate non-provenance public fields without touching root summaries.

    Generic and source-blind policies deliberately call only this function.
    Their executable path therefore does not inspect or derive provenance-only
    state even though the enclosing ``ControllerPublicView`` has been audited.
    """

    config.scenario_type_for(view.scenario_id)
    claim_ids = [claim.claim_id for claim in view.claims]
    node_ids = [node.node_id for node in view.provenance_nodes]
    content_ids = [node.content_id for node in view.provenance_nodes]
    evidence_ids = [card.evidence_id for card in view.evidence_cards]
    if not claim_ids or len(claim_ids) != len(set(claim_ids)):
        raise CascadeControllerValidationError("controller_view_claims_invalid")
    if len(node_ids) != len(set(node_ids)) or len(content_ids) != len(set(content_ids)):
        raise CascadeControllerValidationError("controller_view_nodes_invalid")
    if len(evidence_ids) != len(set(evidence_ids)):
        raise CascadeControllerValidationError("controller_view_evidence_invalid")
    if any(node.scenario_id != view.scenario_id or node.round_id >= view.round_id or node.claim_id not in claim_ids for node in view.provenance_nodes):
        raise CascadeControllerValidationError("controller_view_future_or_cross_scenario_data")
    claim_evidence = {evidence_id for claim in view.claims for evidence_id in claim.evidence_card_ids}
    if not set(evidence_ids).issubset(claim_evidence):
        raise CascadeControllerValidationError("controller_view_evidence_not_exposed_for_claim")
    if any(card.introduced_round >= view.round_id for card in view.evidence_cards):
        raise CascadeControllerValidationError("controller_view_future_evidence")


def _validate_provenance_view_shape(view: ControllerPublicView, config: CascadeControllerPolicyConfig) -> None:
    """Validate root summaries only on the provenance-aware code path."""

    _validate_common_view_shape(view, config)
    node_ids = {node.node_id for node in view.provenance_nodes}
    relation_nodes = {relation.provenance_node_id for relation in view.root_relations}
    if relation_nodes != node_ids:
        raise CascadeControllerValidationError("controller_view_root_relations_invalid")
    known_roots = {root.source_root_id for root in view.source_roots}
    if not known_roots or not {root_id for relation in view.root_relations for root_id in relation.source_root_ids}.issubset(known_roots):
        raise CascadeControllerValidationError("controller_view_root_summary_invalid")


def _public_observation(view: ControllerPublicView, claim_id: str) -> _PublicObservation:
    claim = next((item for item in view.claims if item.claim_id == claim_id), None)
    if claim is None:
        raise CascadeControllerValidationError("claim_not_visible")
    content_ids = tuple(node.content_id for node in view.provenance_nodes if node.claim_id == claim_id)
    evidence_ids = tuple(card.evidence_id for card in view.evidence_cards if card.evidence_id in claim.evidence_card_ids)
    supporting_evidence_ids = tuple(
        card.evidence_id
        for card in view.evidence_cards
        if card.evidence_id in evidence_ids and claim_id in card.supports
    )
    return _PublicObservation(
        claim_id=claim_id,
        verification_status=claim.verification_status,
        visible_content_ids=content_ids,
        visible_evidence_ids=evidence_ids,
        supporting_evidence_ids=supporting_evidence_ids,
    )


def _provenance_observation(view: ControllerPublicView, claim_id: str) -> _ProvenanceObservation:
    base = _public_observation(view, claim_id)
    visible_node_ids = {node.node_id for node in view.provenance_nodes if node.claim_id == claim_id}
    root_ids = tuple(sorted({
        root_id
        for relation in view.root_relations
        if relation.provenance_node_id in visible_node_ids
        for root_id in relation.source_root_ids
    }))
    return _ProvenanceObservation(**base.model_dump(mode="python"), source_root_ids=root_ids)


def _proposal_id(
    *,
    config: CascadeControllerPolicyConfig,
    scenario_id: str,
    round_id: int,
    target_agent_id: str,
    claim_id: str,
    condition_id: CascadeCondition,
    action: CascadeAction,
    reasons: tuple[CascadeReasonCode, ...],
    content_ids: tuple[str, ...],
    evidence_ids: tuple[str, ...],
    root_ids: tuple[str, ...],
) -> str:
    coordinate = {
        "policy_version": config.policy_version,
        "scenario_id": scenario_id,
        "round_id": round_id,
        "target_agent_id": target_agent_id,
        "claim_id": claim_id,
        "condition_id": condition_id.value,
        "action": action.value,
        "reason_codes": [reason.value for reason in reasons],
        "content_ids": list(content_ids),
        "evidence_ids": list(evidence_ids),
        "root_ids": list(root_ids),
    }
    digest = hashlib.sha256(json.dumps(coordinate, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"cascade-proposal-{digest[:24]}"


def _build_proposal(
    *,
    config: CascadeControllerPolicyConfig,
    view: ControllerPublicView,
    claim_id: str,
    condition_id: CascadeCondition,
    action: CascadeAction,
    reasons: tuple[CascadeReasonCode, ...] = (),
    content_ids: tuple[str, ...] = (),
    evidence_ids: tuple[str, ...] = (),
    root_ids: tuple[str, ...] = (),
) -> CascadeInterventionProposal:
    proposal = CascadeInterventionProposal(
        proposal_id=_proposal_id(
            config=config,
            scenario_id=view.scenario_id,
            round_id=view.round_id,
            target_agent_id=view.agent_id,
            claim_id=claim_id,
            condition_id=condition_id,
            action=action,
            reasons=reasons,
            content_ids=content_ids,
            evidence_ids=evidence_ids,
            root_ids=root_ids,
        ),
        scenario_id=view.scenario_id,
        round_id=view.round_id,
        target_agent_id=view.agent_id,
        claim_id=claim_id,
        condition_id=condition_id,
        action=action,
        reason_codes=reasons,
        used_content_ids=content_ids,
        used_evidence_ids=evidence_ids,
        used_source_root_ids=root_ids,
    )
    validate_cascade_proposal(proposal, view, config)
    return proposal


def _abstain(
    *,
    config: CascadeControllerPolicyConfig,
    view: ControllerPublicView,
    claim_id: str,
    condition_id: CascadeCondition,
    reasons: tuple[CascadeReasonCode, ...] = (),
    evidence_ids: tuple[str, ...] = (),
    root_ids: tuple[str, ...] = (),
) -> CascadeInterventionProposal:
    return _build_proposal(
        config=config,
        view=view,
        claim_id=claim_id,
        condition_id=condition_id,
        action=CascadeAction.ABSTAIN,
        reasons=reasons,
        evidence_ids=evidence_ids,
        root_ids=root_ids,
    )


class CascadePolicyController:
    """A deterministic proposal generator; it cannot apply an intervention."""

    def __init__(self, condition_id: CascadeCondition) -> None:
        self.condition_id = condition_id

    def propose(
        self,
        view: ControllerPublicView,
        config: CascadeControllerPolicyConfig,
    ) -> CascadeInterventionProposal:
        if self.condition_id not in config.conditions:
            raise CascadeControllerValidationError("condition_not_in_policy")
        if self.condition_id is CascadeCondition.PROVENANCE_AWARE_CONTROLLER:
            _validate_provenance_view_shape(view, config)
        else:
            _validate_common_view_shape(view, config)
        claim_id = _single_visible_claim(view)
        if self.condition_id is CascadeCondition.NO_INTERVENTION:
            # This condition records no synthetic risk rationale.
            return _abstain(config=config, view=view, claim_id=claim_id, condition_id=self.condition_id)

        scenario_type = config.scenario_type_for(view.scenario_id)
        if self.condition_id is CascadeCondition.PROVENANCE_AWARE_CONTROLLER:
            observation = _provenance_observation(view, claim_id)
            return self._provenance_aware(view, config, scenario_type, observation)
        observation = _public_observation(view, claim_id)
        return self._source_blind_or_generic(view, config, scenario_type, observation)

    def _source_blind_or_generic(
        self,
        view: ControllerPublicView,
        config: CascadeControllerPolicyConfig,
        scenario_type: CascadeScenario,
        observation: _PublicObservation,
    ) -> CascadeInterventionProposal:
        condition_id = self.condition_id
        if scenario_type is CascadeScenario.INDEPENDENT_TRUE_CONSENSUS:
            return _abstain(
                config=config, view=view, claim_id=observation.claim_id, condition_id=condition_id,
                reasons=(CascadeReasonCode.POLICY_ABSTAIN_PROTECTION,),
            )
        if scenario_type is CascadeScenario.UNRESOLVED_DISAGREEMENT:
            return _abstain(
                config=config, view=view, claim_id=observation.claim_id, condition_id=condition_id,
                reasons=(CascadeReasonCode.INSUFFICIENT_PUBLIC_EVIDENCE, CascadeReasonCode.POLICY_ABSTAIN_PROTECTION),
            )
        if observation.supporting_evidence_ids or observation.verification_status is VerificationStatus.SUPPORTED:
            return _abstain(
                config=config, view=view, claim_id=observation.claim_id, condition_id=condition_id,
                reasons=(CascadeReasonCode.VISIBLE_SUPPORTED_CORRECTION, CascadeReasonCode.POLICY_ABSTAIN_PROTECTION),
                evidence_ids=observation.supporting_evidence_ids,
            )
        if observation.verification_status is VerificationStatus.REFUTED:
            return _build_proposal(
                config=config, view=view, claim_id=observation.claim_id, condition_id=condition_id,
                action=CascadeAction.REQUEST_EVIDENCE_BASED_REASONING,
                reasons=(CascadeReasonCode.VISIBLE_REFUTED_CLAIM,),
                content_ids=observation.visible_content_ids,
                evidence_ids=observation.visible_evidence_ids,
            )
        if (
            observation.verification_status is VerificationStatus.UNVERIFIED
            and len(observation.visible_content_ids) >= config.minimum_visible_repetitions
        ):
            action = (
                CascadeAction.REQUEST_EVIDENCE_BASED_REASONING
                if condition_id is CascadeCondition.GENERIC_DISSENT
                else CascadeAction.REQUEST_INDEPENDENT_SOURCE
            )
            return _build_proposal(
                config=config, view=view, claim_id=observation.claim_id, condition_id=condition_id,
                action=action,
                reasons=(CascadeReasonCode.VISIBLE_UNVERIFIED_REPETITION,),
                content_ids=observation.visible_content_ids,
                evidence_ids=observation.visible_evidence_ids,
            )
        return _abstain(
            config=config, view=view, claim_id=observation.claim_id, condition_id=condition_id,
            reasons=(CascadeReasonCode.INSUFFICIENT_PUBLIC_EVIDENCE, CascadeReasonCode.POLICY_ABSTAIN_PROTECTION),
        )

    def _provenance_aware(
        self,
        view: ControllerPublicView,
        config: CascadeControllerPolicyConfig,
        scenario_type: CascadeScenario,
        observation: _ProvenanceObservation,
    ) -> CascadeInterventionProposal:
        if scenario_type is CascadeScenario.INDEPENDENT_TRUE_CONSENSUS:
            return _abstain(
                config=config, view=view, claim_id=observation.claim_id, condition_id=self.condition_id,
                reasons=(CascadeReasonCode.MULTIPLE_VISIBLE_INDEPENDENT_ROOTS, CascadeReasonCode.POLICY_ABSTAIN_PROTECTION),
                root_ids=observation.source_root_ids,
            )
        if scenario_type is CascadeScenario.UNRESOLVED_DISAGREEMENT:
            if config.allow_unresolved_independent_source_request:
                return _build_proposal(
                    config=config, view=view, claim_id=observation.claim_id, condition_id=self.condition_id,
                    action=CascadeAction.REQUEST_INDEPENDENT_SOURCE,
                    reasons=(CascadeReasonCode.INSUFFICIENT_PUBLIC_EVIDENCE,),
                    content_ids=observation.visible_content_ids,
                    root_ids=observation.source_root_ids,
                )
            return _abstain(
                config=config, view=view, claim_id=observation.claim_id, condition_id=self.condition_id,
                reasons=(CascadeReasonCode.INSUFFICIENT_PUBLIC_EVIDENCE, CascadeReasonCode.POLICY_ABSTAIN_PROTECTION),
                root_ids=observation.source_root_ids,
            )
        if observation.supporting_evidence_ids or observation.verification_status is VerificationStatus.SUPPORTED:
            return _abstain(
                config=config, view=view, claim_id=observation.claim_id, condition_id=self.condition_id,
                reasons=(CascadeReasonCode.VISIBLE_SUPPORTED_CORRECTION, CascadeReasonCode.POLICY_ABSTAIN_PROTECTION),
                evidence_ids=observation.supporting_evidence_ids,
                root_ids=observation.source_root_ids,
            )
        if observation.verification_status is VerificationStatus.REFUTED:
            return _build_proposal(
                config=config, view=view, claim_id=observation.claim_id, condition_id=self.condition_id,
                action=CascadeAction.REQUEST_EVIDENCE_BASED_REASONING,
                reasons=(CascadeReasonCode.VISIBLE_REFUTED_CLAIM,),
                content_ids=observation.visible_content_ids,
                evidence_ids=observation.visible_evidence_ids,
                root_ids=observation.source_root_ids,
            )
        if (
            observation.verification_status is VerificationStatus.UNVERIFIED
            and len(observation.visible_content_ids) >= config.minimum_visible_repetitions
            and len(observation.source_root_ids) == 1
        ):
            return _build_proposal(
                config=config, view=view, claim_id=observation.claim_id, condition_id=self.condition_id,
                action=CascadeAction.REQUEST_INDEPENDENT_SOURCE,
                reasons=(CascadeReasonCode.VISIBLE_UNVERIFIED_SAME_ROOT_REPETITION,),
                content_ids=observation.visible_content_ids,
                evidence_ids=observation.visible_evidence_ids,
                root_ids=observation.source_root_ids,
            )
        return _abstain(
            config=config, view=view, claim_id=observation.claim_id, condition_id=self.condition_id,
            reasons=(CascadeReasonCode.INSUFFICIENT_PUBLIC_EVIDENCE, CascadeReasonCode.POLICY_ABSTAIN_PROTECTION),
            root_ids=observation.source_root_ids,
        )


def _single_visible_claim(view: ControllerPublicView) -> str:
    if len(view.claims) != 1:
        raise CascadeControllerValidationError("controller_view_requires_one_claim")
    return view.claims[0].claim_id


def validate_cascade_proposal(
    proposal: CascadeInterventionProposal,
    view: ControllerPublicView,
    config: CascadeControllerPolicyConfig,
) -> None:
    """Validate a proposal without executing it or inspecting private data."""

    if proposal.condition_id is CascadeCondition.PROVENANCE_AWARE_CONTROLLER:
        _validate_provenance_view_shape(view, config)
    else:
        _validate_common_view_shape(view, config)
    if proposal.scenario_id != view.scenario_id:
        raise CascadeControllerValidationError("proposal_scenario_mismatch")
    if proposal.round_id != view.round_id:
        raise CascadeControllerValidationError("proposal_round_mismatch")
    if proposal.target_agent_id != view.agent_id:
        raise CascadeControllerValidationError("proposal_target_not_current_agent")
    if proposal.condition_id not in config.conditions:
        raise CascadeControllerValidationError("proposal_condition_not_in_policy")
    if proposal.claim_id not in {claim.claim_id for claim in view.claims}:
        raise CascadeControllerValidationError("proposal_claim_not_visible")
    visible_content = {node.content_id for node in view.provenance_nodes if node.claim_id == proposal.claim_id}
    visible_evidence = {card.evidence_id for card in view.evidence_cards}
    if not set(proposal.used_content_ids).issubset(visible_content):
        raise CascadeControllerValidationError("proposal_content_not_visible")
    if not set(proposal.used_evidence_ids).issubset(visible_evidence):
        raise CascadeControllerValidationError("proposal_evidence_not_visible")
    if proposal.errors:
        raise CascadeControllerValidationError("valid_proposal_cannot_have_errors")
    if proposal.condition_id in {CascadeCondition.GENERIC_DISSENT, CascadeCondition.SOURCE_BLIND_CONTROLLER}:
        if proposal.used_source_root_ids or {
            CascadeReasonCode.VISIBLE_UNVERIFIED_SAME_ROOT_REPETITION,
            CascadeReasonCode.MULTIPLE_VISIBLE_INDEPENDENT_ROOTS,
        }.intersection(proposal.reason_codes):
            raise CascadeControllerValidationError("source_blind_provenance_access")
    elif proposal.condition_id is CascadeCondition.PROVENANCE_AWARE_CONTROLLER:
        visible_roots = {root_id for relation in view.root_relations for root_id in relation.source_root_ids}
        if not set(proposal.used_source_root_ids).issubset(visible_roots):
            raise CascadeControllerValidationError("proposal_root_not_visible")
    if proposal.condition_id is CascadeCondition.NO_INTERVENTION:
        if proposal.action is not CascadeAction.ABSTAIN or proposal.reason_codes or proposal.used_content_ids or proposal.used_evidence_ids or proposal.used_source_root_ids:
            raise CascadeControllerValidationError("no_intervention_must_be_empty_abstain")
    scenario_type = config.scenario_type_for(view.scenario_id)
    if scenario_type is CascadeScenario.INDEPENDENT_TRUE_CONSENSUS and proposal.action is not CascadeAction.ABSTAIN:
        raise CascadeControllerValidationError("independent_consensus_requires_abstain")
    if scenario_type is CascadeScenario.UNRESOLVED_DISAGREEMENT and proposal.action is not CascadeAction.ABSTAIN:
        if not (
            proposal.condition_id is CascadeCondition.PROVENANCE_AWARE_CONTROLLER
            and config.allow_unresolved_independent_source_request
            and proposal.action is CascadeAction.REQUEST_INDEPENDENT_SOURCE
        ):
            raise CascadeControllerValidationError("unresolved_disagreement_requires_abstain")
    expected_id = _proposal_id(
        config=config,
        scenario_id=proposal.scenario_id,
        round_id=proposal.round_id,
        target_agent_id=proposal.target_agent_id,
        claim_id=proposal.claim_id,
        condition_id=proposal.condition_id,
        action=proposal.action,
        reasons=proposal.reason_codes,
        content_ids=proposal.used_content_ids,
        evidence_ids=proposal.used_evidence_ids,
        root_ids=proposal.used_source_root_ids,
    )
    if proposal.proposal_id != expected_id:
        raise CascadeControllerValidationError("proposal_id_not_deterministic")


def propose(
    condition_id: CascadeCondition,
    view: ControllerPublicView,
    config: CascadeControllerPolicyConfig,
) -> CascadeInterventionProposal:
    """Convenience pure function used by smoke tests and future application code."""

    return CascadePolicyController(condition_id).propose(view, config)


__all__ = [
    "CascadeAction", "CascadeControllerConfigError", "CascadeControllerPolicyConfig", "CascadeControllerPolicyLoader",
    "CascadeControllerValidationError", "CascadeDecisionController", "CascadeInterventionProposal", "CascadePolicyController",
    "CascadePolicyScenarioRule", "CascadeProposalError", "CascadeProposalWarning", "CascadeReasonCode", "propose", "validate_cascade_proposal",
]
