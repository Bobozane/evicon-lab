"""H-G.1 controller comparison based only on the current public view.

Unlike the historical controller, this version never branches on the
pre-registered scenario type.  That label remains study metadata, not an
executable shortcut.  Source-blind and provenance-aware conditions therefore
see matched surface observations and differ only in access to visible root
relations.
"""
from __future__ import annotations

from .cascade_controller import (
    CascadeAction,
    CascadeControllerPolicyConfig,
    CascadeControllerValidationError,
    CascadeInterventionProposal,
    CascadeReasonCode,
    _proposal_id,
)
from .provenance_cascade import VerificationStatus
from .provenance_cascade_exposure import ControllerPublicView
from .provenance_cascade_preregistration import CascadeCondition


def _claim_id(view: ControllerPublicView) -> str:
    if len(view.claims) != 1:
        raise CascadeControllerValidationError("controller_view_requires_one_claim")
    return view.claims[0].claim_id


def _surface(view: ControllerPublicView, claim_id: str):
    claim = next((item for item in view.claims if item.claim_id == claim_id), None)
    if claim is None:
        raise CascadeControllerValidationError("claim_not_visible")
    content = tuple(node.content_id for node in view.provenance_nodes if node.claim_id == claim_id)
    evidence = tuple(card.evidence_id for card in view.evidence_cards if card.evidence_id in claim.evidence_card_ids)
    supporting = tuple(
        card.evidence_id for card in view.evidence_cards
        if card.evidence_id in evidence and claim_id in card.supports
    )
    return claim, content, evidence, supporting


def _roots(view: ControllerPublicView, claim_id: str) -> tuple[str, ...]:
    nodes = {node.node_id for node in view.provenance_nodes if node.claim_id == claim_id}
    return tuple(sorted({
        root_id for relation in view.root_relations
        if relation.provenance_node_id in nodes for root_id in relation.source_root_ids
    }))


def _make(
    condition: CascadeCondition,
    view: ControllerPublicView,
    config: CascadeControllerPolicyConfig,
    claim_id: str,
    action: CascadeAction,
    reasons: tuple[CascadeReasonCode, ...] = (),
    content: tuple[str, ...] = (),
    evidence: tuple[str, ...] = (),
    roots: tuple[str, ...] = (),
) -> CascadeInterventionProposal:
    proposal_id = _proposal_id(
        config=config, scenario_id=view.scenario_id, round_id=view.round_id,
        target_agent_id=view.agent_id, claim_id=claim_id, condition_id=condition,
        action=action, reasons=reasons, content_ids=content,
        evidence_ids=evidence, root_ids=roots,
    )
    proposal = CascadeInterventionProposal(
        proposal_id=proposal_id, scenario_id=view.scenario_id, round_id=view.round_id,
        target_agent_id=view.agent_id, claim_id=claim_id, condition_id=condition,
        action=action, reason_codes=reasons, used_content_ids=content,
        used_evidence_ids=evidence, used_source_root_ids=roots,
    )
    validate_hg1_proposal(proposal, view, config)
    return proposal


def propose_hg1(
    condition: CascadeCondition,
    view: ControllerPublicView,
    config: CascadeControllerPolicyConfig,
) -> CascadeInterventionProposal:
    if condition not in config.conditions:
        raise CascadeControllerValidationError("condition_not_in_policy")
    if not any(rule.scenario_id == view.scenario_id for rule in config.scenario_rules):
        raise CascadeControllerValidationError("scenario_not_in_policy")
    claim_id = _claim_id(view)
    claim, content, evidence, supporting = _surface(view, claim_id)
    if condition is CascadeCondition.NO_INTERVENTION:
        return _make(condition, view, config, claim_id, CascadeAction.ABSTAIN)
    if claim.verification_status is VerificationStatus.CONTESTED:
        return _make(
            condition, view, config, claim_id, CascadeAction.ABSTAIN,
            (CascadeReasonCode.INSUFFICIENT_PUBLIC_EVIDENCE, CascadeReasonCode.POLICY_ABSTAIN_PROTECTION),
        )
    if supporting or claim.verification_status is VerificationStatus.SUPPORTED:
        return _make(
            condition, view, config, claim_id, CascadeAction.ABSTAIN,
            (CascadeReasonCode.VISIBLE_SUPPORTED_CORRECTION, CascadeReasonCode.POLICY_ABSTAIN_PROTECTION),
            evidence=supporting,
            roots=_roots(view, claim_id) if condition is CascadeCondition.PROVENANCE_AWARE_CONTROLLER else (),
        )
    if claim.verification_status is VerificationStatus.REFUTED:
        return _make(
            condition, view, config, claim_id, CascadeAction.REQUEST_EVIDENCE_BASED_REASONING,
            (CascadeReasonCode.VISIBLE_REFUTED_CLAIM,), content=content, evidence=evidence,
            roots=_roots(view, claim_id) if condition is CascadeCondition.PROVENANCE_AWARE_CONTROLLER else (),
        )
    if claim.verification_status is VerificationStatus.UNVERIFIED and len(content) >= config.minimum_visible_repetitions:
        if condition is CascadeCondition.PROVENANCE_AWARE_CONTROLLER:
            roots = _roots(view, claim_id)
            if len(roots) == 1:
                return _make(
                    condition, view, config, claim_id, CascadeAction.REQUEST_INDEPENDENT_SOURCE,
                    (CascadeReasonCode.VISIBLE_UNVERIFIED_SAME_ROOT_REPETITION,),
                    content=content, evidence=evidence, roots=roots,
                )
            if len(roots) > 1:
                return _make(
                    condition, view, config, claim_id, CascadeAction.ABSTAIN,
                    (CascadeReasonCode.MULTIPLE_VISIBLE_INDEPENDENT_ROOTS, CascadeReasonCode.POLICY_ABSTAIN_PROTECTION),
                    roots=roots,
                )
        action = (
            CascadeAction.REQUEST_EVIDENCE_BASED_REASONING
            if condition is CascadeCondition.GENERIC_DISSENT
            else CascadeAction.REQUEST_INDEPENDENT_SOURCE
        )
        return _make(
            condition, view, config, claim_id, action,
            (CascadeReasonCode.VISIBLE_UNVERIFIED_REPETITION,), content=content, evidence=evidence,
        )
    return _make(
        condition, view, config, claim_id, CascadeAction.ABSTAIN,
        (CascadeReasonCode.INSUFFICIENT_PUBLIC_EVIDENCE, CascadeReasonCode.POLICY_ABSTAIN_PROTECTION),
        roots=_roots(view, claim_id) if condition is CascadeCondition.PROVENANCE_AWARE_CONTROLLER else (),
    )


def validate_hg1_proposal(
    proposal: CascadeInterventionProposal,
    view: ControllerPublicView,
    config: CascadeControllerPolicyConfig,
) -> None:
    if proposal.scenario_id != view.scenario_id:
        raise CascadeControllerValidationError("proposal_scenario_mismatch")
    if proposal.round_id != view.round_id:
        raise CascadeControllerValidationError("proposal_round_mismatch")
    if proposal.target_agent_id != view.agent_id:
        raise CascadeControllerValidationError("proposal_target_not_current_agent")
    if proposal.condition_id not in config.conditions:
        raise CascadeControllerValidationError("proposal_condition_not_in_policy")
    if proposal.claim_id not in {item.claim_id for item in view.claims}:
        raise CascadeControllerValidationError("proposal_claim_not_visible")
    visible_content = {node.content_id for node in view.provenance_nodes if node.claim_id == proposal.claim_id}
    visible_evidence = {card.evidence_id for card in view.evidence_cards}
    visible_roots = {root_id for relation in view.root_relations for root_id in relation.source_root_ids}
    if not set(proposal.used_content_ids).issubset(visible_content):
        raise CascadeControllerValidationError("proposal_content_not_visible")
    if not set(proposal.used_evidence_ids).issubset(visible_evidence):
        raise CascadeControllerValidationError("proposal_evidence_not_visible")
    if proposal.condition_id in {CascadeCondition.GENERIC_DISSENT, CascadeCondition.SOURCE_BLIND_CONTROLLER}:
        if proposal.used_source_root_ids or CascadeReasonCode.MULTIPLE_VISIBLE_INDEPENDENT_ROOTS in proposal.reason_codes or CascadeReasonCode.VISIBLE_UNVERIFIED_SAME_ROOT_REPETITION in proposal.reason_codes:
            raise CascadeControllerValidationError("source_blind_provenance_access")
    elif not set(proposal.used_source_root_ids).issubset(visible_roots):
        raise CascadeControllerValidationError("proposal_root_not_visible")
    if proposal.condition_id is CascadeCondition.NO_INTERVENTION and (
        proposal.action is not CascadeAction.ABSTAIN or proposal.reason_codes
        or proposal.used_content_ids or proposal.used_evidence_ids or proposal.used_source_root_ids
    ):
        raise CascadeControllerValidationError("no_intervention_must_be_empty_abstain")
    expected = _proposal_id(
        config=config, scenario_id=proposal.scenario_id, round_id=proposal.round_id,
        target_agent_id=proposal.target_agent_id, claim_id=proposal.claim_id,
        condition_id=proposal.condition_id, action=proposal.action,
        reasons=proposal.reason_codes, content_ids=proposal.used_content_ids,
        evidence_ids=proposal.used_evidence_ids, root_ids=proposal.used_source_root_ids,
    )
    if proposal.proposal_id != expected:
        raise CascadeControllerValidationError("proposal_id_not_deterministic")


__all__ = ["propose_hg1", "validate_hg1_proposal"]
