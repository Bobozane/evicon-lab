"""H-G-specific outcome replay for public cross-claim reasoning.

The H-G Agent may reject one visible claim by citing another visible correction
claim. Historical cascade outcome replay remains unchanged; this validator is
injected only by the H-G Pilot path.
"""
from __future__ import annotations

from typing import Final

from .cascade_application_replay import CascadeApplicationReplayValidator
from .cascade_intervention_application import CascadeApplicationLedger, ControlledRoundContext
from .cascade_outcome_replay import (
    CascadeOutcomeReplayError,
    CascadeOutcomeReplayReport,
    CascadeOutcomeReplayStatus,
)
from .cascade_outcomes import CascadeOutcomeLedger
from .cascade_replay import CascadeReplayValidator
from .provenance_cascade import ProvenanceGraph
from .provenance_cascade_exposure import ExposureLedger

HG_OUTCOME_REPLAY_VERSION: Final[str] = "cascade_hg_outcome_replay.v2.visible_cross_claim_context"


class HGOutcomeReplayValidator:
    """Validate H-G outcomes against the exact public round-start snapshot."""

    @staticmethod
    def validate(
        graph: ProvenanceGraph,
        exposure: ExposureLedger,
        outcomes: CascadeOutcomeLedger,
        application: CascadeApplicationLedger | None = None,
        contexts: tuple[ControlledRoundContext, ...] = (),
    ) -> CascadeOutcomeReplayReport:
        try:
            cascade_report = CascadeReplayValidator.validate(graph, exposure)
            HGOutcomeReplayValidator._validate_outcomes(graph, exposure, outcomes)
            app_report = CascadeApplicationReplayValidator.validate(
                application or CascadeApplicationLedger(), contexts
            )
            return CascadeOutcomeReplayReport(
                status=CascadeOutcomeReplayStatus.PASSED,
                cascade_replay=cascade_report,
                application_replay=app_report,
                outcome_count=len(outcomes.outcomes),
            )
        except CascadeOutcomeReplayError:
            raise
        except Exception as exc:
            raise CascadeOutcomeReplayError("hg_cascade_outcome_replay_invalid") from exc

    @staticmethod
    def _validate_outcomes(
        graph: ProvenanceGraph,
        exposure: ExposureLedger,
        outcomes: CascadeOutcomeLedger,
    ) -> None:
        if graph.scenario_id != outcomes.scenario_id or exposure.scenario_id != outcomes.scenario_id:
            raise ValueError("outcome public scenario mismatch")
        if tuple(exposure.agent_ids) != outcomes.agent_ids:
            raise ValueError("outcome agent set mismatch")
        exposure.validate_against_graph(graph)
        claims = {claim.claim_id: claim for claim in graph.claims}
        nodes = {node.content_id: node for node in graph.nodes}
        cards = {card.evidence_id: card for card in graph.evidence_cards}
        snapshots = {(item.agent_id, item.round_id): item for item in exposure.snapshots}
        max_round = max((item.round_id for item in exposure.snapshots), default=0)
        for outcome in outcomes.outcomes:
            if outcome.round_id > max_round:
                raise ValueError("outcome occurs after the public run")
            target_claim = claims.get(outcome.claim_id)
            if target_claim is None:
                raise ValueError("outcome references unknown claim")
            snapshot = snapshots.get((outcome.agent_id, outcome.round_id))
            if snapshot is None:
                raise ValueError("outcome has no matching round-start snapshot")
            visible_content = set(snapshot.visible_content_ids)
            visible_evidence = set(snapshot.visible_evidence_ids)
            visible_claims = set(snapshot.visible_claim_ids)
            if not set(outcome.content_ids).issubset(visible_content):
                raise ValueError("outcome references content not exposed to agent")
            if not set(outcome.evidence_card_ids).issubset(visible_evidence):
                raise ValueError("outcome references evidence not exposed to agent")
            for content_id in outcome.content_ids:
                node = nodes.get(content_id)
                if node is None or node.round_id > outcome.round_id:
                    raise ValueError("outcome content is unknown or from the future")
                if node.claim_id not in visible_claims:
                    raise ValueError("outcome content claim was not visible")
            for evidence_id in outcome.evidence_card_ids:
                card = cards.get(evidence_id)
                related_to_target = (
                    evidence_id in target_claim.evidence_card_ids
                    or outcome.claim_id in card.supports
                    or outcome.claim_id in card.contradicts
                ) if card is not None else False
                if (
                    card is None
                    or not related_to_target
                    or card.introduced_round > outcome.round_id
                    or not (
                        card.is_visible_to(outcome.agent_id)
                        or "public" in card.visible_to
                        or "*" in card.visible_to
                    )
                ):
                    raise ValueError("outcome evidence is unknown, unrelated, unauthorized, or from the future")


__all__ = ["HG_OUTCOME_REPLAY_VERSION", "HGOutcomeReplayValidator"]
