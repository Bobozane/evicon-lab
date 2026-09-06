"""H-G.1 pre-session exposure timing and independent replay validation."""
from __future__ import annotations

from typing import Final

from .cascade_application_replay import CascadeApplicationReplayValidator
from .cascade_intervention_application import CascadeApplicationLedger, ControlledRoundContext
from .cascade_outcome_replay import CascadeOutcomeReplayReport, CascadeOutcomeReplayStatus
from .cascade_outcomes import CascadeOutcomeLedger
from .cascade_protocol import CascadeProtocolRunner, CascadeScenarioSpec
from .cascade_replay import CascadeReplayReport, CascadeReplayStatus, CascadeReplayValidator
from .provenance_cascade import ProvenanceGraph
from .provenance_cascade_exposure import (
    CascadeExposureSnapshot, ExposureChannel, ExposureEvent, ExposureLedger,
    VisibilityReason, VisibleRootRelation,
)

HG1_REPLAY_VERSION: Final[str] = "cascade_hg1_replay.v1.pre_session_public_exposure"
HG1_OUTCOME_REPLAY_VERSION: Final[str] = "cascade_hg1_outcome_replay.v1.visible_cross_claim_context"
_PRELOAD_MARKER = "-hg1-preload-"


class HG1CascadeProtocolRunner(CascadeProtocolRunner):
    """Opt-in runner whose declared round-zero public content is pre-session."""

    @staticmethod
    def preload_initial_events(scenario: CascadeScenarioSpec) -> tuple[ExposureEvent, ...]:
        nodes = {node.node_id: node for node in scenario.graph.nodes}
        events: list[ExposureEvent] = []
        for index, item in enumerate(scenario.initial_public_content):
            if item.round_id != 0:
                continue
            node = nodes[item.provenance_node_id]
            channel = ExposureChannel.QUOTED_EVIDENCE if item.evidence_card_ids else ExposureChannel.TIMELINE
            reason = VisibilityReason.EVIDENCE_DELIVERY if item.evidence_card_ids else VisibilityReason.TIMELINE_DELIVERY
            for target_index, target in enumerate(item.target_agent_ids):
                events.append(ExposureEvent(
                    event_id=f"{scenario.scenario_id}{_PRELOAD_MARKER}i{index}-t{target_index}",
                    target_agent_id=target, source_actor_id="system", round_id=0,
                    content_id=node.content_id, claim_id=node.claim_id,
                    provenance_node_id=node.node_id, channel=channel,
                    visibility_reason=reason, evidence_card_ids=item.evidence_card_ids,
                ))
        return tuple(events)

    @staticmethod
    def _is_preload(event: ExposureEvent) -> bool:
        return event.source_actor_id == "system" and event.round_id == 0 and _PRELOAD_MARKER in event.event_id

    def build_round_snapshot(self, scenario: CascadeScenarioSpec, events: tuple[ExposureEvent, ...], round_id: int):
        snapshots = []
        for agent_id in scenario.agent_ids:
            prior = tuple(
                event for event in events
                if event.target_agent_id == agent_id
                and (event.round_id < round_id or (round_id == 0 and self._is_preload(event)))
            )
            node_ids = tuple(dict.fromkeys(event.provenance_node_id for event in prior))
            snapshots.append(CascadeExposureSnapshot(
                snapshot_id=f"{scenario.scenario_id}-hg1-snapshot-{agent_id}-r{round_id}",
                scenario_id=scenario.scenario_id, agent_id=agent_id, round_id=round_id,
                captured_event_ids=tuple(event.event_id for event in prior),
                visible_claim_ids=tuple(dict.fromkeys(event.claim_id for event in prior)),
                visible_content_ids=tuple(dict.fromkeys(event.content_id for event in prior)),
                visible_evidence_ids=tuple(dict.fromkeys(eid for event in prior for eid in event.evidence_card_ids)),
                visible_provenance_node_ids=node_ids,
                visible_root_relations=tuple(
                    VisibleRootRelation(
                        provenance_node_id=node_id,
                        source_root_ids=tuple(sorted(scenario.graph.root_sources_for_node(node_id))),
                    ) for node_id in node_ids
                ),
            ))
        return tuple(snapshots)

    @staticmethod
    def _append_initial_content(scenario: CascadeScenarioSpec, round_id: int, events: list[ExposureEvent]) -> None:
        if round_id == 0:
            return
        CascadeProtocolRunner._append_initial_content(scenario, round_id, events)


class HG1CascadeReplayValidator:
    @staticmethod
    def _normalized(ledger: ExposureLedger) -> ExposureLedger:
        events = tuple(
            event if HG1CascadeProtocolRunner._is_preload(event)
            else event.model_copy(update={"round_id": event.round_id + 1})
            for event in ledger.events
        )
        snapshots = tuple(
            snapshot.model_copy(update={"round_id": snapshot.round_id + 1})
            for snapshot in ledger.snapshots
        )
        return ExposureLedger(
            scenario_id=ledger.scenario_id, agent_ids=ledger.agent_ids,
            events=events, snapshots=snapshots,
        )

    @classmethod
    def validate(cls, graph: ProvenanceGraph, ledger: ExposureLedger) -> CascadeReplayReport:
        normalized = cls._normalized(ledger)
        CascadeReplayValidator.validate(graph, normalized)
        return CascadeReplayReport(
            status=CascadeReplayStatus.PASSED, scenario_id=graph.scenario_id,
            agent_count=len(ledger.agent_ids), exposure_count=len(ledger.events),
            snapshot_count=len(ledger.snapshots),
            root_relation_count=sum(len(item.visible_root_relations) for item in ledger.snapshots),
        )


class HG1OutcomeReplayValidator:
    @staticmethod
    def validate(
        graph: ProvenanceGraph,
        exposure: ExposureLedger,
        outcomes: CascadeOutcomeLedger,
        application: CascadeApplicationLedger | None = None,
        contexts: tuple[ControlledRoundContext, ...] = (),
    ) -> CascadeOutcomeReplayReport:
        cascade_report = HG1CascadeReplayValidator.validate(graph, exposure)
        HG1OutcomeReplayValidator._validate_outcomes(graph, exposure, outcomes)
        app_report = CascadeApplicationReplayValidator.validate(application or CascadeApplicationLedger(), contexts)
        return CascadeOutcomeReplayReport(
            status=CascadeOutcomeReplayStatus.PASSED,
            cascade_replay=cascade_report, application_replay=app_report,
            outcome_count=len(outcomes.outcomes),
        )

    @staticmethod
    def _validate_outcomes(graph: ProvenanceGraph, exposure: ExposureLedger, outcomes: CascadeOutcomeLedger) -> None:
        if graph.scenario_id != outcomes.scenario_id or exposure.scenario_id != outcomes.scenario_id:
            raise ValueError("outcome public scenario mismatch")
        if tuple(exposure.agent_ids) != outcomes.agent_ids:
            raise ValueError("outcome agent set mismatch")
        claims = {claim.claim_id: claim for claim in graph.claims}
        nodes = {node.content_id: node for node in graph.nodes}
        cards = {card.evidence_id: card for card in graph.evidence_cards}
        snapshots = {(item.agent_id, item.round_id): item for item in exposure.snapshots}
        for outcome in outcomes.outcomes:
            snapshot = snapshots.get((outcome.agent_id, outcome.round_id))
            if snapshot is None or outcome.claim_id not in claims:
                raise ValueError("outcome coordinate invalid")
            if not set(outcome.content_ids).issubset(snapshot.visible_content_ids):
                raise ValueError("outcome content not visible")
            if not set(outcome.evidence_card_ids).issubset(snapshot.visible_evidence_ids):
                raise ValueError("outcome evidence not visible")
            for content_id in outcome.content_ids:
                node = nodes.get(content_id)
                if node is None or node.claim_id not in snapshot.visible_claim_ids or node.round_id > outcome.round_id:
                    raise ValueError("outcome content invalid")
            for evidence_id in outcome.evidence_card_ids:
                card = cards.get(evidence_id)
                if card is None or card.introduced_round > outcome.round_id:
                    raise ValueError("outcome evidence invalid")
                if not (card.is_visible_to(outcome.agent_id) or "public" in card.visible_to or "*" in card.visible_to):
                    raise ValueError("outcome evidence unauthorized")


__all__ = [
    "HG1_REPLAY_VERSION", "HG1_OUTCOME_REPLAY_VERSION",
    "HG1CascadeProtocolRunner", "HG1CascadeReplayValidator", "HG1OutcomeReplayValidator",
]
