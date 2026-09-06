"""Opt-in deterministic cascade runner with public outcomes.

The historical :class:`~evicon.cascade_protocol.CascadeProtocolRunner` is
left untouched.  This adapter adds a separate path in which a deterministic
actor consumes an already-applied structured directive at the beginning of a
round, records a public stance, and only then lets the fixed public schedule
append exposure events.  No private evaluator fixture, provider, prompt, or
model output is accepted by this module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cascade_application_replay import CascadeApplicationReplayValidator
from .cascade_actors import (
    ActorProfile,
    DeterministicActor,
    DeterministicActorSchedule,
    PublicActorDecision,
)
from .cascade_controller import (
    CascadeControllerPolicyConfig,
    CascadeControllerPolicyLoader,
    CascadeControllerValidationError,
    propose,
)
from .cascade_intervention_application import (
    CascadeApplicationLedger,
    CascadeApplicationStatus,
    ControlledCascadeProtocolRunner,
    ControlledRoundContext,
    CascadeScheduleStatus,
    snapshot_sha256,
)
from .cascade_outcome_replay import (
    CascadeOutcomeReplayReport,
    CascadeOutcomeReplayValidator,
)
from .cascade_outcomes import (
    CascadeOutcomeLedger,
    ClaimStance,
    OutcomeSource,
    PublicClaimOutcome,
)
from .cascade_protocol import (
    ActorAction,
    ActorScheduleEntry,
    CascadeProtocolError,
    CascadeProtocolRunner,
    CascadeScenarioSpec,
)
from .models._validation import normalized_text
from .provenance_cascade_preregistration import CascadeCondition
from .provenance_cascade_exposure import ControllerPublicView, ExposureLedger


_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_POLICY = _ROOT / "configs" / "provenance_cascade" / "cascade_controller_policy.v1.toml"


class CascadeOutcomeRunError(ValueError):
    """Stable, non-sensitive runner error code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class CascadeOutcomeRunRecord(BaseModel):
    """Safe, replayable public record for one scenario/seed/condition run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    scenario_id: str
    seed: int = Field(gt=0)
    condition: CascadeCondition
    agent_ids: tuple[str, ...] = Field(min_length=1)
    round_count: int = Field(ge=1)
    content_event_count: int = Field(ge=0)
    exposure_event_count: int = Field(ge=0)
    snapshot_count: int = Field(ge=0)
    outcome_count: int = Field(ge=0)
    directive_applied_count: int = Field(ge=0)
    proposal_count: int = Field(ge=0)
    exposure_ledger_sha256: str = Field(min_length=64, max_length=64)
    snapshot_sha256: str = Field(min_length=64, max_length=64)
    application_ledger_sha256: str = Field(min_length=64, max_length=64)
    outcome_ledger_sha256: str = Field(min_length=64, max_length=64)
    replay: CascadeOutcomeReplayReport
    exposure_ledger: ExposureLedger
    application_ledger: CascadeApplicationLedger
    outcome_ledger: CascadeOutcomeLedger
    round_contexts: tuple[ControlledRoundContext, ...] = ()
    development_only: bool = True
    not_paper_result: bool = True
    no_causal_conclusion: bool = True

    @model_validator(mode="after")
    def security_and_shape(self) -> "CascadeOutcomeRunRecord":
        if not (self.development_only and self.not_paper_result and self.no_causal_conclusion):
            raise ValueError("cascade outcome records must retain development safety flags")
        if self.exposure_ledger.scenario_id != self.scenario_id or self.outcome_ledger.scenario_id != self.scenario_id:
            raise ValueError("run record scenario mismatch")
        if self.outcome_ledger.agent_ids != self.agent_ids:
            raise ValueError("run record agent mismatch")
        return self


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _condition(value: CascadeCondition | str) -> CascadeCondition:
    try:
        return value if isinstance(value, CascadeCondition) else CascadeCondition(value)
    except (TypeError, ValueError) as exc:
        raise CascadeOutcomeRunError("condition_invalid") from exc


def _schedule_for(
    agent_id: str,
    actor_schedule: DeterministicActorSchedule | Mapping[str, DeterministicActorSchedule] | None,
) -> DeterministicActorSchedule:
    if actor_schedule is None:
        return DeterministicActorSchedule()
    if isinstance(actor_schedule, DeterministicActorSchedule):
        return actor_schedule
    item = actor_schedule.get(agent_id)
    if item is None:
        return DeterministicActorSchedule()
    if not isinstance(item, DeterministicActorSchedule):
        raise CascadeOutcomeRunError("actor_schedule_invalid")
    return item


def _model_hash(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return _sha256_json(value)


class CascadeOutcomeRunner:
    """Run public deterministic actors with optional next-round directives."""

    def __init__(
        self,
        *,
        base_runner: CascadeProtocolRunner | None = None,
        policy_config: CascadeControllerPolicyConfig | None = None,
    ) -> None:
        self.base_runner = base_runner or CascadeProtocolRunner()
        self.policy_config = policy_config or CascadeControllerPolicyLoader.load(_DEFAULT_POLICY)
        self.controlled_runner = ControlledCascadeProtocolRunner(self.base_runner)

    def build_round_snapshot(
        self,
        scenario: CascadeScenarioSpec,
        events: tuple[object, ...],
        round_id: int,
    ) -> tuple[object, ...]:
        """Expose the same round-start snapshot primitive as the base runner."""
        return self.base_runner.build_round_snapshot(scenario, events, round_id)

    def run_round(
        self,
        scenario: CascadeScenarioSpec,
        events: tuple[object, ...],
        round_id: int,
    ) -> tuple[object, ...]:
        """Return an immutable round-start view without mutating a run."""
        return self.build_round_snapshot(scenario, events, round_id)

    def run_scenario(
        self,
        scenario: CascadeScenarioSpec,
        seed: int,
        condition: CascadeCondition | str = CascadeCondition.NO_INTERVENTION,
        *,
        run_id: str | None = None,
        actor_schedule: DeterministicActorSchedule | Mapping[str, DeterministicActorSchedule] | None = None,
        actor_profile: ActorProfile = ActorProfile.REPEAT_ACTOR,
        policy_config: CascadeControllerPolicyConfig | None = None,
    ) -> CascadeOutcomeRunRecord:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
            raise CascadeOutcomeRunError("seed_invalid")
        condition_id = _condition(condition)
        config = policy_config or self.policy_config
        if condition_id not in config.conditions:
            raise CascadeOutcomeRunError("condition_not_in_policy")
        if len(scenario.agent_ids) != 6 or scenario.max_rounds != 3:
            raise CascadeOutcomeRunError("scenario_shape_invalid")
        if scenario.policy != "no_intervention":
            raise CascadeOutcomeRunError("scenario_policy_invalid")
        if scenario.scenario_id not in {rule.scenario_id for rule in config.scenario_rules}:
            raise CascadeOutcomeRunError("scenario_not_in_policy")
        record_id = normalized_text(run_id or f"{scenario.scenario_id}-{seed}-{condition_id.value}", "run_id")

        events: list[Any] = []
        outcomes: list[PublicClaimOutcome] = []
        application = CascadeApplicationLedger()
        contexts: list[ControlledRoundContext] = []
        proposal_count = 0
        actor = DeterministicActor(actor_profile)
        graph_claim_ids = tuple(claim.claim_id for claim in scenario.graph.claims)

        for round_id in range(scenario.max_rounds):
            round_snapshot = tuple(self.base_runner.build_round_snapshot(scenario, tuple(events), round_id))

            # Scheduled proposals are consumed before actor decisions in the
            # next round, never in the round that created them.
            for scheduled in tuple(application.schedules):
                if scheduled.status is not CascadeScheduleStatus.SCHEDULED or scheduled.effective_round_id != round_id:
                    continue
                target_snapshot = next((item for item in round_snapshot if item.agent_id == scheduled.target_agent_id), None)
                if target_snapshot is None:
                    raise CascadeOutcomeRunError("directive_target_unknown")
                target_view = ControllerPublicView.from_snapshot(target_snapshot, scenario.graph)
                application, applied = application.apply_schedule(
                    scheduled.schedule_id,
                    view=target_view,
                    snapshot_hash=scheduled.input_snapshot_sha256,
                    current_round_id=round_id,
                )
                if applied.status is CascadeApplicationStatus.REJECTED:
                    raise CascadeOutcomeRunError("scheduled_directive_rejected")

            contexts_for_round = self.controlled_runner.build_next_round_context(
                scenario, tuple(events), round_id, application
            )
            contexts.extend(contexts_for_round)
            directives_by_agent = {
                context.snapshot.agent_id: (context.directives[0] if context.directives else None)
                for context in contexts_for_round
            }

            # All actors see the same immutable snapshot tuple.  No event is
            # appended until every decision for this round is complete.
            decisions: dict[str, PublicActorDecision] = {}
            for snapshot in round_snapshot:
                view = ControllerPublicView.from_snapshot(snapshot, scenario.graph)
                schedule = _schedule_for(snapshot.agent_id, actor_schedule)
                decisions[snapshot.agent_id] = actor.act(
                    snapshot.agent_id,
                    round_id,
                    view,
                    directives_by_agent.get(snapshot.agent_id),
                    schedule.model_copy(update={"profile": actor_profile}),
                )
                self._append_decision_outcomes(
                    outcomes,
                    scenario=scenario,
                    snapshot=snapshot,
                    decision=decisions[snapshot.agent_id],
                    graph_claim_ids=graph_claim_ids,
                )

            # Public propagation is the fixed declaration from the scenario;
            # actor directives cannot create or hide content in this stage.
            self.base_runner._append_initial_content(scenario, round_id, events)
            schedule_entries = tuple(item for item in scenario.actor_schedule if item.round_id == round_id)
            for index, entry in enumerate(schedule_entries):
                if entry.action is ActorAction.ABSTAIN:
                    continue
                for target_index, target_agent_id in enumerate(entry.target_agent_ids):
                    event = self.base_runner.append_public_event(
                        scenario=scenario,
                        events=tuple(events),
                        snapshots=tuple(context.snapshot for context in contexts_for_round),
                        entry=entry,
                        target_agent_id=target_agent_id,
                        event_id=f"{record_id}-r{round_id}-e{index}-t{target_index}",
                        round_snapshot=round_snapshot,
                    )
                    events.append(event)

            # Proposals are created only after this round's public activity;
            # they are bound to the already-frozen round-start view.
            if round_id < scenario.max_rounds - 1 and condition_id is not CascadeCondition.NO_INTERVENTION:
                for snapshot in round_snapshot:
                    if not snapshot.visible_claim_ids:
                        continue
                    view = ControllerPublicView.from_snapshot(snapshot, scenario.graph)
                    try:
                        proposal = propose(condition_id, view, config)
                    except CascadeControllerValidationError as exc:
                        raise CascadeOutcomeRunError("controller_view_invalid") from exc
                    proposal_count += 1
                    application, result = application.schedule_proposal(
                        proposal,
                        view,
                        snapshot_hash=snapshot_sha256(snapshot),
                        config=config,
                    )
                    if result.status not in {
                        CascadeApplicationStatus.NO_EFFECT,
                        CascadeApplicationStatus.SCHEDULED,
                    }:
                        # A rejected proposal is an auditable public failure;
                        # the runner does not silently downgrade it.
                        raise CascadeOutcomeRunError("proposal_application_rejected")

        exposure = ExposureLedger(
            scenario_id=scenario.scenario_id,
            agent_ids=scenario.agent_ids,
            events=tuple(events),
            snapshots=tuple(context.snapshot for context in contexts),
        )
        outcome_ledger = CascadeOutcomeLedger(
            scenario_id=scenario.scenario_id,
            agent_ids=scenario.agent_ids,
            outcomes=tuple(outcomes),
        )
        try:
            replay = CascadeOutcomeReplayValidator.validate(
                scenario.graph,
                exposure,
                outcome_ledger,
                application,
                tuple(contexts),
            )
        except Exception as exc:
            raise CascadeOutcomeRunError("cascade_outcome_replay_failed") from exc
        exposure_digest = _model_hash(exposure)
        snapshot_digest = _model_hash([context.snapshot.model_dump(mode="json") for context in contexts])
        application_digest = _model_hash(application)
        outcome_digest = outcome_ledger.sha256()
        return CascadeOutcomeRunRecord(
            run_id=record_id,
            scenario_id=scenario.scenario_id,
            seed=seed,
            condition=condition_id,
            agent_ids=scenario.agent_ids,
            round_count=scenario.max_rounds,
            content_event_count=len(events),
            exposure_event_count=len(events),
            snapshot_count=len(exposure.snapshots),
            outcome_count=len(outcome_ledger.outcomes),
            directive_applied_count=sum(1 for item in application.schedules if item.status is CascadeScheduleStatus.APPLIED),
            proposal_count=proposal_count,
            exposure_ledger_sha256=exposure_digest,
            snapshot_sha256=snapshot_digest,
            application_ledger_sha256=application_digest,
            outcome_ledger_sha256=outcome_digest,
            replay=replay,
            exposure_ledger=exposure,
            application_ledger=application,
            outcome_ledger=outcome_ledger,
            round_contexts=tuple(contexts),
        )

    @staticmethod
    def _append_decision_outcomes(
        outcomes: list[PublicClaimOutcome],
        *,
        scenario: CascadeScenarioSpec,
        snapshot: Any,
        decision: PublicActorDecision,
        graph_claim_ids: tuple[str, ...],
    ) -> None:
        visible_claims = set(snapshot.visible_claim_ids)
        if decision.claim_id in visible_claims:
            if not set(decision.used_content_ids).issubset(snapshot.visible_content_ids):
                raise CascadeOutcomeRunError("actor_content_not_visible")
            if not set(decision.used_evidence_card_ids).issubset(snapshot.visible_evidence_ids):
                raise CascadeOutcomeRunError("actor_evidence_not_visible")
        for claim_id in graph_claim_ids:
            if claim_id == decision.claim_id and claim_id in visible_claims:
                outcomes.append(
                    PublicClaimOutcome(
                        scenario_id=scenario.scenario_id,
                        agent_id=decision.agent_id,
                        claim_id=claim_id,
                        round_id=snapshot.round_id,
                        stance=decision.stance,
                        content_ids=decision.used_content_ids,
                        evidence_card_ids=decision.used_evidence_card_ids,
                        source=OutcomeSource.ACTOR_OBSERVATION,
                    )
                )
            else:
                outcomes.append(
                    PublicClaimOutcome(
                        scenario_id=scenario.scenario_id,
                        agent_id=decision.agent_id,
                        claim_id=claim_id,
                        round_id=snapshot.round_id,
                        stance=ClaimStance.NO_POSITION,
                        source=OutcomeSource.ACTOR_OBSERVATION,
                    )
                )


__all__ = [
    "CascadeOutcomeRunError",
    "CascadeOutcomeRunRecord",
    "CascadeOutcomeRunner",
]
