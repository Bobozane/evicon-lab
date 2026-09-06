"""FakeLLM-only adaptive online control loop with auditable budget state."""

from __future__ import annotations

import json
from pathlib import Path

from .adaptive_control import (
    BudgetEntryKind,
    BudgetLedger,
    ControllerHistoryEntry,
    ControllerHistoryStatus,
    ControllerState,
    FakeOnlineStateProvider,
    FakeTargetCandidateProvider,
    OnlineStateProvider,
    TargetCandidateProvider,
    controller_state_from_ledger,
)
from .conformity_monitor import ConformityMonitor, MonitorConfig, MonitorInput
from .controlled_runner import ControlledProtocolRunner
from .events import EventType, JsonlEventLog
from .executor import ExecutionContext, InterventionExecutor
from .fake_llm import FakeLLM, FakeLLMRequest, InterventionMetadata, LocalProvider
from .intervention_application import AppliedIntervention, apply_intervention_plan
from .models import DialogueState, DialogueTurn, EvidenceExposure, RunConfig, RunRecord, RunStatus, ScenarioSpec
from .models.protocols import InterventionAction
from .policy import InterventionDecision, InterventionPolicy, PolicyConfig, PolicyInput


class AdaptiveProtocolRunner(ControlledProtocolRunner):
    """Opt-in online Monitor -> Policy -> Executor loop using only FakeLLM."""

    def __init__(
        self,
        config: RunConfig,
        *,
        scenario: ScenarioSpec,
        online_state_provider: OnlineStateProvider,
        target_candidate_provider: TargetCandidateProvider,
        monitor_config: MonitorConfig,
        policy_config: PolicyConfig,
        provider: LocalProvider | None = None,
    ) -> None:
        super().__init__(
            config,
            scenario=scenario,
            intervention_schedule=None,
            provider=provider or FakeLLM(),
        )
        self.online_state_provider = online_state_provider
        self.target_candidate_provider = target_candidate_provider
        self.monitor_config = monitor_config
        self.policy_config = policy_config
        self.ledger = BudgetLedger.initial(config.intervention_budget)
        self.controller_state = controller_state_from_ledger(self.ledger)
        self._pending_plan: object | None = None

    def run(self) -> RunRecord:
        """Run public turns and generate future plans only after round completion."""
        output_directory = self._prepare_output_directory()
        logger = JsonlEventLog(output_directory / "events.jsonl", self.config.run_id)
        turns: list[DialogueTurn] = []
        evidence_exposures: list[EvidenceExposure] = []
        state: DialogueState | None = None
        active_effect: AppliedIntervention | None = None
        applied_plan_ids: set[str] = set()

        logger.append(EventType.RUN_STARTED, round_id=0, payload=self._run_payload())
        try:
            for round_id in range(self.scenario.max_rounds):
                state = self._build_state(round_id, turns)
                logger.append(EventType.ROUND_STARTED, round_id=round_id, payload={})

                active_effect = self._apply_pending_plan(state, round_id, logger, applied_plan_ids)
                snapshots = self._snapshots_for_round(state, round_id, active_effect)
                turn_order = self._turn_order(state, active_effect)
                responses: list[tuple[object, str]] = []
                for agent_id in turn_order:
                    snapshot = snapshots[agent_id]
                    evidence_exposures.extend(self._record_exposures(snapshot))
                    logger.append(
                        EventType.EXPOSURE_CREATED,
                        round_id=round_id,
                        payload=self._exposure_payload(snapshot),
                    )
                    metadata = self._intervention_metadata_for(agent_id, active_effect)
                    request = FakeLLMRequest(
                        agent_id=agent_id,
                        round_id=round_id,
                        protocol=snapshot.protocol,
                        scenario_context=snapshot.scenario_context,
                        visible_history=snapshot.visible_history,
                        visible_peer_turn_ids=snapshot.visible_peer_turn_ids,
                        visible_evidence_ids=snapshot.visible_evidence_ids,
                        seed=self.config.seed,
                        intervention_metadata=metadata,
                    )
                    logger.append(
                        EventType.LLM_REQUEST,
                        round_id=round_id,
                        payload={
                            **self._exposure_payload(snapshot),
                            "seed": request.seed,
                            "intervention_metadata": (
                                metadata.model_dump(mode="json") if metadata is not None else None
                            ),
                        },
                    )
                    response = self.provider.complete(request)
                    logger.append(
                        EventType.LLM_RESPONSE,
                        round_id=round_id,
                        payload={
                            "agent_id": response.agent_id,
                            "message": response.message,
                            "request_fingerprint": response.request_fingerprint,
                            "visible_peer_turn_ids": response.visible_peer_turn_ids,
                            "visible_evidence_ids": response.visible_evidence_ids,
                        },
                    )
                    responses.append((snapshot, response.message))

                new_turns = [self._build_turn(snapshot, message) for snapshot, message in responses]
                turns.extend(new_turns)
                for turn in new_turns:
                    logger.append(
                        EventType.TURN_COMPLETED,
                        round_id=round_id,
                        payload={
                            "turn_id": turn.turn_id,
                            "speaker_id": turn.speaker_id,
                            "visible_peer_turn_ids": turn.visible_peer_turn_ids,
                            "visible_evidence_ids": turn.visible_evidence_ids,
                        },
                    )
                state = self._build_state(round_id, turns)
                logger.append(
                    EventType.ROUND_COMPLETED,
                    round_id=round_id,
                    payload={"turn_count": len(turns)},
                )
                if round_id < self.scenario.max_rounds - 1:
                    self._control_after_round(state, round_id, logger)

            self.last_state = state
            record = self._record(turns, evidence_exposures, RunStatus.COMPLETED)
            logger.append(
                EventType.RUN_COMPLETED,
                round_id=self.scenario.max_rounds - 1,
                payload={"turn_count": len(turns)},
            )
            self._write_record(output_directory, record)
            self._write_controller_state(output_directory)
            return record
        except Exception as exc:
            if self._pending_plan is not None:
                self._release_pending(round_id if state is not None else 0, logger, "run_failed")
            failed_round = state.current_round if state is not None else 0
            logger.append(
                EventType.RUN_FAILED,
                round_id=failed_round,
                payload={"error_type": type(exc).__name__},
            )
            failed_record = self._record(
                turns,
                evidence_exposures,
                RunStatus.FAILED,
                error_message=f"run failed: {type(exc).__name__}",
            )
            self._write_record(output_directory, failed_record)
            self._write_controller_state(output_directory)
            raise

    def _control_after_round(
        self,
        state: DialogueState,
        round_id: int,
        logger: JsonlEventLog,
    ) -> None:
        observation_config = self.config.model_copy(
            update={"intervention_budget": self.ledger.remaining_budget}
        )
        monitor_input = self.online_state_provider.observe(
            state,
            self.scenario,
            observation_config,
            round_id,
        )
        _ensure_public_monitor_input(monitor_input)
        correlation_id = f"online-{self.config.run_id}-r{round_id}"
        logger.append(
            EventType.ONLINE_STATE_OBSERVED,
            round_id=round_id,
            payload={
                "correlation_id": correlation_id,
                "budget_before": self.ledger.remaining_budget,
                "budget_after": self.ledger.remaining_budget,
                "cooldown_until": self.controller_state.cooldown_until,
                **_monitor_input_payload(monitor_input),
            },
        )
        monitor_result = ConformityMonitor.evaluate(monitor_input, self.monitor_config)
        logger.append(
            EventType.MONITOR_EVALUATED,
            round_id=round_id,
            payload={
                "correlation_id": correlation_id,
                "risk_score": monitor_result.risk_score,
                "risk_level": monitor_result.risk_level.value,
                "should_intervene": monitor_result.should_intervene,
                "evidence_supported_update": monitor_result.evidence_supported_update,
                "reasons": [reason.value for reason in monitor_result.reasons],
                "budget_before": self.ledger.remaining_budget,
                "budget_after": self.ledger.remaining_budget,
                "cooldown_until": self.controller_state.cooldown_until,
                "warnings": [],
                "errors": [],
            },
        )
        candidates = self.target_candidate_provider.build_candidates(
            state,
            monitor_result,
            self.scenario,
            round_id,
        )
        candidates = list(candidates)
        logger.append(
            EventType.TARGET_CANDIDATES_BUILT,
            round_id=round_id,
            payload={
                "correlation_id": correlation_id,
                "candidate_count": len(candidates),
                "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
                "budget_before": self.ledger.remaining_budget,
                "budget_after": self.ledger.remaining_budget,
                "cooldown_until": self.controller_state.cooldown_until,
                "warnings": [],
                "errors": [],
            },
        )
        policy_input = self._policy_input(monitor_result, candidates)
        decision = InterventionPolicy.decide(policy_input, self.policy_config)
        logger.append(
            EventType.POLICY_DECIDED,
            round_id=round_id,
            payload={
                "correlation_id": correlation_id,
                "action": decision.action.value,
                "target_agent_ids": decision.target_agent_ids,
                "risk_score": decision.risk_score,
                "estimated_cost": decision.estimated_cost,
                "reason_codes": [reason.value for reason in decision.reason_codes],
                "budget_before": self.ledger.remaining_budget,
                "budget_after": self.ledger.remaining_budget,
                "cooldown_until": self.controller_state.cooldown_until,
                "warnings": [],
                "errors": [],
            },
        )
        if decision.action is InterventionAction.NO_OP:
            return
        if round_id + 1 >= self.scenario.max_rounds:
            self._append_history(
                plan_id=f"cancelled-r{round_id}",
                action=decision.action,
                round_id=round_id,
                status=ControllerHistoryStatus.CANCELLED,
                cost=decision.estimated_cost,
                reason_codes=["no_future_round"],
            )
            return
        plan = InterventionExecutor.plan(decision, self._execution_context(state))
        logger.append(
            EventType.INTERVENTION_PLAN_CREATED,
            round_id=round_id,
            payload=self._plan_event_payload(
                plan,
                correlation_id=correlation_id,
                scheduled_round=round_id + 1,
                budget_before=self.ledger.remaining_budget,
                budget_after=self.ledger.remaining_budget,
                errors=[error.value for error in plan.validation_errors],
            ),
        )
        if not plan.valid:
            self._append_history(
                plan_id=plan.plan_id,
                action=plan.action,
                round_id=round_id,
                status=ControllerHistoryStatus.REJECTED,
                cost=plan.estimated_cost,
                reason_codes=[error.value for error in plan.validation_errors],
            )
            return
        scheduled_plan = plan.model_copy(update={"round_id": round_id + 1})
        before = self.ledger.remaining_budget
        try:
            self.ledger = self.ledger.reserve(
                plan.estimated_cost,
                plan_id=plan.plan_id,
                round_id=round_id,
            )
        except ValueError:
            return
        self._pending_plan = scheduled_plan
        self._append_history(
            plan_id=plan.plan_id,
            action=plan.action,
            round_id=round_id,
            status=ControllerHistoryStatus.PLANNED,
            cost=plan.estimated_cost,
            reason_codes=[reason.value for reason in plan.reason_codes],
        )
        self._sync_controller_state(scheduled_plan.plan_id)
        logger.append(
            EventType.INTERVENTION_PLAN_SCHEDULED,
            round_id=round_id,
            payload=self._plan_event_payload(
                scheduled_plan,
                correlation_id=correlation_id,
                scheduled_round=round_id + 1,
                budget_before=before,
                budget_after=self.ledger.remaining_budget,
                errors=[],
            ),
        )
        logger.append(
            EventType.BUDGET_RESERVED,
            round_id=round_id,
            payload=self._budget_payload(
                scheduled_plan.plan_id,
                round_id,
                before,
                self.ledger.remaining_budget,
                BudgetEntryKind.RESERVED,
            ),
        )

    def _apply_pending_plan(
        self,
        state: DialogueState,
        round_id: int,
        logger: JsonlEventLog,
        applied_plan_ids: set[str],
    ) -> AppliedIntervention | None:
        if self._pending_plan is None:
            return None
        plan = self._pending_plan
        if plan.plan_id in applied_plan_ids:  # type: ignore[attr-defined]
            raise ValueError("adaptive plan cannot be applied twice")
        result = apply_intervention_plan(state, plan, self.scenario)
        if not result.applied:
            before = self.ledger.remaining_budget
            self.ledger = self.ledger.release(
                plan.estimated_cost,  # type: ignore[attr-defined]
                plan_id=plan.plan_id,  # type: ignore[attr-defined]
                round_id=round_id,
            )
            self._sync_controller_state(None)
            logger.append(
                EventType.INTERVENTION_PLAN_REJECTED,
                round_id=round_id,
                payload=self._plan_event_payload(
                    plan,
                    correlation_id=plan.plan_id,  # type: ignore[attr-defined]
                    scheduled_round=round_id,
                    budget_before=before,
                    budget_after=self.ledger.remaining_budget,
                    errors=result.errors,
                ),
            )
            logger.append(
                EventType.BUDGET_RELEASED,
                round_id=round_id,
                payload=self._budget_payload(
                    plan.plan_id,  # type: ignore[attr-defined]
                    round_id,
                    before,
                    self.ledger.remaining_budget,
                    BudgetEntryKind.RELEASED,
                ),
            )
            self._append_history(
                plan_id=plan.plan_id,  # type: ignore[attr-defined]
                action=plan.action,  # type: ignore[attr-defined]
                round_id=round_id,
                status=ControllerHistoryStatus.REJECTED,
                cost=plan.estimated_cost,  # type: ignore[attr-defined]
                reason_codes=result.errors,
            )
            self._pending_plan = None
            return None
        before = self.ledger.remaining_budget
        self.ledger = self.ledger.spend(
            plan.estimated_cost,  # type: ignore[attr-defined]
            plan_id=plan.plan_id,  # type: ignore[attr-defined]
            round_id=round_id,
        )
        applied_plan_ids.add(plan.plan_id)  # type: ignore[attr-defined]
        self._append_history(
            plan_id=plan.plan_id,  # type: ignore[attr-defined]
            action=plan.action,  # type: ignore[attr-defined]
            round_id=round_id,
            status=ControllerHistoryStatus.APPLIED,
            cost=plan.estimated_cost,  # type: ignore[attr-defined]
            reason_codes=[reason for reason in plan.reason_codes],  # type: ignore[attr-defined]
        )
        self._sync_controller_state(None, last_intervention_round=round_id)
        self.controller_state = self.controller_state.model_copy(
            update={"cooldown_until": round_id + self.policy_config.cooldown_rounds}
        )
        logger.append(
            EventType.INTERVENTION_PLAN_APPLIED,
            round_id=round_id,
            payload=self._plan_event_payload(
                plan,
                correlation_id=plan.plan_id,  # type: ignore[attr-defined]
                scheduled_round=round_id,
                budget_before=before,
                budget_after=self.ledger.remaining_budget,
                errors=[],
                applied=True,
            ),
        )
        logger.append(
            EventType.BUDGET_SPENT,
            round_id=round_id,
            payload=self._budget_payload(
                plan.plan_id,  # type: ignore[attr-defined]
                round_id,
                before,
                self.ledger.remaining_budget,
                BudgetEntryKind.SPENT,
            ),
        )
        logger.append(
            EventType.COOLDOWN_UPDATED,
            round_id=round_id,
            payload={
                "correlation_id": plan.plan_id,  # type: ignore[attr-defined]
                "plan_id": plan.plan_id,  # type: ignore[attr-defined]
                "action": plan.action.value,  # type: ignore[attr-defined]
                "budget_before": self.ledger.remaining_budget,
                "budget_after": self.ledger.remaining_budget,
                "cooldown_until": self.controller_state.cooldown_until,
                "warnings": [],
                "errors": [],
            },
        )
        self._pending_plan = None
        return result

    def _release_pending(self, round_id: int, logger: JsonlEventLog, reason: str) -> None:
        if self._pending_plan is None:
            return
        plan = self._pending_plan
        before = self.ledger.remaining_budget
        self.ledger = self.ledger.release(
            plan.estimated_cost,
            plan_id=plan.plan_id,
            round_id=round_id,
        )
        self._sync_controller_state(None)
        logger.append(
            EventType.BUDGET_RELEASED,
            round_id=round_id,
            payload={
                "correlation_id": plan.plan_id,
                "plan_id": plan.plan_id,
                "action": plan.action.value,
                "reason_codes": [reason],
                "budget_before": before,
                "budget_after": self.ledger.remaining_budget,
                "cooldown_until": self.controller_state.cooldown_until,
                "warnings": [],
                "errors": [],
            },
        )
        self._append_history(
            plan_id=plan.plan_id,
            action=plan.action,
            round_id=round_id,
            status=ControllerHistoryStatus.CANCELLED,
            cost=plan.estimated_cost,
            reason_codes=[reason],
        )
        self._pending_plan = None

    def _policy_input(self, monitor_result: object, candidates: list[object]) -> PolicyInput:
        return PolicyInput(
            run_id=self.config.run_id,
            scenario_id=self.scenario.scenario_id,
            round_id=monitor_result.round_id,
            monitor_result=monitor_result,
            target_candidates=candidates,
            remaining_budget=self.ledger.remaining_budget,
            last_intervention_round=self.controller_state.last_intervention_round,
            cooldown_rounds=self.policy_config.cooldown_rounds,
            intervention_history=[],
            metadata={"source": "adaptive_runner"},
        )

    def _execution_context(self, state: DialogueState) -> ExecutionContext:
        return ExecutionContext(
            run_id=state.run_id,
            scenario_id=state.scenario_id,
            round_id=state.current_round,
            known_agent_ids=[agent.agent_id for agent in state.agents],
            visible_evidence_ids=[card.evidence_id for card in state.evidence_cards],
            visible_evidence_introduced_rounds={
                card.evidence_id: card.introduced_round for card in state.evidence_cards
            },
            candidate_peer_turn_ids=[turn.turn_id for turn in state.turns if turn.round_id < state.current_round],
            candidate_peer_turn_rounds={
                turn.turn_id: turn.round_id for turn in state.turns if turn.round_id < state.current_round
            },
            current_budget=self.ledger.remaining_budget,
            max_target_count=len(state.agents),
            intervention_history=[],
            last_action=None,
            metadata={},
        )

    def _snapshots_for_round(
        self,
        state: DialogueState,
        round_id: int,
        active_effect: AppliedIntervention | None,
    ) -> dict[str, object]:
        snapshots = __import__("evicon.exposure", fromlist=["ExposurePlan"]).ExposurePlan(
            self.config.protocol,
            scenario_context=self.scenario.initial_context,
        ).create(state, round_id=round_id)
        return self._apply_next_round_effect(state, snapshots, active_effect)

    def _intervention_metadata_for(
        self,
        agent_id: str,
        active_effect: AppliedIntervention | None,
    ) -> InterventionMetadata | None:
        if active_effect is None or active_effect.action is InterventionAction.NO_OP:
            return None
        is_target = agent_id in active_effect.affected_agent_ids
        return InterventionMetadata(
            current_action=active_effect.action,
            target_agent=is_target,
            evidence_request=agent_id in active_effect.evidence_request_agent_ids,
            blind_reflection=(
                active_effect.action is InterventionAction.BLIND_EVIDENCE_REFLECTION
                and is_target
            ),
            speaking_priority=(
                active_effect.action is InterventionAction.SOLICIT_DISSENT
                and agent_id in active_effect.next_turn_order[: len(active_effect.affected_agent_ids)]
            ),
            final_report_requirement=agent_id in active_effect.final_report_requirements,
        )

    def _append_history(
        self,
        *,
        plan_id: str,
        action: InterventionAction,
        round_id: int,
        status: ControllerHistoryStatus,
        cost: float,
        reason_codes: list[str],
    ) -> None:
        entry = ControllerHistoryEntry(
            plan_id=plan_id,
            action=action,
            round_id=round_id,
            status=status,
            estimated_cost=cost,
            reason_codes=reason_codes,
        )
        self.controller_state = self.controller_state.model_copy(
            update={"intervention_history": [*self.controller_state.intervention_history, entry]}
        )

    def _sync_controller_state(
        self,
        scheduled_plan_id: str | None,
        *,
        last_intervention_round: int | None = None,
    ) -> None:
        scheduled = list(self.controller_state.scheduled_plan_ids)
        if scheduled_plan_id is not None and scheduled_plan_id not in scheduled:
            scheduled.append(scheduled_plan_id)
        elif scheduled_plan_id is None and self._pending_plan is None:
            scheduled = []
        self.controller_state = controller_state_from_ledger(
            self.ledger,
            last_intervention_round=(
                last_intervention_round
                if last_intervention_round is not None
                else self.controller_state.last_intervention_round
            ),
            cooldown_until=self.controller_state.cooldown_until,
            history=list(self.controller_state.intervention_history),
            scheduled_plan_ids=scheduled,
        )

    def _plan_event_payload(
        self,
        plan: object,
        *,
        correlation_id: str,
        scheduled_round: int,
        budget_before: float,
        budget_after: float,
        errors: list[str],
        applied: bool = False,
    ) -> dict[str, object]:
        return {
            "correlation_id": correlation_id,
            "plan_id": plan.plan_id,
            "action": plan.action.value,
            "target_agent_ids": plan.target_agent_ids,
            "scheduled_round": scheduled_round,
            "applied": applied,
            "reason_codes": [reason for reason in plan.reason_codes],
            "estimated_cost": plan.estimated_cost,
            "budget_before": budget_before,
            "budget_after": budget_after,
            "cooldown_until": self.controller_state.cooldown_until,
            "warnings": [],
            "errors": errors,
        }

    def _budget_payload(
        self,
        plan_id: str,
        round_id: int,
        before: float,
        after: float,
        kind: BudgetEntryKind,
    ) -> dict[str, object]:
        return {
            "correlation_id": plan_id,
            "plan_id": plan_id,
            "action": self._history_action(plan_id),
            "budget_transition": kind.value,
            "budget_before": before,
            "budget_after": after,
            "cooldown_until": self.controller_state.cooldown_until,
            "warnings": [],
            "errors": [],
        }

    def _history_action(self, plan_id: str) -> str:
        for entry in reversed(self.controller_state.intervention_history):
            if entry.plan_id == plan_id:
                return entry.action.value
        return InterventionAction.NO_OP.value

    def _run_payload(self) -> dict[str, object]:
        return {
            "scenario_id": self.config.scenario_id,
            "scenario_title": self.scenario.title,
            "protocol": self.config.protocol.value,
            "model_name": self.config.model_name,
            "agent_count": self.config.agent_count,
            "max_rounds": self.config.max_rounds,
            "seed": self.config.seed,
            "adaptive": True,
        }

    def _write_controller_state(self, output_directory: Path) -> None:
        with (output_directory / "controller_state.json").open("x", encoding="utf-8") as handle:
            json.dump(self.controller_state.model_dump(mode="json"), handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")


def _monitor_input_payload(monitor_input: MonitorInput) -> dict[str, object]:
    return {
        "run_id": monitor_input.run_id,
        "scenario_id": monitor_input.scenario_id,
        "current_diversity": monitor_input.current_diversity,
        "previous_diversity": monitor_input.previous_diversity,
        "current_coverage": monitor_input.current_coverage,
        "previous_coverage": monitor_input.previous_coverage,
        "minority_loss": monitor_input.minority_loss,
        "evidence_gain": monitor_input.evidence_gain,
        "evidence_quality": monitor_input.evidence_quality,
        "harm_risk": monitor_input.harm_risk,
        "task_quality": monitor_input.task_quality,
        "observable_peer_count": monitor_input.observable_peer_count,
    }


def _ensure_public_monitor_input(monitor_input: MonitorInput) -> None:
    if not isinstance(monitor_input, MonitorInput):
        raise TypeError("OnlineStateProvider must return MonitorInput")
    serialized = monitor_input.model_dump_json().lower()
    if any(token in serialized for token in ("hidden_probe", "hidden_profile", "valueprobe", "prompt")):
        raise ValueError("online monitor input contains prohibited private or prompt data")


def run_adaptive(
    run_config: RunConfig,
    scenario: ScenarioSpec,
    online_state_provider: OnlineStateProvider,
    target_candidate_provider: TargetCandidateProvider,
    monitor_config: MonitorConfig,
    policy_config: PolicyConfig,
    provider: LocalProvider | None = None,
) -> RunRecord:
    """Run the explicit local adaptive control loop."""
    return AdaptiveProtocolRunner(
        run_config,
        scenario=scenario,
        online_state_provider=online_state_provider,
        target_candidate_provider=target_candidate_provider,
        monitor_config=monitor_config,
        policy_config=policy_config,
        provider=provider,
    ).run()


__all__ = ["AdaptiveProtocolRunner", "run_adaptive"]
