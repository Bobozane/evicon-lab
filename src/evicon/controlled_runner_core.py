"""Opt-in, plan-driven deterministic protocol execution with FakeLLM only."""

from __future__ import annotations

import json

from .events import EventType, JsonlEventLog
from .exposure import ExposurePlan, ExposureSnapshot
from .fake_llm import FakeLLM, FakeLLMRequest, InterventionMetadata, LocalProvider
from .intervention_application import (
    AppliedIntervention,
    InterventionSchedule,
    apply_intervention_plan,
)
from .models import DialogueState, DialogueTurn, EvidenceExposure, RunRecord, RunStatus, ScenarioSpec
from .runner import ProtocolRunner


class ControlledProtocolRunner(ProtocolRunner):
    """Run an explicit schedule without auto-invoking monitor, policy, or executor."""

    def __init__(
        self,
        config: object,
        *,
        scenario: ScenarioSpec,
        intervention_schedule: InterventionSchedule | None = None,
        provider: LocalProvider | None = None,
    ) -> None:
        super().__init__(config, scenario=scenario, provider=provider or FakeLLM())  # type: ignore[arg-type]
        self.intervention_schedule = intervention_schedule
        self._validate_schedule()
        self.applied_interventions: list[AppliedIntervention] = []

    def _validate_schedule(self) -> None:
        schedule = self.intervention_schedule
        if schedule is None:
            return
        if schedule.run_id != self.config.run_id:
            raise ValueError("schedule.run_id must match config.run_id")
        if schedule.scenario_id != self.scenario.scenario_id:
            raise ValueError("schedule.scenario_id must match scenario.scenario_id")
        if any(plan.round_id >= self.config.max_rounds for plan in schedule.plans):
            raise ValueError("schedule plan round_id must be within configured rounds")

    def run(self) -> RunRecord:
        """Use the original path unless a non-empty explicit schedule was supplied."""
        if self.intervention_schedule is None or not self.intervention_schedule.plans:
            return super().run()
        return self._run_scheduled()

    def _run_scheduled(self) -> RunRecord:
        output_directory = self._prepare_output_directory()
        logger = JsonlEventLog(output_directory / "events.jsonl", self.config.run_id)
        turns: list[DialogueTurn] = []
        evidence_exposures: list[EvidenceExposure] = []
        state: DialogueState | None = None
        future_effects: dict[int, AppliedIntervention] = {}
        final_report_requirements: set[str] = set()
        applied_plan_ids: set[str] = set()

        logger.append(
            EventType.RUN_STARTED,
            round_id=0,
            payload={
                "scenario_id": self.config.scenario_id,
                "scenario_title": self.scenario.title,
                "protocol": self.config.protocol.value,
                "model_name": self.config.model_name,
                "agent_count": self.config.agent_count,
                "max_rounds": self.config.max_rounds,
                "seed": self.config.seed,
            },
        )
        try:
            for round_id in range(self.scenario.max_rounds):
                state = self._build_state(round_id, turns)
                logger.append(EventType.ROUND_STARTED, round_id=round_id, payload={})

                plan = self.intervention_schedule.plan_for_round(round_id)  # type: ignore[union-attr]
                if plan is not None:
                    self._log_plan_seen(logger, plan)
                    applied = apply_intervention_plan(state, plan, self.scenario)
                    if not applied.applied:
                        self._log_plan_rejected(logger, plan, applied)
                        raise ValueError(f"intervention plan rejected: {','.join(applied.errors)}")
                    if plan.plan_id in applied_plan_ids:
                        raise ValueError("intervention plan may not be applied more than once")
                    applied_plan_ids.add(plan.plan_id)
                    self.applied_interventions.append(applied)
                    self._log_plan_applied(logger, plan, applied)
                    self._log_effect_events(logger, plan, applied)
                    future_effects[round_id + 1] = applied
                    final_report_requirements.update(applied.final_report_requirements)

                active_effect = future_effects.get(round_id)
                snapshots = ExposurePlan(
                    self.config.protocol,
                    scenario_context=self.scenario.initial_context,
                ).create(state, round_id=round_id)
                snapshots = self._apply_next_round_effect(
                    state,
                    snapshots,
                    active_effect,
                )
                turn_order = self._turn_order(state, active_effect)
                responses: list[tuple[ExposureSnapshot, str]] = []
                for agent_id in turn_order:
                    snapshot = snapshots[agent_id]
                    evidence_exposures.extend(self._record_exposures(snapshot))
                    logger.append(
                        EventType.EXPOSURE_CREATED,
                        round_id=round_id,
                        payload=self._exposure_payload(snapshot),
                    )
                    metadata = self._intervention_metadata(
                        snapshot.agent_id,
                        active_effect,
                        final_report_requirements,
                    )
                    request = FakeLLMRequest(
                        agent_id=snapshot.agent_id,
                        round_id=snapshot.round_id,
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
                    try:
                        response = self.provider.complete(request)
                    except Exception as exc:
                        logger.append(
                            EventType.LLM_RESPONSE,
                            round_id=round_id,
                            payload={
                                "agent_id": snapshot.agent_id,
                                "error_type": type(exc).__name__,
                                "status": "failed",
                                "visible_peer_turn_ids": snapshot.visible_peer_turn_ids,
                                "visible_evidence_ids": snapshot.visible_evidence_ids,
                            },
                        )
                        raise
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

            self.last_state = state
            record = self._record(turns, evidence_exposures, RunStatus.COMPLETED)
            logger.append(
                EventType.RUN_COMPLETED,
                round_id=self.scenario.max_rounds - 1,
                payload={"turn_count": len(turns)},
            )
            self._write_record(output_directory, record)
            self._write_schedule(output_directory)
            return record
        except Exception as exc:
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
            self._write_schedule(output_directory)
            raise

    def _record(
        self,
        turns: list[DialogueTurn],
        evidence_exposures: list[EvidenceExposure],
        status: RunStatus,
        error_message: str | None = None,
    ) -> RunRecord:
        return RunRecord(
            config=self.config,
            scenario=self.scenario,
            turns=turns,
            value_profiles=[],
            evidence_exposures=evidence_exposures,
            intervention_decisions=[],
            status=status,
            error_message=error_message,
        )

    def _write_schedule(self, output_directory: object) -> None:
        path = output_directory / "intervention_schedule.json"  # type: ignore[operator]
        with path.open("x", encoding="utf-8") as handle:
            json.dump(
                self.intervention_schedule.model_dump(mode="json"),  # type: ignore[union-attr]
                handle,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

    def _apply_next_round_effect(
        self,
        state: DialogueState,
        snapshots: dict[str, ExposureSnapshot],
        active_effect: AppliedIntervention | None,
    ) -> dict[str, ExposureSnapshot]:
        if active_effect is None or not active_effect.next_visibility_overrides:
            return snapshots
        turns_by_id = {turn.turn_id: turn for turn in state.turns}
        for override in active_effect.next_visibility_overrides:
            own_ids = {turn.turn_id for turn in state.turns if turn.speaker_id == override.agent_id}
            visible_ids = own_ids | set(override.visible_peer_turn_ids)
            snapshots[override.agent_id] = snapshots[override.agent_id].model_copy(
                update={
                    "visible_history": [turn for turn in state.turns if turn.turn_id in visible_ids],
                    "visible_peer_turn_ids": list(override.visible_peer_turn_ids),
                    "visible_evidence_ids": list(override.visible_evidence_ids),
                }
            )
            if any(turn_id not in turns_by_id for turn_id in override.visible_peer_turn_ids):
                raise ValueError("visibility override references an unavailable peer turn")
        return snapshots

    @staticmethod
    def _turn_order(state: DialogueState, active_effect: AppliedIntervention | None) -> list[str]:
        return (
            active_effect.next_turn_order
            if active_effect is not None and active_effect.next_turn_order
            else [agent.agent_id for agent in state.agents]
        )

    @staticmethod
    def _intervention_metadata(
        agent_id: str,
        active_effect: AppliedIntervention | None,
        final_report_requirements: set[str],
    ) -> InterventionMetadata | None:
        if active_effect is None and agent_id not in final_report_requirements:
            return None
        action = active_effect.action if active_effect is not None else None
        if action is None:
            return None
        is_target = agent_id in active_effect.affected_agent_ids
        receives_request = (
            agent_id in active_effect.evidence_request_agent_ids
            or (
                active_effect.action.value == "request_evidence"
                and not active_effect.evidence_request_agent_ids
            )
        )
        is_blind = any(
            override.agent_id == agent_id for override in active_effect.next_visibility_overrides
        ) and active_effect.action.value == "blind_evidence_reflection"
        priority = (
            agent_id in active_effect.next_turn_order
            and active_effect.action.value == "solicit_dissent"
        )
        return InterventionMetadata(
            current_action=action,
            target_agent=is_target,
            evidence_request=receives_request,
            blind_reflection=is_blind,
            speaking_priority=priority,
            final_report_requirement=agent_id in final_report_requirements,
        )

    def _log_plan_seen(self, logger: JsonlEventLog, plan: object) -> None:
        logger.append(
            EventType.INTERVENTION_PLAN_SEEN,
            round_id=plan.round_id,  # type: ignore[attr-defined]
            payload=self._plan_payload(plan, applied=False),
        )

    def _log_plan_applied(
        self,
        logger: JsonlEventLog,
        plan: object,
        applied: AppliedIntervention,
    ) -> None:
        logger.append(
            EventType.INTERVENTION_PLAN_APPLIED,
            round_id=plan.round_id,  # type: ignore[attr-defined]
            payload=self._plan_payload(plan, applied=True, result=applied),
        )

    def _log_plan_rejected(
        self,
        logger: JsonlEventLog,
        plan: object,
        applied: AppliedIntervention,
    ) -> None:
        logger.append(
            EventType.INTERVENTION_PLAN_REJECTED,
            round_id=plan.round_id,  # type: ignore[attr-defined]
            payload=self._plan_payload(plan, applied=False, result=applied),
        )

    def _log_effect_events(
        self,
        logger: JsonlEventLog,
        plan: object,
        applied: AppliedIntervention,
    ) -> None:
        base = self._plan_payload(plan, applied=True, result=applied)
        if applied.next_visibility_overrides:
            logger.append(EventType.VISIBILITY_OVERRIDE_APPLIED, round_id=plan.round_id, payload=base)
        if applied.next_turn_order:
            logger.append(EventType.TURN_ORDER_OVERRIDE_APPLIED, round_id=plan.round_id, payload=base)
        if applied.evidence_request_agent_ids or applied.requested_evidence_ids:
            logger.append(EventType.EVIDENCE_REQUEST_CREATED, round_id=plan.round_id, payload=base)
        if applied.final_report_requirements:
            logger.append(
                EventType.MINORITY_REPORT_REQUIREMENT_CREATED,
                round_id=plan.round_id,
                payload=base,
            )

    @staticmethod
    def _plan_payload(
        plan: object,
        *,
        applied: bool,
        result: AppliedIntervention | None = None,
    ) -> dict[str, object]:
        return {
            "scenario_id": plan.scenario_id,  # type: ignore[attr-defined]
            "plan_id": plan.plan_id,  # type: ignore[attr-defined]
            "action": plan.action.value,  # type: ignore[attr-defined]
            "target_agent_ids": plan.target_agent_ids,  # type: ignore[attr-defined]
            "applied": applied,
            "reason_codes": plan.reason_codes,  # type: ignore[attr-defined]
            "warnings": [] if result is None else result.warnings,
            "errors": [] if result is None else result.errors,
            "visibility_overrides": [
                item.model_dump(mode="json") for item in plan.visibility_overrides  # type: ignore[attr-defined]
            ],
            "turn_order_override": plan.turn_order_override,  # type: ignore[attr-defined]
            "requested_evidence_ids": plan.requested_evidence_ids,  # type: ignore[attr-defined]
            "final_report_requirements": (
                [] if result is None else result.final_report_requirements
            ),
        }


def run_controlled(
    run_config: object,
    scenario: ScenarioSpec,
    intervention_schedule: InterventionSchedule | None = None,
    provider: LocalProvider | None = None,
) -> RunRecord:
    """Convenience entry point for an explicit local controlled run."""
    return ControlledProtocolRunner(
        run_config,
        scenario=scenario,
        intervention_schedule=intervention_schedule,
        provider=provider,
    ).run()


__all__ = ["ControlledProtocolRunner", "InterventionSchedule", "run_controlled"]
