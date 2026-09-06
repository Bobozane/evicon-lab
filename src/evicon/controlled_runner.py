"""Public controlled runner with no-op-neutral metadata and complete audit events."""

from __future__ import annotations

from . import controlled_runner_core as _core
from .events import EventType, JsonlEventLog
from .fake_llm import InterventionMetadata, LocalProvider
from .intervention_application import AppliedIntervention, InterventionSchedule
from .models import InterventionAction, RunRecord, ScenarioSpec


class ControlledProtocolRunner(_core.ControlledProtocolRunner):
    """Opt-in plan execution that leaves no-op runs observationally unchanged."""

    @staticmethod
    def _intervention_metadata(
        agent_id: str,
        active_effect: AppliedIntervention | None,
        final_report_requirements: set[str],
    ) -> InterventionMetadata | None:
        if active_effect is not None and active_effect.action is InterventionAction.NO_OP:
            return None
        return _core.ControlledProtocolRunner._intervention_metadata(
            agent_id,
            active_effect,
            final_report_requirements,
        )

    def _log_effect_events(
        self,
        logger: JsonlEventLog,
        plan: object,
        applied: AppliedIntervention,
    ) -> None:
        base = self._plan_payload(plan, applied=True, result=applied)
        if applied.next_visibility_overrides:
            logger.append(EventType.VISIBILITY_OVERRIDE_APPLIED, round_id=plan.round_id, payload=base)  # type: ignore[attr-defined]
        if applied.next_turn_order:
            logger.append(EventType.TURN_ORDER_OVERRIDE_APPLIED, round_id=plan.round_id, payload=base)  # type: ignore[attr-defined]
        if applied.action is InterventionAction.REQUEST_EVIDENCE:
            logger.append(EventType.EVIDENCE_REQUEST_CREATED, round_id=plan.round_id, payload=base)  # type: ignore[attr-defined]
        if applied.final_report_requirements:
            logger.append(
                EventType.MINORITY_REPORT_REQUIREMENT_CREATED,
                round_id=plan.round_id,  # type: ignore[attr-defined]
                payload=base,
            )


def run_controlled(
    run_config: object,
    scenario: ScenarioSpec,
    intervention_schedule: InterventionSchedule | None = None,
    provider: LocalProvider | None = None,
) -> RunRecord:
    """Run a local controlled protocol without auto-invoking monitor or policy."""
    return ControlledProtocolRunner(
        run_config,
        scenario=scenario,
        intervention_schedule=intervention_schedule,
        provider=provider,
    ).run()


__all__ = ["ControlledProtocolRunner", "InterventionSchedule", "run_controlled"]
