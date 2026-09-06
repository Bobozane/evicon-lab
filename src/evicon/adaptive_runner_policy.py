"""Public adaptive runner with JSON-stable policy reason handling."""

from __future__ import annotations

from . import adaptive_runner_core as _core
from .conformity_monitor import ConformityMonitor
from .events import EventType, JsonlEventLog
from .executor import InterventionExecutor
from .intervention_application import AppliedIntervention
from .models.protocols import InterventionAction
from .adaptive_control import (
    BudgetEntryKind,
    ControllerHistoryStatus,
    OnlineStateProvider,
    TargetCandidateProvider,
)
from .fake_llm import LocalProvider
from .models import DialogueState, RunConfig, RunRecord, ScenarioSpec
from .policy import PolicyConfig
from .conformity_monitor import MonitorConfig


class AdaptiveProtocolRunner(_core.AdaptiveProtocolRunner):
    """FakeLLM-only adaptive loop with string-stable audit reason codes."""

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
            state, self.scenario, observation_config, round_id
        )
        _core._ensure_public_monitor_input(monitor_input)
        correlation_id = f"online-{self.config.run_id}-r{round_id}"
        logger.append(
            EventType.ONLINE_STATE_OBSERVED,
            round_id=round_id,
            payload={
                "correlation_id": correlation_id,
                "budget_before": self.ledger.remaining_budget,
                "budget_after": self.ledger.remaining_budget,
                "cooldown_until": self.controller_state.cooldown_until,
                **_core._monitor_input_payload(monitor_input),
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
        candidates = list(
            self.target_candidate_provider.build_candidates(
                state, monitor_result, self.scenario, round_id
            )
        )
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
        decision = self._policy_input(monitor_result, candidates)
        from .policy import InterventionPolicy

        policy_decision = InterventionPolicy.decide(decision, self.policy_config)
        logger.append(
            EventType.POLICY_DECIDED,
            round_id=round_id,
            payload={
                "correlation_id": correlation_id,
                "action": policy_decision.action.value,
                "target_agent_ids": policy_decision.target_agent_ids,
                "risk_score": policy_decision.risk_score,
                "estimated_cost": policy_decision.estimated_cost,
                "reason_codes": [reason.value for reason in policy_decision.reason_codes],
                "budget_before": self.ledger.remaining_budget,
                "budget_after": self.ledger.remaining_budget,
                "cooldown_until": self.controller_state.cooldown_until,
                "warnings": [],
                "errors": [],
            },
        )
        if policy_decision.action is InterventionAction.NO_OP:
            return
        if round_id + 1 >= self.scenario.max_rounds:
            self._append_history(
                plan_id=f"cancelled-r{round_id}",
                action=policy_decision.action,
                round_id=round_id,
                status=ControllerHistoryStatus.CANCELLED,
                cost=policy_decision.estimated_cost,
                reason_codes=["no_future_round"],
            )
            return
        plan = InterventionExecutor.plan(policy_decision, self._execution_context(state))
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
            reason_codes=list(plan.reason_codes),
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


def run_adaptive(
    run_config: RunConfig,
    scenario: ScenarioSpec,
    online_state_provider: OnlineStateProvider,
    target_candidate_provider: TargetCandidateProvider,
    monitor_config: MonitorConfig,
    policy_config: PolicyConfig,
    provider: LocalProvider | None = None,
) -> RunRecord:
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
