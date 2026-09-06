"""Observable providers and immutable budget/controller contracts for adaptive runs."""

from __future__ import annotations

import copy
import math
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from .conformity_monitor import MonitorInput, MonitorResult
from .intervention_executor import TargetCandidate
from .models import DialogueState, RunConfig, ScenarioSpec
from .models._validation import Metadata, normalized_text
from .models.protocols import InterventionAction


class OnlineStateProvider(Protocol):
    """Read-only public-state observer for one completed round."""

    def observe(
        self,
        dialogue_state: DialogueState,
        scenario: ScenarioSpec,
        run_config: RunConfig,
        round_id: int,
    ) -> MonitorInput:
        """Return validated online-observable monitor input."""


class TargetCandidateProvider(Protocol):
    """Read-only public-state candidate builder."""

    def build_candidates(
        self,
        dialogue_state: DialogueState,
        monitor_result: MonitorResult,
        scenario: ScenarioSpec,
        round_id: int,
    ) -> list[TargetCandidate]:
        """Return deterministic structured target candidates."""


class FakeOnlineStateProvider:
    """Scripted local provider; it has no model, network, probe, or profile access."""

    def __init__(self, scripted_signals: dict[int, dict[str, JsonValue]] | None = None) -> None:
        self._scripted_signals = copy.deepcopy(scripted_signals or {})

    def observe(
        self,
        dialogue_state: DialogueState,
        scenario: ScenarioSpec,
        run_config: RunConfig,
        round_id: int,
    ) -> MonitorInput:
        if dialogue_state.run_id != run_config.run_id or dialogue_state.scenario_id != scenario.scenario_id:
            raise ValueError("online observation identifiers must match the public state")
        values: dict[str, object] = dict(self._scripted_signals.get(round_id, {}))
        current_diversity = float(values.pop("current_diversity", 0.6))
        current_coverage = float(values.pop("current_coverage", 0.8))
        values.setdefault("previous_diversity", current_diversity)
        values.setdefault("previous_coverage", current_coverage)
        values.setdefault("minority_loss", 0.0)
        values.setdefault("evidence_gain", 0.0)
        values.setdefault("evidence_quality", 0.0)
        values.setdefault("harm_risk", 0.0)
        values.setdefault("task_quality", 0.8)
        values.setdefault("observable_peer_count", len(dialogue_state.turns))
        values.setdefault("metadata", {"source": "fake_online_state_provider"})
        values.update(
            {
                "run_id": run_config.run_id,
                "scenario_id": scenario.scenario_id,
                "round_id": round_id,
                "current_diversity": current_diversity,
                "current_coverage": current_coverage,
                "remaining_budget": run_config.intervention_budget,
            }
        )
        return MonitorInput.model_validate(values)


class FakeTargetCandidateProvider:
    """Scripted public-agent candidates for tests and smoke runs."""

    def __init__(self, scripted_candidates: dict[int, list[TargetCandidate]] | None = None) -> None:
        self._scripted_candidates = copy.deepcopy(scripted_candidates or {})

    def build_candidates(
        self,
        dialogue_state: DialogueState,
        monitor_result: MonitorResult,
        scenario: ScenarioSpec,
        round_id: int,
    ) -> list[TargetCandidate]:
        if dialogue_state.scenario_id != scenario.scenario_id:
            raise ValueError("candidate scenario_id must match dialogue state")
        scripted = self._scripted_candidates.get(round_id)
        if scripted is not None:
            return [candidate.model_copy() for candidate in scripted]
        return [
            TargetCandidate(
                agent_id=agent.agent_id,
                marginal_coverage_gain=0.5,
                minority_representation_score=0.5,
                safety_risk=0.1,
                estimated_cost=0.1,
                eligible=True,
                reason_codes=["public_agent_fixture"],
            )
            for agent in scenario.agents
        ]


class BudgetEntryKind(str, Enum):
    RESERVED = "reserved"
    SPENT = "spent"
    RELEASED = "released"


class BudgetEntry(BaseModel):
    """One immutable budget transition suitable for JSONL auditing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: BudgetEntryKind
    plan_id: str
    round_id: int = Field(ge=0)
    amount: float = Field(ge=0.0)
    remaining_before: float = Field(ge=0.0)
    remaining_after: float = Field(ge=0.0)
    reserved_before: float = Field(ge=0.0)
    reserved_after: float = Field(ge=0.0)
    spent_before: float = Field(ge=0.0)
    spent_after: float = Field(ge=0.0)

    @field_validator("plan_id")
    @classmethod
    def validate_plan_id(cls, value: str) -> str:
        return normalized_text(value, "plan_id")


class BudgetLedger(BaseModel):
    """Immutable budget ledger; remaining equals initial minus reserved and spent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    initial_budget: float = Field(ge=0.0)
    reserved_budget: float = Field(ge=0.0)
    spent_budget: float = Field(ge=0.0)
    remaining_budget: float = Field(ge=0.0)
    entries: list[BudgetEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_balance(self) -> "BudgetLedger":
        expected = self.initial_budget - self.reserved_budget - self.spent_budget
        if expected < -1e-9 or abs(expected - self.remaining_budget) > 1e-9:
            raise ValueError("budget ledger balance is inconsistent")
        return self

    @classmethod
    def initial(cls, amount: float) -> "BudgetLedger":
        return cls(
            initial_budget=amount,
            reserved_budget=0.0,
            spent_budget=0.0,
            remaining_budget=amount,
            entries=[],
        )

    def reserve(self, amount: float, *, plan_id: str, round_id: int) -> "BudgetLedger":
        _validate_amount(amount)
        if amount == 0.0:
            return self
        if amount > self.remaining_budget + 1e-9:
            raise ValueError("budget reservation exceeds remaining_budget")
        entry = self._entry(
            BudgetEntryKind.RESERVED,
            amount,
            plan_id,
            round_id,
            remaining_after=self.remaining_budget - amount,
            reserved_after=self.reserved_budget + amount,
            spent_after=self.spent_budget,
        )
        return self.model_copy(
            update={
                "reserved_budget": self.reserved_budget + amount,
                "remaining_budget": self.remaining_budget - amount,
                "entries": [*self.entries, entry],
            }
        )

    def spend(self, amount: float, *, plan_id: str, round_id: int) -> "BudgetLedger":
        _validate_amount(amount)
        if amount > self.reserved_budget + 1e-9:
            raise ValueError("budget spend exceeds reserved_budget")
        entry = self._entry(
            BudgetEntryKind.SPENT,
            amount,
            plan_id,
            round_id,
            remaining_after=self.remaining_budget,
            reserved_after=self.reserved_budget - amount,
            spent_after=self.spent_budget + amount,
        )
        return self.model_copy(
            update={
                "reserved_budget": self.reserved_budget - amount,
                "spent_budget": self.spent_budget + amount,
                "entries": [*self.entries, entry],
            }
        )

    def release(self, amount: float, *, plan_id: str, round_id: int) -> "BudgetLedger":
        _validate_amount(amount)
        if amount > self.reserved_budget + 1e-9:
            raise ValueError("budget release exceeds reserved_budget")
        entry = self._entry(
            BudgetEntryKind.RELEASED,
            amount,
            plan_id,
            round_id,
            remaining_after=self.remaining_budget + amount,
            reserved_after=self.reserved_budget - amount,
            spent_after=self.spent_budget,
        )
        return self.model_copy(
            update={
                "reserved_budget": self.reserved_budget - amount,
                "remaining_budget": self.remaining_budget + amount,
                "entries": [*self.entries, entry],
            }
        )

    def _entry(
        self,
        kind: BudgetEntryKind,
        amount: float,
        plan_id: str,
        round_id: int,
        *,
        remaining_after: float,
        reserved_after: float,
        spent_after: float,
    ) -> BudgetEntry:
        return BudgetEntry(
            kind=kind,
            plan_id=plan_id,
            round_id=round_id,
            amount=amount,
            remaining_before=self.remaining_budget,
            remaining_after=remaining_after,
            reserved_before=self.reserved_budget,
            reserved_after=reserved_after,
            spent_before=self.spent_budget,
            spent_after=spent_after,
        )


class ControllerHistoryStatus(str, Enum):
    PLANNED = "planned"
    APPLIED = "applied"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class ControllerHistoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str
    action: InterventionAction
    round_id: int = Field(ge=0)
    status: ControllerHistoryStatus
    estimated_cost: float = Field(ge=0.0)
    reason_codes: list[str] = Field(default_factory=list)


class ControllerState(BaseModel):
    """Serializable adaptive controller state, separate from DialogueState."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    remaining_budget: float = Field(ge=0.0)
    reserved_budget: float = Field(ge=0.0)
    spent_budget: float = Field(ge=0.0)
    last_intervention_round: int | None = Field(default=None, ge=0)
    cooldown_until: int | None = Field(default=None, ge=0)
    intervention_history: list[ControllerHistoryEntry] = Field(default_factory=list)
    scheduled_plan_ids: list[str] = Field(default_factory=list)


def controller_state_from_ledger(
    ledger: BudgetLedger,
    *,
    last_intervention_round: int | None = None,
    cooldown_until: int | None = None,
    history: list[ControllerHistoryEntry] | None = None,
    scheduled_plan_ids: list[str] | None = None,
) -> ControllerState:
    return ControllerState(
        remaining_budget=ledger.remaining_budget,
        reserved_budget=ledger.reserved_budget,
        spent_budget=ledger.spent_budget,
        last_intervention_round=last_intervention_round,
        cooldown_until=cooldown_until,
        intervention_history=history or [],
        scheduled_plan_ids=scheduled_plan_ids or [],
    )


def _validate_amount(amount: float) -> None:
    if isinstance(amount, bool) or not math.isfinite(float(amount)) or float(amount) < 0.0:
        raise ValueError("budget amount must be a finite non-negative number")


__all__ = [
    "BudgetEntry",
    "BudgetEntryKind",
    "BudgetLedger",
    "ControllerHistoryEntry",
    "ControllerHistoryStatus",
    "ControllerState",
    "FakeOnlineStateProvider",
    "FakeTargetCandidateProvider",
    "OnlineStateProvider",
    "TargetCandidateProvider",
    "controller_state_from_ledger",
]
