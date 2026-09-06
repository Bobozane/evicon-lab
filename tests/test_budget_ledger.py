"""Pure immutable budget accounting coverage."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evicon.adaptive_control import BudgetEntryKind, BudgetLedger


def test_reserve_spend_and_release_preserve_the_ledger_invariant() -> None:
    initial = BudgetLedger.initial(1.0)
    reserved = initial.reserve(0.3, plan_id="plan-a", round_id=0)
    spent = reserved.spend(0.3, plan_id="plan-a", round_id=1)
    second_reserved = spent.reserve(0.2, plan_id="plan-b", round_id=2)
    released = second_reserved.release(0.2, plan_id="plan-b", round_id=3)

    assert initial.remaining_budget == 1.0
    assert reserved.reserved_budget == pytest.approx(0.3)
    assert reserved.remaining_budget == pytest.approx(0.7)
    assert spent.spent_budget == pytest.approx(0.3)
    assert spent.remaining_budget == pytest.approx(0.7)
    assert released.reserved_budget == 0.0
    assert released.spent_budget == pytest.approx(0.3)
    assert released.remaining_budget == pytest.approx(0.7)
    assert [entry.kind for entry in released.entries] == [
        BudgetEntryKind.RESERVED,
        BudgetEntryKind.SPENT,
        BudgetEntryKind.RESERVED,
        BudgetEntryKind.RELEASED,
    ]


def test_no_op_amount_does_not_create_a_budget_transition() -> None:
    ledger = BudgetLedger.initial(0.5)

    assert ledger.reserve(0.0, plan_id="plan-no-op", round_id=0) == ledger


def test_ledger_rejects_negative_or_unavailable_transitions_and_serializes() -> None:
    ledger = BudgetLedger.initial(0.2)
    with pytest.raises(ValueError, match="exceeds remaining"):
        ledger.reserve(0.3, plan_id="plan-a", round_id=0)
    with pytest.raises(ValueError, match="amount"):
        ledger.reserve(-0.1, plan_id="plan-a", round_id=0)
    with pytest.raises(ValidationError, match="balance"):
        BudgetLedger(
            initial_budget=1.0,
            reserved_budget=0.1,
            spent_budget=0.1,
            remaining_budget=0.9,
            entries=[],
        )
    assert BudgetLedger.model_validate_json(ledger.model_dump_json()) == ledger
