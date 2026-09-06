# Budget Ledger

`BudgetLedger` is an immutable Pydantic model with:

- `initial_budget`
- `reserved_budget`
- `spent_budget`
- `remaining_budget`
- append-only structured `entries`

The invariant is:

```text
remaining_budget = initial_budget - reserved_budget - spent_budget
```

Creating a valid non-`no_op` plan reserves its estimated cost. A plan that is
actually applied moves that reservation to `spent_budget`; because the amount
was already unavailable, `remaining_budget` does not increase during that
transition. A rejected, cancelled, or failed pending plan releases its
reservation. `no_op` has no ledger entry.

`ControllerState` mirrors the current remaining, reserved, and spent values and
also records cooldown, intervention history, and scheduled plan IDs. Cooldown
starts only after an applied non-`no_op` plan. Plan creation alone cannot start
cooldown, and the initial `RunConfig.intervention_budget` is never modified.

Every transition records before/after numeric values in the JSONL audit log as
`budget_reserved`, `budget_spent`, or `budget_released`. Replay validation can
therefore distinguish planned, applied, rejected, and cancelled states without
assuming an intervention was successful.

This ledger tracks control-flow accounting, not token billing or an empirical
cost estimate. Those integrations remain future work.
