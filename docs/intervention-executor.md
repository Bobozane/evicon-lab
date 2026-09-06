# Intervention Executor

`InterventionExecutor` is a pure plan compiler. `InterventionPolicy` decides
whether a structured action is eligible and which agents are targeted;
`InterventionExecutor` checks the decision against an `ExecutionContext` and
returns an `InterventionPlan`. Neither component executes an intervention.

## Contracts

`ExecutionContext` contains only current observable identifiers: known agents,
visible evidence with introduction rounds, candidate peer turns with their
rounds, budget, target limit, intervention history, last action, and metadata.
Peer turns must be strictly earlier than the current round. Evidence declared
visible must have been introduced by the current round. Unknown fields and
private prompt, hidden-probe, hidden-profile, value-profile, or dialogue-state
payloads are rejected.

`InterventionPlan` records a deterministic `plan_id`, source run/scenario/round,
action, target agents, mediator event type, requested evidence identifiers,
visibility overrides, speaking priority, future turn-order override, response
and minority-report flags, cost, policy reason codes and version, validity, and
structured validation errors. The ID is a deterministic hash of the policy
decision's auditable fields, so it can be traced back without storing prompts.

Invalid inputs return an inert `valid=false` plan with `mediator_event_type`
`invalid` and stable error codes. No partial visibility, order, or report
effect is created.

## Action Semantics

| Action | Structured plan effect |
| --- | --- |
| `no_op` | `none`, no targets, zero cost, no overrides. |
| `request_evidence` | Requests a target response but does not create or assert an EvidenceCard. |
| `blind_evidence_reflection` | Next turn retains each target's own history and visible evidence, with peer turns explicitly empty. |
| `solicit_dissent` | Gives selected targets speaking priority; it does not require a contrary position. |
| `adaptive_exposure` | Records only existing, visible, earlier peer-turn IDs for selected targets. |
| `minority_report` | Records targets to retain in the final report without changing their positions. |
| `restructure` | Records a future target-first order using only known agents; historical turns remain unchanged. |

The first four visibility-related actions only describe prospective next-turn
constraints. `minority_report` affects only a future final-report requirement.
No plan changes a current dialogue turn.

## Validation And Boundaries

`validate_intervention_plan(plan, context)` is a pure check that returns
`valid`, stable `errors`, and `warnings`. It validates context identity, agent
existence and limits, budget, visible evidence, earlier peer turns, and each
action's structural semantics. Cooldown and restructure eligibility remain
Policy responsibilities; the Executor never bypasses them or selects an
alternative action.

This phase deliberately generates no natural-language prompt, invokes no model
or network, writes no event log, and does not attach to `ProtocolRunner` or
modify `DialogueState`. A future runtime integration must consume only valid
plans, apply prospective visibility/order changes at a round boundary, record
the resulting action separately, and preserve existing event-log replay
semantics.

## Local Smoke

```bash
uv run python -m evicon.executor_smoke
```

The command prints JSON for all seven actions plus unknown-target,
invisible-evidence, and over-budget rejections. It creates no files.
