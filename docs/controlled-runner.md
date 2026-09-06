# Controlled Runner

`ProtocolRunner` remains the baseline runner for the four visibility protocols.
`ControlledProtocolRunner` is an opt-in local-only runner that consumes an
explicit, prevalidated `InterventionSchedule`. With no schedule or an empty
schedule it delegates to the original runner path and emits no intervention
events.

## Application Boundary

Each schedule has at most one valid `InterventionPlan` per decision round. The
runner records `intervention_plan_seen` at that round, validates it against the
round-start `DialogueState`, then records either an applied or rejected result.
A valid plan affects only the next round: exposure snapshots are still built
once before any response in that next round, so changing response order never
creates same-round leakage. Existing `DialogueTurn` records are never edited.

`no_op` is observationally neutral. `request_evidence` creates a structured
next-round request marker without creating an EvidenceCard. Blind reflection
sets a target's next peer-turn list to empty while retaining its own history and
already visible evidence. Solicit dissent and restructure only alter a future
turn order. Adaptive exposure can use only logged peer turns from strictly
earlier rounds. Minority report creates only a final-report retention
requirement and never changes a message or value profile.

## Audit And Replay

Controlled logs add plan-seen, applied/rejected, visibility, order, evidence
request, and minority-report events. They contain safe identifiers and
structured fields, never prompts, hidden probe data, complete value profiles,
or fabricated evidence. Invalid plans are logged as rejected and fail the run;
they are not silently ignored. The schedule snapshot is stored next to the run
record as `intervention_schedule.json`.

Replay validation checks that every applied plan was first seen, is applied at
most once, cannot follow a rejection, and retains its schedule fields. It also
checks blind-reflection next-round peer visibility, adaptive chronological peer
references, and future turn-order effects.

## Boundaries

The runner sends only structured `InterventionMetadata` to `FakeLLM`; there is
no mediator prompt or real provider. It does not call `ConformityMonitor`,
`InterventionPolicy`, or `InterventionExecutor` automatically. Plans must be
created and validated before scheduling. This stage verifies action application
and replayability, not intervention quality, causal effects, or empirical
effectiveness.

## Local Smoke

```bash
uv run python -m evicon.controlled_smoke
```

The smoke schedule contains `no_op`, blind reflection, solicit dissent,
adaptive exposure, and restructure. It prints per-round visible peer turns,
turn order, intervention counts, output directory, and replay status.
