# Provenance-Cascade Intervention Calibration

## Purpose

The earlier 24G deterministic runs replayed correctly but produced zero
`directive_applied_count`: their propagation schedules did not expose two
content nodes with the same root to one target before proposal generation.
This stage adds a separate, versioned, synthetic fixture to calibrate that
observability condition. It is **development-only**, **calibration-only**,
not a paper result, and not a causal experiment.

## Fixture

`configs/provenance_cascade/calibration/` contains a public-only manifest, a
separate controller policy, four scenario declarations, and four synthetic
provenance graphs. The `false_majority` graph gives `network-agent-01` two
round-0 content nodes that resolve to one `source_root_id`; therefore its
round-1 snapshot has repetition count 2 and root count 1. The policy proposal
is created after round 1 and the structured directive is applied only to the
round-2 context. No evaluator truth is present in the fixture.

The other scenarios preserve their safety behavior:

- `true_minority_correction` exposes a supported correction with an independent
  public root; it is never hidden, suppressed, downgraded, or removed.
- `independent_true_consensus` exposes two independent roots and defaults to
  `abstain`.
- `unresolved_disagreement` exposes conflicting public material and defaults
  to `abstain` without a truth decision.

All scenarios use six deterministic agents, three rounds, the fixed
`ring_6_bidirectional` topology, and seeds `20260901`, `20260902`, and
`20260903`. Conditions are run in the preregistered order.

## Safety and interpretation

The calibration runner reuses the existing `CascadeOutcomeRunner`, controller,
application ledger, and sidecar replay validator. It does not alter their
default behavior and writes no `results/` files. A directive is a structured
next-round request, not a fact, evidence, source root, or truth label.

The calibration only demonstrates that the trigger can be observed and that
proposal-to-schedule-to-application timing is auditable. It must not be used
for confirmatory paper analysis or interpreted as evidence of method
effectiveness. The 24A preregistration is not silently amended; any future use
of this fixture in a formal study requires an explicit human-approved
amendment.

Before a 24H real pilot, still required are an independent preflight, explicit
network and request/token budget gates, run-directory and resume protection,
human approval of the fixture and policy, and a separate analysis plan.
