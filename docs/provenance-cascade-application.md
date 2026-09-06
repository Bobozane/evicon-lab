# Provenance-Cascade Proposal Application (24E.1)

Stage 24E.1 is an opt-in, next-round-only application boundary. It converts a
validated `CascadeInterventionProposal` into an immutable schedule and a
structured directive sidecar. It does not call a model, create a prompt, alter
an exposure ledger, mutate a historical snapshot, or execute an intervention.

## Contracts

`ScheduledCascadeIntervention` binds the proposal ID, scenario, target, claim,
creation round, effective round, action, visible audit IDs, source snapshot
SHA-256, source public-view SHA-256, status, and stable reason codes. A non-
`abstain` action is valid only when `effective_round_id = created_round_id + 1`.
`abstain` returns `no_effect`: no schedule and no audit event.

`CascadeInterventionAuditEvent` records schedule creation, application,
rejection, or cancellation. It contains only coordinates, hashes, status, and
stable codes. It never stores prompt text, messages, evaluator truth, API
secrets, provider metadata, or private fixture paths.

`CascadeApplicationLedger` is immutable. Each scheduling/application operation
returns a new ledger; callers may persist its append-only schedules and audit
records without changing the historical cascade ledger. Duplicate proposals,
conflicting target/claim/effective-round schedules, invalid hashes, invalid
coordinates, and invalid public references are rejected.

## Fixed action semantics

- `request_independent_source` creates a `verification_request` directive. It
  carries only the claim, already visible content/evidence IDs, visible same-
  root repetition count, and the request type. It does not add a source or
  assert truth.
- `request_evidence_based_reasoning` creates a `reasoning_request` directive
  limited to evidence/content already listed by the proposal.
- `evidence_first_exposure` creates `priority_evidence` only for evidence IDs
  already authorized and visible in the proposal. It cannot introduce new
  evidence, hide a correction, remove a consensus message, or change public
  verification status.
- `abstain` is zero effect.

These are structured system-context transformations, not natural-language
mediator instructions and not evidence. A future actor may read a directive as
an instruction to request or organize public reasoning, but may not treat it as
an additional source, claim, or truth label.

## Timing and runner isolation

The current-round snapshot remains unchanged. A proposal made from round `r`
can only be scheduled for `r + 1`. `ControlledCascadeProtocolRunner` is an
opt-in adapter that builds the ordinary base snapshot first and attaches
validated directives in a separate `ControlledRoundContext`. It does not alter
`CascadeProtocolRunner`'s default `no_intervention` behavior.

The next-round public view can have a different hash from the source view. The
schedule retains the source hash for audit binding, while application checks
the next-round scenario, target, round, and visible IDs. No proposal can use a
provenance root relation to discover unexposed ancestor content.

## Replay

`CascadeApplicationReplayValidator` is a sidecar validator, separate from the
historical `ReplayValidator` and `CascadeReplayValidator`. It checks:

- schedule creation precedes application;
- effective round is exactly the next round;
- at most one application exists;
- status and audit action/coordinates/hashes agree;
- applied directives have a transformation hash and come from an applied
  schedule;
- directive content/evidence IDs are subsets of the proposal schedule;
- rejected or cancelled schedules produce no directive.

It reports counts and stable error codes only. It never reads evaluator-private
fixtures or attempts to assess truth.

The local smoke command is:

```bash
uv run python -m evicon.cascade_application_smoke
```

It uses the four synthetic public fixtures, schedules the provenance-aware
false-majority request for the next round, and leaves supported correction,
independent consensus, and unresolved disagreement at `abstain`. It writes no
`results/` files and performs no network calls.

## 24F handoff

A future offline evaluator can consume the safe run record interface consisting
of: source proposal/view/snapshot hashes, immutable schedule and audit events,
next-round directive hashes, application status, and the existing public
exposure/replay report. It must not consume prompt text, model output, private
truth, or treat a directive as a measured outcome.
