# Provenance-Cascade Controller Proposals

Stage 24E defines a pure, deterministic decision boundary over one agent's
current-round `ControllerPublicView`. It does not apply an action. It cannot
append an exposure event, alter a snapshot, edit the provenance graph, reorder
actors, spend a budget, or write a result.

## Public input boundary

The only runtime inputs are one current public `ControllerPublicView` and the
frozen local `provenance_cascade_controller.v1` policy configuration. Evaluator
truth records, private source-independence labels, global unexposed graph data,
future content, unexposed evidence, private fixture paths, prompts, and model
outputs are outside the interface.

Visible root relations are opaque audit summaries attached to nodes already in
the snapshot. They support same-root counting, but do not expose or authorize
an ancestor content ID, ancestor node, evidence card, or message text.

## Four conditions

- `no_intervention` returns an empty `abstain` proposal and invents no risk
  reason.
- `generic_dissent` sees current public claim status, exposed evidence, and
  visible repetition. It does not inspect root summaries. Its request is fixed
  and content-neutral.
- `source_blind_controller` uses the same non-provenance projection. Its code
  path neither reads nor derives root IDs, root counts, root relations, parent
  topology, or source-independence labels.
- `provenance_aware_controller` additionally uses root relationships for nodes
  actually visible in the view. It may distinguish repeated content from one
  root from multiple visible independent roots.

None of the conditions uses majority or minority membership as a trigger.

## Stable proposal actions

`CascadeAction` contains `abstain`, `request_independent_source`,
`request_evidence_based_reasoning`, and `evidence_first_exposure`. These are
proposal labels only. No prompt or free-text instruction is generated.

A `CascadeInterventionProposal` records a deterministic proposal ID, current
scenario/round/agent/claim coordinate, condition, action, stable public reason
codes, and only the visible content/evidence/root IDs actually used. It rejects
unknown fields and cannot carry ledger, snapshot, graph, runner, prompt,
private-label, or message mutation fields.

## Protective constraints

- A visible supported correction produces `abstain`; no hide, suppress,
  downgrade, remove, or retract action exists in the action enum.
- `independent_true_consensus` always produces `abstain`. Multiple visible
  roots are an audit reason, not an automatic dissent trigger.
- `unresolved_disagreement` defaults to `abstain` and makes no truth judgment.
  The frozen policy can only opt into requesting an additional independent
  source, never into declaring a winner.
- Source-blind and generic proposals cannot record provenance-only IDs or
  reason codes.
- All proposal audit coordinates must be subsets of the current view; cross
  scenario, future round, wrong target, and unexposed references are rejected.

## Proposal versus application

Stage 24E stops at `propose(view, config) -> CascadeInterventionProposal`.
A future 24E.1 application boundary would need the proposal, the exact snapshot
ID/hash it was derived from, an explicit next-round target, an allowed exposure
transformation, and a separate validator. That future layer must not treat the
proposal as permission to discover unexposed content and must not mutate the
round whose snapshot produced the proposal.

The local smoke command is:

```bash
uv run python -m evicon.cascade_controller_smoke
```

It prints only scenario/condition/action/reason codes and public coordinate
counts. It does not load evaluator-private fixtures, construct a provider,
access the network, or write `results/`.

Stage 24E.1 defines the separate schedule/application boundary in `provenance-cascade-application.md`; proposals remain non-executing until that later opt-in layer.
