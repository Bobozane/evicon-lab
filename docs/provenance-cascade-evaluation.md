# Provenance-Cascade Evaluation Boundary

## Scope

Stage 24F freezes a public outcome contract and an evaluator-only offline metric layer. It does not change `CascadeProtocolRunner`, consume directives, call a provider, run a batch, or claim intervention effects. The smoke fixtures are deterministic synthetic data only.

## Public outcome ledger

`PublicClaimOutcome` records `scenario_id`, `agent_id`, `claim_id`, `round_id`, one of `endorses`, `rejects`, `uncertain`, or `no_position`, and optional IDs for content and evidence that were actually visible. It contains no rationale text, ground truth, source-independence annotation, prompt, or private path. `CascadeOutcomeLedger` is immutable and append-only by returning a new ledger; duplicate coordinates and time reversal are rejected. Before evaluation, every reference is checked against the public exposure ledger and graph.

A stance is an observable behavior, not a fact judgment. Evaluator truth is joined only after public validation through stable claim/root IDs.

## Fixed metric definitions

The nine preregistered metrics retain their exact names and denominators:

1. `false_cascade_adoption_rate`: final `endorses` among eligible agents for evaluator-labelled false/refuted claims.
2. `supported_correction_retention`: final `endorses` among agents that were actually exposed the supported, independently sourced correction content.
3. `beneficial_receptivity`: among agents initially endorsing an evaluator-labelled false claim, the proportion that were later exposed supported correction content and ended rejecting the error (or endorsing the correction). This is an exposure-linked behavioral transition, not a causal estimate.
4. `harmful_conformity`: final endorsement of a false/refuted claim among agents exposed social propagation of that claim but not independently supported evidence. This is an operational signal, not proof of psychological or causal conformity.
5. `intervention_false_positive_rate`: applied non-`abstain` interventions targeting evaluator-labelled independent consensus or supported correction claims, divided by all applied non-`abstain` interventions. With no applied intervention the result is `not_applicable`, never zero.
6. `provenance_diversity`: mean number of distinct visible `source_root_id` values per Agent x Claim exposure unit; same-root reposts count once.
7. `replay_audit_pass_rate`: complete runs passing cascade replay (and application replay when supplied) divided by runs included.
8. `token_cost`: sum of known `total_tokens` from a real provider ledger. Deterministic/FakeLLM or missing usage is `not_applicable`.
9. `latency_ms`: mean of known provider latencies. Without real calls it is `not_applicable`.

Missing observations are `not_applicable`; the evaluator never substitutes zero or guesses a denominator.

## Data boundary

Public inputs are a validated `CascadeRunRecord`, graph, exposure ledger, outcome ledger and (optionally) application audit. `EvaluatorTruthFixture` is loaded in a separate offline evaluator call and is never nested into a public model, run record, proposal, directive, prompt, event, hash summary, or CLI output. Reports contain only metric values/status, counts, safe hashes, warnings, replay status, and mandatory flags: `development_only`, `not_paper_result`, and `no_causal_conclusion`.

No evaluator report ranks conditions, performs significance tests, or claims a method works. No matched pilot batch is created in this stage. `CascadeMatchedGroup` only validates a future four-condition coordinate contract (scenario, seed, Agent order, rounds, configuration and outcome-contract hashes).

## Future interface

A later stage may have an Agent consume a validated next-round structured directive and emit a public `CascadeOutcomeLedger` entry. That integration must keep the directive separate from evidence and source roots, validate exposure/replay first, and keep evaluator truth offline.
