# Source-Behavior Qualification v2

This is an isolated, development-only protocol gate for the future Conformity
Identification study. It supersedes neither the earlier source-root
comprehension gate nor any H-G, WVS, Pilot, ledger, receipt, or analysis
artifact.

For each of four synthetic scenarios, the same two public message summaries
are presented under three public projections: source-free, a single shared
root, and two distinct roots. The gate stores only a safe per-case semantic
fact set: case/scenario/projection IDs, a valid parser status, public adoption
and sharing decision categories, used-content count, and visible-root count.
It stores neither prompts nor raw replies.

This design makes a later behavioral measurement auditable without exposing
private truth labels. It does **not** determine whether a model changes its
behavior under a source projection: the offline FakeProvider deliberately uses
the same withhold/do-not-share response in every case. A later behavior run is
still capped at 12 logical requests and remains separately unauthorized. It
would be a qualification gate, not a causal result or a paper result.

The companion approval was accepted on 2026-08-28. It binds the current
configuration and protocol hashes but does not authorize network use. The
offline approval CLI therefore reports only
`provider_compatibility_not_requested`. The separate one-shot compatibility
gate is documented in
`provenance-cascade-conformity-source-behavior-compatibility-v2.md`; passing it
would not authorize a behavior study or establish a source effect.

That compatibility gate has now completed under technical amendment v2.1.
The independent execution sidecar described in
`provenance-cascade-conformity-source-behavior-execution-v2.md` was accepted on
2026-08-28. The separately authorized 12-case run then completed with 12 safe
case audits and no parser-invalid response. Its receipt and audit are now
frozen inputs to the post-collection, outcome-blind descriptive analysis lock
documented in
`provenance-cascade-conformity-source-behavior-analysis-lock-v2.md`. Completion
of the qualification is not evidence of a source effect or a causal result.

The subsequent aggregate-only descriptive check found constant adoption and
sharing across all 12 cases. A read-only diagnosis established that v2 omitted
the scenario-specific decision task, correction, evidence, and conflict
material from the behavior request. V2 is therefore frozen as a successfully
executed but behaviorally non-identifiable development qualification. It must
not be rerun or interpreted as evidence of no provenance effect. The isolated
repair is documented in
`provenance-cascade-conformity-source-behavior-qualification-v3.md`.
