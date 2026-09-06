# H-G1.2 Observability Amendment

H-G1.2 is a new development-only pilot design. It does not alter the completed
H-G11 batch, its ledger, checkpoints, receipt, or evaluator-only analysis.
H-G11 remains `non_identifiable_pilot`; the two batches must never be pooled.

## Problem and Scope

The H-G11 public trajectory passed replay, but real Agents were predominantly
`uncertain` in the round-zero false-majority and correction scenarios. The
pre-registered eligibility chain therefore had no initial false-endorsement or
correction-transition denominator. Separately, the second same-root message was
delivered only when a round-zero Agent elected to share it, so the intended
source-aware decision opportunity was not guaranteed to be observable.

This amendment changes only declared public exposure timing and the round-two
measurement target for the correction scenario. It does not change evaluator
truth, the nine preregistered metrics, controller access limits, model settings,
the JSON schema, topology, condition set, or policy thresholds.

## New Timing

- Round 0: each Agent receives the pre-session public initial claim and returns
  a provisional public stance.
- After round 0: a declared second public content item is delivered to the five
  Agents that previously saw the first item. It shares the same root in the
  false-majority and correction scenarios, but has a distinct root in the
  independent-consensus scenario.
- Round 1: proposals may be created from the resulting view; any non-abstain
  proposal can affect only round 2.
- After round 1: each scenario's existing declared correction/evidence exposure
  is delivered according to its public scenario file.
- Round 2: the correction scenario measures its correction claim for all Agents;
  the other scenarios retain their principal public claim.

The false-majority schedule gives `network-agent-02` two visible public
same-root messages at round 1. They resolve to one root. The independent-
consensus counterpart gives that Agent two visible messages resolving to two
roots. This is the intended information-access difference: source-blind policy
receives matching visible message counts, while provenance-aware policy may use
only the visible root relation.

The amendment does not force a real model to endorse an unverified initial
claim. The FakeProvider smoke establishes timing, eligibility *opportunity*,
and replay mechanics only. A future real execution remains conditional on a new
human approval and a new one-shot compatibility check; it cannot claim effects
unless the observed denominators are nonzero.

## Safety

All content remains synthetic and public. Evaluator truth and manual source-
independence labels remain evaluator-only. Prompts, full model responses,
private truth, credentials, and provider metadata are excluded from receipts,
ledgers, and smoke output. H-G1.2 is not a paper result and implies no causal
conclusion.
