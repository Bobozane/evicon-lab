# Conformity Identification Study v1.1

Version 1.1 is a development-only amendment to the offline v1 design. It does not modify v1 or any H-G, WVS, calibration, Pilot, ledger, receipt, or analysis artifact. It introduces a shared initial observation, freezes the analysis contrasts, and adds an independent source-structure comprehension gate. No real model has been called.

## Shared T0 branch design

Each `scenario x seed` matched group first records six Agent judgments under one `initial_private` checkpoint. That checkpoint is immutable and projected into all six condition branches. It is not requested again. Every continuation fingerprint binds the branch ID and shared-T0 observation hash, while branch identity remains absent from the Agent prompt.

There are 12 matched groups and 72 condition branches. Shared T0 requires 72 logical requests; four continuation observations for six Agents across 72 branches require 1,728. The total is 1,800 logical requests. At 512 reserved completion tokens each, the reservation is 921,600 completion tokens. This is neither a total-token bound nor a price commitment.

This structure replaces repeated sampling of the initial state with a literal common counterfactual starting point. The original v1 plan remains preserved as an earlier offline design.

## Frozen analysis

Ordinal judgments map to `-2, -1, 0, 1, 2`. Five paired contrasts are registered:

```text
natural_instability                  = self_reflection - private_baseline
text_repetition_effect              = source_free_repetition - self_reflection
social_source_increment             = same_root_social - source_free_repetition
independent_corroboration_increment = independent_roots - same_root_social
evidence_receptivity_increment      = verified_evidence - self_reflection
```

The pairing unit is `scenario x seed x agent`. Agents are aggregated before inference; they are not treated as independent replicates. Exact sign-flip analysis is planned on `scenario x seed` aggregates, accompanied by descriptive statistics, missingness audit, scenario/seed breakdowns, and leave-one-scenario-out results.

Conventional answer flip toward visible content is diagnostic only. The report must show where it disagrees with the factorized estimates. No significance or Go/No-Go threshold is introduced in this version.

## Source manipulation gate

The gate presents the same two messages and text under three public projections: no visible root, one shared root, and two distinct roots. It asks only for visible message count, visible source-root count, and the supplied public assignments. It never asks for truth, reliability, or evaluator annotations, and its responses are not joined to behavioral outcomes.

The offline FakeProvider passes 12 deterministic cases across the four development scenarios. A future real check remains separately approval-gated, has at most 12 logical requests, uses no behavioral Pilot material beyond the synthetic matched text, and cannot make an effectiveness claim.

## Remaining gates

Network execution remains blocked by exact-hash human approval, a real source-manipulation check, a protocol-stability probe, and a one-shot provider compatibility check. None of those gates authorizes the 1,800-request study. A later real study requires a separate runner, ledger/no-overwrite contract, budget confirmation, and explicit network authorization.

The executable sidecar contracts are documented in `provenance-cascade-conformity-identification-qualification-gates.md`. Design approval and per-command network authorization are deliberately separate. The compatibility gate is one request; the source manipulation gate is 12 requests; the stability gate is 30 requests. All default CLI paths are offline and create no output.
