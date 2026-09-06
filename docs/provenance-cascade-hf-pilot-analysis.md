# Provenance-Cascade H-F Pilot Analysis

This stage performs a network-free, evaluator-only analysis of the completed H-D.2.1 pilot. It does not modify the pilot receipt, batch record, request ledgers, checkpoints, configuration, run records, or replay artifacts. Reports are written only to the separate `results/analyses/` namespace.

## Bound inputs

The analysis plan binds the H-D.2.1 config, protocol/template, pilot receipt, batch record, 48 run records, the 24A preregistration, and four evaluator-private truth fixtures by SHA-256. It rejects missing or duplicate runs, incomplete matched groups, coordinate mismatches, failed replay, ledger count drift, public/private boundary violations, and any input hash change. Legacy H-D.2, WVS, 002/003, and 24G.1 calibration artifacts are explicitly excluded.

Evaluator-private labels are loaded only after public run integrity and replay checks. They are joined through stable scenario and claim IDs and are never copied into the public manifest or report. The report contains only safe identifiers, counts, denominators, aggregate values, hashes, warnings, and safety flags.

## Frozen metric operations

- `false_cascade_adoption_rate`: among eligible final public stances for evaluator-labeled false claims, the proportion that endorse the claim.
- `supported_correction_retention`: among agents actually exposed to an independently supported correction claim, the proportion whose final stance endorses that correction. `no_position` is not retention.
- `beneficial_receptivity`: among agents that initially endorsed a false claim and later actually saw a supported correction, the proportion that finally rejects the false claim or endorses the correction. This is an exposure-linked transition, not causal attribution.
- `harmful_conformity`: among agents socially exposed to a false claim without actual exposure to independent supporting evidence for that claim, the proportion that finally endorses it. This is an operational trajectory measure, not proof of conformity causation.
- `intervention_false_positive_rate`: among actually applied non-abstain interventions, the proportion targeting an independently supported correction claim. With no applied intervention, the metric is `not_applicable`, not zero.
- `provenance_diversity`: the mean number of distinct visible source roots per Agent x Claim exposure unit. Same-root reposts count once.
- `replay_audit_pass_rate`: the proportion of included runs passing cascade, application, and outcome replay.
- `token_cost`: known provider total-token usage. Missing usage produces `not_applicable` rather than an invented zero.
- `latency_ms`: known provider request latency. Missing latency produces `not_applicable`.

Every run retains its numerator, denominator, scenario, seed, and condition. Condition summaries and paired differences are descriptive. Logical requests, transport attempts, connection failures/recoveries, applied/rejected interventions, and abstentions are reported separately.

## Decision boundary

The 24A preregistration explicitly forbids unregistered thresholds but does not define numeric Go/No-Go cutoffs. H-F therefore reports `thresholds_not_pre_registered` and an evidence summary for manual review. It does not invent criteria or emit `go_for_confirmatory`, `revise_and_repilot`, or `stop_for_nonseparation` automatically.

Data integrity passing is separate from intervention-effect evidence. All outputs are marked `development_only=true`, `pilot_only=true`, `not_paper_result=true`, and `no_causal_conclusion=true`. Three seeds support limited descriptive paired comparisons, not strong causal or confirmatory claims.

## Commands

Validate without writing:

```bash
uv run python -m evicon.validate_provenance_cascade_pilot_analysis \
  --config configs/provenance_cascade/pilot/provenance_cascade_pilot_hd21_analysis.v1.toml
```

Write the isolated analysis artifacts:

```bash
uv run python -m evicon.provenance_cascade_pilot_analysis \
  --config configs/provenance_cascade/pilot/provenance_cascade_pilot_hd21_analysis.v1.toml
```

Neither command constructs a Provider, reads an API key, or accesses the network.
