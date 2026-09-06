# Protocol-Blind v2 Confirmatory Analysis

## Scope

This stage performs an offline, seed-level analysis of the locked English protocol-blind v2 confirmatory batch. It reads only the confirmatory batch receipt, completeness audit, initial/final offline evaluation report, batch manifest, and the twelve public `run_record.json` files. Run records are inspected only for safe completion and configuration metadata; turns, value profiles, and any other content fields are not consumed.

The analysis never reads prompts, Agent messages, WVS text, raw probe answers, private checkpoints, request-ledger provider metadata, or environment variables. It does not construct a Provider or access the network. The v1 condition-aware `002` run and v2 development calibration `003` are excluded by both plan contract and artifact validation.

## Analysis Unit And Inputs

The statistical unit is one matched group / seed, not an Agent, probe item, individual model reply, or event. The fixed seeds are `20260820`, `20260821`, and `20260822`; every seed must contain exactly one run for `independent`, `social_only`, `evidence_only`, and `evidence_social`.

Before any result is emitted, the gate requires a `confirmatory` receipt, 12 completed runs, 3 complete matched groups, and 12 `passed` replay entries. It verifies the plan, plan-lock, frozen ProbeSet, and test-scenario SHA-256 values against the receipt and batch binding. It also validates that every safe run record matches its explicit batch coordinate.

## Pre-Registered Metrics

Primary metrics are fixed to:

- `pairwise_diversity`
- `social_influence_loss`
- `profile_drift`

The report gives the per-seed, per-condition `pairwise_diversity` and `profile_drift` values, four-condition mean, sample standard deviation, and median. `social_influence_loss` is preserved as its explicit matched contrast, `evidence_only - evidence_social`; it is not re-labelled as a single-condition measure.

The fixed contrasts are `social_only - independent`, `evidence_only - independent`, `evidence_social - evidence_only`, and the registered social-influence-loss contrast. Each includes all three seed differences, mean difference, sample standard deviation, and direction.

With `n=3`, the inferential display is an exact two-sided paired sign-flip/randomization calculation over all eight sign assignments. Its p-value is recorded only as a finite-sample descriptive reference and is explicitly not treated as strong significance evidence. The analysis does not introduce new thresholds, assert proof, infer a human value, or claim a strict causal effect. In particular, `social_influence_loss` remains a matched counterfactual estimate rather than a strict causal estimand.

## Robustness

`robustness_report.json` contains leave-one-seed-out summaries for each registered contrast. Each leaves out one entire matched group and reports the remaining two-seed mean, sample standard deviation, and direction. It is deterministic and does not select a favorable subset.

The source evaluation warnings are retained unchanged as structured strings: no explicit minority dimensions, no holdout social-influence pair, and no numeric token count in events. This stage does not reinterpret those warnings or silently fill missing measurements.

## Commands

Validate without writing an analysis directory:

```bash
uv run python -m evicon.validate_wvs7_confirmatory_analysis \
  --config configs/studies/wvs7_english_protocol_blind_v2_confirmatory_analysis.toml
```

Write the deterministic, safe reports after validation:

```bash
uv run python -m evicon.validate_wvs7_confirmatory_analysis \
  --config configs/studies/wvs7_english_protocol_blind_v2_confirmatory_analysis.toml \
  --write-reports
```

The writer produces `analysis_manifest.json`, `analysis_report.json`, and `robustness_report.json` beneath the configured `results/analyses/` directory. Existing non-identical files are rejected rather than overwritten.

All reports carry `confirmatory=true`, `statistical_analysis=true`, `not_causal_conclusion=true`, `v1_002_excluded=true`, and `v2_003_excluded=true`. These labels describe the locked design and analysis boundary; they do not turn this finite-sample confirmatory pilot into a paper conclusion.
