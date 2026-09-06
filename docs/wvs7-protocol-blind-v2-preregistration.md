# WVS English Protocol-Blind v2 Preregistration

`wvs7_english_protocol_blind_v2_calibration.toml` is a local, auditable plan
for a future English calibration run. It declares the four fixed protocol
conditions, `contextual_value_probe.v2`, the frozen English 23-item material
receipt, and an explicitly future `003` run identifier. Validation only hashes
the frozen file bytes and reads its metadata receipt; it does not parse or
output probe text and does not create a results directory.

The executable settings are also preregistered: `gpt-5.6-luna`, Agent
temperature `0.2` and `256` maximum completion tokens, probe temperature `0.0`
and `128` maximum completion tokens, `reasoning_effort = none`, one bounded
provider retry, and result paths under `results/`. A future execution gate must
reject environment or CLI values that attempt to replace these settings.

The configuration is deliberately `calibration`, `development_only = true`,
and `not_paper_result = true`. Its single seed and one development scenario can
validate engineering integration and cost assumptions only. It must not be
reported as a paper result.

The earlier `wvs7-real-baseline-pilot-seed-002` used
`contextual_value_probe.v1` with condition-aware behavior. It is explicitly
excluded from v2 main analysis, so the two records must not be pooled or used
as a common statistical sample.

The fixed primary metrics are `pairwise_diversity`,
`social_influence_loss`, and `profile_drift`. Structural diversity and value
dimension coverage are explanatory. Minority retention is not a primary metric
without preregistered minority dimensions; holdout drift is not a primary
metric without a separately declared holdout ProbeSet.

## Cost terms

For one matched group, `completion_reservation_cap = 51200` reserves bounded
completion capacity. It is neither total-token usage nor a price cap. The
estimate of `190602` total tokens is an observational reference from audited
v1 `002`, not a hard limit or a result expected from v2. The plan also records
384 provider requests per matched group.

## Before Confirmation

A later confirmatory preregistration must define at least three seeds, provide
separate non-overlapping development and test scenario IDs, include a holdout
instrument before any holdout claims, and predefine the statistical analysis.
Those requirements are represented here as future gates, not as claims that a
confirmatory study already exists.

Run the local-only check with:

```bash
uv run python -m evicon.validate_study_preregistration \
  --config configs/studies/wvs7_english_protocol_blind_v2_calibration.toml
```

The command does not read API credentials, construct a provider, call a model,
or write to `results/`.
