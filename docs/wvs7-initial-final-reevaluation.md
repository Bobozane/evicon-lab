# WVS7 Initial/Final Re-evaluation

Every completed WVS condition has one public trajectory but two isolated probe
files: `pre_probe_results.jsonl` and `post_probe_results.jsonl`. The offline
evaluation manifest now declares them as separate, explicit `initial` and
`final` entries that share the same physical run ID. The entry identifiers are
unique only within the evaluation manifest; the loader still verifies that both
entries refer to the same persisted `RunRecord` and event stream.

The standard-scope report can therefore compute `profile_drift` per condition,
in addition to final-profile diversity and the matched social-influence
contrast. Audit metrics remain counted once per public trajectory, rather than
once per measurement phase.

For an already completed batch, this no-network command creates a new,
versioned manifest and report. It never modifies the prior report or any model
output:

```bash
uv run python -m evicon.wvs7_re_evaluate \
  --config configs/studies/wvs7_english_baseline_pilot.toml \
  --batch-id wvs7-real-baseline-pilot-002
```

The revised report still does not create a holdout result when the frozen
ProbeSet has no declared holdout items. It also does not infer minority
dimensions; those remain an explicit future analysis choice.
