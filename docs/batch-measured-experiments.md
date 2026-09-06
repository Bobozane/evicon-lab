# Batch Measured Experiments

`BatchExperimentManifest` is an auditable plan for four matched baseline
conditions: `independent`, `social_only`, `evidence_only`, and
`evidence_social`. For every declared seed it contains exactly one run for each
condition, a shared `matched_group_id`, explicit run IDs, the complete run
order, agent IDs, model and probe-model names, selected probe item IDs, probe
mode, rounds, seeds, non-secret request settings, and expected artifact paths.
Pairing is never inferred from a directory name, JSONL ordering, or a result.

Within each matched group, only `condition` and `run_id` may differ. Scenario,
ordered Agent IDs, model settings, seed, rounds, ProbeSet, selected probe
items, phase rounds, and output parent must match exactly. The manifest rejects
duplicate run IDs, duplicate condition/seed entries, unknown seeds, incomplete
groups, and unsafe metadata.

`BatchMeasuredExperimentRunner` receives providers and runtimes through
dependency injection. It reads no environment variables, constructs no
provider, and performs no network configuration. It calls the existing
single-condition `MeasuredProtocolExperimentRunner` serially in the explicit
manifest order. Public run directories and their events, records, and isolated
probe files retain their existing meanings.

The batch layer writes only safe audit files:

```text
results/batches/<batch-id>/batch_manifest.json
results/batches/<batch-id>/batch_record.json
```

They reference expected paths and safe statuses but do not copy prompts, public
turn text, probe item text, raw probe answers, provider metadata, hidden data,
or credentials. The runner rejects an existing batch directory or any planned
run directory before it starts.

Execution is fail-fast. A failed single condition updates `batch_record.json`,
leaves remaining runs as planned, and marks the affected matched group
incomplete. An incomplete group is an audit state, not an evaluable comparison.

The default smoke remains network-disabled:

```bash
uv run python -m evicon.batch_measured_smoke
uv run python -m evicon.batch_measured_smoke --plan-only --batch-id local-batch-plan-smoke
```

The explicit network path is limited to a fixed two-agent, two-round,
one-seed plan with at most 32 provider calls. It is an engineering boundary,
not a registered four-condition experiment. Actual four-condition runs require
pre-registered data, model, seeds, and parameters. This stage computes no
diversity, social-influence loss, or paper conclusion.
