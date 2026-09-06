# Offline Evaluation

`evicon.evaluation` is a local, read-only orchestration layer. It loads completed
`RunRecord` files, JSONL event logs, and separately stored `ProbeResult` files;
it does not call a provider, execute prompts, mutate `DialogueState`, or feed a
metric back to `ProtocolRunner`.

## Manifest

An `EvaluationManifest` is the only source of pairing intent. It contains an
evaluation and scenario identifier, selected metric suites, declared probe-set
IDs, run entries, optional metric options, and metadata. Unknown fields are
rejected. Every run entry declares its run ID, condition, underlying protocol,
record path, event path, optional private probe-result path, seed, model, role,
and `counterfactual_group_id`.

Paths are resolved relative to the manifest. Before evaluating, the loader
checks each path, run ID, scenario snapshot, protocol, seed, model name, agent
count, agent IDs, probe-set ID, and probe round consistency. Each file read is
included in `input_audit` with an absolute path, byte count, and SHA-256 hash.
Inputs are never modified.

`ProbeResult` files are read only here, after online execution. Their raw
responses and probe text are not copied into the evaluation report.

## Matching

The matcher never derives a pairing from a filename or list position. A pair
requires the same scenario ID, `counterfactual_group_id`, model name, sorted
agent IDs, seed, and probe-set ID. The protocol is intentionally excluded from
cross-condition keys because it is the experimental treatment being compared.
Ambiguous or incomplete groups remain in `unmatched_entries` and produce a
warning.

Supported explicit comparisons are:

- `independent` to `social_only`;
- `evidence_only` to `evidence_social`, for standard and holdout profiles;
- `generic_mediator` to `evicon`;
- `initial` to `final` runs with the same condition and probe set.

An `evidence_only` / `evidence_social` pair is required for social influence
loss. An initial/final pair is required for profile drift. Missing data skips
only the affected metric and records the reason in `warnings`.

## Metric Suites

`baseline_diversity` runs pairwise and structural diversity for each completed
final run with standard profiles. `representation` runs value-dimension coverage
and, only when both a threshold and explicitly nominated minority dimensions
are supplied, minority retention. `holdout` runs holdout profile drift.
`counterfactual` runs exposed and holdout social influence loss. `audit` reports
run completeness, evidence-exposure consistency, intervention count, and token
or latency totals only when those numeric fields exist in the event log. `all`
selects every suite.

The evaluator delegates mathematical calculations to the existing pure
functions in `evicon.metrics`; it does not duplicate their formulas.

## Failed Runs

Completed runs are the normal evaluation inputs. A failed run is rejected unless
its manifest entry sets `allow_failed_for_audit = true`; even then it contributes
only audit output, marked as partial in the report. It cannot contribute to
diversity, representation, holdout, or counterfactual metrics.

## Command

From the project root, evaluate the local example with:

```bash
uv run python -m evicon.evaluate_offline \
  --manifest configs/evaluations/smoke_evaluation.toml \
  --output results/evaluations/scenario-smoke-evaluation.json
```

The output path must be new. The supplied smoke manifest contains only one
`evidence_social` run, so comparison-dependent metrics are explicitly skipped
with warnings. This is expected example behavior, not an experimental result.

## Not Implemented

This stage does not implement a real model provider, online metric feedback,
`ConformityMonitor`, mediator strategy, statistical inference, or paper-level
claims. It does not alter protocol execution or the hidden-probe isolation
boundary established in earlier stages.
