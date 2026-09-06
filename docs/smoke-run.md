# Local Smoke Run

## Command

From the EviCon-Lab repository root, run:

```bash
uv run python -m evicon.run_smoke --config configs/smoke.toml
```

The checked-in configuration uses `fake-llm`, two agents, two rounds, and an
`evidence_social` condition. It loads participants, initial public context,
and two synthetic evidence cards from `configs/scenarios/smoke_scenario.toml`.
It does not make a network request or require credentials.

On first execution it prints the run ID, scenario ID, protocol, round count,
turn count, and output path. It creates:

```text
results/scenario-smoke-run/events.jsonl
results/scenario-smoke-run/run_record.json
```

The directory is intentionally not overwritten. To repeat the smoke run, copy
the configuration and choose a new `run_id`; do not reuse `scenario-smoke-run`.

The persisted run record includes a scenario snapshot. Validate it without
rerunning the provider:

```bash
uv run python -m evicon.replay_validate \
  --events results/scenario-smoke-run/events.jsonl \
  --record results/scenario-smoke-run/run_record.json
```

## Configuration loading

`evicon.config.load_run_config` accepts TOML and JSON. It validates the strict
`RunConfig` contract and reports clear failures for unsupported protocols,
unknown fields, non-positive agent counts or rounds, and non-integer seeds.
No configuration value is read from the environment.

## Not implemented

This command is a protocol and logging smoke test only. It does not implement
real LLM providers, caching, prompt generation, hold-out probe execution,
value-profile updates, diversity or representation metrics, conformity
monitoring, mediator policy, statistical analysis, or experimental claims.
