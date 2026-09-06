# Probe Format

`ProbeSet` is a strict TOML or JSON configuration for offline value measurement. It contains a stable `probe_set_id`, ordered `dimensions`, a `version`, metadata, and ordered `items`.

```toml
probe_set_id = "smoke-value-probes"
dimensions = ["fairness", "transparency"]
version = "1.0"

[[items]]
probe_id = "fairness-standard"
text = "How strongly should comparable cases be treated consistently?"
dimension = "fairness"
response_scale = ["1", "2", "3", "4", "5", "6", "7"]
is_holdout = false
reverse_scored = false
```

Probe IDs are unique. Every item dimension must appear in `dimensions`, and each dimension must have at least one item. `response_scale` is ordered and contains at least two unique labels. The checked-in example, `configs/probes/smoke_probe.toml`, has two dimensions, two items per dimension, and one standard plus one holdout item for each dimension.

`ProbeRunConfig` selects either standard or holdout items through `is_holdout`. `ProbeRunner` gives each agent only one item request at a time. It has no `DialogueState`, evidence, peer history, mediator prompt, or social-exposure input.

`ProbeProfileBuilder` maps scale positions to `[0, 1]`, applies `reverse_scored`, averages items by dimension, and emits `ValueProfile` in the declared `dimensions` order. It neither compares agents nor interprets a score as correct or desirable.

Run the local standard-item smoke command:

```bash
uv run python -m evicon.probe_smoke \
  --probe-config configs/probes/smoke_probe.toml \
  --run-id probe-smoke-run
```

Use `--holdout` to select holdout items. The command uses only `FakeLLM` and never makes an API request.
