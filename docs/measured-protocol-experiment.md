# Measured Protocol Experiment

`MeasuredProtocolExperimentRunner` is an opt-in, single-condition engineering
orchestrator for one existing `RealAgentProtocolRunner` trajectory. It does not
implement a four-condition baseline, a mediator, online monitoring, policy,
intervention, diversity metric, or causal analysis.

Its fixed order is:

1. Build initial public-only measurement contexts and run private pre probes in memory.
2. Run the unchanged public real-Agent protocol runner.
3. Only after a completed public `RunRecord`, reconstruct each agent's offline public context and run private post probes.
4. Only when both probe phases succeed, write phase-separated probe JSONL files and a safe measurement summary.

The initial context contains only each agent identity, role, initial labels, and
the scenario's public initial context. Post contexts come only from
`OfflineMeasurementContextBuilder`: own prior turns, appropriately visible
prior peer turns, and actually exposed evidence. Neither phase can enter Agent
prompts, `DialogueState`, public JSONL events, `RunRecord`, Monitor, Policy,
Executor, Mediator, or any controller.

`probe_seed`, selected holdout mode, probe-set ID, selected probe item IDs,
Agent IDs, and pre/post target rounds are explicit and recorded in
`measurement_record.json`; pairing is never inferred from file names or line
order. Private raw answers are only persisted, after complete success, in:

```text
results/<run-id>/probes/pre_probe_results.jsonl
results/<run-id>/probes/post_probe_results.jsonl
```

The adjacent `measurement_record.json` has only safe identifiers, phase states,
counts, SHA-256 values of the two result files, token aggregates, and stable
error codes. It contains no item text, answers, prompts, public turn text,
provider metadata, or secrets.

Pre-probe failure creates no run directory and prevents the public trajectory.
Agent failure preserves the existing failed public-run record but skips post
probes. Post-probe failure preserves the completed public trajectory and marks
the measurement incomplete; it does not write partial probe files as a complete
measurement.

The default smoke is network-disabled:

```bash
uv run python -m evicon.measured_protocol_smoke
```

The explicit `--allow-network` path requires a unique `--run-id`, makes at most
eight requests for its fixed two-agent, two-round fixture, and uses zero retry
configurations. This is an engineering pilot, not a four-condition replication,
not a value-homogenization measurement, and not a causal or paper result.
