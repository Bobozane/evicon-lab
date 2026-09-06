# Real Mediator Pilot

`MediatorRuntime` is a small, isolated boundary for one already validated
`InterventionPlan`. It accepts only that immutable plan, its matching public
`PromptContext`, and an injected `LLMProvider`. Its fixed sequence is:

1. `render_action_instruction()`
2. `LLMProvider.complete()`
3. `parse_mediator_response()`

The runtime does not choose an action or target, alter evidence permissions,
change visibility or speaking order, update budget or cooldown state, write an
event, or create a file. A `no_op` plan returns `not_invoked` without rendering
a request or calling a provider.

`MediatorRuntimeResult.response` may retain a valid parsed mediator message in
memory for the immediate caller. `audit_summary()` deliberately omits both
prompts, the provider's full response, the parsed message, provider metadata,
hidden probes and profiles, offline evaluation data, and credentials. Provider
and parser errors are returned only as stable error codes.

## Opt-in smoke

```bash
uv run python -m evicon.real_mediator_smoke
```

The default command prints `network_disabled` and performs no network activity.
The only opt-in command is:

```bash
uv run python -m evicon.real_mediator_smoke --allow-network
```

It reads the existing provider environment configuration, creates a fixed
`solicit_dissent` plan via the existing Policy/Executor contract fixture, and
makes at most one OpenAI-compatible HTTP request. It forces `max_retries=0`, a
bounded 256-token response budget, and `reasoning_effort="none"`. These are
controlled parameters intended only to improve the reliability of one strict
JSON response; they are not an experimental setting or an intervention-effect
claim. Its output contains only the status, fixed action/template identifiers,
model/request identifiers, finish reason, token usage, latency, parser status,
and an error code. It cannot print a prompt, a full mediator response, or a
key, and it does not write `results/`, `events.jsonl`, `run_record.json`, or
model-output files.

This is an engineering boundary check only. It is not connected to
`AdaptiveProtocolRunner`; ordinary agents remain on FakeLLM, no real agent is
implemented, and a successful call is not evidence of intervention quality or
an experimental result.
