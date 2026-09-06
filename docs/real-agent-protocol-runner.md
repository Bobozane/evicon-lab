# Real Agent Protocol Runner

`RealAgentProtocolRunner` is an opt-in baseline runner for the four existing
base conditions: `independent`, `social_only`, `evidence_only`, and
`evidence_social`. It accepts an already constructed `RunConfig`,
`ScenarioSpec`, `AgentRuntime`, and injected `LLMProvider`. It never reads an
environment variable, constructs a Provider, enables networking, falls back to
FakeLLM, or retries a failed Agent call.

At every round boundary it first creates every `ExposureSnapshot` from one
shared `DialogueState`. Only then does it invoke agents in a deterministic
order. An AgentPromptContext is constructed solely from its own snapshot and
the exact evidence cards it lists, so a turn generated earlier in a round
cannot be visible to a later Agent in that same round. A parsed successful
response becomes a normal public `DialogueTurn`; its message is a research
trajectory artifact and is therefore written into the turn, RunRecord and
existing `llm_response` event. Prompts, provider metadata, API keys, hidden
probes/profiles and unparsed provider content are never logged.

If rendering, provider completion, or strict response parsing fails, the runner
records a paired, redacted request/response audit entry, writes a failed
`RunRecord`, appends `run_failed`, and terminates without creating a substitute
turn. The existing replay validator can validate both completed runs and these
partially completed failed runs.

## Opt-in smoke

```bash
uv run python -m evicon.real_protocol_smoke
```

The default command prints `network_disabled` and creates no directory. The
network-enabled command requires a unique output name:

```bash
uv run python -m evicon.real_protocol_smoke --allow-network --run-id local-pilot-001
```

It creates a fixed two-Agent, two-round `social_only` public scenario and makes
at most four requests. Its Provider configuration forces zero retries,
256 output tokens, `temperature=0.2`, and `reasoning_effort="none"`. Successful
runs write the ordinary `events.jsonl` and `run_record.json`, then validate the
result with the existing ReplayValidator. The command refuses to overwrite an
existing run directory and prints only safe aggregate information.

This runner deliberately has no Mediator, Monitor, Policy, Executor,
intervention plan, budget, cooldown, probe, metric, or offline-evaluation
integration. It verifies an engineering call boundary only; it is not a
baseline reproduction, a real multi-agent experiment, or evidence about value
homogenization or intervention effectiveness.
