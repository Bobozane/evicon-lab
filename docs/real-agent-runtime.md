# Real Agent Runtime

`AgentRuntime` is an isolated boundary for one normal agent turn. It is not a
runner and does not accept `DialogueState`, value profiles, probe results,
offline evaluation data, Policy/Executor decisions, budgets, cooldowns, or any
hidden probe/profile data.

The only runtime input is an `AgentPromptContext` built from public data: the
current agent identity, role and initial value labels; round, protocol and
scenario context; an `ExposureSnapshot`; and the exact visible `EvidenceCard`
objects. The context validates that snapshot agent, round, protocol and
scenario agree; every peer turn is explicitly exposed, earlier than the round,
and visible to the agent; and every evidence card exactly matches a visible ID,
has been introduced, and grants the agent access. The four protocol conditions
also prohibit peer turns or evidence where their visibility rules do not allow
them.

For a valid context, the sequence is fixed:

1. `render_agent_turn()`
2. `LLMProvider.complete()`
3. `parse_agent_response()`

The prompt treats peer turns as viewpoints rather than facts and permits only
visible evidence. The parser accepts exactly
`{"message":"...","evidence_ids_used":[]}` and rejects controller fields,
unavailable evidence, malformed JSON, and hidden content. `AgentRuntime` does
not create a `DialogueTurn`, mutate any state, write an event, or write a file.
The valid message remains only in the in-memory `AgentRuntimeResult`; its audit
summary excludes prompts, messages, provider metadata, hidden content and
credentials.

## Opt-in smoke

```bash
uv run python -m evicon.real_agent_smoke
```

The default prints `network_disabled` and does not construct a provider. The
only opt-in form is:

```bash
uv run python -m evicon.real_agent_smoke --allow-network
```

It uses one fixed round-0, independent-protocol, single-agent public fixture.
The request makes at most one HTTP call, forces zero retries, a 256-token
budget, `temperature=0.2`, and `reasoning_effort="none"`. It prints only safe
status and usage fields plus validated evidence IDs, and never writes
`results/`, JSONL, run records, prompts, or model output.

This is a call-boundary implementation only. It is not connected to
`ProtocolRunner`, `ControlledProtocolRunner`, or `AdaptiveProtocolRunner`.
Ordinary agents continue to use FakeLLM in every existing runner, and this
module establishes neither a real multi-agent experiment nor an intervention
effect.
