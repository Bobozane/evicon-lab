# Base Protocols

## Scope

EviCon-Lab currently implements four deterministic, no-network base protocols.
They control only what a simulated agent can see. They do not measure
diversity, judge factuality, estimate conformity, select an intervention, or
call a real LLM.

| Condition | Own prior turns | Prior-round peer turns | Evidence introduced by current round |
| --- | --- | --- | --- |
| `independent` | Yes | No | No |
| `social_only` | Yes | Yes | No |
| `evidence_only` | Yes | No | Yes |
| `evidence_social` | Yes | Yes | Yes |

The public-audience marker `"*"` and per-agent evidence visibility are still
enforced by `DialogueState`. A card is included only when it was introduced by
the current round and its audience permits that agent.

## Round snapshot rule

For every round, `ProtocolRunner` first reconstructs one `DialogueState` from
all earlier turns and builds every agent's `ExposureSnapshot`. Only after all
snapshots exist does it call the local provider and append new `DialogueTurn`
records. A snapshot permits social exposure only from `turn.round_id <
current_round`.

This prevents same-round order leakage: a later provider call cannot see a
previously generated answer from the same round, even though calls are executed
in a deterministic agent order.

## FakeLLM

`FakeLLM` implements the local `LocalProvider` interface. Its request includes
agent ID, round, protocol, visible dialogue history, visible peer-turn IDs,
visible evidence IDs, and seed. Its structured response is a deterministic
text rendering of those visibility-safe facts plus a request fingerprint.

It does not import a provider SDK, open a network connection, read an API key
or environment variable, render value probes, or produce a hold-out probe
response. It is a test double, not a behavioral model of an LLM.

## Runner boundary

`ProtocolRunner` creates default participant `AgentSpec` values from
`RunConfig.agent_count` unless explicit specs are supplied. It emits dialogue
turns, evidence-exposure records, a `RunRecord`, and JSONL events. It does not
produce value profiles, value-probe responses, metrics, monitor signals,
mediator decisions, prompts, caches, or real-model calls.
