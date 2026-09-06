# H-D.2.1 Agent Output Truncation Amendment

H-D.2.1 is a new, development-only provenance-cascade Pilot design. It does
not resume, delete, overwrite, or merge the historical H-D.2 Pilot at
`results/provenance-cascade-pilot-hd2-v1`.

The historical batch retains its three completed runs, failed request ledger,
checkpoint, batch record, parser diagnostic receipt, and failed fingerprint.
The append-only sidecar `hd2_abort_receipt.json` marks that batch
`aborted_protocol_truncation`, prohibits resume under H-D.2.1, and marks the
data `not_paper_result` and `no_causal_conclusion`. The historical batch
record itself is unchanged.

## Frozen change

The only generation allowance changed by this amendment is the Agent
`max_tokens` value:

- H-D.2: 256
- H-D.2.1: 512

The public prompt semantics, four-field JSON output shape, claims, provenance
graphs, public exposure schedules, four scenarios, four conditions, three
seeds, six Agents, three rounds, ring topology, evaluator-private truth
boundary, controller rules, application timing, outcomes, and evaluator
metrics are unchanged.

The new identifiers are:

- protocol: `provenance_cascade_agent_protocol.v2_1`
- template: `cascade_agent_turn.v2_1.strict_json_512`
- response schema: `cascade_agent_response_v2_1`
- output root: `results/provenance-cascade-pilot-hd21-v1`

Every run ID and matched-group ID uses the `hd21-` prefix. Rendering under
the new protocol produces new request IDs and request fingerprints. Old
fingerprints are never eligible for replay or recovery.

## Budget and execution boundary

The Pilot still contains 48 runs and 864 logical Agent requests. The
completion reservation is recalculated rather than reused:

```text
48 runs x 18 requests x 512 completion tokens = 442368
```

This is a completion reservation, not a total-token or price guarantee.
Provider behavior remains strict JSON Schema, temperature 0.2, one configured
Provider retry, and a 15-second timeout. Parser-invalid output remains
terminal: H-D.2.1 has no semantic parser recovery.

The new formal result directory must not exist before a non-resume execution
and cannot be overwritten. Resume is limited to the same config, protocol,
template, model, scenario, seed, condition, hashes, approval, compatibility
receipt, and output binding. Completed logical requests cannot be replayed.

## Offline evidence

The FakeProvider integration executes all 48 runs under a temporary directory.
It verifies 864 logical requests, 12 complete matched groups, strict parsing,
append-only ledgers, next-round directive timing, and passed cascade,
application, and outcome replay. It is an engineering regression only, not a
Pilot result.

The compatibility preflight separately exercises a valid 512-token strict JSON
response and a deliberately truncated 512-token response. A real Provider
compatibility receipt remains pending until a separately authorized one-shot
network check succeeds. The repository design approval is accepted, while network authorization remains false.

Default commands are offline:

```bash
uv run python -m evicon.cascade_agent_protocol_v21_compatibility
uv run python -m evicon.cascade_agent_protocol_v21_compatibility --fake-smoke
uv run python -m evicon.cascade_agent_protocol_v21_pilot --mode fake-smoke
uv run python -m evicon.cascade_agent_protocol_v21_real_preflight
```

The optional one-shot real Provider check is:

```bash
uv run python -m evicon.cascade_agent_protocol_v21_compatibility --allow-network
```

It uses the H-D.2.1 renderer and strict schema
`cascade_agent_response_v2_1` with `max_tokens=512`, temperature `0.2`,
seed `20260911`, `max_retries=0`, and a 5-second timeout. It can make at
most one HTTP request. It does not read WVS or Pilot scenario material, create
a result directory, write a request ledger, or retain prompts, full responses,
headers, Provider metadata, or credentials. It never switches to
`json_object`, repairs output, or changes the token limit.

The one-shot command only prints a redacted compatibility result. It does not
rewrite the registered receipt. A successful result must be registered through
a separate safe receipt update before the final Pilot preflight can clear
`compatibility_network_check_pending`.

No default command constructs a Provider, reads an API key, calls a network,
or creates `results/provenance-cascade-pilot-hd21-v1`.

## Required future approvals

Before any real H-D.2.1 Pilot, a successful one-shot compatibility receipt
