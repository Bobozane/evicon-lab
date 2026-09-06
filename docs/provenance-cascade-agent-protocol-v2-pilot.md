# H-E H-D.2 Strict Agent Pilot

This is an isolated, versioned execution path for the
`cascade_agent_turn.v2.strict_json` protocol. It does not modify the historical
H-D runner, its results, request ledgers, approval, or recovery amendment. The
formal output root is `results/provenance-cascade-pilot-hd2-v1`, protected by a
no-overwrite gate.

The fixed scope is four synthetic scenarios, four conditions, three seeds, six
Agents, three rounds, 48 runs, 12 matched groups, 864 logical requests, and a
221184 completion-token reservation. Provider requests use
`response_format=json_schema`, schema `cascade_agent_response_v2`, temperature
`0.2`, `max_tokens=256`, timeout 15 seconds, and at most one Provider retry.
Parser-invalid responses stop the run and have no automatic semantic recovery.

## Approval and compatibility binding

The signed approval binds the H-D.2 config, protocol/template, successful
one-shot compatibility receipt, scope, budgets, retry policy, append-only
ledger, resume semantics, no-overwrite rule, and exclusion of historical H-D,
WVS, `002`, `003`, and calibration artifacts. Design approval is not network
authorization. Default preflight remains offline and never constructs a
Provider, reads an API key, calls transport, or creates the formal result root.

The compatibility receipt is stored at
`outputs/study-locks/provenance_cascade_agent_v2_compatibility_receipt.json`. It
contains only safe observed status, token counts, generation parameters, and
hash bindings.

## Execution gate

The `real-pilot` branch calls `HD2PilotRunner.run_all()` only when all of the
following are present and valid:

- `--allow-network`
- `--confirm-run`
- `--confirm-request-cap 864`
- `--confirm-completion-reservation-cap 221184`
- accepted, hash-matched approval and compatibility receipt
- successful final preflight
- absent formal output root, or explicit `--resume` with an identical batch
  binding

Provider construction occurs only after these checks. Each run has its own
append-only `request_ledger.jsonl`, public parsed checkpoint, and safe run
record. Resume loads completed run records without constructing a Provider or
replaying their fingerprints. A changed model, config, protocol, approval,
compatibility receipt, output root, seed, scenario, or condition invalidates the
binding.

A failed run leaves a safe partial batch record and no complete pilot receipt.
Only 48 completed runs with passed cascade, application, and outcome replay can
produce `pilot_receipt.json`. Neither batch records nor receipts contain
prompts, full model responses, API keys, Provider metadata, or evaluator-private
truth.

## Offline validation

```bash
uv run python -m evicon.cascade_agent_protocol_v2_pilot --mode fake-smoke
uv run python -m evicon.cascade_agent_protocol_v2_real_preflight
```

The FakeProvider smoke executes the same `run_all()` path below a temporary
directory. It does not create the formal output root and is not a paper result.
No network command is executed by tests or default smoke.
