# Source-Behavior v3 Provider Compatibility Gate

This gate makes at most one narrowly scoped OpenAI-compatible request to test
whether a future Provider accepts and returns the frozen v3 ordinal behavior
contract. It is separate from the 36-case qualification. Passing it cannot run
that qualification, estimate a provenance effect, authorize a behavior batch,
or create a paper result.

## Default boundary

Without `--allow-network`, the CLI returns `network_disabled` before it reads
environment variables, constructs a Provider, resolves or creates a receipt,
or touches a transport. It writes no request ledger, result, prompt, response,
or output directory.

The future explicitly enabled path requires the accepted v3 design sidecar and
an absent receipt target. Its fixed wire contract is:

- exactly one HTTP attempt;
- `response_format=json_schema` in strict mode;
- schema derived directly from `BehaviorQualificationResponseV3`;
- `max_tokens=256`;
- `temperature=0.2`;
- `seed=20261201`;
- `max_retries=0`;
- timeout of 5 seconds;
- `reasoning_effort=none`.

The compatibility request uses four short, author-original statements about a
fictional reading-room schedule. They are defined inside the compatibility
module and are not loaded from the v3 36-case corpus, v2 artifacts, historical
Pilots, WVS material, or evaluator-private truth. The request exercises the v3
target claim, decision task, public root relations, ordinal judgment, public
action, sharing, confidence, and visible-content reference contract.

## Safe output and receipt

CLI output has a strict allowlist: status, model, finish reason, parser validity,
HTTP/transport category, token usage, latency, schema identity, attempt count,
and explicit safety flags. Provider error messages and response bodies are
never copied into the output.

A successful response must be 2xx, end with `finish_reason=stop`, pass the v3
semantic parser, provide complete consistent token usage, and use exactly one
attempt. Only then may the command create the append-only receipt:

```text
outputs/study-locks/conformity_source_behavior_compatibility.v3.json
```

The receipt binds the v3 config, corpus, protocol, accepted design approval,
compatibility module, protocol/template/schema versions, schema SHA, generation
parameters, model, safe transport status, token usage, and latency. It contains
no prompt, public request text, complete response, API key, headers, Provider
metadata, request ledger, case outcomes, private truth, or behavior result.
An existing receipt blocks before environment access and is never overwritten.

FakeTransport coverage includes valid 2xx, malformed provider and inner JSON,
unknown response fields, unavailable content IDs, inconsistent judgment/action,
`finish_reason=length`, missing environment values, timeout, connection failure,
401, 429, 500, unsupported response format, the one-call cap, output redaction,
duplicate receipt fields, strict receipt binding, no overwrite, and no unintended
files.

The default offline command is:

```bash
.venv/bin/python -m evicon.conformity_source_behavior_compatibility_v3
```

The future one-time network command is not authorized by implementation or by
the accepted design sidecar. It may be run only after separate researcher
authorization:

```bash
.venv/bin/python -m evicon.conformity_source_behavior_compatibility_v3 \
  --allow-network
```
