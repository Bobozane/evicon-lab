# Source-Behavior v2 Provider Compatibility Gate

This gate makes one narrowly scoped OpenAI-compatible request to establish
whether a future Provider can accept and return the frozen v2 adoption/share
contract. It is isolated from the 12-case behavior qualification. Passing it
does not execute that qualification, estimate a source effect, authorize a
behavior run, or create a paper result.

## Default boundary

Without `--allow-network`, the CLI returns `network_disabled` before reading
environment variables, constructing a Provider, resolving a receipt path, or
writing a file. It does not create a request ledger or a behavior results
directory.

The network path is available only through an explicit `--allow-network`. It
requires the accepted exact-hash v2 approval and an absent receipt target. The
fixed transport contract is:

- exactly one HTTP attempt;
- `response_format=json_schema` with strict mode;
- schema derived directly from `BehaviorQualificationResponse`;
- `max_tokens=128`;
- `temperature=0.0`;
- `seed=20261110`;
- `max_retries=0`;
- timeout of 5 seconds;
- `reasoning_effort=none`.

The explicit reasoning setting is compatibility amendment v2.1. The first
authorized v2 attempt returned 2xx but ended at exactly 128 completion tokens
with `finish_reason=length` and parser-invalid content. It made one request and
wrote no receipt, behavior output, or request ledger. The v2.1 amendment treats
that outcome as a truncation diagnostic. The output is consistent with the
previously uncontrolled Provider-default reasoning consuming the bounded
completion budget, so the amended request now disables reasoning explicitly.
It does not increase the completion cap, change the prompt/schema contract, add
a retry, or authorize another network request.

The request uses two short, fictional public scheduling statements created in
the compatibility module. It does not load request content from any historical
Pilot, WVS material, or evaluator-private truth. Loading the accepted sidecar
and its bound current-study configuration is a precondition check only; those
materials are not placed in the request.

## Safe output and receipt

CLI output is restricted by a strict Pydantic allowlist to status, model,
finish reason, parser validity, HTTP/transport category, token usage, latency,
schema identity, attempt count, and explicit safety flags. Unknown finish
reasons are reduced to `other`. Provider error messages and response bodies are
never copied into the output.

A valid 2xx response must have `finish_reason=stop`, pass the existing v2
semantic parser, report complete and consistent token usage, and consume one
attempt. Only then does the CLI create the append-only redacted receipt at:

```text
outputs/study-locks/conformity_source_behavior_compatibility.v2_1.json
```

The receipt binds the current config SHA, protocol SHA, approval SHA,
protocol/template/schema versions, schema SHA, the v2.1 compatibility module,
and all fixed generation
parameters. It contains no prompt, complete model response, API key, headers,
Provider metadata, request ledger, private truth, or behavior output. An
existing receipt blocks the command before environment access or transport use
and is never overwritten.

FakeTransport tests cover valid 2xx, malformed provider and model JSON, unknown
response fields, unavailable content IDs, `finish_reason=length`, timeout,
connection failure, 401, 429, 500, unsupported `response_format`, the one-call
cap, redaction, no-overwrite behavior, and absence of unintended files.

The future one-time command, to be run only after separate network
authorization, is:

```bash
.venv/bin/python -m evicon.conformity_source_behavior_compatibility --allow-network
```
