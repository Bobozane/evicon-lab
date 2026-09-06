# Source-Behavior v2 Execution Gate

The v2 Provider compatibility receipt is complete and bound to the frozen
source-behavior protocol. The next stage is a separate 12-case behavior
qualification, not a paper experiment and not a causal test. This execution
gate keeps that stage separately authorized and auditable.

## Offline approval

The execution sidecar is:

```text
configs/provenance_cascade/identification/conformity_source_behavior_execution_approval.v2.toml
```

It was accepted by the researcher on 2026-08-28. It binds the current v2
config, protocol, accepted design approval, the completed compatibility
receipt, the response schema, and the runner SHA. The sidecar does not itself
authorize network use; the future command still needs `--allow-network`,
`--confirm-run`, and exact cap confirmations.

The offline preflight is:

```bash
.venv/bin/python -m evicon.conformity_source_behavior_execution_approval
```

It must not read an API key, construct a Provider, create an output directory,
or execute a case. With the accepted sidecar it reports only
`behavior_qualification_network_authorization_required`.

## Frozen run contract

The runner uses the already-qualified model from the compatibility receipt and
the exact v2 case set: four synthetic scenarios under source-free, same-root,
and independent-roots projections, for 12 logical requests. It uses:

- `max_tokens=128`;
- `temperature=0.0`;
- `seed=20261110`;
- `reasoning_effort=none`;
- `max_retries=0`;
- five-second timeout;
- 12-request and 1,536-token reservation caps.

An unavailable content ID, unknown response field, parser-invalid response, or
non-`stop` finish reason stops the run immediately. There is no retry, resume,
repair, fallback schema, or output overwrite path.

The runner writes only this new output root, after confirming it does not exist:

```text
outputs/conformity-source-behavior-qualification-v2/
```

It contains `safe_case_audits.jsonl` with only case/scenario/projection IDs,
parser status, adoption/share categories, used-content count, and visible-root
count, plus `qualification_receipt.json` with aggregate counts, token usage,
latency, hashes, generation parameters, and safety flags. It does not write a
request ledger, prompts, full responses, credentials, headers, provider
metadata, private truth, or historical results.

## Completed bounded execution

After independent sidecar acceptance and separate network authorization, the
bounded command was run once:

```bash
.venv/bin/python -m evicon.conformity_source_behavior_runner \
  --allow-network \
  --confirm-run \
  --confirm-request-cap 12 \
  --confirm-completion-reservation-cap 1536
```

It completed all 12 logical requests, wrote 12 parser-valid safe audits and one
bound receipt, and must not be run again. This establishes only that the
bounded protocol produced safe, parseable public decision observations across
its source projections. It does not establish a source effect, intervention
effect, causal effect, model quality, or paper conclusion. The next gate is the
post-collection, outcome-blind lock in
`provenance-cascade-conformity-source-behavior-analysis-lock-v2.md`; it remains
isolated from all H-G, H-D, WVS, calibration, ledger, receipt, and historical
pilot artifacts.
