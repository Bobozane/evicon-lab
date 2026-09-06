# WVS7 Real Baseline Pilot

This module is a guarded, single-seed engineering pilot for the frozen English
core 23-item ProbeSet. It is not a formal WVS study, a multi-seed experiment,
or a causal analysis. It runs exactly the four existing baseline conditions in
the pre-registered order: `independent`, `social_only`, `evidence_only`, and
`evidence_social`. No Mediator, Monitor, Policy, Executor, adaptive runner, or
online use of probe results is enabled.

## Default Boundary

`wvs7_real_preflight` is local-only. It verifies the frozen file and manifest
hash, accepted English authoring/transcription gate, scenario and matching
coordinates, and output-directory availability. It only reports whether a
model and base URL are configured; it never prints an API key and sends no HTTP
request by default.

The optional `--allow-network --connection-check-only` mode is limited to one
minimal request, writes no experimental directory, and is not run
automatically. It exists only to diagnose an explicitly requested connection
boundary.

## Explicit Run Gate

A real run requires all of the following:

```bash
--allow-network --confirm-run --confirm-request-cap 384 --confirm-completion-reservation-cap 51200
```

The model name comes exclusively from `EVICON_LLM_MODEL`; `gpt-5.6-luna` is the
recommended development-pilot value, not a hard-coded experiment setting. API
credentials are read only by the existing protected Provider and are not fields in the pilot
configuration, batch record, summary, request ledger, or CLI output. Agent
requests use the declared `temperature=0.2`; probe requests use
`temperature=0.0`. Both use the one declared seed. The real path uses finite
`max_retries=1` by default; timeout, transient connection, 429, and 5xx paths
are retryable. No 4xx retry or unlimited retry loop is introduced.

## Fixed Request And Completion Reservation

The fixed plan has two Agents, two rounds, 23 pre probes and 23 post probes per
condition: 16 Agent requests and 368 probe requests. The cap is therefore 384
unique planned request fingerprints and 51,200 completion tokens reserved from
the declared per-request `max_tokens` values. The ledger records every provider
attempt separately, but a resumed retry of the exact failed fingerprint does
not create a new planned request coordinate. This makes recovery possible
without silently expanding the fixed plan.

The completion reservation is not a total-token or monetary-cost cap: provider
input tokens depend on rendered public context and are only known after a
response. The ledger reports actual prompt, completion, and total token counts
separately when the provider supplies them. Any future cost study must set and
state a provider-specific input-token estimation policy rather than treating
the completion reservation as a total-cost guarantee.

## Request Ledger and Recovery

`request_ledger.jsonl` is append-only and contains only safe facts: request ID,
hash fingerprint, content hash, model, template version, condition, phase,
Agent/round/probe coordinates, logical-attempt counter, token and latency
fields, stable error codes, and timestamp. It has no prompt, answer, WVS item text, provider
metadata, Authorization header, or key.

On `--resume`, a completed condition is skipped and a completed request
fingerprint cannot make another provider call. A failed fingerprint may be
retried only with `--resume`; a changed fingerprint for the same logical
coordinate is rejected. A started receipt with no terminal receipt, such as an
interrupted process, also requires `--resume`. A failure stops later conditions in the matched group.

During a real-pilot probe phase, each successfully parsed probe response is
atomically checkpointed under
`results/batches/<batch-id>/private_probe_recovery/<condition>/`. The checkpoint
contains only the response, item/Agent identifiers, phase coordinates, and
hashes of public context. It contains no prompt, WVS item text, public turn,
event log, provider metadata, or credential. The entire `results/` tree is
ignored by version control. On a successful condition, its checkpoint is
deleted after the ordinary isolated probe result files are written.

Consequently, a pre-probe or post-probe provider failure can resume without
replaying completed probe calls. Post-probe recovery also reuses the completed
public run record and does not replay Agent turns. Recovery does not replay a
failed public Agent trajectory: that remains deliberately blocked and requires
a fresh, explicit run decision. Batches created before this checkpoint feature
cannot be safely resumed after a partial probe phase, because their redacted
ledger does not contain the private choices needed to reconstruct the profile.

## Completion

Only after all four conditions complete does the pilot rerun replay validation,
verify two pre and two post 23-item response sets per condition, recheck the
frozen SHA-256, and invoke the existing offline evaluator. The safe batch
summary marks `pilot_only=true`, `single_seed=true`, `not_paper_result=true`,
and `no_causal_conclusion=true`.

The resulting metric names may include pairwise diversity, structural
diversity, value-dimension coverage, and matched social-influence loss. These
are engineering-pilot outputs only. They are not conclusions about human
values, value homogenization, treatment effectiveness, or strict causality.
