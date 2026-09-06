# Protocol-Blind v2 Confirmatory Real Run

This stage adds an explicit execution gate for the locked English WVS protocol-blind v2 plan. It is infrastructure for a future confirmatory study, not a statistical result or a claim about value change.

## Scope

The runner consumes only the locked plan, its lock receipt, the accepted synthetic English test scenario, and the frozen 23-item English ProbeSet. It executes the fixed order of three seeds (`20260820`, `20260821`, `20260822`) and four conditions (`independent`, `social_only`, `evidence_only`, `evidence_social`). Each run uses the existing Real Agent Runtime, Contextual Probe Runtime, measured experiment orchestration, replay validator, and an independent safe request ledger. Mediator, Monitor, Policy, Executor, and AdaptiveProtocolRunner are not enabled.

The planned upper bound is 1,152 logical requests and 153,600 completion-token reservations. A reservation is not actual total-token usage and is not a price or billing cap. `571,806` is an observational total-token reference inherited from calibration; actual provider billing depends on the intermediary API.

## Offline Preflight

```bash
uv run python -m evicon.wvs7_confirmatory_real_preflight \
  --config configs/studies/wvs7_english_protocol_blind_v2_confirmatory_plan.toml
```

Preflight verifies the plan lock, source hashes, accepted scenario review, all 12 explicit run entries, seed/condition completeness, v2 probe template, and output collision protection. It does not read provider environment variables, construct a provider, make HTTP requests, or create `results/`.

## Explicit Run Gate

A real run is intentionally impossible without all of the following confirmations:

```text
--allow-network
--confirm-run
--confirm-request-cap 1152
--confirm-completion-reservation-cap 153600
--confirm-estimated-total-token-reference 571806
```

The command is not run automatically during development. When enabled by a researcher in a separately reviewed environment, the runner rejects an existing batch unless `--resume` is explicit. A completed run is skipped on resume. A failed run can only be retried with resume; a changed plan, lock, scenario, or ProbeSet binding rejects recovery.

## Audit and Completion

Each run writes a separate JSONL request ledger containing only request coordinates, content hashes, attempt state, usage counters, latency, and stable error codes. Batch receipts contain input hashes, run/group counts, replay statuses, and safe token aggregates. They never contain prompts, model replies, WVS text, raw probe answers, provider metadata, or secrets.

No confirmatory receipt, completeness audit, or initial/final offline report is produced until all 12 runs are completed, every matched group is complete, every replay passes, and the frozen ProbeSet and test-scenario hashes remain unchanged. The generated offline report is descriptive infrastructure output; statistical analysis is deferred to a later stage and no causal conclusion is made here. The v1 `002` pilot and v2 calibration `003` are explicitly excluded from confirmatory statistics.

No real command with `--allow-network` was executed while developing this stage.

## Parser-Invalid Probe Recovery Amendment

A parser-invalid probe response is a transport that completed at the provider but did not produce a valid structured probe response. The normal completed-fingerprint rule remains strict: a completed fingerprint is never replayed by ordinary resume. The confirmatory runner has one additional, explicit recovery path only for a post-probe `invalid_schema`-class failure:

- the run must be resumed explicitly;
- the private post checkpoint must identify the next missing `agent_id x probe_id x round_id` coordinate;
- the existing completed v2 probe ledger fingerprint must be the exact allow-listed coordinate;
- agent requests and unrelated completed probe fingerprints remain ineligible;
- one semantic recovery transport attempt is allowed per fingerprint.

Recovery appends new ledger rows and increments `attempt_count`; it never rewrites the original transport receipt. Logical request counts remain 1,152, while audit summaries distinguish unique logical requests, total transport attempts, parser-recovery attempts, and actual token usage. A second parser-invalid response for the same fingerprint stops safely and cannot produce a confirmatory receipt or analysis report.

`confirmatory_parser_recovery_amendment.json` is a safe, immutable-input-bound receipt. It contains only batch/run coordinates, the fingerprint, stable error code, recovery strategy version, attempt limit, and plan/lock/scenario/ProbeSet hashes. It contains no prompt, response, WVS text, provider metadata, or secret. This is a runtime fault-recovery patch only; it does not alter stimuli, templates, generation parameters, protocol order, plans, locks, or metric definitions.
