# H-G.2.2 real eligibility calibration runner

The H-G.2.2 runner is an isolated, one-seed development calibration for the explicit share contract. It uses 16 runs, four matched groups, 288 logical Agent requests, and a 294,912 completion reservation. It is not an effectiveness Pilot or paper result.

The default commands are offline:

```bash
uv run python -m evicon.provenance_cascade_hg22_calibration --mode preflight
uv run python -m evicon.provenance_cascade_hg22_calibration --mode fake-smoke
```

Real execution is unavailable until the exact runner hash and execution scope are accepted and network execution is separately authorized. It then requires all network and cap confirmations. The locked Provider model is `gpt-5.6-luna`; a different environment model is rejected before Provider construction.

Each run uses its own append-only request ledger and parsed-decision checkpoint. Explicit resume skips completed runs and completed fingerprints only when all bindings match. Existing output is never overwritten. Parser-invalid output stops immediately and cannot be retried or semantically recovered under this protocol version.

The runner records no prompt, full model response, API key, Provider metadata, evaluator truth, or private source-independence label. A calibration receipt is written only after 16/16 runs complete and all cascade, application, and outcome replay checks pass.
