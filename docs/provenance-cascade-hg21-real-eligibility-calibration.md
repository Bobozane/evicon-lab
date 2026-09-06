# H-G.2.1 real eligibility calibration gate

This is a one-seed, development-only eligibility calibration. It is not a full Pilot and cannot establish intervention effectiveness.

The fixed scope is four synthetic scenarios, four conditions, six Agents, three rounds, 16 runs, four matched groups, 288 logical Agent requests, and a 294,912 completion-token reservation. Each run has an independent append-only request ledger, parsed-response checkpoint, content-free behavior-decision sidecar, and three-layer replay audit.

The behavior sidecar stores only request fingerprints, public Agent/round coordinates, epistemic and behavioral enums, permitted-ID counts, and whether sharing was requested. It excludes prompts, model text, claim text, keys, Provider metadata, and evaluator-private truth. It allows completed parsed behavior to remain auditable across explicit resume without replaying completed requests.

Parser-invalid output stops the calibration and has no automatic semantic recovery. Existing output is rejected unless `--resume` is explicit and all config, model, scenario, seed, condition, protocol, approval, receipt, and output bindings match. A complete receipt is written only after 16/16 runs and replay pass.

The default preflight and FakeProvider smoke are offline:

```bash
uv run python -m evicon.provenance_cascade_hg21_calibration --mode preflight
uv run python -m evicon.provenance_cascade_hg21_calibration --mode fake-smoke
```

Real execution additionally requires an exact-hash execution approval and all explicit network and cap confirmations. Even successful calibration only determines whether the eligibility chain and behavior variation are observable enough to justify a later Pilot amendment.
