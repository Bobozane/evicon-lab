# H-G.2.3.2 protocol-stability probe gate

This gate is a one-run, development-only transport and parser stability check. It is not the 16-run eligibility calibration and its ledger, fingerprints, run ID, output root, and receipt are never merged with calibration data.

The frozen coordinate is `cascade-hg1-true-minority-correction` under `generic_dissent`, seed `20261031`, six Agents, and three rounds. It issues exactly 18 logical Agent requests with `max_tokens=2048` and a completion reservation of 36,864. It reuses the locked H-G.2.3.2 public prompt semantics and strict JSON Schema contract.

Each terminal request appends a content-free audit entry containing only the logical fingerprint, Agent/round coordinate, finish reason, HTTP status class, token counts, token-limit status, latency, parser status, and stable error code. Prompt text, response content, headers, credentials, provider metadata, and evaluator-private truth are prohibited.

Any parser-invalid response, `finish_reason=length`, or completion count at the 2,048-token limit stops the probe immediately. There is no permissive parsing, string repair, automatic retry, semantic recovery, or fallback response format. A passing receipt requires 18 completed logical requests, 18 transport attempts, cascade/application/outcome replay all passed, and a sensitive-field scan passed.

The probe requires its own exact-hash execution approval and four explicit CLI confirmations. A successful receipt removes only the full calibration's `protocol_stability_probe_required` blocker. The separate 16-run calibration execution approval remains required. Neither the probe nor FakeProvider smoke supports an effectiveness or causal claim.
