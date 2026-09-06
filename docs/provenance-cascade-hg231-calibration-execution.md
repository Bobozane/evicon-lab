# H-G.2.3.1 Provider-subset eligibility calibration execution gate

This development-only gate executes the H-G.2.3.1 provider-subset protocol without changing the H-G.2.3 prompt or local semantic parser. The provider JSON Schema omits only `uniqueItems`; duplicate content and evidence IDs remain invalid under the local parser.

The execution path is isolated in `provenance_cascade_hg231_calibration`. It binds the approved design, compatibility receipt, runner hash, model, run coordinates, and output root. It uses 16 runs, four matched groups, 288 logical requests, and a 294,912-token completion reservation. Each run owns an append-only request ledger and parsed-decision checkpoint. The parsed audit stores epistemic stance, adoption decision, sharing decision, ID counts, and fingerprints, but no prompt or provider response text.

Execution requires a separately accepted execution approval plus explicit network, run, request-cap, and completion-reservation confirmations. Resume accepts only an identical binding, never replays completed fingerprints, and does not recover parser-invalid output. A failed or incomplete calibration cannot produce the completion receipt. Existing H-G.2.2, H-G.2.3, WVS, and prior pilot outputs are excluded and are never overwritten.

The FakeProvider smoke is an engineering and eligibility check only. It does not establish effectiveness, statistical separation, causality, or a paper result.
