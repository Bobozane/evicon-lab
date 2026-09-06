# H-G.1.1 Truncation Amendment

H-G.1 stopped after eight completed runs when a ninth run produced a provider response that the strict JSON parser classified as malformed. The safe request ledger records 512 completion tokens for that logical request, exactly matching the H-G.1 output limit. This is treated as an engineering truncation risk, not as an experimental result.

H-G.1 and all of its batch, ledger, checkpoint, approval, compatibility, and completed-run artifacts remain immutable. The batch is not resumed and is not combined with H-G.1.1.

H-G.1.1 changes only the Agent completion reservation from 512 to 1024 tokens and introduces new protocol, template, run ID, request ID, fingerprint, approval, compatibility, and output-root bindings. Public prompt semantics, strict JSON schema, response parser, scenarios, exposure schedules, conditions, seeds, topology, controller rules, evaluator-private truth boundary, outcome rules, metrics, retry policy, and timeout remain unchanged. Parser-invalid responses still stop immediately and have no automatic recovery.

The fixed design remains 48 runs, 12 matched groups, and 864 logical requests. The completion reservation is therefore 884,736 tokens. This reservation is not a total-token or price cap.

H-G.1.1 remains development-only, pilot-only, not a paper result, and carries no causal conclusion. It requires a new one-shot compatibility check and separate researcher approval before any network execution.
