# H-G.2.3.1 parser-failure diagnostic

This read-only diagnostic examines only the append-only request ledger, parsed checkpoint, batch record, completed run records, and locked hashes. It never reads or reconstructs the prompt or provider response body.

The failed logical request has a completed transport ledger terminal, one attempt, and 1,024 completion tokens against the locked 1,024-token maximum. Its parsed checkpoint entry is absent and the batch parser category is `malformed_json`. The ledger does not persist finish reason or HTTP status class, so those facts remain unavailable. Reaching the exact output cap makes truncation more consistent than a non-truncated sporadic malformed response; this is an operational diagnosis, not a claim about response content.

The five completed runs, 106 unique logical requests, and 107 transport attempts reconstruct from six append-only ledgers. All five completed run records retain passed replay status and match their recorded hashes. The extra transport attempt is consistent with the earlier explicit resume after an HTTP server failure; no parser retry or repair occurred.

Because the diagnosis is `truncation_more_consistent`, the current batch must not resume and no single-fingerprint recovery amendment is drafted. A future proposal must use a new protocol/config version, run namespace, fingerprints, compatibility receipt, approval, and output directory, then restart all 16 runs. A reasonable draft is a 2,048-token Agent output cap with a recalculated 589,824-token completion reservation (`288 * 2048`), while leaving prompt semantics, schema fields, model, seed, scenarios, conditions, controller, metrics, and evaluator boundary unchanged. This draft requires separate review and is not approved or executable.

The failed and completed H-G.2.3.1 artifacts remain development-only and cannot be merged with a future rerun or used as a paper or causal result.
