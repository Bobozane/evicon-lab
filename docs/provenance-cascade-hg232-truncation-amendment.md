# H-G.2.3.2 truncation amendment

H-G.2.3.1 stopped after five completed runs when one completed transport response reached the locked 1,024-token output cap and failed strict JSON parsing. The content-free diagnostic receipt classifies this as `truncation_more_consistent`. H-G.2.3.1 is sealed: it must not resume, be overwritten, or be merged with this version.

H-G.2.3.2 raises only Agent `max_tokens` from 1,024 to 2,048. For 288 logical requests, completion reservation becomes 589,824. The research prompt bytes, response fields and local semantic parser, model, temperature, seed, scenarios, exposure schedules, four conditions, controller, evaluator boundary, metric definitions, and calibration purpose are unchanged. The protocol/config identity, schema name, run namespace, fingerprints, and output root are new.

The new runtime writes a separate append-only safe response audit for each provider terminal. It may contain fingerprint, finish reason, HTTP status class, prompt/completion/total token counts, whether completion tokens reached 2,048, latency, parser status, and stable error code. It never contains prompt text, response content, keys, headers, provider metadata, or evaluator-private truth. Parser-invalid remains fail-fast, with no string repair, permissive parsing, retry, or recovery.

Two gates precede execution: an offline protocol-stability probe receipt and a one-shot 2,048-token Provider compatibility receipt. Execution approval remains pending until both receipts and their exact hashes are accepted. No effectiveness or causal conclusion is implied.
