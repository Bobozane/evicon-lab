# H-G.2.1 execution environment amendment v2

This technical amendment isolates an execution-environment binding error from the H-G.2.1 eligibility calibration. The original `v1` output remains append-only and is not resumed, moved, deleted, or treated as research data.

The failed attempt used the local `OPENAI_MODEL_NAME` value `gemini-3.6-flash`, while the accepted one-shot compatibility receipt binds `gpt-5.6-luna`. It stopped on the first request with a safe `http_client_error`, produced no completed run, no parsed behavior observation, no token usage, and no calibration receipt.

Version 2 changes only execution identity and validation:

- require the Provider environment model to equal the compatibility receipt model before Provider construction;
- use a new output root, batch binding, run-ID namespace, and request fingerprints;
- bind the unchanged H-G.2.1 protocol, prompt template, response schema, public scenarios, controller, replay rules, seed, conditions, metrics, and evaluator boundary;
- retain 16 runs, four matched groups, 288 logical requests, 1,024 completion tokens per request, and a 294,912 completion reservation;
- retain `max_retries=0`, a 15-second timeout, append-only ledgers, explicit resume, no overwrite, and parser-invalid stop without semantic recovery.

This amendment does not alter any experimental stimulus or outcome definition. It is development-only eligibility calibration infrastructure, not an effectiveness experiment or paper result. A new exact-hash approval and explicit network authorization are required before execution.
