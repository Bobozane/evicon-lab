# Conformity Identification v1.1 Qualification Gates

These gates qualify the frozen development design; they do not estimate an intervention effect and do not authorize the 1,800-request behavior study.

The exact-hash approval is a sidecar because changing the v1.1 study config to reference later receipts would change the already reported study hash. Human acceptance of that sidecar approves the scientific inputs only. Every network command still requires `--allow-network`, `--confirm-run`, and exact request and completion-reservation caps.

## Gate order

1. Validate the pending exact-hash approval offline.
2. Run one minimal Provider compatibility request after separate authorization.
3. Run 12 source-structure checks. They test whether the model can report zero, one, or two visible public roots without assessing truth or reliability.
4. Run the independent 30-request stability probe. It uses one scenario and one condition, writes a separate ledger, and is never joined to behavioral outcomes.
5. Only after all receipts validate may a later stage prepare the 1,800-request study runner.

Default commands are network-free:

```bash
uv run python -m evicon.validate_conformity_identification_qualification
uv run python -m evicon.conformity_identification_compatibility
uv run python -m evicon.conformity_source_manipulation_gate
uv run python -m evicon.conformity_identification_stability_probe
```

The source gate is capped at 12 requests and 1,536 reserved completion tokens. The stability gate is capped at 30 requests and 15,360 reserved completion tokens. Both use `max_retries=0`, stop on parser-invalid output, use append-only safe ledgers, and refuse existing output directories.

Receipts and ledgers contain hashes, request coordinates, status, token usage, latency, and stable error codes. They do not contain prompts, complete responses, credentials, private truth, or source-independence annotations.

The qualification approval template remains `pending`. It must not be marked accepted by automation. Network authorization for any gate is a separate human decision, and passing a gate is not evidence that the proposed method works.
