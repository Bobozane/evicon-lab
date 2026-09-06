# H-G.2 Provider compatibility gate

This gate checks whether the configured OpenAI-compatible endpoint accepts the frozen H-G.2 five-field strict JSON Schema. It is a one-request transport and parser check, not a Pilot run and not evidence that the intervention works.

The default command is offline. It does not construct a Provider, read an API key, write a request ledger, create `results/`, or load evaluator-private truth. The explicit network path uses a synthetic library notice unrelated to Pilot scenarios, `max_tokens=1024`, `temperature=0.2`, seed `20261021`, `max_retries=0`, and a five-second timeout.

A successful check writes only a content-free receipt under `outputs/study-locks/` and binds its SHA-256 into the accepted approval. The receipt excludes prompts, responses, request IDs, headers, keys, Provider metadata, and private labels. A receipt does not authorize the real eligibility calibration.

```bash
uv run python -m evicon.provenance_cascade_hg2_compatibility
uv run python -m evicon.provenance_cascade_hg2_compatibility --fake-smoke
uv run python -m evicon.provenance_cascade_hg2_compatibility --allow-network
```

After a successful one-shot check, the offline preflight remains blocked until the separately authorized one-seed real eligibility calibration is completed. Full Pilot execution remains unauthorized.
