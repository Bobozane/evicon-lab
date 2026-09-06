# H-G Outcome Replay Technical Amendment Receipt

This amendment records a replay-validation repair for the existing H-G Pilot batch. It is not a change to the experimental design and does not authorize network continuation by itself.

## Scope

The historical validator remains unchanged. H-G resume validation opts into `cascade_hg_outcome_replay.v2.visible_cross_claim_context`, which permits an Agent stance on one claim to cite correction content from another claim only when that correction content was present in the same immutable round-start snapshot.

The repair does not change prompts, scenario materials, conditions, seeds, metrics, budgets, request caps, model settings, outcome generation, controller decisions, or directives. It does not expose global provenance content or evaluator-private truth.

## Receipt

The safe receipt is stored at:

`outputs/study-locks/provenance_cascade_hg_outcome_replay_technical_amendment_receipt.json`

It binds the existing batch record, H-G config, protocol, public-identifiability amendment, approval, compatibility receipt, scenario material hashes, and both validator file hashes. It stores counts and replay statuses only. It does not contain prompts, model replies, API keys, provider metadata, claim text, or evaluator-private labels.

The receipt was created against twelve completed run records and the failed run's eighteen completed logical request fingerprints. A read-only reconstruction from the saved checkpoint passes cascade, application, and H-G outcome replay. Because all eighteen coordinates are already checkpointed and completed in the append-only ledger, resuming that run requires zero Provider calls for those coordinates.

## Resume Gate

Resume preflight is offline and now requires both the technical amendment receipt and the current H-G validator hash:

```bash
uv run python -m evicon.provenance_cascade_hg_pilot --mode preflight --resume
```

After a completed resume, the receipt remains bound to the historical pre-resume batch hash while the current completed batch is validated separately. If the receipt, validator, current batch design binding, config, protocol, approval, compatibility receipt, scenario bindings, checkpoint, or ledger coverage changes, the preflight is blocked. The preflight does not construct a Provider, read an API key, issue a request, or alter existing Pilot artifacts.

An actual continuation remains a separate, explicitly authorized network operation. This receipt is development-only, not a paper result, and supports no causal conclusion.
