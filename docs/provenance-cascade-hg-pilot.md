# Provenance-Cascade H-G Pilot Execution Gate

The H-G Pilot is an isolated, opt-in execution path for the public-content-identifiable provenance-cascade protocol. It does not replace or resume H-D.2.1, and it does not modify WVS, 002/003, calibration, or H-F analysis artifacts.

## Frozen scope

The registered design contains four synthetic scenarios, four conditions, three seeds, six Agents, three rounds, 48 runs, 12 matched groups, and 864 logical Agent requests. The completion reservation is 442,368 tokens. Agent requests use `cascade_agent_turn.v2_1.public_identifiable`, strict `json_schema`, schema `cascade_agent_response_v2_1`, `max_tokens=512`, temperature `0.2`, `max_retries=1`, and timeout 15 seconds.

Each run owns an append-only request ledger and checkpoint. A completed logical request fingerprint cannot be replayed. Resume is accepted only when config, protocol, model, scenario, seed, condition, run plan, approval, compatibility receipt, amendment, and output-root bindings are unchanged. Parser-invalid responses stop the run; H-G has no semantic parser recovery path.

## Offline checks

The full FakeProvider path uses a temporary directory and writes no formal Pilot result:

```bash
uv run python -m evicon.provenance_cascade_hg_pilot --mode fake-smoke
```

The final preflight is also offline. It does not construct a Provider, read an API key, or create `results/`:

```bash
uv run python -m evicon.provenance_cascade_hg_pilot \
  --mode preflight \
  --config configs/provenance_cascade/pilot/provenance_cascade_pilot_hg.v1.toml \
  --approval configs/provenance_cascade/pilot/provenance_cascade_pilot_hg_approval_template.toml
```

It can report `ready_for_real_pilot` only after the accepted approval is bound to a completed, parser-valid one-shot compatibility receipt and the formal output root does not exist.

## Future real invocation

A later, separately authorized real Pilot would use:

```bash
uv run python -m evicon.provenance_cascade_hg_pilot \
  --mode real-pilot \
  --config configs/provenance_cascade/pilot/provenance_cascade_pilot_hg.v1.toml \
  --approval configs/provenance_cascade/pilot/provenance_cascade_pilot_hg_approval_template.toml \
  --allow-network \
  --confirm-run \
  --confirm-request-cap 864 \
  --confirm-completion-reservation-cap 442368
```

For an interrupted execution, the same bindings and explicit `--resume` are required. The command is invalid before compatibility receipt registration and separate network authorization. It must never be run merely because offline preflight passes.

No prompt, complete model response, API key, Provider metadata, evaluator-private truth, or private source-independence label is written to the Pilot ledger, run record, batch record, receipt, or CLI summary. All artifacts retain `development_only`, `pilot_only`, `not_paper_result`, and `no_causal_conclusion` boundaries.

## H-G outcome replay repair

The H-G execution path uses the isolated `cascade_hg_outcome_replay.v2.visible_cross_claim_context` validator. This aligns replay with the frozen Agent response contract: a stance on one claim may cite a different claim's content only when that content was actually present in the same immutable round-start snapshot. The historical outcome validator remains unchanged. See `docs/provenance-cascade-hg-outcome-replay-fix.md` for the exact retained safety checks.
