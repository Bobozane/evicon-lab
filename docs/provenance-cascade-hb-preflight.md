# Provenance-Cascade H-B Pilot Preflight

## Purpose

This stage freezes an offline execution gate for a future synthetic provenance-cascade pilot. It does not run Agents, controllers, Providers, or experiments. The preflight is separate from the 24A preregistration and does not alter the 24D baseline, 24E proposal logic, 24E.1 application ledger, 24F evaluator, 24G outcome runner, or 24G.1 calibration fixture.

The manifest is `configs/provenance_cascade/pilot/provenance_cascade_pilot_hb.v1.toml`. It explicitly names four public scenario configs and graph hashes, four conditions, three seeds, twelve matched groups, and 48 stable run IDs. Each matched group has exactly one run per condition, with the fixed six-Agent order, three rounds, and `ring_6_bidirectional` topology. Pairing is never inferred from directories or result filenames.

## Scope and budget

Each planned run has 6 Agents x 3 rounds = 18 logical Agent requests. Across 4 scenarios x 4 conditions x 3 seeds this is 48 runs and 864 logical requests. The per-request reservation is 256 completion tokens, so the completion reservation is 4,608 per run and 221,184 for the planned pilot. A completion reservation is a scheduling/capacity guard; it is not actual total token usage, a price cap, or a success guarantee. Actual token counts and latency belong to a later request ledger.

Provider settings are declared without reading credentials: OpenAI-compatible, Agent temperature `0.2`, `max_tokens=256`, finite retry count `1`, and a 15 second timeout. The default network state is disabled. The preflight checks only whether model/base-url environment names are configured; it never reads the API-key value.

## CLI and network boundary

Offline validation:

```bash
uv run python -m evicon.validate_provenance_cascade_preflight \
  --config configs/provenance_cascade/pilot/provenance_cascade_pilot_hb.v1.toml
```

The command validates the 24A preregistration hash, public scenario and graph hashes, topology, run coordinates, matched groups, provider settings, caps, and result-root availability. It does not create `results/`, construct a Provider, inspect private truth, or make HTTP requests.

A future network gate must provide all of the following together: `--allow-network`, `--confirm-run`, `--confirm-request-cap 864`, and `--confirm-token-cap 221184`. In H-B, even a complete set only returns `network_gate_validated_no_execution`; network execution is deliberately deferred to H-C/I after FakeProvider integration and a separate user-approved gate. Existing output roots are never overwritten.

## Information boundary

Evaluator labels and private fixtures are not loaded. Public Agent/controller inputs remain limited to public verification state, public evidence, exposed source-root relations, and actual exposure. Majority/minority identity is not a trigger. The safety flags remain `development_only`, `pilot_only`, `not_paper_result`, `no_causal_conclusion`, `no_mediator`, and `no_online_feedback`. This manifest is a plan and preflight artifact, not a result and not a paper conclusion.

## Next gates

H-C must first connect `CascadeAgentRuntime` to a separate opt-in outcome runner under FakeProvider and pass runtime/application/cascade replay tests. Before H-I, the project still needs explicit provider environment preflight, request-ledger/resume semantics, bounded retry and output-directory checks, amendment approval for the exact pilot fixture, and user confirmation of network, request, and token caps. H-B does not authorize real calls.
