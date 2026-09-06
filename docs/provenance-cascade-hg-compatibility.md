# Provenance-Cascade H-G Compatibility Gate

This gate is separate from the H-G Pilot runner. It validates only whether one OpenAI-compatible endpoint can accept the H-G strict JSON Schema request contract. It does not run a Pilot, use a Pilot scenario, load evaluator-private truth, write a request ledger, or create a result directory.

## Accepted design

The researcher approval binds H-G config SHA-256 `cd335f5e575b0fcc760cc21efc07bb948b3ebf23965e8d3d3256ea32e038a8ed`, protocol/template SHA-256 `eea59e37e41804de063144602f44404abaa1a635007e7a7d97346eab9cda8c3a`, amendment SHA-256 `f840925cd46a27741c2245e011acf68725b4654c85a2a9ce899d69cb4b34e49a`, and amendment receipt SHA-256 `666c33830e2b79f6afe884b3557b8429d96cdf876a3ddd5747f9c51b75f8ed00`.

It also accepts 48 runs, 864 logical requests, 442,368 completion reservation units, strict `json_schema`, schema `cascade_agent_response_v2_1`, 512 Agent completion tokens, one Pilot retry, a 15-second Pilot timeout, no overwrite, append-only request ledgers, and explicit resume semantics. This design approval is not network authorization.

## One-shot check

The default command is offline and does not read Provider environment variables:

```bash
uv run python -m evicon.cascade_hg_compatibility
```

A future explicitly authorized check is:

```bash
uv run python -m evicon.cascade_hg_compatibility --allow-network
```

The opt-in path sends at most one request with `max_retries=0`, timeout 5 seconds, `max_tokens=512`, temperature `0.2`, seed `20260911`, `response_format=json_schema`, and schema `cascade_agent_response_v2_1`. It uses a minimal synthetic public library-schedule context. It does not load any H-G scenario, Pilot prompt, WVS material, evaluator truth, private fixture, or historical result.

CLI output is restricted to status, stable error category, status class, model, finish reason, parser validity, token usage, latency, response contract, attempt count, network state, and safety flags. Prompt text, complete response, headers, request ID, credentials, Provider metadata, and private labels are never emitted or stored.

## Receipt registration

After a successful one-shot result has been reviewed, save only that safe JSON output to a temporary local file and register it explicitly:

```bash
uv run python -m evicon.cascade_hg_compatibility_receipt \
  --register-result /tmp/hg_compatibility_result.json \
  --amendment-sha256 f840925cd46a27741c2245e011acf68725b4654c85a2a9ce899d69cb4b34e49a
```

Registration is accepted only for a completed, parser-valid, `finish_reason=stop`, HTTP 2xx, one-attempt result with consistent token usage. It writes only `outputs/study-locks/provenance_cascade_hg_compatibility_receipt.json` and binds that file's SHA-256 into the already accepted H-G approval. Existing receipts are never overwritten.

The final preflight remains offline:

```bash
uv run python -m evicon.validate_provenance_cascade_identifiability \
  --config configs/provenance_cascade/pilot/provenance_cascade_pilot_hg.v1.toml
```

It returns `ready_for_real_pilot=true` only when config, protocol, amendment, amendment receipt, accepted approval, compatibility receipt, and approval receipt hash all match, and the H-G result root does not exist. Even then, a separate explicit Pilot network and budget authorization is required.
