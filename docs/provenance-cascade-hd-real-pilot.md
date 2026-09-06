# Provenance-Cascade H-D Real Pilot Execution

This execution path is an opt-in batch sidecar around `CascadeRealAgentRunner`.
It does not change the H-C runner, controller, application, replay, or WVS
contracts.

The execution authorization binds the approved H-D v2 config, researcher design
approval, schedule amendment, technical receipt, four scenarios, 48 explicit run
coordinates, and the fixed caps of 864 logical requests and 221,184 reserved
completion tokens. Completion reservation is neither a total-token limit nor a
price guarantee.

Network execution requires all of `--allow-network`, `--confirm-run`,
`--confirm-request-cap 864`, and
`--confirm-completion-reservation-cap 221184`. Before these checks pass, the CLI
does not construct a provider and does not create the results root. Provider
credentials remain environment-only and are never serialized.

Each run owns an append-only `request_ledger.jsonl`, a parsed public-response
checkpoint, and a safe run record. A first failure stops all later runs. Explicit
`--resume` reuses checkpointed successful coordinates, retries only ledger states
already permitted by `LedgeredProvider`, and skips completed runs. A changed
config, approval, authorization, model, scenario binding, seed, condition, or
generation contract changes the batch/checkpoint binding and blocks reuse.

The batch receipt is produced only after all 48 runs complete and all cascade,
application, and outcome replay reports pass. It contains aggregate request and
token accounting but no prompts, raw model replies, provider metadata, private
truth, or credentials. This remains a development pilot and cannot support a
paper or causal conclusion by itself.
