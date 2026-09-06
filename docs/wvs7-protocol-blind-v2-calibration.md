# WVS English Protocol-Blind v2 Calibration Gate

`evicon.wvs7_protocol_blind_v2_calibration` is an execution gate for the
offline preregistration. Its default path is deliberately network-disabled:
it validates the preregistration and prints a small safe receipt without
constructing a Provider, reading credentials, or creating a directory.

The opt-in path requires all of the following:

```text
--allow-network
--confirm-run
--confirm-request-cap 384
--confirm-completion-reservation-cap 51200
--confirm-estimated-total-token-reference 190602
```

The `190602` confirmation is the audited actual total-token observation from
the condition-aware v1 `002` engineering pilot. It is an observational
reference, not a total-token cap, price cap, success guarantee, or v2 result.
The separate `51200` field is the completion-reservation cap only.

The execution gate uses only the preregistered future `003` identifier and the
four fixed baseline conditions. It delegates to the existing Real Agent,
contextual probe, request ledger, recovery, replay, and offline evaluation
components. It does not introduce a Mediator, Monitor, Policy, Executor, or
online feedback from probe measurements.

On completion it verifies four completed conditions, replay status, frozen
material integrity, 384 logical request fingerprints, v2 probe templates, and
four initial/final `profile_drift` entries. Its calibration receipt separates
reserved completion capacity from actual prompt, completion, and total token
usage. Any total-token difference from `190602` is recorded only as a
direction and relative deviation.

All receipts carry `development_only`, `not_paper_result`,
`no_causal_conclusion`, and `protocol_blind_v2` markers. The v1 `002` record
is explicitly excluded from v2 main analysis and must not be pooled with v2
data. A successful calibration demonstrates only controlled engineering
execution, not a substantive research finding.

Default safe invocation:

```bash
uv run python -m evicon.wvs7_protocol_blind_v2_calibration \
  --config configs/studies/wvs7_english_protocol_blind_v2_calibration.toml
```
