# Contextual Value Probes

`OfflineMeasurementContext` reconstructs a frozen, public-only view of one
completed run for one agent and one measurement round. It includes the agent's
role and initial labels, scenario public context, that agent's earlier turns,
peer turns strictly before the round and visible to that agent, and evidence
with a recorded `EvidenceExposure` no later than that round. It rejects failed
runs and trajectories that violate their stored protocol.

Protocol limits remain in force during reconstruction. `independent` contexts
contain neither peer turns nor evidence; `social_only` contains no evidence;
and `evidence_only` contains no peer turns. The context has no fields for
`ValueProfile`, `ProbeResult`, hidden probe content, Monitor, Policy,
Mediator, budget, cooldown, or offline evaluation reports.

`ContextualProbeRuntime` accepts exactly one such context and one
`ValueProbeItem`, renders a private request, calls an injected `LLMProvider`,
and strictly parses `{"choice":"..."}`. The selected choice must exactly
match the item's declared response scale. Prompts, items, raw answers, full
provider responses, and private `ValueProbeResponse.raw_response` remain in
memory only. The runtime writes no files, events, run records, or probe result
files. Its audit summary contains only safe status, usage, and stable error
codes.

The current `contextual_value_probe.v2` renderer does not send an internal
protocol label to the model. This makes round-0 probe prompts identical across
otherwise identical conditions, so a pre-probe measures the shared initial
public context rather than anticipation of a named experimental arm. The
protocol remains local request metadata for ledger coordinates, recovery, and
offline matching; it is not transmitted to the provider. Post-probe differences
come only from the public history and evidence actually visible in the rendered
context.

The `real_probe_smoke` command is network-disabled by default:

```bash
uv run python -m evicon.real_probe_smoke
```

Its explicit `--allow-network` path is limited to one provider request with
zero retries, `max_tokens=128`, `temperature=0.0`, and
`reasoning_effort="none"`. It is an engineering boundary check, not an
experiment, and it writes no result artifacts.

These probes are offline-only measurements conditioned on a recorded public
trajectory. They never feed the online controller and do not establish human
values, model psychological state, value drift, social conformity, or causal
effects.
