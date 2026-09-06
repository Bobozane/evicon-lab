# Provenance-Cascade Agent Runtime

This is an opt-in, one-call boundary for a future real Agent. It is isolated from CascadeOutcomeRunner, deterministic actors, controllers, application, and WVS runners. This stage is offline engineering validation only: it does not run a real model, create results, or claim intervention effects.

## Boundary

CascadeAgentRuntime accepts a validated ControllerPublicView, target claim, optional already-applied StructuredCascadeDirective, frozen non-secret runtime settings, and an injected LLMProvider. Context construction projects only currently exposed claims, content, evidence, and public root summaries. It never accepts evaluator truth, private fixtures, a global graph, an exposure ledger, budget, cooldown, future data, or hidden probes.

The fixed sequence is: validated context -> cascade_agent_turn.v1 renderer -> one provider.complete call -> strict JSON parser -> safe audit summary. The runtime does not create events, nodes, outcomes, files, schedules, or ledger mutations. Provider retries remain provider-owned; runtime execute makes at most one call.

## Directive semantics

The renderer treats directives as process constraints, never facts or new evidence. verification_request asks the Agent to distinguish repeated public material from an independent source and permits uncertainty. reasoning_request asks it to compare only already visible evidence. priority_evidence asks it to review already authorized evidence first. With no directive, no synthetic control message is injected. Directives cannot add claims, evidence, roots, verification status, unseen content, or change exposure/order.

The response is one JSON object containing only stance, content_ids_used, evidence_ids_used, and nullable share_content_id. Stance is a public position, not a factual verdict. IDs must be subsets of the current visible snapshot. Control fields, private labels, malformed JSON, duplicate keys, and unavailable IDs are rejected with stable redacted codes.

## Audit and next gate

The audit summary contains only status, template/model/request IDs, token usage, latency, parse status, stable error code, and permitted-ID counts. It never stores prompts, complete responses, public message text, provider metadata, credentials, evaluator truth, hidden probe/profile, or private paths. The local FakeProvider smoke is not confirmatory evidence. Before 24H-B, require an approved amendment/fixture decision, scenario and contract hashes, offline preflight, explicit network and request/token budget confirmations, bounded retry/resume rules, and a private-truth leakage audit. No real Agent, batch pilot, or causal conclusion is implemented here.
