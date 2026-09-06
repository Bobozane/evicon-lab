# Provenance-Cascade H-C: Opt-In Agent Integration

H-C adds `CascadeRealAgentRunner`, an isolated development-only path that
connects the existing `CascadeAgentRuntime` to the public cascade protocol.
It does not change `CascadeProtocolRunner`, `CascadeOutcomeRunner`, any WVS
component, or an existing result artifact.

## Round Boundary

For each agent and round, the runner performs:

1. builds the immutable round-start `CascadeExposureSnapshot`;
2. consumes only a directive already applied for that next round;
3. projects that one snapshot into `ControllerPublicView` and
   `CascadeAgentPromptContext`;
4. renders one request, calls the injected provider once, and strictly parses
   a public response;
5. writes a `PublicClaimOutcome` using only parsed stance and visible IDs;
6. permits a predeclared propagation slot only when `share_content_id` is its
   already visible parent/content; and
7. produces proposals after the round. A proposal can only schedule a
   directive for the following round.

The LLM cannot create a claim, provenance node, root, evidence card, parent,
or directive. It cannot access a parent merely because it can see a root
relationship. The runner never treats a directive as evidence or truth.

At a coordinate with no exposed claim, the runtime receives an empty public
context and the runner emits `no_position`. This preserves the fixed H-B
18-agent-request run scope without exposing a claim before the exposure ledger
allows it.

## Safety and Recovery

`RequestLedger` remains append-only and content-free. It records hashes,
coordinates, status, attempts, known usage, latency, and stable error codes;
it never records prompts, raw provider content, API keys, or provider
metadata. The injected provider owns its bounded transport retry policy.

A local `CascadeAgentCheckpoint` stores only a successfully parsed public
stance plus permitted content/evidence IDs. It is bound to scenario hash,
seed, condition, run ID, model, temperature, and token limit. Under explicit
`resume`, matching completed checkpoints avoid replaying completed logical
requests; failed coordinates are retried through the ledger. A parser-invalid
transport response does not become a completed checkpoint and is not replayed
by default. Any future parser-recovery exception needs an explicit, separately
audited amendment.

## FakeProvider Scope

`cascade_agent_integration_smoke` runs the H-B 48-run plan in a temporary
folder with a local FakeProvider: four scenarios, four conditions, three
seeds, six agents, and three rounds (864 logical Agent requests). It is not a
real pilot and does not create formal result artifacts.

The historical H-B `false_majority` schedule does not place two same-root
items in one target's proposal-time view, so its provenance-aware directives
remain zero. The separately versioned 24G.1 calibration fixture demonstrates
that the exact same controller/application chain can produce and apply one
next-round `verification_request`. It is calibration-only and cannot enter
matched pilot analysis.

Before real-pilot execution, an approved amendment must version a
separation-capable pilot scenario/schedule and update its hashes and
preregistration/preflight binding. H-C makes no claim about intervention
performance, causality, or real social systems.
