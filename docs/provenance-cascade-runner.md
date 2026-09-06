# Provenance-Cascade Protocol Runner

`CascadeProtocolRunner` is an offline, deterministic social-network harness for
synthetic provenance-cascade fixtures. It is deliberately separate from the
WVS runners and does not call an LLM, provider, controller, mediator or
network.

## Public contract

A `CascadeScenarioSpec` is loaded from a local TOML declaration and a public
JSON provenance graph. It fixes six agents, three rounds, the preregistered
six-node undirected ring, a declared actor schedule, and `no_intervention`.
The scenario loader checks graph identity and SHA-256, agent order, topology,
claim IDs, and all schedule references. Evaluator-private fixtures are not an
input to the loader.

At the start of each round the runner creates one immutable
`CascadeExposureSnapshot` per agent from events strictly earlier than that
round. System-seeded content is then delivered, and scheduled actors can emit
only a declared public provenance node. A repost, reply, or quoted-evidence action requires the actor to have seen the
actual content being propagated in the current round-start snapshot. For a new
child node, its direct parent content must also be visible and is recorded only
as a safe `parent_content_id` audit coordinate. A public provenance ancestor is
never an authorization shortcut: an unseen parent text, content ID, evidence
card, or propagation target cannot be recovered from a root relation. Events
created in a round therefore cannot enter that round's snapshots or influence
another actor until the next round.

The append-only `ExposureLedger` and the independent `CascadeReplayValidator`
provide the replay boundary. Replay requires every non-system event to cite
only evidence and content in the source agent's current snapshot. A new child
must have a directly visible parent content; an ancestor found only through a
public graph edge is rejected. The runner returns a safe `CascadeRunRecord` with
counts, root-resolution counts, a trajectory hash, and a replay report; it does
not persist output files.

## Synthetic smoke

Run the local-only smoke command:

```bash
uv run python -m evicon.cascade_protocol_smoke
```

It runs the four synthetic scenarios for three deterministic seeds and prints
only IDs, counts, hashes, root counts, and replay status. It never creates a
`results/` directory. The scenario type is a fixture label, not a claim about
real public opinion.

The four fixtures exercise distinct public graph shapes: repeated same-root
propagation, an independently sourced minority correction, independent roots
supporting a consensus, and unresolved conflicting evidence. Public
verification statuses are retained as observable data; evaluator truth labels
remain offline-only and cannot enter a snapshot, event, prompt, or controller
input.

## 24E boundary

A future controller can consume one agent's `ControllerPublicView` built from
its current snapshot. It may count exposed public roots and inspect public
verification/evidence state, but it must not receive evaluator truth. This
stage intentionally does not choose interventions, calculate primary metrics,
or make claims about causal effects.


## Stage 24E remains outside the runner

The controller introduced in Stage 24E receives a separate immutable
`ControllerPublicView` and returns a proposal only. `CascadeProtocolRunner`
continues to accept only `no_intervention`; it does not import, execute, or
schedule controller proposals. No 24E decision changes an exposure event,
actor order, provenance edge, or round-start snapshot.
