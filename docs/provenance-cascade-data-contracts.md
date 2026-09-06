# Provenance Cascade Data Contracts

This stage defines only offline data boundaries. It does not implement an
agent runner, controller, provider, or intervention strategy.

## Public graph

`Claim` exposes a stable ID, a short public summary, a four-valued
`VerificationStatus` (`unverified`, `supported`, `refuted`, or `contested`),
and the evidence cards that may be cited. `SourceRoot` exposes a stable root ID,
a coarse public source category, and allowed evidence-card IDs. `ProvenanceNode`
binds public content to a claim, source root, scenario, and round. Directed
`ProvenanceEdge` records `originates`, `repost`, `reply`, or `quotes_evidence`.

`ProvenanceGraph` rejects cycles, unknown IDs, cross-claim or cross-scenario
edges, time reversal, future evidence citations, and inconsistent source roots.
An edge connecting nodes with different claimed roots is rejected: this avoids
letting a shared upstream parent masquerade as independent evidence. Root
resolution deduplicates repeated propagation, so three reposts from one root
count as one root.

## Private evaluator boundary

`EvaluatorTruthRecord` is loaded by a separate loader and contains only a claim
ID, ground-truth label, source-independence label, review metadata, and rationale.
It is never nested in a public graph, evidence card, controller view, event, or
prompt. Offline validation joins public and private files only by stable
`scenario_id` and `claim_id`; the CLI reports counts and hashes but never labels
or source text.

The public graph is the only future controller input for source-root reasoning.
Models and runtime code cannot create an evaluator-provided “independent” flag.
Four synthetic fixtures cover false-majority repetition, supported minority
correction, independent true consensus, and unresolved disagreement. They are
development-only authored material and do not use WVS data.

Validate all local fixtures with:

```bash
uv run python -m evicon.validate_provenance_cascade
```

The next stage may define exposure events and audit records. It must preserve
this public/private separation and still keep evaluator truth offline.

## Exposure Ledger And Replay

Stage 24C adds `ExposureEvent`, immutable `CascadeExposureSnapshot`, `AgentTimeline`, and `ExposureLedger`. An event is recorded at the end of round `r`; a snapshot for round `r` contains only that agent's events from rounds strictly before `r`. Snapshot IDs, captured event IDs, visible content, evidence, provenance nodes, and root relations are checked against the ledger, so a later historical insertion is rejected and a same-round event cannot alter the existing snapshot.

`ControllerPublicView` is constructed only from one validated snapshot and contains only exposed claims, exposed evidence cards, exposed provenance nodes, and root ID/category summaries for those visible nodes. It omits root evidence-card lists and unseen ancestor content, so root relationships remain audit metadata rather than propagation authorization. It cannot enumerate the remaining global graph and rejects evaluator labels or other unknown fields. `CascadeReplayValidator` is a separate read-only validator; it does not alter the historical WVS `ReplayValidator` and never loads evaluator-private fixtures.

Run the local smoke audit with:

```bash
uv run python -m evicon.cascade_exposure_smoke --fixture-dir configs/provenance_cascade/fixtures
```


## 24E pure controller proposal boundary

`CascadeControllerPolicyConfig` freezes the four preregistered condition paths.
`CascadeInterventionProposal` is immutable and records only stable public reason
codes plus current-view content, evidence, and (provenance-aware only) root IDs.
It contains no execution command, prompt, message, budget operation, exposure
mutation, evaluator annotation, or truth label. Proposal validation is separate
from any future action-application validator. See `provenance-cascade-controller.md`.
