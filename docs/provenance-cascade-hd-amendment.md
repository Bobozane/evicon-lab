# Provenance-Cascade H-D Research Design Amendment

## Status and lineage

`provenance_cascade_hb_amendment.v2` is a proposed, development-only amendment
to the historical H-B v1 preflight. H-B remains unchanged and auditable. The
amendment is technically validated but remains `pending_human_approval`; it is
not permission to run a networked pilot.

The 24G.1 calibration materials are excluded. They established only that the
controller/application mechanism was triggerable. They are neither copied
into this pilot nor eligible for pilot comparison or later paper analysis.

## What changes

Only the public observability schedule for `cascade-false-majority` changes.
The v2 public graph adds one synthetic round-0 repost node that resolves to the
existing `root-fm`. The existing claim, public verification status, evidence
card, source root, evaluator boundary, controller policy, metrics, seeds,
topology, Agent order, and generation limits do not change. The other three
formal scenarios retain their H-B scenario and graph hashes.

At round 0, the system delivers `fm-content-0` and
`fm-content-repeat-observation-v2` to `network-agent-01`. They are not available
in that Agent's round-0 start snapshot. At the round-1 start snapshot, that
Agent has actually seen two public content IDs but only one public root. The
provenance-aware controller may therefore create a proposal after round 1 with
`visible_unverified_same_root_repetition`. An accepted schedule is effective
only in round 2. It cannot affect the round-1 snapshot, Runtime input, outcome,
or propagation that created it.

This change does not assert that the claim is true or false and introduces no
evaluator label to an Agent or controller. Two reposts remain one independent
root; Agent count is never treated as source independence.

## Protection cases

The unchanged `true_minority_correction` material retains its supported,
independent correction and no application action can hide, suppress, remove,
or downgrade it. `independent_true_consensus` and `unresolved_disagreement`
retain their policy-level abstain protections. The latter receives no public
truth verdict.

## Matched plan and costs

The H-D plan retains four scenarios, four conditions, three seeds, six Agents,
three rounds, and `ring_6_bidirectional`: 48 runs in 12 matched groups. It
reserves 18 logical requests and 4,608 completion tokens per run, for 864
logical requests and 221,184 completion tokens in total. Completion
reservation is not actual total-token usage or a price cap.

The offline FakeProvider smoke runs in a temporary directory. It verifies
cascade/application/outcome replay, request-ledger safety, next-round timing,
and protected abstention. It is not an intervention-effect result.

## Approval and execution boundary

The H-D preflight returns `ready_for_human_approval`, never
`ready_for_real_pilot`. A future approval artifact must bind the amendment,
plan, scenario, graph, exposure, policy, and output hashes. Real execution must
then require a separate network gate, explicit request/token confirmation,
Provider/model confirmation, retry/timeout confirmation, and a non-existing
result root. Changing this design requires another versioned amendment rather
than editing this one.

## Safe amendment receipt

The completed offline smoke is bound by
`outputs/study-locks/provenance_cascade_pilot_hd_amendment_receipt.json`. The
receipt stores hashes, counts, safety flags, and technical replay status only.
It does not contain prompts, Agent messages, evaluator labels, provider
metadata, credentials, or model outputs, and it explicitly remains pending
human approval.

## Researcher design approval and final offline preflight

The researcher approval is recorded separately in
`configs/provenance_cascade/pilot/provenance_cascade_pilot_hd_approval_2026-08-21.toml`.
It binds the amendment, pilot config, technical receipt, four scenario configs,
public provenance graphs, exposure ledgers, request cap, completion reservation,
Provider contract, retry/timeout settings, append-only ledger, resume semantics,
and no-overwrite rule.

This approval does not authorize a network request. The final offline preflight
returns `passed_waiting_for_network_authorization`, with
`network_execution_authorized=false` and `ready_for_execution=false`. It does
not inspect environment variables or API-key values, construct a Provider, or
create the planned result directory. A later command must require separate
explicit network and run confirmations before any real request.
