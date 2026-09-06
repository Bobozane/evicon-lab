# Source-Behavior Qualification v3

Version 3 is a development-only repair for the non-identifiable v2 behavior
qualification. It does not modify, resume, overwrite, merge, or reinterpret the
completed v2 requests, audits, receipts, or descriptive summary. Version 2
remains evidence that the transport, strict parser, safe audit, and hash-bound
analysis path operated correctly; its constant behavior is not evidence for or
against a provenance effect.

## Diagnosed v2 limitation

The v2 case builder selected only `repeat_a` and `repeat_b`. The scenario's
initial material, correction, evidence, and scenario-specific decision task did
not enter the behavior request. All four nominal scenarios therefore reduced to
two aligned public messages plus an underspecified binary adoption/share task.
All 12 valid replies selected adoption and sharing. This is a measurement
identifiability failure, not an aggregation failure and not a null effect.

## Isolated v3 repair

V3 uses four new, author-original synthetic cases. Every public request contains:

- a literal target claim;
- a scenario-specific reversible decision task;
- two focal messages whose text is fixed across source projections;
- two fixed public counter, caveat, or record messages;
- public root assignments for every visible message.

The projection changes only whether the two focal messages show no provenance,
one shared opaque root, or two opaque roots. `null` explicitly means provenance
is not shown, not that the message has no source. The prompt permits use of the
supplied public root relation to distinguish repetition from corroboration, but
states that roots alone do not establish truth or source quality.

The two fixed counter/record messages always retain two distinct opaque roots.
Only the focal pair changes across projections, so the correction and record
context remains independently traceable in every case.

Internal scenario, case, projection, seed, and order labels are absent from the
model prompt. Public content IDs are `content-01` through `content-04`, and root
IDs are `root-01` and `root-02`; none encodes a condition or expected answer.

## Measurement surface

The strict v3 response records a five-level ordinal judgment, consistent public
action, three-level sharing decision, confidence from 1 to 7, visible content
references, and an optional shared content ID. A safe audit persists only opaque
case/scenario coordinates, projection, seed, order position, parsed behavior
categories, confidence, counts, and focal root count. It never stores prompts,
public text, full responses, credentials, headers, provider metadata, private
truth, or a request ledger.

The design uses seeds `20261201`, `20261202`, and `20261203`. Projection order is
rotated as a Latin square inside each scenario, producing 36 future logical
requests. Each projection appears once in each order position per scenario.
This reduces the fixed-order confound present in v2; it does not by itself make
the future qualification a causal experiment or a paper result.

## Current gate

The configuration, public corpus, protocol, schema, and accepted human approval
are independently hash-bound. Default commands are offline:

```bash
.venv/bin/python -m evicon.conformity_source_behavior_qualification_v3
.venv/bin/python -m evicon.conformity_source_behavior_qualification_v3_smoke
```

The FakeProvider sensitivity smoke deliberately emits different valid responses
from public root counts. It verifies that the new response/audit pipeline can
retain behavioral separation; it does not estimate model behavior or validate
the scientific effect.

The approval sidecar was accepted on 2026-08-28 for this offline design only.
The separate one-shot compatibility CLI is implemented and documented in
`provenance-cascade-conformity-source-behavior-compatibility-v3.md`; its default
path is offline and its real one-request path still requires explicit researcher
authorization. A future bounded 36-case runner requires separate implementation,
receipt review, execution approval, and explicit network authorization. No real
v3 request is authorized by this document or sidecar.
