# Provenance-Cascade H-G Identifiability Amendment

H-G is a versioned, development-only correction to the public input and behavior-observation contract. It does not resume, overwrite, or merge the H-D.2.1 Pilot. The H-D.2.1 batch remains intact and is registered separately as `non_identifiable_pilot`, `not_paper_result=true`, and `no_causal_conclusion=true`.

## Diagnosed defect

The pre-H-G cascade prompt projection preserved public IDs but lost their public semantics. `Claim.public_summary` existed in the public graph, yet the controller view did not carry it, and the old Agent context used `claim_id` and `content_id` as the corresponding summaries. A real Agent therefore received stable audit coordinates without an understandable synthetic claim or message statement.

The earlier outcome observation also selected the first round-zero `no_position` event as the initial stance. That pre-exposure placeholder obscured later substantive transitions. In addition, the earlier correction scenario did not represent the rumor and its supported correction as separately observable claims. These are identifiability defects, not adverse findings about intervention effectiveness.

## Versioned public input

H-G adds optional public text fields without changing old fixtures or runners:

- `Claim.public_summary` is projected into the H-G prompt.
- `ProvenanceNode.public_statement` carries a short, synthetic public message statement.
- Public evidence retains its existing summary plus its structured `supports` and `contradicts` relation.
- IDs and visible root relationships remain audit metadata; they do not substitute for public meaning.

Only claims, content, evidence, nodes, and root relationships in the current Agent's verified round-start `ControllerPublicView` are projected. A global graph node that was not exposed cannot be enumerated or recovered through the H-G builder. Public material rejects evaluator-truth, source-independence, private-fixture, credential, and authorization markers.

The false-majority fixture contains two distinguishable public retellings that resolve to the same root. The independent-consensus fixture contains distinguishable public statements from two roots, without exposing the evaluator's independence label. The true-minority fixture separates the public rumor claim from the independently evidenced public correction claim. The unresolved fixture retains conflicting, insufficient public accounts.

## Directive semantics

The directive remains a next-round process constraint. A `verification_request` asks the Agent to compare already visible repetition and sources; it does not state truth. A `reasoning_request` and `priority_evidence` can reference only already visible, authorized material. The H-G tests hold public claims, content, and evidence constant while showing that an applied directive can change the FakeProvider stance from endorsement to uncertainty. No directive creates a claim, evidence card, source root, or verification status.

## Behavior observation

H-G defines the initial behavioral observation as the first substantive stance other than `no_position`. This preserves round-zero no-exposure records but prevents them from being mistaken for a measured initial position. The evaluator-only truth fixture is used only by the offline smoke to count whether the redesigned fixtures contain eligible units. It never enters an Agent context, controller view, directive, public ledger, audit receipt, or preflight output.

The 48-run FakeProvider smoke checks that eligible units now exist and that replay remains valid. Its response rules are deterministic engineering fixtures. Their transition counts are not estimates of model behavior or intervention effects.

## Frozen boundaries

The four scenario types, four conditions, three seeds, six Agents, three rounds, ring topology, controller thresholds, preregistered metrics, request cap, and token reservation remain unchanged. The limits remain 864 logical requests and 442,368 completion-token reservation units. H-G uses new scenario IDs, material hashes, protocol/template versions, run IDs, fingerprints, and output root `results/provenance-cascade-pilot-hg-v1`.

No prompt or full response is written to audit, ledger, receipt, or report. H-D.2.1 and H-G data must not be merged. The H-G research design is now accepted, but final preflight remains blocked until a new one-shot Provider compatibility receipt is bound to the H-G config, protocol, template, and amendment. A later real Pilot would still require an independent explicit network and budget authorization.
