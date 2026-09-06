# H-G Outcome Replay Contract Fix

## Scope

This is an implementation repair for the H-G public-content-identifiable Pilot. It does not change the H-G prompt, strict JSON schema, scenario material, public exposure schedule, controller, directive semantics, model parameters, seeds, metrics, evaluator truth, request fingerprints, or request/token caps.

The historical `CascadeOutcomeReplayValidator` remains unchanged. H-G now opts into `cascade_hg_outcome_replay.v2.visible_cross_claim_context` through an explicitly injected `HGOutcomeReplayValidator`.

## Repaired invariant

An H-G response has one stance attached to the current target claim, but it may cite any content that was actually visible in the same immutable round-start snapshot. This is necessary when an Agent rejects an unverified rumor by citing a separately represented, publicly visible correction claim.

The H-G validator still rejects:

- content or evidence absent from the exact round-start snapshot;
- unknown or future content;
- content whose claim was not visible to the Agent;
- unknown, unrelated, unauthorized, or future evidence;
- scenario, Agent-set, cascade replay, or application replay mismatch.

Cross-claim visibility is not a truth label and does not permit global graph access. Evaluator-private truth and source-independence labels remain unavailable to the Agent, controller, application layer, replay output, and public records.

## Failure diagnosis

The failed coordinate was `hg-cascade-hg-true-minority-correction-20260911-no_intervention`. All 18 Provider responses had completed and were safely parsed. The Agent rejected the rumor while citing the visible correction content and the visible public schedule evidence. Cascade and application replay passed; only the old same-claim outcome-reference check failed.

A read-only reconstruction of the saved checkpoint passes all three replay layers with the H-G validator. An isolated copy also resumes with zero Provider calls, proving that completed fingerprints are not replayed. The original batch, ledger, checkpoint, completed run records, and compatibility artifacts are not modified by this repair.

A future explicit `--resume` continues the existing output root. It must not use parser recovery and must retain all original config, approval, compatibility receipt, model, seed, condition, hash, ledger, and no-overwrite bindings.

## Resume gate

The default preflight continues to reject an existing output root. Only an explicit resume preflight permits the existing directory while retaining all receipt and hash checks:

```bash
uv run python -m evicon.provenance_cascade_hg_pilot --mode preflight --resume
```

A future continuation requires the original real-run confirmations plus `--resume`. Existing completed run records are loaded without Provider construction. The failed run's completed checkpoint coordinates are reused without transport calls; subsequent incomplete runs may call the Provider only after explicit network authorization.
