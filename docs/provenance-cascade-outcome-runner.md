# Provenance Cascade Outcome Runner

Stage 24G adds an opt-in, deterministic public-outcome path. It is separate from the historical `CascadeProtocolRunner`, which remains the no-intervention baseline, and it does not call an LLM or a provider.

## Round contract

For every round, the runner first freezes one `CascadeExposureSnapshot` per Agent. The deterministic actor receives only its own `ControllerPublicView` and an optional structured directive that was applied for that round. All actor decisions and public outcomes are created before any same-round propagation event is appended. A proposal made after a round can therefore only schedule a directive for the next round.

A directive is metadata, not evidence. `verification_request` asks the actor to compare already visible public sources; `reasoning_request` asks it to reason over already visible evidence; `priority_evidence` changes the order of already authorized evidence. None can add a claim, source root, verification status, evidence card, or hidden content.

## Actors and outcomes

`repeat_actor` repeats a visible claim without inventing content. `verification_responsive_actor` may become uncertain or use already visible evidence when the structured directive permits it. Both are pure deterministic functions. `PublicClaimOutcome` records only the public stance and IDs actually visible in that round. It is not a truth label.

`CascadeOutcomeRunRecord` stores safe hashes, counts, public ledgers, replay reports and development safety flags. It does not store prompts, model output, evaluator truth, credentials or provider metadata. The smoke command reads only local synthetic scenario/preregistration files and writes nothing.

The four scenarios remain synthetic development fixtures. A passing replay confirms ordering and visibility contracts, not intervention benefit, causal effect or any claim about real public opinion.

## Future 24H input

A future real pilot would need an explicit preflight that binds the scenario/config hashes, actor schedule, condition, seed, request/cost budget and a separate run directory. It would also need an explicit decision about how real Agent outputs are converted to the public outcome contract, plus a second replay gate before any offline evaluator is allowed to inspect the run.
