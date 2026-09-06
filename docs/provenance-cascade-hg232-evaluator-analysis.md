# H-G.2.3.2 evaluator-only analysis

This stage is a read-only integrity audit and descriptive analysis of the completed H-G.2.3.2 development calibration. It does not run an Agent, construct a Provider, use the network, or modify the calibration batch, ledgers, checkpoints, run records, or receipt.

The original evaluator-v1 artifact remains unchanged. A coordinate audit found that it classified social exposure solely from `source_actor_id != system`. The frozen calibration uses `system` as the deterministic delivery actor for some public repost/recommendation content, while its public channel and provenance edge still identify social propagation. Evaluator-v2 corrects only this offline join; it changes no registered metric or experimental input.

## Integrity boundary

The final audit binds the v4 timeout-resume runner, its technical amendment and approval, the final batch record, the calibration receipt, and every run-record hash. It reconstructs each append-only request ledger by logical fingerprint. A valid batch has 16 completed runs, four complete matched groups, 288 unique logical requests, 289 transport attempts, and one authorized timeout fingerprint with exactly two attempts. Every other fingerprint has one started/terminal pair and exactly one completed terminal. All 16 cascade, application, and outcome replays must pass.

The timeout resume is an engineering recovery only. It did not change the prompt, schema, scenario, seed, condition, controller, metric definition, or parser policy. Parser recovery remained disabled.

## Adoption-only operationalization

`false_cascade_adoption_rate` uses only the explicit `adoption_decision` value. `adopt_claim` counts as adoption; `reject_claim` and `withhold_judgment` do not. `epistemic_stance` is reported nowhere as a substitute for behavior and is not used to derive false adoption.

The report preserves denominator and N/A semantics for all nine pre-registered metrics:

- false adoption: final Agent × false-claim adoption decisions;
- supported correction retention: exposed Agent × supported-correction adoption decisions;
- beneficial receptivity: initial adopters subsequently exposed to a supported correction;
- harmful conformity: socially exposed false-claim units without independent supporting evidence;
- intervention false positives: applied non-abstain interventions;
- provenance diversity: visible root count per Agent × Claim exposure;
- replay pass rate: included runs;
- token and latency: completed provider requests with known usage.

Zero eligible units produce `not_applicable`, never an invented zero. `harmful_conformity` remains an operational, non-causal label. With one seed, the output is descriptive calibration evidence only: no significance test, Go/No-Go threshold, causal inference, or paper conclusion is produced.

The separate eligibility-funnel report follows each Agent from its round-0 false-claim decision through later supported-correction exposure and the final decision. It reports stable exclusion codes by condition, scenario, public Agent role, and round. The diagnosis is mixed because the old harmful-conformity denominator had a join defect, but the remaining primary blocker is a behavioral floor: no round-0 false-claim observation produced `adopt_claim`, even though all true-minority correction units later received correction exposure.

Private evaluator annotations are loaded only after public run hashes and replay integrity pass. Reports contain only aggregate values, safe coordinates, hashes, warnings, and safety flags. They exclude annotation labels, claim text, prompts, model replies, keys, headers, and provider metadata.
