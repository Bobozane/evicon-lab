# H-G.2.2 Integrity Audit and Evaluator-Only Analysis

H-G.2.2 is a one-seed, development-only eligibility calibration. It is not an effectiveness experiment, a confirmatory study, or a paper result.

## Resume-cap technical amendment

The real batch began before the shared request-ledger guard was corrected. The original guard counted a failed fingerprint as occupying the final logical slot and then rejected an explicit retry of that same fingerprint. The amendment changes only cap accounting: retrying an already registered failed fingerprint does not allocate a second logical coordinate or completion reservation. It still appends another transport attempt. New fingerprints remain subject to the original cap.

The change does not alter rendered requests, fingerprints, prompts, response schema, parser, scenario, condition, seed, controller, metrics, Agent outputs, checkpoints, or replay rules. Completed fingerprints remain non-replayable. Parser-invalid responses remain non-recoverable.

The technical receipt records the reconstructed pre-fix implementation hash and the current post-fix hash, plus the implementation timestamp relative to the append-only ledger. No pre-fix snapshot hash of the complete evolving ledger was captured, so the receipt records this limitation explicitly. Current ledgers reconstruct 288 unique logical requests, 292 transport attempts, four failed transport attempts followed by four same-fingerprint completions, and no duplicate completed fingerprint.

## Analysis boundary

The evaluator validates all public hashes, run coordinates, request ledgers, checkpoints, parsed decision records, and cascade/application/outcome replay before loading evaluator-private truth. Private labels are joined only in memory by stable scenario and claim IDs. They are never written to the analysis manifest or report.

The report contains the nine preregistered metrics, eligible denominators, N/A warnings, epistemic and behavioral decision distributions, sharing counts, proposal/application counts, replay results, token/latency aggregates, and recovery counts. It does not define new thresholds, run significance tests, make causal claims, or rank controllers as effective.

Because there is one seed, any observed condition difference is descriptive calibration evidence only. The calibration can inform an engineering decision about whether a later pilot is identifiable, but cannot establish a behavioral mechanism or research conclusion.
