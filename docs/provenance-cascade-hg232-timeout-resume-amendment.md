# H-G.2.3.2 timeout-resume technical amendment

The original H-G.2.3.2 eligibility calibration stopped after seven completed runs. The eighth run, `hg232-cascade-hg1-true-minority-correction-20261031-provenance_aware_controller`, contains eight completed logical requests followed by one terminal `timeout` for the fingerprint `5f2a4650dad79abe1a2ff28f57092228f892d5f9d09248278eced91c1b1c747f` at Agent `network-agent-03`, round 1. This diagnosis uses only content-free request-ledger coordinates and does not inspect prompt or response text.

This amendment changes no prompt, schema, model, seed, condition, scenario, controller, metric, request fingerprint, request cap, or completion reservation. It authorizes only an explicit resume path that preserves the original batch binding and historical runner hashes. The existing seven run records are replayed locally and never rerun. Completed fingerprints are rejected by the append-only request ledger. The sole existing failed fingerprint may receive a new transport attempt only under explicit `--resume`; subsequent planned requests proceed in the pre-registered order.

The logical request cap remains 288 and the completion reservation remains 589,824. Existing batch, ledger, checkpoint, response-audit, behavior-decision, and run-record bytes are not rewritten by preparation or preflight. Runtime resume may append new request-ledger/audit/checkpoint state and update the batch record through the already approved atomic runner behavior.

This is an engineering recovery amendment. It does not permit parser recovery, response repair, permissive parsing, replay of a completed fingerprint, or a change to the experiment. Parser-invalid, `finish_reason=length`, and token-limit events remain fail-fast. The batch remains development-only, not a paper result, and not a causal conclusion.
# H-G.2.3.2 timeout-resume technical amendment

The H-G.2.3.2 eligibility calibration stopped after seven completed runs. In the eighth run, `true_minority_correction × provenance_aware_controller`, the ninth logical Agent request ended with the stable transport error `timeout`. The failed coordinate is Agent `network-agent-03`, round 1, fingerprint `5f2a4650dad79abe1a2ff28f57092228f892d5f9d09248278eced91c1b1c747f`.

This amendment changes no prompt, schema, model, seed, scenario, condition, controller, metric, request cap, or completion reservation. It restores and binds the approved historical calibration runner and stability-probe runner hashes. Existing batch, request-ledger, checkpoint, behavior-decision, response-audit, and seven completed run records remain in place. Ledger lines remain append-only.

One explicit resume invocation may retry the single timeout-failed logical fingerprint once through the standard transport resume path, then continue previously unstarted coordinates. The eight completed fingerprints in the failed run and every completed run are checkpoint-replayed locally with zero Provider calls. Completed fingerprints may not be replayed. The logical request cap remains 288; the retry increases transport attempts, not unique logical requests. Completion reservation remains 589,824.

This is not parser recovery or semantic recovery. Parser-invalid, `finish_reason=length`, or token-limit responses still stop immediately. If the authorized timeout fingerprint fails again, or any bound input changes before resume, this amendment cannot be reused and no complete receipt is produced.

The amendment is development-only and supports no effectiveness, causal, or paper conclusion. Preparing and validating it is offline. Network resume requires a separate exact-hash approval after the amendment receipt and resume runner are finalized.
