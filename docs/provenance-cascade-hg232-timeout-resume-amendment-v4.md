# H-G.2.3.2 timeout-resume technical amendment v4

The H-G.2.3.2 eligibility calibration stopped after seven completed runs. The eighth run has eight checkpointed logical requests and one terminal `timeout` at Agent `network-agent-03`, round 1. The failed logical fingerprint is locked in the amendment receipt. This diagnosis uses only safe ledger coordinates and stable error categories.

This amendment changes no prompt, response schema, model, seed, scenario, condition, controller, metric, logical request cap, or completion reservation. It locks the original batch record, failed-run ledger, checkpoint, response audit, seven completed run records, historical calibration runner, and stability-probe runner by SHA-256.

One separately authorized resume invocation may retry only the existing timeout-failed logical fingerprint, then continue previously unstarted coordinates in the registered order. The retry is an additional transport attempt for an existing logical coordinate; it does not increase the 288 logical-request cap. The completion reservation remains 589,824. Provider-level retry behavior remains the approved finite setting.

The eight completed fingerprints in the partial run and every completed run remain protected by checkpoint and ledger semantics and must not be replayed. Ledger updates are append-only. Preparation, receipt generation, and preflight are offline and do not modify the batch or any run artifact.

This is not parser or semantic recovery. Parser-invalid, `finish_reason=length`, and token-limit outcomes remain fail-fast and cannot be repaired or retried under this amendment. A repeated timeout or any changed binding requires a new amendment.

Technical-design acceptance and network-resume authorization are separate gates. This amendment can be accepted and hash-locked while network authorization remains false. The batch is development-only, not a paper result, and supports no causal conclusion.
