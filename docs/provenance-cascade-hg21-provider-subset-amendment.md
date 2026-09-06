# H-G.2.1 Provider-subset schema amendment

The accepted H-G.2 one-shot compatibility request returned a safe `4xx response_format_unsupported` classification before inference. No compatibility receipt or calibration result was created.

H-G.2.1 changes only the JSON Schema sent to the Provider: it removes `uniqueItems` from the two identifier arrays because the configured intermediary appears to implement a restricted JSON Schema subset. The local strict parser remains unchanged and still rejects duplicate content or evidence identifiers. The five output fields, enums, prompt text, public input boundary, scenario design, controller rules, seed, metrics, request cap, and completion reservation are unchanged.

The transport protocol, schema name, config, request identity, hashes, approval, receipt, and future output root are versioned independently. H-G.2 remains immutable and its failed compatibility attempt is not treated as a Pilot result. There is no fallback to `json_object`, permissive parsing, response repair, or parser recovery.

This amendment is development-only and calibration-only. It does not claim intervention effectiveness, authorize the real eligibility calibration, or authorize a full Pilot. A new one-shot compatibility check and an exact-hash researcher approval are required before any network calibration.
