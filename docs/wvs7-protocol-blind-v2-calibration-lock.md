# Protocol-Blind v2 Calibration Lock

The lock validator is a read-only audit over the completed `003` calibration. It checks the four completed conditions, replay status, the frozen ProbeSet hash, 384 unique logical request fingerprints, terminal completion after any recoverable failed attempt, and equality of pre-probe public-content hashes for every `agent_id x probe_id` coordinate across protocols.

The receipt under `outputs/study-locks/` contains only input SHA-256 digests, counts, protocol-blind audit flags, and development safety flags. It never stores prompts, replies, WVS text, probe answers, provider metadata, or secrets. Existing `results/` files are not rewritten.

The lock does not turn the single-seed calibration into a paper result. It remains `development_only`, has `no_causal_conclusion = true`, and explicitly excludes the condition-aware v1 `002` pilot from v2 analysis.
