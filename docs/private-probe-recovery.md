# Private Probe Recovery

`PrivateProbeRecoveryStore` exists only for explicitly enabled real-pilot
recovery. It is not used by `DialogueState`, event logging, online control,
offline metrics, or ordinary fake/offline runs.

For each pre- or post-probe phase, the store writes one local JSON checkpoint
under the real batch's ignored `results/` directory. It validates the run,
scenario, protocol, frozen ProbeSet identity, model/seed, Agent order, item
order, target round, holdout scope, and SHA-256 hashes of public contexts before
allowing a resume. A changed identity is rejected.

The checkpoint retains the private response values necessary to rebuild a
`ProbeResult`; it never retains prompt text, question text, full provider
responses, dialogue turns, event content, provider metadata, or credentials.
Every successful response update is written through an atomic local replacement.
The checkpoint is deleted only after the condition's ordinary private
`probes/pre_probe_results.jsonl` and `probes/post_probe_results.jsonl` have been
written successfully.

`--resume` requires the pre-phase checkpoint for a condition that previously
failed before its public trajectory. This avoids silently sending a duplicate
request whose completed response was intentionally omitted from the ledger.
For a completed public trajectory with an interrupted post phase, the runner
uses the existing completed `run_record.json` and only resumes the missing post
probes. A failed public Agent trajectory is not replayed by this feature.
