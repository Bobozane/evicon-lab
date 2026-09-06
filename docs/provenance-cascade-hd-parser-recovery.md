# H-D Agent Parser-Invalid Recovery Amendment

The first authorized H-D transport completed for
`hd-cascade-false-majority-20260901-no_intervention`,
`network-agent-01`, round 0, but the strict public Agent response parser
returned `invalid_schema`. No parsed checkpoint, public outcome, or run record
was created.

The approved recovery is deliberately narrower than ordinary resume:

- it is available only with explicit `--resume` and
  `--confirm-parser-recovery`;
- it accepts only the fingerprint, coordinate, original failure, batch hash,
  ledger hash, config hash, and execution-authorization hash recorded in the
  versioned amendment;
- the checkpoint coordinate must still be missing;
- it appends at most one additional transport attempt for that same logical
  fingerprint;
- the 864 logical-request cap is unchanged, while actual transport attempts and
  token usage include both the parser-rejected response and its recovery;
- prompts, templates, model parameters, seed, condition order, scenarios,
  controller policy, and metrics are unchanged;
- a second parser-invalid result stops the batch and cannot be retried through
  this amendment.

The safe amendment receipt contains only IDs, hashes, the stable error code,
and safety flags. It contains no prompt, raw reply, evaluator truth, provider
metadata, or credential. This is runtime fault recovery for a development
pilot, not an experimental-design change or evidence of method effectiveness.
