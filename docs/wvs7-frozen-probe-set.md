# English Core WVS 7 Frozen ProbeSet

Stage 22B-4 adds an explicit local freeze gate for the 23-item English core
authoring plan. It does not read PDFs, contact WVS, call a model, run a
ProbeRunner, or create a Chinese ProbeSet.

The gate requires accepted researcher signoff, the exact 23 non-Q48 variables,
manual transcription provenance, complete text/options/endpoints, matching page
and SHA-256 metadata, and the special-code rules for Q111, Q149/Q150, and the
democracy items. Q158-Q163 remain marked as secondary science/technology
attitude analysis, and Q246 retains its cross-language semantic-scope warning.

The explicit command is:

```bash
uv run python -m evicon.freeze_wvs7_probe_set \
  --candidate configs/studies/wvs7_candidate_24.toml \
  --review configs/studies/wvs7_candidate_24_codex_preliminary_review.toml \
  --acceptance configs/studies/wvs7_authoring_acceptance_researcher_2026-08-18.toml \
  --transcriptions private_wvs_transcriptions/english_core_wvs7_23.toml \
  --output outputs/wvs7-frozen/english_core_wvs7_23_frozen.json \
  --confirm-freeze
```

Without `--confirm-freeze`, the command returns `blocked` with
`confirm_freeze_required` and writes nothing. Existing output files are never
overwritten. A successful run writes the ProbeSet and an audit-only manifest
beside it; the manifest contains hashes, IDs, source prefixes, review receipts,
and special-code summaries, but no question text, options, prompts, answers,
model data, or secrets.

Any extraction draft with `human_verified=false` is intentionally blocked until
a researcher completes a separate manual review and marks every item with
`manual_researcher_transcription`. This stage does not claim a benchmark, value
convergence, or experimental result.
