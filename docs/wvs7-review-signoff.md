# WVS Wave 7 Review Signoff

`wvs7_candidate_24_review_template.toml` is a draft, manual pre-freeze
signoff template. It mirrors the 24 IDs and five EviCon operational groups from
the candidate mapping, but is not a `ProbeSet`, a frozen instrument, or an
experiment configuration.

Every candidate has two independent review arms:

- `english_core` is the prospective English main-study arm.
- `chinese_applied` is the prospective Chinese application arm.

An inclusion decision in one arm does not establish translation quality,
semantic equivalence, scale equivalence, or inclusion in the other arm. Each
arm must independently record its reviewer, date, questionnaire and codebook
locators, scale bounds, high-score meaning, reverse-scoring decision, and
rationale before it can be included. Exclusions also require a reviewer, date,
codebook locator, and rationale.

The checked-in template deliberately leaves both language arms `pending`. Empty
review fields and empty codebook-page lists signal missing human signoff rather
than a completed decision. `pending` produces `blocked`; it is not a successful
validation result. When all 24 items in one arm are independently included or
excluded with complete review fields, that arm can be reported as
`ready_for_authoring`. This only means it may enter a later human-approved
authoring phase. It never creates a ProbeSet, calls a model, or establishes a
research result.

The five groups remain EviCon operational groupings, not a WVS official factor
structure. Source hashes are matched against the draft candidate manifest and
the local receipt in `external-wvs7-sources.md`. The validator reads only the
candidate and review TOML files; it does not read WVS PDFs, follow URLs, or use
the network.

Run the review check with:

```bash
uv run python -m evicon.validate_wvs7_review \
  --candidate configs/studies/wvs7_candidate_24.toml \
  --review configs/studies/wvs7_candidate_24_review_template.toml
```

Use `--strict` when a blocked language arm should cause a nonzero exit code.
CLI output intentionally contains only statuses, counts, reason codes, source
receipt checks, and candidate IDs. It excludes WVS wording, response options,
prompts, answers, and credentials.
