# WVS Wave 7 Candidate Selection

`wvs7_candidate_24.toml` is a draft-only manual review checklist. It is not a
`ProbeSet`, is not a pilot or frozen material, and is not a final benchmark.
The checklist records short topics, response-scale categories, source filenames,
and questionnaire page locators without copying question wording or options.

The candidate IDs are: `Q48`, `Q106`-`Q111`, `Q149`-`Q150`, `Q158`-`Q163`,
`Q196`-`Q198`, `Q241`, `Q243`, and `Q246`-`Q249`.

EviCon-Lab uses the following operational groupings for review, not a WVS
official factor structure:

- `autonomy_security`: 3 candidates (`Q48`, `Q149`, `Q150`).
- `economic_distribution`: 6 candidates (`Q106`-`Q111`).
- `science_knowledge`: 6 candidates (`Q158`-`Q163`).
- `surveillance_privacy`: 3 candidates (`Q196`-`Q198`).
- `democratic_order`: 6 candidates (`Q241`, `Q243`, `Q246`-`Q249`).

Each entry declares local USA and China source presence, country-specific
questionnaire page locators, and paired country-source filenames. The four
local source receipts and SHA-256 digests are maintained in
`external-wvs7-sources.md`; the candidate manifest repeats the same digests.
The candidate validator checks this static receipt only. It does not read PDF
content at runtime, follow links, download material, or call a model.

Before a final `frozen` ProbeSet can be considered, a human reviewer must
verify exact item wording, scale direction, reverse-scoring choices, missing
value rules, USA/China language equivalence, WVS citation, and applicable use
terms. A candidate mapping alone establishes none of those properties.
