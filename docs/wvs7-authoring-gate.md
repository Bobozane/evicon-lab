# WVS7 Authoring Gate

This stage turns the technical preliminary review into two independent,
metadata-only plans: `english_core` and `chinese_applied`. Their five EviCon
groups are operational groupings, not WVS official factors. Neither plan is a
ProbeSet.

## Acceptance

`wvs7_candidate_24_codex_preliminary_review.toml` is a technical pre-review,
not a researcher signature. The separate acceptance template remains `pending`
until a researcher supplies an identifier and date, sets `accepted`, and turns
on all four confirmations. Each language arm then needs its own complete local
manual transcription before it can become `ready_for_freeze`.

English and Chinese are never inferred equivalent from a shared variable ID.
Each arm independently validates page locators, source hash prefixes, endpoint
codes, option codes, and special-code handling. The validator does not translate
or inspect PDFs.

## Protected Review Decisions

`Q48` remains excluded because the preliminary review classifies it as a
perceived agency/control measure rather than a direct normative tradeoff. The
researcher must confirm that exclusion. `Q111` preserves volunteered code `3`
as non-ordinal. `Q149` and `Q150` are binary categories, not continuous scales.
For `Q241`, `Q243`, and `Q246` through `Q249`, volunteered code `0` is outside
ordinary `1`--`10` distance calculations. `Q158` through `Q163` retain a note
that they are secondary science/technology-attitude analyses, not pure value
factors. `Q246` retains a cross-language semantic-scope warning.

## Manual Input and Freeze

Full text and response labels may exist only in the ignored local
`private_wvs_transcriptions/` directory. They are never printed by this CLI,
stored in an authoring plan, or committed as test fixtures. A plan alone cannot
create a ProbeSet.

The validation CLI is assessment-only and never writes a formal WVS ProbeSet,
even with `--confirm-freeze`. A separate API can write only a marked
`synthetic_fixture` with explicit confirmation, solely to test this gate; formal
WVS authoring and final instrument freezing are not implemented here.

```bash
uv run python -m evicon.validate_wvs7_authoring \
  --candidate configs/studies/wvs7_candidate_24.toml \
  --review configs/studies/wvs7_candidate_24_codex_preliminary_review.toml \
  --acceptance configs/studies/wvs7_authoring_acceptance_template.toml \
  --language english
```

This stage is local-only infrastructure. It does not call models, access the
network or PDFs, run experiments, or provide empirical conclusions.
