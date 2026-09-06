# Private English-Core Transcription

Stage 22B-3 creates a local, ignored transcription input for the accepted
English-core authoring plan. The generator copies only plan metadata for the 23
included variables:

`Q106`--`Q111`, `Q149`--`Q150`, `Q158`--`Q163`, `Q196`--`Q198`, `Q241`, `Q243`,
and `Q246`--`Q249`. `Q48` is excluded and no Chinese arm is present.

The generated file is:

```text
private_wvs_transcriptions/english_core_wvs7_23.toml
```

That directory is ignored by Git. The file contains source locators, scale
metadata, SHA-256 prefixes, and special-code rules, while question text,
response labels, endpoints, transcriber identity, and date are left as
`TODO_MANUAL_ENTRY` or empty values. A researcher must enter those fields from
their own local transcription of the audited source. The generator does not
read `/date/`, translate text, call a model, or access the network.

## Validation

```bash
uv run python -m evicon.transcription_template_smoke

uv run python -m evicon.validate_wvs7_authoring \
  --candidate configs/studies/wvs7_candidate_24.toml \
  --review configs/studies/wvs7_candidate_24_codex_preliminary_review.toml \
  --acceptance configs/studies/wvs7_authoring_acceptance_researcher_2026-08-18.toml \
  --language english \
  --transcriptions private_wvs_transcriptions/english_core_wvs7_23.toml
```

An untouched template is intentionally reported as `blocked` with
`manual_transcription_invalid`. Completion requires exactly one English item
for each included variable, non-empty question text and options, endpoint
labels for both scale endpoints, a transcriber and a date, and exact page/hash
agreement with the authoring plan.

## Protected Coding Rules

The validator preserves the existing authoring plan rules. Q111 code `3` is a
volunteered-other response and is reported separately rather than used in
ordinary ordered distances. Q149 and Q150 remain binary categories. Code `0`
for Q241, Q243, Q246, Q247, Q248 and Q249 is a volunteered
against-democracy response excluded from ordinary `1`--`10` distances and
recorded separately. Q158--Q163 remain secondary science/technology-attitude
items, not WVS official factors.

This stage does not create a frozen WVS ProbeSet. Full source text is never
placed in `configs/`, `src/`, `tests/`, `docs/`, logs, or CLI output.
