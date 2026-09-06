# Study Materials

The study-materials layer records local source provenance, language, status,
split allocation, and SHA-256 digests before a material is used in an
experiment. `validate_study_materials` reads only local TOML files. It never
downloads a source, follows a citation URL, calls a model, or reads run results.

The checked-in English probe set is original, balanced, development-only value
measurement. Each declared dimension has one value-forward item and one
reasonable competing tradeoff item with reverse scoring. It is explicitly
`author_original`, `pilot`, and `not_wvs_pilot`: it is not WVS, a WVS
derivative, an external survey, or a final paper item bank.

The two checked-in Chinese scenarios are synthetic development materials with
explicit public decision options, uncertainty, and evidence-change conditions.
Together with the English probes, they support only a cross-language engineering
pilot. They must not be reported as primary experimental results, evidence of
real platform-governance effects, public behavior, or policy effects.

The manifest has explicit `development` and `test` allocations. The current
pilot scenarios are development-only, and no pilot scenario may be placed in a
test split. Material pairing and source identity are not inferred from a file
name or a result directory.

The final research design must establish an English core arm and a Chinese
applied arm, then validate language consistency separately. The English core
must use an independently citable WVS-compatible probe after provenance review.
Before any final material becomes `frozen`, a human review must verify source
identity, version, stable citation, permission or license, language, dimension
mapping, factual claims, and the recorded SHA-256. Frozen status is a governance
state, not a statement that a benchmark is valid or that an experimental
conclusion has been obtained.

Run the local validator with:

```bash
uv run python -m evicon.validate_study_materials \
  --manifest configs/studies/evicon_pilot_materials.toml
```

Its output is limited to study identity, status, language, counts, split counts,
and hash prefixes. It intentionally excludes item text, scenario text,
responses, prompts, and credentials.
