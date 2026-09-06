# WVS 7 Source-Extraction Draft

Stage 22B-3.1 adds a deterministic, local-only extraction step for the English
core 23-item authoring plan. It reads only the approved USA English
questionnaire PDF and only the questionnaire pages already declared in the
metadata plan. The implementation uses an optional local PDF text parser and a
small standard-library FlateDecode/TJ fallback; it never uses a model,
translation service, network, or the Chinese PDF.

The output is `private_wvs_transcriptions/english_core_wvs7_23.toml`, which is
ignored by Git. It contains the extracted question and option text, but every
item is marked `transcription_origin = "local_pdf_text_extraction_draft"` and
`human_verified = false`. The existing authoring gate therefore returns
`status=blocked` with `reason=human_verification_required`, even where text was
successfully extracted. PDF layout ambiguities are recorded as item warnings;
the extractor does not invent missing endpoints or reorder uncertain options.

The smoke command is:

```bash
uv run python -m evicon.source_transcription_smoke
```

Its JSON output contains only variable IDs, page locators, extraction warnings,
and missing-field names. It does not print question text, option labels,
answers, prompts, provider data, or secrets. A researcher must inspect and
correct each private draft item before changing `human_verified` to `true` in a
separate manual workflow. This stage does not create a frozen ProbeSet, modify
existing runners, or produce research results.
