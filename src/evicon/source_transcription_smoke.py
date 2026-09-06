"""Offline smoke command for the English WVS source-extraction draft."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Sequence

from .wvs7_authoring import WVS7AuthoringAcceptanceLoader, WVS7LanguageArm, build_wvs7_authoring_plan
from .wvs7_candidates import WVS7CandidateLoader
from .wvs7_review import WVS7ReviewLoader
from .wvs7_source_transcriber import (
    safe_source_draft_summary,
    write_english_core_source_draft,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract an unverified English WVS draft from local PDF text.")
    parser.add_argument("--candidate", default="configs/studies/wvs7_candidate_24.toml")
    parser.add_argument("--review", default="configs/studies/wvs7_candidate_24_codex_preliminary_review.toml")
    parser.add_argument("--acceptance", default="configs/studies/wvs7_authoring_acceptance_researcher_2026-08-18.toml")
    parser.add_argument("--pdf", default="date/F00008646-WVS7_Questionnaire_USA_2017_English.pdf")
    parser.add_argument("--output", default="private_wvs_transcriptions/english_core_wvs7_23.toml")
    arguments = parser.parse_args(argv)
    candidate = WVS7CandidateLoader.load(arguments.candidate)
    review = WVS7ReviewLoader.load(arguments.review)
    acceptance = WVS7AuthoringAcceptanceLoader.load(arguments.acceptance)
    plan = build_wvs7_authoring_plan(candidate, review, acceptance, WVS7LanguageArm.ENGLISH_CORE)
    draft = write_english_core_source_draft(plan, arguments.pdf, arguments.output, overwrite=True)
    print(json.dumps(safe_source_draft_summary(draft), ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
