"""Offline smoke command for the private English transcription template."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from .wvs7_authoring import WVS7AuthoringAcceptanceLoader, WVS7LanguageArm, build_wvs7_authoring_plan
from .wvs7_candidates import WVS7CandidateLoader
from .wvs7_review import WVS7ReviewLoader
from .wvs7_transcription_template import (
    ENGLISH_CORE_VARIABLE_IDS,
    build_english_core_template,
    load_blank_template,
    safe_template_summary,
    write_english_core_template,
)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    candidate_path = root / "configs/studies/wvs7_candidate_24.toml"
    review_path = root / "configs/studies/wvs7_candidate_24_codex_preliminary_review.toml"
    acceptance_path = root / "configs/studies/wvs7_authoring_acceptance_researcher_2026-08-18.toml"
    output_path = root / "private_wvs_transcriptions/english_core_wvs7_23.toml"
    candidate = WVS7CandidateLoader.load(candidate_path)
    review = WVS7ReviewLoader.load(review_path)
    acceptance = WVS7AuthoringAcceptanceLoader.load(acceptance_path)
    plan = build_wvs7_authoring_plan(candidate, review, acceptance, WVS7LanguageArm.ENGLISH_CORE)
    if output_path.exists():
        # Stage 22B-3.1 may replace this ignored path with an explicitly
        # unverified PDF extraction draft.  Keep the original blank-template
        # smoke deterministic by validating a fresh in-memory blank shape.
        payload = tomllib.loads(output_path.read_text(encoding="utf-8"))
        if any("transcription_origin" in item for item in payload.get("items", [])):
            template = build_english_core_template(plan, acceptance)
        else:
            template = load_blank_template(output_path)
    else:
        template = write_english_core_template(plan, acceptance, output_path)
    if tuple(item.variable_id for item in template.items) != ENGLISH_CORE_VARIABLE_IDS:
        raise RuntimeError("template smoke variable coverage mismatch")
    print(json.dumps(safe_template_summary(template, output_path), ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
