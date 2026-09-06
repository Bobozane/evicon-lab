"""Offline smoke check for the WVS Wave 7 authoring gate."""

from __future__ import annotations

import json
from pathlib import Path

from .wvs7_authoring import WVS7LanguageArm, load_and_evaluate_wvs7_authoring_gate, safe_gate_summary


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    candidate = root / "configs/studies/wvs7_candidate_24.toml"
    review = root / "configs/studies/wvs7_candidate_24_codex_preliminary_review.toml"
    acceptance = root / "configs/studies/wvs7_authoring_acceptance_template.toml"
    reports = []
    for language in (WVS7LanguageArm.ENGLISH_CORE, WVS7LanguageArm.CHINESE_APPLIED):
        _, report = load_and_evaluate_wvs7_authoring_gate(candidate, review, acceptance, language)
        reports.append(safe_gate_summary(report))
    print(json.dumps({"mode": "local_only", "reports": reports}, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
