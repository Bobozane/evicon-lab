"""CLI for safe, local WVS Wave 7 manual-signoff validation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .wvs7_review import WVS7ReviewError, load_and_validate_wvs7_review


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate local WVS Wave 7 review signoff without PDF access or network calls.")
    parser.add_argument("--candidate", required=True, help="Path to the local draft WVS candidate manifest.")
    parser.add_argument("--review", required=True, help="Path to the local WVS review TOML manifest.")
    parser.add_argument("--strict", action="store_true", help="Return nonzero while either language arm is blocked.")
    arguments = parser.parse_args(argv)
    try:
        report = load_and_validate_wvs7_review(arguments.candidate, arguments.review)
    except WVS7ReviewError as error:
        print(json.dumps({"status": "invalid", "error_code": _error_code(error)}, ensure_ascii=True, sort_keys=True))
        return 1

    payload = {
        "review_id": report.review_id,
        "artifact_status": report.artifact_status.value,
        "candidate_count": report.candidate_count,
        "status": report.status.value,
        "next_step": report.next_step,
        "english_core": _safe_language_summary(report.english_core),
        "chinese_applied": _safe_language_summary(report.chinese_applied),
    }
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 1 if arguments.strict and report.status.value == "blocked" else 0


def _safe_language_summary(summary: object) -> dict[str, object]:
    return {
        "blocked": getattr(summary, "blocked"),
        "ready_for_authoring": getattr(summary, "ready_for_authoring"),
        "included_count": getattr(summary, "included_count"),
        "excluded_count": getattr(summary, "excluded_count"),
        "pending_count": getattr(summary, "pending_count"),
        "blocking_reasons": getattr(summary, "blocking_reasons"),
        "blocking_candidate_ids": getattr(summary, "blocking_candidate_ids"),
    }


def _error_code(error: WVS7ReviewError) -> str:
    message = str(error)
    if "hash" in message:
        return "hash_mismatch"
    if "pages" in message:
        return "locator_mismatch"
    if "candidate" in message or "variable_id" in message:
        return "candidate_mismatch"
    return "validation_failed"


if __name__ == "__main__":
    raise SystemExit(main())
