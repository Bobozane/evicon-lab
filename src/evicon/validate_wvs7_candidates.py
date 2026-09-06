"""CLI for safe validation of the local WVS Wave 7 candidate mapping."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .wvs7_candidates import WVS7CandidateError, load_and_validate_wvs7_candidates


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate local WVS Wave 7 candidate mappings without network or PDF reads.")
    parser.add_argument("--manifest", required=True, help="Path to a local WVS candidate TOML manifest.")
    arguments = parser.parse_args(argv)
    try:
        report = load_and_validate_wvs7_candidates(arguments.manifest)
    except WVS7CandidateError as error:
        print(json.dumps({"status": "invalid", "error_code": _error_code(error)}, ensure_ascii=True, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "study_id": report.study_id,
                "manifest_version": report.manifest_version,
                "material_status": report.material_status.value,
                "candidate_count": report.candidate_count,
                "domain_counts": report.domain_counts,
                "source_hash_prefixes": report.source_hash_prefixes,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


def _error_code(error: WVS7CandidateError) -> str:
    message = str(error)
    if "hash" in message:
        return "hash_mismatch"
    if "status" in message:
        return "invalid_status"
    return "validation_failed"


if __name__ == "__main__":
    raise SystemExit(main())
