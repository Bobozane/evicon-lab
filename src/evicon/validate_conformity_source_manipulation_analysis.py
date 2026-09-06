"""Offline CLI for the source-root manipulation gate audit."""
from __future__ import annotations

import argparse
from typing import Sequence

from .conformity_identification import safe_json
from .conformity_source_manipulation_analysis import DEFAULT_ANALYSIS_CONFIG, analyze_source_manipulation_gate, safe_analysis_preflight, write_analysis_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a source-root manipulation gate analysis")
    parser.add_argument("--config", default=DEFAULT_ANALYSIS_CONFIG)
    parser.add_argument("--write-report", default=None)
    args = parser.parse_args(argv)
    if args.write_report is None:
        result = safe_analysis_preflight(args.config)
    else:
        try:
            result = analyze_source_manipulation_gate(args.config)
            result["report_path"] = str(write_analysis_report(result, args.write_report))
        except Exception as exc:
            result = {"status": "blocked", "blocking_reasons": [getattr(exc, "code", "source_manipulation_analysis_failed")], "network": "disabled", "provider_constructed": False, "api_key_read": False, "results_written": False, "private_truth_exposed": False}
    print(safe_json(result))
    return 0 if result.get("status") == "completed_with_semantic_audit_limit" else 1


if __name__ == "__main__":
    raise SystemExit(main())
