"""Offline CLI for the protocol-blind v2 confirmatory analysis gate."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .wvs7_confirmatory_analysis import (
    WVS7ConfirmatoryAnalysisError,
    run_confirmatory_analysis,
    validate_confirmatory_analysis,
)
from .wvs7_confirmatory_analysis_plan import (
    ConfirmatoryAnalysisPlanError,
    load_confirmatory_analysis_plan,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate or write safe protocol-blind v2 confirmatory analysis outputs offline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--write-reports", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        plan = load_confirmatory_analysis_plan(arguments.config)
        result = run_confirmatory_analysis(plan) if arguments.write_reports else validate_confirmatory_analysis(plan)
    except (ConfirmatoryAnalysisPlanError, WVS7ConfirmatoryAnalysisError, OSError, ValueError):
        print(json.dumps({"status": "blocked", "blocking_reasons": ["confirmatory_analysis_validation_failed"]}, sort_keys=True))
        return 1
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
