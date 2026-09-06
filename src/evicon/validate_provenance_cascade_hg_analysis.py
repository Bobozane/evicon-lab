"""Network-free validator for the evaluator-only H-G Pilot analysis."""
from __future__ import annotations

import argparse
import json
from typing import Sequence

from .provenance_cascade_hg_analysis import run_hg_analysis


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate H-G evaluator-only analysis inputs")
    parser.add_argument(
        "--config",
        default="configs/provenance_cascade/pilot/provenance_cascade_pilot_hg_analysis.v1.toml",
    )
    args = parser.parse_args(argv)
    result = run_hg_analysis(args.config, write_outputs=False)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))
    return 0 if result.status == "validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
