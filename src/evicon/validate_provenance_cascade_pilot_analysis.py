"""Network-free validator CLI for the H-F Pilot analysis."""
from __future__ import annotations

import argparse
import json
from typing import Sequence

from .provenance_cascade_pilot_analysis import run_hf_analysis


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate H-D.2.1 evaluator-only analysis inputs")
    parser.add_argument("--config", default="configs/provenance_cascade/pilot/provenance_cascade_pilot_hd21_analysis.v1.toml")
    args = parser.parse_args(argv)
    result = run_hf_analysis(args.config, write_outputs=False)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))
    return 0 if result.status == "validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
