"""Network-free CLI for the Conformity Identification Study analysis plan."""
from __future__ import annotations

import argparse
from typing import Sequence

from .conformity_identification_analysis_plan import DEFAULT_ANALYSIS_PLAN, safe_analysis_preflight, safe_json


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the conformity identification analysis plan")
    parser.add_argument("--config", default=DEFAULT_ANALYSIS_PLAN)
    args = parser.parse_args(argv)
    result = safe_analysis_preflight(args.config)
    print(safe_json(result))
    return 0 if result.get("status") == "analysis_plan_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
