"""Network-free CLI for the conformity identification study design."""
from __future__ import annotations

import argparse
from typing import Sequence

from .conformity_identification import DEFAULT_CONFIG, safe_json, safe_preflight


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the offline conformity identification study")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = safe_preflight(args.config)
    print(safe_json(result))
    return 0 if result.get("status") == "offline_design_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
