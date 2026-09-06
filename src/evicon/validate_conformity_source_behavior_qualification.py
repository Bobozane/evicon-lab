"""Safe local preflight for source-behavior qualification v2."""
from __future__ import annotations

import argparse
from typing import Sequence

from .conformity_identification import safe_json
from .conformity_source_behavior_qualification import DEFAULT_CONFIG, safe_preflight


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate source-behavior qualification v2")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = safe_preflight(args.config)
    print(safe_json(result))
    return 0 if result.get("status") == "offline_behavior_gate_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
