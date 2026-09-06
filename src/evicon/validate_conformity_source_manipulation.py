"""Network-free preflight for the source-root manipulation gate."""
from __future__ import annotations

import argparse
from typing import Sequence

from .conformity_identification import safe_json
from .conformity_source_manipulation import DEFAULT_MANIPULATION_CONFIG, safe_preflight


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the source-root manipulation gate")
    parser.add_argument("--config", default=DEFAULT_MANIPULATION_CONFIG)
    args = parser.parse_args(argv)
    result = safe_preflight(args.config)
    print(safe_json(result))
    return 0 if result.get("status") == "offline_gate_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
