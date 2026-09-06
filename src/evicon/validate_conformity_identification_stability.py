"""Network-free preflight for the 30-request stability probe."""
from __future__ import annotations

import argparse
from typing import Sequence

from .conformity_identification_stability import DEFAULT_STABILITY_CONFIG, preflight, safe_json


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the identification stability probe")
    parser.add_argument("--config", default=DEFAULT_STABILITY_CONFIG)
    args = parser.parse_args(argv)
    result = preflight(args.config)
    print(safe_json(result))
    return 0 if result.get("status") == "offline_probe_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
