"""CLI for the offline 30-request stability FakeProvider smoke."""
from __future__ import annotations

import argparse
from typing import Sequence

from .conformity_identification_stability import DEFAULT_STABILITY_CONFIG, run_fake_smoke, safe_json


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline identification stability smoke")
    parser.add_argument("--config", default=DEFAULT_STABILITY_CONFIG)
    args = parser.parse_args(argv)
    print(safe_json(run_fake_smoke(args.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
