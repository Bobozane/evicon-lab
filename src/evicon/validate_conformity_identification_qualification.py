"""Offline qualification preflight for Conformity Identification Study v1.1."""
from __future__ import annotations

import argparse
from typing import Sequence

from .conformity_identification_qualification import (
    DEFAULT_QUALIFICATION_APPROVAL,
    safe_json,
    safe_qualification_preflight,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate conformity identification qualification gates")
    parser.add_argument("--approval", default=DEFAULT_QUALIFICATION_APPROVAL)
    args = parser.parse_args(argv)
    result = safe_qualification_preflight(args.approval)
    print(safe_json(result))
    return 0 if result["status"] == "qualification_gates_offline_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
