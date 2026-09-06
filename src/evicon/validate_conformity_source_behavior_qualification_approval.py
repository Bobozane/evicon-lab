"""Safe offline CLI for the source-behavior approval gate."""
from __future__ import annotations
import argparse
import json
from typing import Sequence
from .conformity_source_behavior_qualification_approval import DEFAULT_APPROVAL, safe_preflight

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval", default=DEFAULT_APPROVAL)
    args = parser.parse_args(argv)
    payload = safe_preflight(args.approval)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if not payload.get("blocking_reasons") else 1

if __name__ == "__main__":
    raise SystemExit(main())
