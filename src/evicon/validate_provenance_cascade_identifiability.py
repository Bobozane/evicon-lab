"""Network-free H-G identifiability amendment preflight."""
from __future__ import annotations
import argparse
import json
from typing import Sequence
from .provenance_cascade_identifiability import hg_preflight

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate H-G public-content identifiability amendment")
    parser.add_argument("--config", default="configs/provenance_cascade/pilot/provenance_cascade_pilot_hg.v1.toml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = hg_preflight(args.config, allow_existing_output=args.resume)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result.get("ready_for_real_pilot") is True else 1

if __name__ == "__main__":
    raise SystemExit(main())
