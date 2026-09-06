"""Default-offline H-G.1 final preflight CLI."""
from __future__ import annotations
import argparse, json
from .provenance_cascade_hg1 import DEFAULT_CONFIG, hg1_preflight

def main(argv=None) -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args=parser.parse_args(argv)
    print(json.dumps(hg1_preflight(args.config), ensure_ascii=True, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
