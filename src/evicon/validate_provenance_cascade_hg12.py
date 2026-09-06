"""Safe offline final preflight CLI for H-G.1.2."""
from __future__ import annotations

import argparse
import json
from typing import Sequence

from .provenance_cascade_hg12_governance import DEFAULT_CONFIG, final_preflight


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the H-G.1.2 final offline gate")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    payload = final_preflight(args.config)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if payload["status"] in {"blocked", "ready_for_real_pilot"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
