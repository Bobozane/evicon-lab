"""Safe offline H-D amendment preflight CLI."""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .provenance_cascade_amendment import preflight_hd


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the offline provenance-cascade H-D amendment.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    report = preflight_hd(args.config)
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))
    return 0 if report.status == "ready_for_human_approval" else 1


if __name__ == "__main__":
    raise SystemExit(main())
