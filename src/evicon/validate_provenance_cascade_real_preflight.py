"""Final, network-disabled provenance-cascade H-D real-pilot preflight CLI."""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .provenance_cascade_real_preflight import final_preflight_hd


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the approved H-D pilot without enabling network execution.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--approval", required=True)
    args = parser.parse_args(argv)
    report = final_preflight_hd(args.config, args.approval)
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))
    return 0 if report.status == "passed_waiting_for_network_authorization" else 1


if __name__ == "__main__":
    raise SystemExit(main())
