"""Safe offline CLI for the provenance-cascade H-B preflight."""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .provenance_cascade_preflight import preflight_hb


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the offline provenance-cascade pilot preflight.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-network", action="store_true", help="validate the future network gate only; never executes it")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--confirm-request-cap", type=int)
    parser.add_argument("--confirm-token-cap", type=int)
    args = parser.parse_args(argv)
    report = preflight_hb(
        args.config,
        allow_network=args.allow_network,
        confirm_run=args.confirm_run,
        confirm_request_cap=args.confirm_request_cap,
        confirm_token_cap=args.confirm_token_cap,
    )
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))
    return 0 if report.status == "ready_for_offline_preflight" else 1


if __name__ == "__main__":
    raise SystemExit(main())
