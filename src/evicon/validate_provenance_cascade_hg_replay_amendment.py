"""Offline CLI for the H-G outcome replay technical amendment receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .provenance_cascade_hg_replay_amendment import (
    DEFAULT_RECEIPT_PATH,
    HGReplayAmendmentError,
    safe_summary,
    validate_technical_amendment_receipt,
    write_technical_amendment_receipt,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the H-G outcome replay technical amendment")
    parser.add_argument("--receipt", default=str(DEFAULT_RECEIPT_PATH))
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.create:
            receipt, digest = write_technical_amendment_receipt(Path(args.receipt))
        else:
            receipt, digest = validate_technical_amendment_receipt(Path(args.receipt))
        payload = safe_summary(receipt, digest)
    except HGReplayAmendmentError as exc:
        payload = {
            "status": "blocked",
            "error_code": exc.code,
            "network": "disabled",
            "provider_constructed": False,
            "results_written": False,
        }
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if payload["status"] == "offline_validated_ready_for_resume" else 1


if __name__ == "__main__":
    raise SystemExit(main())
