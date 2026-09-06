"""No-network planning entry point for the WVS English baseline pilot."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .wvs7_baseline_pilot import (
    WVS7BaselinePilotError,
    dry_run_wvs7_baseline_pilot,
    prepare_wvs7_baseline_pilot,
    safe_dry_run_summary,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and budget a WVS English baseline plan without model calls.")
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args(argv)
    try:
        prepared = prepare_wvs7_baseline_pilot(arguments.config)
        print(json.dumps(safe_dry_run_summary(dry_run_wvs7_baseline_pilot(prepared)), ensure_ascii=True, sort_keys=True))
        return 0
    except (WVS7BaselinePilotError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "blocking_reasons": ["baseline_pilot_validation_failed"]}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
