"""Offline-only preflight CLI for the locked protocol-blind v2 confirmatory plan."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .wvs7_confirmatory_real_runner import (
    WVS7ConfirmatoryRealError,
    _safe_error_code,
    preflight_wvs7_confirmatory_real_experiment,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the confirmatory v2 plan locally without a Provider.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        _, report = preflight_wvs7_confirmatory_real_experiment(arguments.config, resume=arguments.resume)
    except (WVS7ConfirmatoryRealError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "blocking_reasons": [_safe_error_code(exc)], "provider_call_count": 0}, sort_keys=True))
        return 1
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))
    return 0 if report.status == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
