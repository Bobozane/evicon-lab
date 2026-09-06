"""CLI for the local-only WVS protocol-blind v2 preregistration gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .study_preregistration import StudyPreregistrationError, load_and_validate_study_preregistration


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an offline WVS protocol-blind preregistration.")
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args(argv)
    try:
        summary = load_and_validate_study_preregistration(arguments.config)
    except (OSError, ValueError, StudyPreregistrationError):
        print(json.dumps({"status": "blocked", "blocking_reasons": ["preregistration_validation_failed"]}, sort_keys=True))
        return 1
    print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
