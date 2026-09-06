"""Safe offline CLI for the provenance-cascade pilot preregistration."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .provenance_cascade_preregistration import (
    ProvenanceCascadePreregistrationError,
    load_and_validate_provenance_cascade_preregistration,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an offline provenance-cascade preregistration.")
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args(argv)
    try:
        summary = load_and_validate_provenance_cascade_preregistration(arguments.config)
    except (OSError, ValueError, ProvenanceCascadePreregistrationError):
        print(json.dumps({"status": "blocked", "blocking_reasons": ["provenance_cascade_preregistration_validation_failed"]}, sort_keys=True))
        return 1
    print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
