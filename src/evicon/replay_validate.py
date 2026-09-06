"""CLI for read-only EviCon-Lab JSONL replay validation."""

from __future__ import annotations

import argparse

from .replay import ReplayValidationError, validate_replay


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an EviCon-Lab event log against its run record.")
    parser.add_argument("--events", required=True, help="Path to events.jsonl.")
    parser.add_argument("--record", required=True, help="Path to run_record.json.")
    args = parser.parse_args()
    try:
        report = validate_replay(args.events, args.record)
    except (ReplayValidationError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"run_id={report.run_id}")
    print(f"status={report.status.value}")
    print(f"events={report.event_count}")
    print(f"rounds={report.round_count}")
    print(f"turns={report.turn_count}")
    print(f"validation={report.validation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
