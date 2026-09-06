"""Offline smoke for the isolated intervention-trigger calibration fixture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cascade_intervention_calibration import (
    DEFAULT_MANIFEST,
    CalibrationRunError,
    CascadeInterventionCalibrationRunner,
)


def run_smoke(manifest: Path = DEFAULT_MANIFEST) -> list[dict[str, object]]:
    runner = CascadeInterventionCalibrationRunner.from_file(manifest)
    summaries, _records = runner.run_all()
    return [summary.model_dump(mode="json") for summary in summaries]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline provenance-cascade trigger calibration")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    try:
        rows = run_smoke(args.manifest)
    except CalibrationRunError as exc:
        print(json.dumps({"status": "blocked", "error_code": exc.code, "network": "disabled", "results_written": False}, sort_keys=True))
        return 2
    print(json.dumps({
        "status": "offline_intervention_calibration",
        "network": "disabled",
        "results_written": False,
        "private_truth_exposed": False,
        "development_only": True,
        "calibration_only": True,
        "not_paper_result": True,
        "no_causal_conclusion": True,
        "run_count": len(rows),
        "runs": rows,
    }, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
