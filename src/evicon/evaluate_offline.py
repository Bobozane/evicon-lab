"""Command-line entry point for read-only offline metric evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation import OfflineEvaluator, write_evaluation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate declared EviCon-Lab artifacts offline.")
    parser.add_argument("--manifest", required=True, help="Path to a JSON or TOML EvaluationManifest.")
    parser.add_argument("--output", required=True, help="Path for a new JSON evaluation report.")
    args = parser.parse_args()

    evaluator = OfflineEvaluator.from_manifest_file(args.manifest)
    report = evaluator.evaluate()
    output_path = write_evaluation_report(report, args.output)
    print(f"evaluation_id={report.evaluation_id}")
    print(f"scenario_id={report.scenario_id}")
    print(f"metric_count={len(report.metrics)}")
    print(f"warning_count={len(report.warnings)}")
    print(f"output_path={Path(output_path)}")


if __name__ == "__main__":
    main()
