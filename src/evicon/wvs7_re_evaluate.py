"""No-network regeneration of a completed WVS pilot's offline evaluation report."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from .batch_experiment import BatchExperimentManifest, BatchExperimentRecord, BatchExperimentStatus
from .evaluation import OfflineEvaluator, write_evaluation_report
from .frozen_wvs7_manifest import sha256_file
from .wvs7_baseline_pilot import WVS7BaselinePilotError, prepare_wvs7_baseline_pilot
from .wvs7_real_pilot import WVS7RealPilotError, build_wvs7_evaluation_manifest


class WVS7ReevaluationError(ValueError):
    """Stable, safe error for read-only evaluation regeneration."""


def re_evaluate_wvs7_batch(
    config_path: str | Path,
    *,
    batch_id: str,
) -> dict[str, object]:
    """Build a versioned initial/final report without touching model-facing artifacts."""
    prepared = prepare_wvs7_baseline_pilot(config_path)
    return re_evaluate_prepared_wvs7_batch(prepared, batch_id=batch_id)


def re_evaluate_prepared_wvs7_batch(
    prepared: object,
    *,
    batch_id: str,
) -> dict[str, object]:
    """Reuse the same offline initial/final evaluation for an already prepared plan.

    The public path-based wrapper above retains its existing behavior.  This
    dependency-injection form lets a separately validated study plan reuse the
    evaluation logic without writing a surrogate baseline configuration file.
    """
    batch_directory = Path(prepared.config.batch_output_dir) / batch_id
    try:
        manifest = BatchExperimentManifest.model_validate_json(
            (batch_directory / "batch_manifest.json").read_text(encoding="utf-8")
        )
        record = BatchExperimentRecord.model_validate_json(
            (batch_directory / "batch_record.json").read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as exc:
        raise WVS7ReevaluationError("batch_artifacts_invalid") from exc
    if record.status is not BatchExperimentStatus.COMPLETED:
        raise WVS7ReevaluationError("completed_batch_required")
    if (
        manifest.batch_id != batch_id
        or manifest.scenario_id != prepared.scenario.scenario_id
        or manifest.probe_set_id != prepared.probe_set.probe_set_id
        or sha256_file(prepared.config.frozen_probe_set_path) != prepared.config.frozen_probe_set_sha256
    ):
        raise WVS7ReevaluationError("batch_or_frozen_instrument_mismatch")

    evaluation_manifest = build_wvs7_evaluation_manifest(
        manifest,
        evaluation_id=f"{batch_id}-offline-initial-final-v2",
        coverage_threshold=prepared.config.coverage_threshold,
    )
    manifest_path = batch_directory / "offline_evaluation_manifest_initial_final_v2.json"
    report_path = batch_directory / "offline_evaluation_report_initial_final_v2.json"
    if manifest_path.exists() or report_path.exists():
        raise WVS7ReevaluationError("versioned_evaluation_output_already_exists")
    try:
        manifest_path.write_text(evaluation_manifest.model_dump_json(indent=2), encoding="utf-8")
        report = OfflineEvaluator(evaluation_manifest, manifest_directory=batch_directory).evaluate()
        write_evaluation_report(report, report_path)
    except (OSError, ValueError) as exc:
        raise WVS7ReevaluationError("offline_evaluation_failed") from exc

    return {
        "status": report.status.value,
        "batch_id": batch_id,
        "evaluation_id": report.evaluation_id,
        "metric_names": [metric.metric_name for metric in report.metrics],
        "warning_count": len(report.warnings),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
    }


def _safe_error_code(error: BaseException) -> str:
    message = str(error)
    for code in (
        "batch_artifacts_invalid",
        "completed_batch_required",
        "batch_or_frozen_instrument_mismatch",
        "versioned_evaluation_output_already_exists",
        "offline_evaluation_failed",
    ):
        if code in message:
            return code
    return "offline_reevaluation_failed"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate a completed WVS pilot's initial/final offline report without network access.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-id", required=True)
    arguments = parser.parse_args(argv)
    try:
        print(json.dumps(
            re_evaluate_wvs7_batch(arguments.config, batch_id=arguments.batch_id),
            ensure_ascii=True,
            sort_keys=True,
        ))
        return 0
    except (WVS7BaselinePilotError, WVS7RealPilotError, WVS7ReevaluationError, OSError, ValidationError, ValueError) as error:
        print(json.dumps({"status": "blocked", "error_code": _safe_error_code(error)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "WVS7ReevaluationError",
    "re_evaluate_prepared_wvs7_batch",
    "re_evaluate_wvs7_batch",
]
