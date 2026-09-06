"""Public policy boundary and persistence API for offline evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ..models import RunStatus
from . import evaluator_core as _core
from .loaders import LoadedEvaluationRun
from .matching import MatchingResult
from .models import EvaluationReport


def write_evaluation_report(report: EvaluationReport, path: str | Path) -> Path:
    """Write a new UTF-8 JSON report without changing any source artifact."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as handle:
            json.dump(report.model_dump(mode="json"), handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite existing evaluation report: {output_path}") from None
    return output_path


_match_completed_runs = _core.match_runs


def _match_evaluable_runs(runs: Iterable[LoadedEvaluationRun]) -> MatchingResult:
    """Keep failed artifacts, when explicitly allowed, in the audit-only path."""
    return _match_completed_runs(
        run for run in runs if run.record.status is RunStatus.COMPLETED
    )


# The core calculates metrics; this public boundary supplies persistence and
# prevents a failed audit input from being selected for a comparison metric.
_core.write_evaluation_report = write_evaluation_report
_core.match_runs = _match_evaluable_runs
OfflineEvaluator = _core.OfflineEvaluator

__all__ = ["OfflineEvaluator", "write_evaluation_report"]
