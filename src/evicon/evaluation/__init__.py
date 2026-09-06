"""Read-only orchestration for offline EviCon-Lab metric evaluation."""

from .evaluator import OfflineEvaluator, write_evaluation_report
from .loaders import (
    EvaluationInputError,
    EvaluationManifestLoader,
    ProbeResultLoader,
    RunRecordLoader,
)
from .matching import match_runs
from .models import (
    EvaluationCondition,
    EvaluationManifest,
    EvaluationMetricOptions,
    EvaluationReport,
    EvaluationRunEntry,
    EvaluationStatus,
    InputFileAudit,
    MatchedRunPair,
    UnmatchedEntry,
)

__all__ = [
    "EvaluationCondition",
    "EvaluationInputError",
    "EvaluationManifest",
    "EvaluationManifestLoader",
    "EvaluationMetricOptions",
    "EvaluationReport",
    "EvaluationRunEntry",
    "EvaluationStatus",
    "InputFileAudit",
    "MatchedRunPair",
    "OfflineEvaluator",
    "ProbeResultLoader",
    "RunRecordLoader",
    "UnmatchedEntry",
    "match_runs",
    "write_evaluation_report",
]
