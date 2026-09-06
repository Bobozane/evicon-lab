"""Failed inputs admitted for audit never enter counterfactual metrics."""

from __future__ import annotations

import json
from pathlib import Path

from evicon.evaluation import OfflineEvaluator
from evicon.models import ProtocolCondition

from test_offline_evaluation import create_artifacts, entry, manifest


def _mark_failed(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "failed"
    payload["error_message"] = "fixture failure"
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_failed_counterfactual_arms_are_audit_only(tmp_path: Path) -> None:
    left_record, left_events, left_probe = create_artifacts(
        tmp_path,
        run_id="failed-evidence-only",
        protocol=ProtocolCondition.EVIDENCE_ONLY,
    )
    right_record, right_events, right_probe = create_artifacts(
        tmp_path,
        run_id="failed-evidence-social",
        protocol=ProtocolCondition.EVIDENCE_SOCIAL,
    )
    _mark_failed(left_record)
    _mark_failed(right_record)
    left = entry(
        run_id="failed-evidence-only",
        condition="evidence_only",
        protocol=ProtocolCondition.EVIDENCE_ONLY,
        record_path=left_record,
        events_path=left_events,
        probe_path=left_probe,
    ).model_copy(update={"allow_failed_for_audit": True})
    right = entry(
        run_id="failed-evidence-social",
        condition="evidence_social",
        protocol=ProtocolCondition.EVIDENCE_SOCIAL,
        record_path=right_record,
        events_path=right_events,
        probe_path=right_probe,
    ).model_copy(update={"allow_failed_for_audit": True})

    report = OfflineEvaluator(
        manifest([left, right], ["counterfactual", "audit"]),
        manifest_directory=tmp_path,
    ).evaluate()

    assert "social_influence_loss" not in [metric.metric_name for metric in report.metrics]
    assert len([metric for metric in report.metrics if metric.metric_name == "run_completeness"]) == 2
    assert any("failed run was retained only for partial audit" in warning for warning in report.warnings)
