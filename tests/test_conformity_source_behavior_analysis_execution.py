from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from evicon.conformity_identification import IdentificationError
from evicon.conformity_source_behavior_analysis_execution import (
    DEFAULT_EXECUTION_APPROVAL,
    RECEIPT_NAME,
    SUMMARY_NAME,
    execute_analysis,
    load_execution_receipt,
    main,
    safe_preflight,
)
from evicon.conformity_source_behavior_analysis_lock import load_analysis_lock
from evicon.conformity_source_behavior_qualification import (
    AdoptionDecision,
    SafeBehaviorCaseAudit,
    SharingDecision,
)
from evicon.conformity_source_behavior_qualification_smoke import build_cases


ROOT = Path(__file__).resolve().parents[1]


def _accepted_approval(tmp_path: Path) -> Path:
    raw = (ROOT / DEFAULT_EXECUTION_APPROVAL).read_text(encoding="utf-8")
    raw = raw.replace('acceptance_status = "pending"', 'acceptance_status = "accepted"')
    raw = raw.replace('accepted_by = ""', 'accepted_by = "test-reviewer"')
    raw = raw.replace('accepted_on = ""', 'accepted_on = "2026-08-28"')
    raw = re.sub(r"(confirm_[a-z0-9_]+) = false", r"\1 = true", raw)
    path = tmp_path / "analysis-execution-approval.toml"
    path.write_text(raw, encoding="utf-8")
    return path


def _fake_audits() -> tuple[SafeBehaviorCaseAudit, ...]:
    return tuple(
        SafeBehaviorCaseAudit(
            case_id=f"fake-{case.case_id}",
            scenario_id=case.scenario_id,
            projection=case.projection,
            parser_status="valid",
            adoption_decision=AdoptionDecision.WITHHOLD,
            sharing_decision=SharingDecision.DO_NOT_SHARE,
            used_content_count=0,
            visible_root_count=len({
                item.source_root_id
                for item in case.public_root_assignments
                if item.source_root_id
            }),
        )
        for case in build_cases()
    )


def test_default_preflight_is_offline_and_does_not_parse_audits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "evicon.conformity_source_behavior_analysis_execution._read_audits",
        lambda path: (_ for _ in ()).throw(AssertionError("audit parser must not run")),
    )
    report = safe_preflight()
    assert report["status"] == "source_behavior_analysis_execution_gate_offline_ready"
    assert report["approval_status"] == "accepted"
    assert report["blocking_reasons"] == [
        "source_behavior_analysis_execution_authorization_required"
    ]
    assert report["ready_for_analysis_execution"] is True
    assert report["case_outcomes_loaded"] is False
    assert report["descriptive_statistics_computed"] is False
    assert report["results_written"] is False


def test_default_cli_uses_offline_preflight(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        "evicon.conformity_source_behavior_analysis_execution._read_audits",
        lambda path: (_ for _ in ()).throw(AssertionError("audit parser must not run")),
    )
    assert main([]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "source_behavior_analysis_execution_gate_offline_ready"
    assert output["approval_status"] == "accepted"
    assert output["blocking_reasons"] == [
        "source_behavior_analysis_execution_authorization_required"
    ]
    assert output["case_outcomes_loaded"] is False


def test_explicit_flag_requires_confirmation() -> None:
    result = execute_analysis(allow_analysis=True)
    assert result.status == "blocked"
    assert result.error_code == "source_behavior_analysis_explicit_confirmation_required"
    assert result.case_outcomes_loaded is False


def test_accepted_fake_execution_writes_only_summary_and_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approval = _accepted_approval(tmp_path)
    output = tmp_path / "run"
    monkeypatch.setattr(
        "evicon.conformity_source_behavior_analysis_execution._read_audits",
        lambda path: _fake_audits(),
    )
    monkeypatch.setattr(
        "evicon.conformity_source_behavior_analysis_execution._new_output",
        lambda path: output if not output.exists() else (_ for _ in ()).throw(
            IdentificationError("source_behavior_analysis_output_exists")
        ),
    )
    result = execute_analysis(
        allow_analysis=True,
        confirm_run=True,
        approval_path=approval,
        output_root=output,
    )
    assert result.status == "completed"
    assert result.case_outcomes_loaded is True
    assert result.descriptive_statistics_computed is True
    assert set(path.name for path in output.iterdir()) == {SUMMARY_NAME, RECEIPT_NAME}
    receipt = load_execution_receipt(output / RECEIPT_NAME, approval_path=approval)
    assert receipt.case_count == 12
    assert receipt.summary_path == SUMMARY_NAME
    payload = json.loads((output / SUMMARY_NAME).read_text(encoding="utf-8"))
    assert len(payload["paired_summaries"]) == 4
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in output.iterdir()).lower()
    for forbidden in (
        "case_id",
        "scenario_id",
        "adopt_visible_claim",
        "withhold",
        "share_visible_content",
        "do_not_share",
        "raw_response",
        "system_prompt",
        '"api_key":',
    ):
        assert forbidden not in serialized


def test_execution_is_one_shot_and_does_not_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approval = _accepted_approval(tmp_path)
    output = tmp_path / "run"
    monkeypatch.setattr(
        "evicon.conformity_source_behavior_analysis_execution._read_audits",
        lambda path: _fake_audits(),
    )
    monkeypatch.setattr(
        "evicon.conformity_source_behavior_analysis_execution._new_output",
        lambda path: output if not output.exists() else (_ for _ in ()).throw(
            IdentificationError("source_behavior_analysis_output_exists")
        ),
    )
    first = execute_analysis(
        allow_analysis=True, confirm_run=True, approval_path=approval, output_root=output
    )
    assert first.status == "completed"
    marker = (output / SUMMARY_NAME).read_text(encoding="utf-8")
    second = execute_analysis(
        allow_analysis=True, confirm_run=True, approval_path=approval, output_root=output
    )
    assert second.status == "blocked"
    assert second.error_code == "source_behavior_analysis_output_exists"
    assert (output / SUMMARY_NAME).read_text(encoding="utf-8") == marker


def test_loader_rejects_tampered_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approval = _accepted_approval(tmp_path)
    output = tmp_path / "run"
    monkeypatch.setattr(
        "evicon.conformity_source_behavior_analysis_execution._read_audits",
        lambda path: _fake_audits(),
    )
    monkeypatch.setattr(
        "evicon.conformity_source_behavior_analysis_execution._new_output",
        lambda path: output,
    )
    result = execute_analysis(
        allow_analysis=True, confirm_run=True, approval_path=approval, output_root=output
    )
    assert result.status == "completed"
    summary = output / SUMMARY_NAME
    summary.write_text(summary.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(IdentificationError) as error:
        load_execution_receipt(output / RECEIPT_NAME, approval_path=approval)
    assert error.value.code == "source_behavior_analysis_execution_receipt_binding_mismatch"


def test_accepted_sidecar_still_requires_separate_command_authorization(
    tmp_path: Path,
) -> None:
    approval = _accepted_approval(tmp_path)
    report = safe_preflight(approval)
    assert report["approval_status"] == "accepted"
    assert report["blocking_reasons"] == [
        "source_behavior_analysis_execution_authorization_required"
    ]
    assert report["ready_for_analysis_execution"] is True
    assert report["case_outcomes_loaded"] is False


def test_sidecar_binds_current_analysis_lock() -> None:
    assert load_analysis_lock().analysis_id == "evicon-conformity-source-behavior-analysis-v2"
