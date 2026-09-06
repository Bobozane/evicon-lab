from __future__ import annotations

from pathlib import Path

import pytest

from evicon.conformity_identification import IdentificationError
from evicon.conformity_source_behavior_analysis_lock import (
    DEFAULT_ANALYSIS_APPROVAL,
    DEFAULT_ANALYSIS_LOCK,
    load_analysis_approval,
    load_analysis_lock,
    safe_preflight,
)


ROOT = Path(__file__).resolve().parents[1]


def test_default_lock_is_bound_and_outcome_blind_accepted_approval() -> None:
    plan = load_analysis_lock()
    approval = load_analysis_approval()
    report = safe_preflight()
    assert plan.status == "post_collection_outcome_blind_lock"
    assert approval.acceptance_status == "accepted"
    assert report["status"] == "source_behavior_analysis_lock_ready"
    assert report["blocking_reasons"] == [
        "source_behavior_analysis_execution_authorization_required"
    ]
    assert report["ready_for_analysis_execution"] is True
    assert report["case_outcomes_loaded"] is False
    assert report["descriptive_statistics_computed"] is False
    assert report["results_written"] is False
    assert report["network"] == "disabled"


def test_preflight_does_not_call_case_audit_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("case-level outcomes must remain unloaded")

    monkeypatch.setattr(
        "evicon.conformity_source_behavior_runner._read_safe_audits", forbidden
    )
    report = safe_preflight()
    assert report["status"] == "source_behavior_analysis_lock_ready"
    assert report["case_outcomes_loaded"] is False


def test_analysis_lock_rejects_unknown_field(tmp_path: Path) -> None:
    raw = (ROOT / DEFAULT_ANALYSIS_LOCK).read_text(encoding="utf-8")
    path = tmp_path / "analysis-lock.toml"
    path.write_text(raw + "\nunregistered_test = true\n", encoding="utf-8")
    with pytest.raises(IdentificationError) as error:
        load_analysis_lock(path)
    assert error.value.code == "source_behavior_analysis_lock_invalid"


def test_analysis_lock_rejects_input_hash_drift(tmp_path: Path) -> None:
    raw = (ROOT / DEFAULT_ANALYSIS_LOCK).read_text(encoding="utf-8")
    raw = raw.replace(
        "3ae8f0590132f938666c56eb8392bb7c0945f8a509647c87ab35d35d6c102b8b",
        "0ae8f0590132f938666c56eb8392bb7c0945f8a509647c87ab35d35d6c102b8b",
    )
    path = tmp_path / "analysis-lock.toml"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(IdentificationError) as error:
        load_analysis_lock(path)
    assert error.value.code == "source_behavior_analysis_input_hash_mismatch"


def test_analysis_lock_rejects_projection_order_change(tmp_path: Path) -> None:
    raw = (ROOT / DEFAULT_ANALYSIS_LOCK).read_text(encoding="utf-8")
    raw = raw.replace(
        '["source_free", "same_root", "independent_roots"]',
        '["same_root", "source_free", "independent_roots"]',
    )
    path = tmp_path / "analysis-lock.toml"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(IdentificationError) as error:
        load_analysis_lock(path)
    assert error.value.code == "source_behavior_analysis_lock_invalid"


def test_analysis_approval_rejects_unknown_field(tmp_path: Path) -> None:
    raw = (ROOT / DEFAULT_ANALYSIS_APPROVAL).read_text(encoding="utf-8")
    path = tmp_path / "approval.toml"
    path.write_text(raw + "\ncase_values_reviewed = true\n", encoding="utf-8")
    with pytest.raises(IdentificationError) as error:
        load_analysis_approval(path)
    assert error.value.code == "source_behavior_analysis_approval_invalid"


def test_accepted_analysis_approval_requires_every_confirmation(tmp_path: Path) -> None:
    raw = (ROOT / DEFAULT_ANALYSIS_APPROVAL).read_text(encoding="utf-8")
    raw = raw.replace(
        "confirm_fixed_order_confounding = true",
        "confirm_fixed_order_confounding = false",
    )
    path = tmp_path / "approval.toml"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(IdentificationError) as error:
        load_analysis_approval(path)
    assert error.value.code == "source_behavior_analysis_approval_invalid"


def test_accepted_lock_still_requires_execution_authorization(tmp_path: Path) -> None:
    raw = (ROOT / DEFAULT_ANALYSIS_APPROVAL).read_text(encoding="utf-8")
    path = tmp_path / "approval.toml"
    path.write_text(raw, encoding="utf-8")
    report = safe_preflight(approval_path=path)
    assert report["approval_status"] == "accepted"
    assert report["blocking_reasons"] == [
        "source_behavior_analysis_execution_authorization_required"
    ]
    assert report["ready_for_analysis_execution"] is True
    assert report["case_outcomes_loaded"] is False
    assert report["descriptive_statistics_computed"] is False
    assert report["results_written"] is False
