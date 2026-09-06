"""Pure-function and online-boundary coverage for ConformityMonitor."""

from __future__ import annotations

import copy
import socket
from pathlib import Path

import pytest
from pydantic import ValidationError

from evicon.evaluation import OfflineEvaluator
from evicon.monitor import (
    ConformityMonitor,
    MonitorConfig,
    MonitorInput,
    MonitorReason,
    MonitorResult,
    RiskLevel,
)


def config() -> MonitorConfig:
    return MonitorConfig(
        collapse_weight=0.6,
        minority_weight=0.5,
        evidence_weight=0.5,
        evidence_quality_weight=0.1,
        harm_weight=0.6,
        collapse_trigger=0.3,
        minority_trigger=0.4,
        evidence_sufficient_threshold=0.7,
        high_risk_threshold=0.5,
        medium_risk_threshold=0.2,
        task_quality_sufficient_threshold=0.6,
        version="test-v1",
    )


def monitor_input(**overrides: object) -> MonitorInput:
    values: dict[str, object] = {
        "run_id": "monitor-run",
        "scenario_id": "monitor-scenario",
        "round_id": 1,
        "current_diversity": 0.6,
        "previous_diversity": 0.6,
        "current_coverage": 0.8,
        "previous_coverage": 0.8,
        "minority_loss": 0.0,
        "evidence_gain": 0.0,
        "evidence_quality": 0.0,
        "harm_risk": 0.0,
        "task_quality": 0.8,
        "remaining_budget": 0.0,
        "observable_peer_count": 2,
        "metadata": {"fixture": True},
    }
    values.update(overrides)
    return MonitorInput.model_validate(values)


def test_stable_diversity_is_low_risk_without_intervention() -> None:
    result = ConformityMonitor.evaluate(monitor_input(), config())

    assert result.risk_score == 0.0
    assert result.risk_level is RiskLevel.LOW
    assert not result.should_intervene
    assert not result.evidence_supported_update
    assert result.reasons == []


def test_diversity_and_coverage_loss_without_evidence_can_be_high_risk() -> None:
    result = ConformityMonitor.evaluate(
        monitor_input(
            current_diversity=0.0,
            previous_diversity=1.0,
            current_coverage=0.1,
            previous_coverage=0.9,
            minority_loss=0.8,
        ),
        config(),
    )

    assert result.risk_level is RiskLevel.HIGH
    assert result.should_intervene
    assert result.signals.collapse_amount == 1.0
    assert result.signals.coverage_loss == pytest.approx(0.8)
    assert MonitorReason.INSUFFICIENT_EVIDENCE in result.reasons


def test_sufficient_reliable_evidence_suppresses_default_intervention() -> None:
    result = ConformityMonitor.evaluate(
        monitor_input(
            current_diversity=0.0,
            previous_diversity=1.0,
            evidence_gain=1.0,
            evidence_quality=1.0,
            task_quality=0.8,
        ),
        config(),
    )

    assert result.evidence_supported_update
    assert not result.should_intervene
    assert MonitorReason.EVIDENCE_SUPPORTED_UPDATE in result.reasons


def test_minority_loss_can_raise_risk_without_average_diversity_change() -> None:
    result = ConformityMonitor.evaluate(monitor_input(minority_loss=1.0), config())

    assert result.risk_score == pytest.approx(0.5)
    assert result.signals.collapse_amount == 0.0
    assert MonitorReason.MINORITY_LOSS in result.reasons


def test_harm_risk_raises_risk_and_records_a_structured_reason() -> None:
    result = ConformityMonitor.evaluate(monitor_input(harm_risk=1.0), config())

    assert result.risk_score == pytest.approx(0.6)
    assert MonitorReason.HARM_RISK in result.reasons


def test_task_quality_is_recorded_without_being_labelled_as_homogeneity() -> None:
    result = ConformityMonitor.evaluate(monitor_input(task_quality=0.2), config())

    assert result.signals.task_quality == pytest.approx(0.2)
    assert MonitorReason.TASK_QUALITY_CONCERN in result.reasons
    assert result.risk_score == 0.0


def test_risk_score_is_clamped_to_the_unit_interval() -> None:
    high_weight_config = config().model_copy(
        update={"collapse_weight": 100.0, "minority_weight": 100.0, "harm_weight": 100.0}
    )
    result = ConformityMonitor.evaluate(
        monitor_input(
            current_diversity=0.0,
            previous_diversity=1.0,
            minority_loss=1.0,
            harm_risk=1.0,
        ),
        high_weight_config,
    )

    assert result.risk_score == 1.0
    assert 0.0 <= result.risk_score <= 1.0


def test_threshold_changes_can_change_the_operational_risk_level() -> None:
    monitor_case = monitor_input(current_diversity=0.0, previous_diversity=1.0)
    high = ConformityMonitor.evaluate(monitor_case, config())
    adjusted = ConformityMonitor.evaluate(
        monitor_case,
        config().model_copy(update={"high_risk_threshold": 0.9, "medium_risk_threshold": 0.3}),
    )

    assert high.risk_level is RiskLevel.HIGH
    assert adjusted.risk_level is RiskLevel.MEDIUM
    assert not adjusted.should_intervene


def test_same_input_and_config_produce_the_same_json_serializable_result() -> None:
    monitor_case = monitor_input(minority_loss=0.2, harm_risk=0.1)
    first = ConformityMonitor.evaluate(monitor_case, config())
    second = ConformityMonitor.evaluate(monitor_case, config())

    assert first == second
    assert MonitorResult.model_validate_json(first.model_dump_json()) == first


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("round_id", -1),
        ("observable_peer_count", -1),
        ("remaining_budget", -0.1),
        ("harm_risk", 1.1),
        ("current_diversity", float("nan")),
    ],
)
def test_invalid_monitor_input_is_rejected_with_a_clear_validation_error(
    field: str, value: object
) -> None:
    with pytest.raises(ValidationError, match=field):
        monitor_input(**{field: value})


def test_hidden_probe_fields_are_forbidden_from_monitor_input() -> None:
    values = monitor_input().model_dump()
    values["hidden_probe_response"] = "private"
    with pytest.raises(ValidationError, match="hidden_probe_response"):
        MonitorInput.model_validate(values)

    values = monitor_input().model_dump()
    values["hidden_value_profile"] = {"agent_id": "agent-1"}
    with pytest.raises(ValidationError, match="hidden_value_profile"):
        MonitorInput.model_validate(values)


def test_invalid_config_is_rejected_without_silent_threshold_reordering() -> None:
    with pytest.raises(ValidationError, match="medium_risk_threshold"):
        MonitorConfig.model_validate(
            {
                **config().model_dump(),
                "medium_risk_threshold": 0.8,
                "high_risk_threshold": 0.5,
            }
        )


def test_monitor_does_not_modify_its_input() -> None:
    monitor_case = monitor_input(current_diversity=0.2, previous_diversity=0.8)
    before = copy.deepcopy(monitor_case.model_dump())

    ConformityMonitor.evaluate(monitor_case, config())

    assert monitor_case.model_dump() == before


def test_monitor_creates_no_files(tmp_path: Path) -> None:
    before = list(tmp_path.iterdir())

    ConformityMonitor.evaluate(monitor_input(), config())

    assert list(tmp_path.iterdir()) == before


def test_monitor_does_not_open_network_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("ConformityMonitor attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", reject_connection)

    assert ConformityMonitor.evaluate(monitor_input(), config()).valid


def test_monitor_result_neither_generates_nor_contains_a_prompt() -> None:
    result = ConformityMonitor.evaluate(monitor_input(), config())

    assert "prompt" not in MonitorResult.model_fields
    assert "prompt" not in result.model_dump_json()


def test_offline_evaluator_is_not_automatically_called(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_evaluation(*args: object, **kwargs: object) -> object:
        raise AssertionError("ConformityMonitor invoked OfflineEvaluator")

    monkeypatch.setattr(OfflineEvaluator, "evaluate", reject_evaluation)

    assert ConformityMonitor.evaluate(monitor_input(), config()).valid
