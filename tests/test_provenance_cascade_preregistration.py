from __future__ import annotations

import copy
import json
import tomllib
from pathlib import Path

import pytest

from evicon.provenance_cascade_preregistration import (
    ProvenanceCascadePilotPreregistration,
    ProvenanceCascadePreregistrationLoader,
    load_and_validate_provenance_cascade_preregistration,
)
from evicon.validate_provenance_cascade_preregistration import main


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/studies/provenance_cascade_pilot_preregistration.toml"


def payload() -> dict[str, object]:
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))


def test_valid_pilot_summary_is_safe_and_fixed() -> None:
    summary = load_and_validate_provenance_cascade_preregistration(CONFIG)
    assert summary.scenario_count == 4
    assert summary.condition_count == 4
    assert summary.agent_count == 6
    assert summary.round_count == 3
    assert summary.seed_count == 3
    assert summary.planned_logical_run_count == 48
    assert summary.evaluator_labels_isolated is True
    assert summary.abstain_scenarios == ["independent_true_consensus", "unresolved_disagreement"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scenario_types", []),
        ("conditions", ["no_intervention"]),
        ("primary_metrics", ["token_cost"]),
        ("max_rounds", 2),
        ("seeds", [20260901, 20260902]),
    ],
)
def test_fixed_shape_rejects_missing_declarations(field: str, value: object) -> None:
    data = payload()
    data[field] = value
    with pytest.raises(ValueError):
        ProvenanceCascadePilotPreregistration.model_validate(data)


def test_duplicate_scenario_is_rejected() -> None:
    data = payload()
    data["scenario_types"][1]["scenario_type"] = "false_majority"  # type: ignore[index]
    with pytest.raises(ValueError):
        ProvenanceCascadePilotPreregistration.model_validate(data)


def test_private_truth_cannot_be_public_controller_input() -> None:
    data = payload()
    data["information_boundary"]["controller_allowed_public_fields"].append("ground_truth_label")  # type: ignore[index]
    with pytest.raises(ValueError):
        ProvenanceCascadePilotPreregistration.model_validate(data)


def test_majority_or_minority_trigger_is_rejected() -> None:
    data = payload()
    data["conditions"][0]["triggers_on_majority_count"] = True  # type: ignore[index]
    with pytest.raises(ValueError):
        ProvenanceCascadePilotPreregistration.model_validate(data)


def test_uncertain_scenarios_must_declare_abstain() -> None:
    data = payload()
    data["success_criterion"]["abstention_is_correct_for"] = ["false_majority"]  # type: ignore[index]
    with pytest.raises(ValueError):
        ProvenanceCascadePilotPreregistration.model_validate(data)


@pytest.mark.parametrize("field", ["wvs_config_path", "results_path", "legacy_run_record_path", "unexpected_prompt"])
def test_wvs_and_old_result_paths_are_not_allowed(field: str) -> None:
    data = payload()
    data[field] = "must not be mixed"  # type: ignore[index]
    with pytest.raises(ValueError):
        ProvenanceCascadePilotPreregistration.model_validate(data)


def test_cli_emits_only_safe_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(CONFIG)]) == 0
    output = capsys.readouterr().out
    parsed = json.loads(output)
    assert parsed["planned_logical_run_count"] == 48
    assert parsed["runnable_offline"] is True
    lowered = output.lower()
    assert "prompt" not in lowered
    assert "api_key" not in lowered
    assert "ground_truth_label" not in lowered
    assert "results" not in lowered


def test_missing_config_is_blocked(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(ROOT / "missing-cascade.toml")]) == 1
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["status"] == "blocked"
