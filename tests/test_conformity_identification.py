from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from evicon.conformity_identification import (
    DEFAULT_CONFIG,
    ConformityIdentificationConfig,
    IdentificationCondition,
    IdentificationError,
    IdentificationStage,
    load_identification_config,
    load_scenario,
    safe_preflight,
    visible_roles,
)
from evicon.conformity_identification_protocol import (
    IdentificationPromptContext,
    parse_identification_response,
    render_identification_turn,
)
from evicon.conformity_identification_smoke import _context, run_fake_smoke

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / DEFAULT_CONFIG


def _config_data() -> dict[str, object]:
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))


def test_locked_design_has_six_conditions_five_stages_and_seventy_two_runs():
    config, scenarios = load_identification_config(CONFIG)
    assert config.conditions == tuple(IdentificationCondition)
    assert config.stages == tuple(IdentificationStage)
    assert len(scenarios) == 4
    assert config.run_count == 72
    assert config.matched_group_count == 12
    assert config.logical_request_cap == 2160
    assert config.completion_reservation_cap == 1_105_920
    assert config.controller_enabled is False
    assert config.intervention_enabled is False


def test_unknown_config_field_is_rejected():
    data = _config_data()
    data["provider_api_key"] = "forbidden"
    with pytest.raises(ValidationError):
        ConformityIdentificationConfig.model_validate(data)


def test_missing_condition_and_wrong_contrast_are_rejected():
    data = _config_data()
    data["conditions"] = data["conditions"][:-1]
    with pytest.raises(ValidationError):
        ConformityIdentificationConfig.model_validate(data)
    data = _config_data()
    data["contrasts"][0]["subtrahend"] = "private_baseline"
    with pytest.raises(ValidationError):
        ConformityIdentificationConfig.model_validate(data)


def test_scenario_hash_mismatch_is_rejected(tmp_path: Path):
    data = _config_data()
    data["scenario_bindings"][0]["sha256"] = "0" * 64
    path = tmp_path / "bad.toml"
    text = CONFIG.read_text(encoding="utf-8").replace(
        data["scenario_bindings"][1]["sha256"], data["scenario_bindings"][1]["sha256"]
    )
    original = _config_data()["scenario_bindings"][0]["sha256"]
    path.write_text(text.replace(original, "0" * 64), encoding="utf-8")
    with pytest.raises(IdentificationError, match="scenario_hash_mismatch"):
        load_identification_config(path)


def test_public_scenario_rejects_private_truth_marker(tmp_path: Path):
    source = ROOT / "configs/provenance_cascade/identification/scenarios/false-majority.v1.toml"
    path = tmp_path / "leak.toml"
    path.write_text(
        source.read_text(encoding="utf-8").replace(
            "A neighborhood bulletin says", "ground_truth_label false; a neighborhood bulletin says"
        ),
        encoding="utf-8",
    )
    with pytest.raises(IdentificationError, match="scenario_invalid"):
        load_scenario(path)


def test_repetition_arms_match_text_and_timing_but_not_source_projection():
    config, scenarios = load_identification_config(CONFIG)
    scenario = scenarios[0]
    stage = IdentificationStage.SOCIAL_EXPOSURE
    contexts = {
        condition: _context(scenario, condition, config.seeds[0], config.agent_ids[0], stage)
        for condition in (
            IdentificationCondition.SOURCE_FREE_REPETITION,
            IdentificationCondition.SAME_ROOT_SOCIAL,
            IdentificationCondition.INDEPENDENT_ROOTS,
        )
    }
    assert len({context.visible_stimulus_ids for context in contexts.values()}) == 1
    assert len({context.visible_public_summaries for context in contexts.values()}) == 1
    assert contexts[IdentificationCondition.SOURCE_FREE_REPETITION].visible_source_root_ids == ()
    assert len(set(contexts[IdentificationCondition.SAME_ROOT_SOCIAL].visible_source_root_ids)) == 1
    assert len(set(contexts[IdentificationCondition.INDEPENDENT_ROOTS].visible_source_root_ids)) == 2


def test_initial_private_inputs_are_condition_invariant():
    config, scenarios = load_identification_config(CONFIG)
    scenario = scenarios[0]
    contexts = [
        _context(scenario, condition, config.seeds[0], config.agent_ids[0], IdentificationStage.INITIAL_PRIVATE)
        for condition in config.conditions
    ]
    assert len({item.visible_stimulus_ids for item in contexts}) == 1
    assert all(item.visible_source_root_ids == () for item in contexts)
    assert all(item.visible_evidence_ids == () for item in contexts)


def test_rendered_request_does_not_expose_condition_or_private_labels():
    config, scenarios = load_identification_config(CONFIG)
    context = _context(
        scenarios[0], IdentificationCondition.SAME_ROOT_SOCIAL,
        config.seeds[0], config.agent_ids[0], IdentificationStage.SOCIAL_EXPOSURE,
    )
    request = render_identification_turn(context)
    serialized = request.user_prompt.lower()
    assert "same_root_social" not in serialized
    assert "ground_truth_label" not in serialized
    assert "source_independence_label" not in serialized
    assert "evaluator_private" not in serialized


def test_strict_parser_rejects_unknown_and_unavailable_fields():
    context = IdentificationPromptContext(
        scenario_id="scenario", agent_id="agent", seed=1,
        stage=IdentificationStage.INITIAL_PRIVATE, target_claim_id="claim",
        decision_task="Make a reversible public planning recommendation from visible material.",
        visible_stimulus_ids=("visible",), visible_public_summaries=("A sufficiently long public synthetic statement is visible.",),
        visible_source_root_ids=(), visible_evidence_ids=(), reflection_only=False,
    )
    valid = {
        "ordinal_judgment": "lean_adopt", "public_action": "adopt",
        "share_decision": "do_not_share", "confidence": 4,
        "content_ids_used": ["visible"], "evidence_ids_used": [],
    }
    assert parse_identification_response(json.dumps(valid), context).public_action.value == "adopt"
    with pytest.raises(ValueError, match="invalid_schema"):
        parse_identification_response(json.dumps({**valid, "truth": "false"}), context)
    with pytest.raises(ValueError, match="unavailable_content_id"):
        parse_identification_response(json.dumps({**valid, "content_ids_used": ["hidden"]}), context)


def test_fake_smoke_validates_eligibility_without_estimating_effect():
    report = run_fake_smoke(str(CONFIG))
    assert report["status"] == "qualification_smoke_passed"
    assert report["run_count"] == 72
    assert report["logical_request_count"] == 2160
    assert report["provider_call_count"] == 2160
    assert report["initial_substantive_count"] == 288
    assert report["correction_transition_eligible_count"] == 24
    assert report["social_conformity_contrast_estimable"] is True
    assert report["evidence_receptivity_contrast_estimable"] is True
    assert report["effect_estimated"] is False
    assert report["private_truth_exposed"] is False


def test_preflight_is_safe_and_remains_blocked_for_network():
    report = safe_preflight(CONFIG)
    assert report["status"] == "offline_design_ready"
    assert report["ready_for_network"] is False
    assert report["blocking_reasons"] == ["human_approval_required", "provider_compatibility_not_requested"]
    assert report["network"] == "disabled"
    assert report["provider_constructed"] is False
    assert report["api_key_read"] is False
    assert report["results_written"] is False


def test_no_historical_or_results_paths_are_inputs():
    config, _ = load_identification_config(CONFIG)
    serialized = config.model_dump_json().lower()
    for forbidden in ("hg232", "wvs", "002", "003", "results/batches", "evaluator_private"):
        assert forbidden not in serialized
    assert not (ROOT / config.output_root).exists()


def test_stage_visibility_is_monotonic_and_evidence_is_late():
    stages = tuple(IdentificationStage)
    verified = [set(visible_roles(IdentificationCondition.VERIFIED_EVIDENCE, stage)) for stage in stages]
    assert all(earlier.issubset(later) for earlier, later in zip(verified, verified[1:]))
    assert "evidence" not in verified[2]
    assert "evidence" in verified[3]
