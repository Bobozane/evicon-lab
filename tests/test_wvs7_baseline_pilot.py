"""Tests for the local-only matched WVS English baseline pilot plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evicon.batch_experiment import BatchExperimentManifest, BatchExperimentStatus, BatchRunStatus
from evicon.freeze_wvs7_probe_set import EXPECTED_VARIABLE_IDS
from evicon.frozen_wvs7_manifest import FrozenWVS7Manifest, sha256_file
from evicon.models import ProbeSet
from evicon.replay import validate_replay
from evicon.wvs7_baseline_pilot import (
    PreparedWVS7BaselinePilot,
    WVS7BaselinePilotConfig,
    WVS7BaselinePilotError,
    WVS7BaselinePilotRunner,
    WVS7BaselineFakeLLMProvider,
    _build_batch_manifest,
    _validate_frozen_probe_set,
    dry_run_wvs7_baseline_pilot,
    load_wvs7_baseline_pilot_config,
    prepare_wvs7_baseline_pilot,
)
from evicon import wvs7_baseline_dry_run as dry_run_cli


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/studies/wvs7_english_baseline_pilot.toml"


def _prepared(tmp_path: Path) -> PreparedWVS7BaselinePilot:
    """Use immutable production inputs but direct generated artifacts to tmp_path."""
    original = prepare_wvs7_baseline_pilot(CONFIG)
    raw = original.config.model_dump(mode="python")
    raw["run_output_dir"] = str(tmp_path / "runs")
    raw["batch_output_dir"] = str(tmp_path / "batches")
    config = WVS7BaselinePilotConfig.model_validate(raw)
    return PreparedWVS7BaselinePilot(
        config=config,
        scenario=original.scenario,
        probe_set=original.probe_set,
        frozen_manifest=original.frozen_manifest,
        batch_manifest=_build_batch_manifest(config, original.scenario, original.probe_set),
    )


def _write_probe_and_manifest(tmp_path: Path, probe_set: ProbeSet, manifest: FrozenWVS7Manifest) -> tuple[Path, FrozenWVS7Manifest]:
    path = tmp_path / "frozen.json"
    path.write_text(probe_set.model_dump_json(indent=2), encoding="utf-8")
    updated = manifest.model_copy(update={
        "probe_set_sha256": sha256_file(path),
        "item_count": len(probe_set.items),
        "variable_ids": [item.probe_id.removeprefix("wvs7-english-core-") for item in probe_set.items],
    })
    return path, updated


def test_frozen_probe_hash_and_item_contracts_are_checked(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    config = prepared.config
    probe_path = ROOT / config.frozen_probe_set_path

    with pytest.raises(WVS7BaselinePilotError, match="SHA-256"):
        _validate_frozen_probe_set(
            config.model_copy(update={"frozen_probe_set_sha256": "0" * 64}),
            probe_path,
            prepared.frozen_manifest,
        )

    without_item = prepared.probe_set.model_copy(update={"items": prepared.probe_set.items[:-1]})
    altered_path, altered_manifest = _write_probe_and_manifest(tmp_path, without_item, prepared.frozen_manifest)
    altered_config = config.model_copy(update={"frozen_probe_set_sha256": sha256_file(altered_path)})
    with pytest.raises(WVS7BaselinePilotError, match="invalid item contract"):
        _validate_frozen_probe_set(altered_config, altered_path, altered_manifest)

    q48_item = prepared.probe_set.items[0].model_copy(update={"probe_id": "wvs7-english-core-Q48"})
    q48_probe = prepared.probe_set.model_copy(update={"items": [q48_item, *prepared.probe_set.items[1:]]})
    q48_path, q48_manifest = _write_probe_and_manifest(tmp_path, q48_probe, prepared.frozen_manifest)
    q48_config = config.model_copy(update={"frozen_probe_set_sha256": sha256_file(q48_path)})
    with pytest.raises(WVS7BaselinePilotError, match="invalid item contract"):
        _validate_frozen_probe_set(q48_config, q48_path, q48_manifest)


def test_config_and_batch_manifest_reject_incomplete_or_mismatched_conditions(tmp_path: Path) -> None:
    config = load_wvs7_baseline_pilot_config(CONFIG)
    raw = config.model_dump(mode="python")
    raw["runs"] = raw["runs"][:-1]
    with pytest.raises(ValueError, match="at least 4"):
        WVS7BaselinePilotConfig.model_validate(raw)

    manifest = _prepared(tmp_path).batch_manifest
    mismatched = manifest.runs[1].model_copy(update={"agent_ids": ["different-agent"]})
    payload = manifest.model_dump(mode="python")
    payload["runs"] = [manifest.runs[0], mismatched, *manifest.runs[2:]]
    with pytest.raises(ValueError, match="ordered agent IDs"):
        BatchExperimentManifest.model_validate(payload)

    for field, replacement, expected in (
        ("scenario_id", "different-scenario", "scenario_id"),
        ("model_name", "different-model", "model_name"),
        ("probe_set_id", "different-probe-set", "probe_set_id"),
    ):
        payload = manifest.model_dump(mode="python")
        changed = manifest.runs[1].model_copy(update={field: replacement})
        payload["runs"] = [manifest.runs[0], changed, *manifest.runs[2:]]
        with pytest.raises(ValueError, match=expected):
            BatchExperimentManifest.model_validate(payload)

    changed_seed = manifest.runs[2].model_copy(update={"seed": 99, "probe_seed": 99})
    payload = manifest.model_dump(mode="python")
    payload["runs"] = [manifest.runs[0], manifest.runs[1], changed_seed, manifest.runs[3]]
    with pytest.raises(ValueError, match="explicitly listed"):
        BatchExperimentManifest.model_validate(payload)

    changed_parameters = manifest.runs[3].model_copy(
        update={"runtime_parameters": manifest.runs[3].runtime_parameters.model_copy(update={"agent_max_tokens": 17})}
    )
    payload = manifest.model_dump(mode="python")
    payload["runs"] = [*manifest.runs[:3], changed_parameters]
    with pytest.raises(ValueError, match="inconsistent non-condition parameters"):
        BatchExperimentManifest.model_validate(payload)


def test_dry_run_is_deterministic_and_has_no_provider_requests(tmp_path: Path) -> None:
    summary = dry_run_wvs7_baseline_pilot(_prepared(tmp_path))

    assert summary.condition_count == 4
    assert summary.seed_count == 1
    assert summary.agent_turn_count == 16
    assert summary.pre_probe_request_count == 184
    assert summary.post_probe_request_count == 184
    assert summary.provider_request_upper_bound == 384
    assert summary.token_upper_bound == 51200
    assert summary.matched_groups[0]["complete"] is True
    assert not list(tmp_path.iterdir())


def test_dry_run_cli_emits_only_safe_summary(capsys: pytest.CaptureFixture[str]) -> None:
    assert dry_run_cli.main(["--config", str(CONFIG)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider_request_upper_bound"] == 384
    assert payload["token_upper_bound"] == 51200
    assert "question_text" not in json.dumps(payload)
    assert "prompt" not in json.dumps(payload)


def test_fake_smoke_completes_four_conditions_replays_and_writes_safe_offline_report(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    runner = WVS7BaselinePilotRunner(prepared, batch_id="fake-smoke")

    record, evaluation = runner.run_fake_smoke()

    assert record.status is BatchExperimentStatus.COMPLETED
    assert record.completed_run_count == 4
    assert record.failed_run_count == 0
    assert all(item.status is BatchRunStatus.COMPLETED for item in record.runs)
    assert all(group.is_complete for group in record.matched_groups)
    assert evaluation is not None
    assert {"pairwise_diversity", "structural_diversity", "value_dimension_coverage", "social_influence_loss"}.issubset(
        set(evaluation["metric_names"])
    )
    # The WVS frozen set has no independent holdout subset, so a holdout drift
    # metric is intentionally not claimed for this engineering smoke run.
    assert "holdout_profile_drift" not in evaluation["metric_names"]

    for spec in prepared.batch_manifest.runs:
        replay = validate_replay(spec.artifacts.events_path, spec.artifacts.run_record_path)
        assert replay.validation == "passed"
        run_record = json.loads(Path(spec.artifacts.run_record_path).read_text(encoding="utf-8"))
        by_turn_id = {turn["turn_id"]: turn for turn in run_record["turns"]}
        for turn in run_record["turns"]:
            assert all(by_turn_id[turn_id]["round_id"] < turn["round_id"] for turn_id in turn["visible_peer_turn_ids"])
        if spec.condition.value == "independent":
            assert all(not turn["visible_peer_turn_ids"] and not turn["visible_evidence_ids"] for turn in run_record["turns"])
        if spec.condition.value == "social_only":
            assert all(not turn["visible_evidence_ids"] for turn in run_record["turns"])
            assert any(turn["visible_peer_turn_ids"] for turn in run_record["turns"] if turn["round_id"] == 1)
        if spec.condition.value == "evidence_only":
            assert all(not turn["visible_peer_turn_ids"] for turn in run_record["turns"])
            assert any(turn["visible_evidence_ids"] for turn in run_record["turns"])
        if spec.condition.value == "evidence_social":
            assert any(turn["visible_peer_turn_ids"] for turn in run_record["turns"] if turn["round_id"] == 1)
            assert any(turn["visible_evidence_ids"] for turn in run_record["turns"])

        events_text = Path(spec.artifacts.events_path).read_text(encoding="utf-8")
        for unsafe_key in ("system_prompt", "user_prompt", "provider_metadata", "api_key", "hidden_probe"):
            assert unsafe_key not in events_text
        assert all(item.text not in events_text for item in prepared.probe_set.items)

    report_text = (runner.batch_directory / "offline_evaluation_report.json").read_text(encoding="utf-8")
    for unsafe_key in ("system_prompt", "user_prompt", "provider_metadata", "api_key"):
        assert unsafe_key not in report_text


def test_failed_group_keeps_planned_followups_and_resumes_only_safe_incomplete_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = _prepared(tmp_path)
    failing = WVS7BaselinePilotRunner(prepared, batch_id="failed-group")

    def fail_once(*_: object, **__: object) -> object:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(failing, "_run_condition", fail_once)
    record, _ = failing.run_fake_smoke()
    assert record.status is BatchExperimentStatus.FAILED
    assert [item.status for item in record.runs] == [
        BatchRunStatus.FAILED,
        BatchRunStatus.PLANNED,
        BatchRunStatus.PLANNED,
        BatchRunStatus.PLANNED,
    ]
    assert record.matched_groups[0].is_complete is False

    # A batch interrupted after a completed condition can resume its remaining
    # planned work without overwriting the completed run directory.
    resumed_prepared = _prepared(tmp_path / "resumable")
    resume_runner = WVS7BaselinePilotRunner(resumed_prepared, batch_id="resumable-group")
    first_spec = resumed_prepared.batch_manifest.runs[0]
    resume_runner._run_condition(first_spec, provider=WVS7BaselineFakeLLMProvider())
    initial_records = resume_runner._initial_or_resumed_records(resume=False)
    partial_records = [
        record.model_copy(update={"status": BatchRunStatus.COMPLETED, "measurement_status": "completed", "error_code": None})
        if record.run_id == first_spec.run_id else record
        for record in initial_records
    ]
    resume_runner._write_record(resume_runner._record(partial_records, BatchExperimentStatus.RUNNING, error_code=None))
    resumed, _ = resume_runner.run_fake_smoke(resume=True)
    assert resumed.status is BatchExperimentStatus.COMPLETED
    assert resumed.completed_run_count == 4


def test_expected_variable_contract_remains_23_non_q48_items() -> None:
    prepared = prepare_wvs7_baseline_pilot(CONFIG)
    variables = [item.probe_id.removeprefix("wvs7-english-core-") for item in prepared.probe_set.items]
    assert tuple(variables) == EXPECTED_VARIABLE_IDS
    assert "Q48" not in variables
    assert hashlib.sha256((ROOT / "outputs/wvs7-frozen/english_core_wvs7_23_frozen.json").read_bytes()).hexdigest() == prepared.config.frozen_probe_set_sha256
