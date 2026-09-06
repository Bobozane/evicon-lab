"""Tests for explicit four-condition matched batch orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.batch_experiment import (
    BASELINE_PROTOCOLS,
    BatchExperimentManifest,
    BatchExperimentStatus,
    BatchRunStatus,
)
from evicon.llm_contract import LLMProviderError, ProviderErrorCode

from batch_experiment_cases import BatchLocalProvider, batch_runner, batch_setup
from measured_experiment_cases import AGENT_MESSAGE, PROBE_TEXT, RAW_ANSWER


def test_manifest_explicitly_contains_all_four_conditions_and_deterministic_order(tmp_path) -> None:
    manifest, _, _, _, _ = batch_setup(tmp_path)

    assert tuple(manifest.conditions) == BASELINE_PROTOCOLS
    assert len(manifest.runs) == 4
    assert [run.condition for run in manifest.runs] == list(BASELINE_PROTOCOLS)
    assert manifest.run_order == [run.run_id for run in manifest.runs]
    assert len({run.matched_group_id for run in manifest.runs}) == 1
    assert all(run.status is BatchRunStatus.PLANNED for run in manifest.runs)
    assert all(run.artifacts.events_path.endswith("events.jsonl") for run in manifest.runs)
    assert all(run.artifacts.pre_probe_results_path.endswith("pre_probe_results.jsonl") for run in manifest.runs)


def test_manifest_rejects_duplicate_ids_mismatched_group_parameters_and_unsafe_metadata(tmp_path) -> None:
    manifest, _, _, _, _ = batch_setup(tmp_path)
    duplicate = manifest.runs[1].model_copy(
        update={"run_id": manifest.runs[0].run_id, "artifacts": manifest.runs[0].artifacts}
    )
    data = manifest.model_dump(mode="python")
    data["runs"] = [manifest.runs[0], duplicate, *manifest.runs[2:]]
    with pytest.raises(ValueError, match="duplicate run_id"):
        BatchExperimentManifest.model_validate(data)

    mismatched = manifest.runs[1].model_copy(update={"probe_item_ids": ["different-item"]})
    data = manifest.model_dump(mode="python")
    data["runs"] = [manifest.runs[0], mismatched, *manifest.runs[2:]]
    with pytest.raises(ValueError, match="probe item selection"):
        BatchExperimentManifest.model_validate(data)

    data = manifest.model_dump(mode="python")
    data["metadata"] = {"system_prompt": "not allowed"}
    with pytest.raises(ValueError, match="unsafe"):
        BatchExperimentManifest.model_validate(data)


def test_batch_runner_serially_completes_and_references_existing_single_run_outputs(tmp_path) -> None:
    runner, agent_provider, probe_provider = batch_runner(tmp_path)

    record = runner.run()

    assert record.status is BatchExperimentStatus.COMPLETED
    assert record.completed_run_count == 4
    assert record.failed_run_count == 0
    assert all(item.status is BatchRunStatus.COMPLETED for item in record.runs)
    assert all(group.is_complete for group in record.matched_groups)
    assert [request.metadata["protocol"] for request in agent_provider.requests] == [
        protocol.value for protocol in BASELINE_PROTOCOLS for _ in range(4)
    ]
    assert len(probe_provider.requests) == 16
    assert (runner.output_directory / "batch_manifest.json").is_file()
    assert (runner.output_directory / "batch_record.json").is_file()
    for item in record.runs:
        assert Path(item.artifacts.run_record_path).is_file()
        assert Path(item.artifacts.measurement_record_path).is_file()
    batch_manifest_text = (runner.output_directory / "batch_manifest.json").read_text(encoding="utf-8")
    batch_record_text = (runner.output_directory / "batch_record.json").read_text(encoding="utf-8")
    for forbidden in (PROBE_TEXT, RAW_ANSWER, AGENT_MESSAGE, "provider_metadata"):
        assert forbidden not in batch_manifest_text
        assert forbidden not in batch_record_text
    assert json.loads(batch_record_text)["status"] == "completed"


def test_batch_runner_fails_fast_and_marks_group_incomplete(tmp_path) -> None:
    failed_agent = BatchLocalProvider(
        kind="agent",
        error_at=5,
        error=LLMProviderError(ProviderErrorCode.HTTP_SERVER_ERROR, "api_key=do-not-leak"),
    )
    runner, agent_provider, _ = batch_runner(tmp_path, batch_id="fail-fast", agent_provider=failed_agent)

    record = runner.run()

    assert record.status is BatchExperimentStatus.FAILED
    assert record.completed_run_count == 1
    assert record.failed_run_count == 1
    assert [item.status for item in record.runs] == [
        BatchRunStatus.COMPLETED,
        BatchRunStatus.FAILED,
        BatchRunStatus.PLANNED,
        BatchRunStatus.PLANNED,
    ]
    assert record.matched_groups[0].is_complete is False
    assert len(agent_provider.requests) == 5
    assert "api_key" not in (runner.output_directory / "batch_record.json").read_text(encoding="utf-8")


def test_batch_and_run_directory_collisions_are_rejected_before_provider_calls(tmp_path) -> None:
    runner, agent_provider, _ = batch_runner(tmp_path, batch_id="occupied-batch")
    runner.output_directory.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="batch directory"):
        runner.run()
    assert agent_provider.requests == []

    runner, agent_provider, _ = batch_runner(tmp_path, batch_id="occupied-run")
    Path(runner.manifest.runs[0].artifacts.run_directory).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="run directory"):
        runner.run()
    assert agent_provider.requests == []
