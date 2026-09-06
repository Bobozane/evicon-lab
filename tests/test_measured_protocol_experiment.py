"""Tests for staged, offline-only measurement around one real-Agent baseline."""

from __future__ import annotations

import hashlib
import json

import pytest

from evicon.llm_contract import LLMProviderError, ProviderErrorCode
from evicon.measured_protocol_experiment import MeasurementExperimentStatus, MeasurementPhaseStatus
from evicon.models import ProtocolCondition, RunStatus
from evicon.probe_results import read_probe_results
from evicon.replay import validate_replay

from measured_experiment_cases import AGENT_MESSAGE, PROBE_TEXT, RAW_ANSWER, RecordedProvider, experiment


def test_pre_trajectory_post_sequence_writes_only_phase_isolated_results(tmp_path) -> None:
    call_order: list[str] = []
    runner = experiment(tmp_path, call_order=call_order)

    result = runner.run()

    assert result.status is MeasurementExperimentStatus.COMPLETED
    assert call_order == ["probe", "probe", "agent", "agent", "agent", "agent", "probe", "probe"]
    assert result.run_record is not None and result.run_record.value_profiles == []
    assert result.measurement_record is not None
    measurement = result.measurement_record
    assert measurement.pre_probe_status is MeasurementPhaseStatus.COMPLETED
    assert measurement.post_probe_status is MeasurementPhaseStatus.COMPLETED
    assert measurement.probe_item_ids == ["private-probe-id-marker"]
    assert measurement.probe_seed == 29
    assert measurement.pre_probe_result_path == "probes/pre_probe_results.jsonl"
    assert measurement.post_probe_result_path == "probes/post_probe_results.jsonl"

    pre_path = runner.output_directory / "probes" / "pre_probe_results.jsonl"
    post_path = runner.output_directory / "probes" / "post_probe_results.jsonl"
    assert len(read_probe_results(pre_path)) == 2
    assert len(read_probe_results(post_path)) == 2
    assert measurement.pre_probe_sha256 == hashlib.sha256(pre_path.read_bytes()).hexdigest()
    assert measurement.post_probe_sha256 == hashlib.sha256(post_path.read_bytes()).hexdigest()

    measurement_json = (runner.output_directory / "measurement_record.json").read_text(encoding="utf-8")
    events_json = (runner.output_directory / "events.jsonl").read_text(encoding="utf-8")
    record_json = (runner.output_directory / "run_record.json").read_text(encoding="utf-8")
    for private_value in (PROBE_TEXT, RAW_ANSWER, AGENT_MESSAGE):
        assert private_value not in measurement_json
    assert PROBE_TEXT not in events_json
    assert PROBE_TEXT not in record_json
    assert AGENT_MESSAGE in events_json
    assert AGENT_MESSAGE in record_json
    agent_requests = getattr(runner.agent_provider, "requests")
    assert PROBE_TEXT not in "\n".join(request.user_prompt for request in agent_requests)
    assert validate_replay(runner.output_directory / "events.jsonl", runner.output_directory / "run_record.json").validation == "passed"


@pytest.mark.parametrize(
    ("protocol", "peer_visible", "evidence_visible"),
    [
        (ProtocolCondition.INDEPENDENT, False, False),
        (ProtocolCondition.SOCIAL_ONLY, True, False),
        (ProtocolCondition.EVIDENCE_ONLY, False, True),
        (ProtocolCondition.EVIDENCE_SOCIAL, True, True),
    ],
)
def test_post_contexts_preserve_all_four_protocol_visibility_boundaries(
    tmp_path,
    protocol: ProtocolCondition,
    peer_visible: bool,
    evidence_visible: bool,
) -> None:
    call_order: list[str] = []
    probe_provider = RecordedProvider("probe", call_order, content=json.dumps({"choice": RAW_ANSWER}))
    runner = experiment(tmp_path, protocol=protocol, run_id=f"run-{protocol.value}", probe_provider=probe_provider, call_order=call_order)

    result = runner.run()
    post_contexts = [
        json.loads(request.user_prompt)["measurement_context"]
        for request in probe_provider.requests[2:]
    ]
    pre_contexts = [
        json.loads(request.user_prompt)["measurement_context"]
        for request in probe_provider.requests[:2]
    ]

    assert result.status is MeasurementExperimentStatus.COMPLETED
    assert any(context["visible_peer_turns"] for context in post_contexts) is peer_visible
    assert any(context["visible_evidence_cards"] for context in post_contexts) is evidence_visible
    assert all(not context["own_history"] for context in pre_contexts)
    assert all(not context["visible_peer_turns"] and not context["visible_evidence_cards"] for context in pre_contexts)


def test_pre_failure_does_not_start_public_run_or_create_output_directory(tmp_path) -> None:
    call_order: list[str] = []
    probe_provider = RecordedProvider(
        "probe",
        call_order,
        content=json.dumps({"choice": RAW_ANSWER}),
        error_at=1,
        error=LLMProviderError(ProviderErrorCode.HTTP_SERVER_ERROR, "api_key=do-not-leak"),
    )
    agent_provider = RecordedProvider("agent", call_order, content=json.dumps({"message": AGENT_MESSAGE, "evidence_ids_used": []}))
    runner = experiment(tmp_path, probe_provider=probe_provider, agent_provider=agent_provider, call_order=call_order)

    result = runner.run()

    assert result.status is MeasurementExperimentStatus.PRE_PROBE_FAILED
    assert result.measurement_record is None
    assert call_order == ["probe"]
    assert agent_provider.requests == []
    assert not runner.output_directory.exists()


def test_pre_probe_resume_reuses_private_checkpoint_without_replaying_completed_requests(tmp_path) -> None:
    checkpoint = tmp_path / "private-recovery" / "pre-resume"
    first_order: list[str] = []
    first_probe = RecordedProvider(
        "probe",
        first_order,
        content=json.dumps({"choice": RAW_ANSWER}),
        error_at=2,
        error=LLMProviderError(ProviderErrorCode.HTTP_SERVER_ERROR, "transient failure"),
    )
    first_agent = RecordedProvider("agent", first_order, content=json.dumps({"message": AGENT_MESSAGE, "evidence_ids_used": []}))
    failed = experiment(
        tmp_path,
        run_id="pre-resume",
        probe_provider=first_probe,
        agent_provider=first_agent,
        call_order=first_order,
        recovery_checkpoint_dir=checkpoint,
    ).run()

    assert failed.status is MeasurementExperimentStatus.PRE_PROBE_FAILED
    assert failed.error_code == ProviderErrorCode.HTTP_SERVER_ERROR.value
    assert len(first_probe.requests) == 2
    assert first_agent.requests == []
    assert not (tmp_path / "results" / "pre-resume").exists()
    checkpoint_text = (checkpoint / "pre_responses.json").read_text(encoding="utf-8")
    assert RAW_ANSWER in checkpoint_text
    assert PROBE_TEXT not in checkpoint_text

    resumed_order: list[str] = []
    resumed_probe = RecordedProvider("probe", resumed_order, content=json.dumps({"choice": RAW_ANSWER}))
    resumed_agent = RecordedProvider("agent", resumed_order, content=json.dumps({"message": AGENT_MESSAGE, "evidence_ids_used": []}))
    resumed = experiment(
        tmp_path,
        run_id="pre-resume",
        probe_provider=resumed_probe,
        agent_provider=resumed_agent,
        call_order=resumed_order,
        recovery_checkpoint_dir=checkpoint,
    ).run(resume=True)

    assert resumed.status is MeasurementExperimentStatus.COMPLETED
    assert len(resumed_probe.requests) == 3  # One remaining pre probe, then two post probes.
    assert len(resumed_agent.requests) == 4
    assert not checkpoint.exists()


def test_post_probe_resume_reuses_private_checkpoint_without_replaying_public_trajectory(tmp_path) -> None:
    checkpoint = tmp_path / "private-recovery" / "post-resume"
    first_order: list[str] = []
    first_probe = RecordedProvider(
        "probe",
        first_order,
        content=json.dumps({"choice": RAW_ANSWER}),
        error_at=3,
        error=LLMProviderError(ProviderErrorCode.HTTP_SERVER_ERROR, "transient failure"),
    )
    first_agent = RecordedProvider("agent", first_order, content=json.dumps({"message": AGENT_MESSAGE, "evidence_ids_used": []}))
    failed = experiment(
        tmp_path,
        run_id="post-resume",
        probe_provider=first_probe,
        agent_provider=first_agent,
        call_order=first_order,
        recovery_checkpoint_dir=checkpoint,
    ).run()

    assert failed.status is MeasurementExperimentStatus.POST_PROBE_FAILED
    assert len(first_probe.requests) == 3
    assert len(first_agent.requests) == 4
    assert (checkpoint / "pre_responses.json").is_file()
    assert (checkpoint / "post_responses.json").is_file()

    resumed_order: list[str] = []
    resumed_probe = RecordedProvider("probe", resumed_order, content=json.dumps({"choice": RAW_ANSWER}))
    resumed_agent = RecordedProvider("agent", resumed_order, content=json.dumps({"message": AGENT_MESSAGE, "evidence_ids_used": []}))
    resumed = experiment(
        tmp_path,
        run_id="post-resume",
        probe_provider=resumed_probe,
        agent_provider=resumed_agent,
        call_order=resumed_order,
        recovery_checkpoint_dir=checkpoint,
    ).run(resume=True)

    assert resumed.status is MeasurementExperimentStatus.COMPLETED
    assert len(resumed_probe.requests) == 2
    assert resumed_agent.requests == []
    assert not checkpoint.exists()


def test_agent_and_post_probe_failures_preserve_distinct_safe_semantics(tmp_path) -> None:
    agent_order: list[str] = []
    failed_agent_provider = RecordedProvider(
        "agent",
        agent_order,
        content=json.dumps({"message": AGENT_MESSAGE, "evidence_ids_used": []}),
        error_at=1,
        error=LLMProviderError(ProviderErrorCode.HTTP_SERVER_ERROR, "provider failed"),
    )
    agent_failure = experiment(tmp_path, run_id="agent-failed", agent_provider=failed_agent_provider, call_order=agent_order).run()
    assert agent_failure.status is MeasurementExperimentStatus.AGENT_RUN_FAILED
    assert agent_failure.run_record is not None and agent_failure.run_record.status is RunStatus.FAILED
    assert agent_failure.measurement_record is not None
    assert agent_failure.measurement_record.error_code == "agent_run_failed"
    assert not (tmp_path / "results" / "agent-failed" / "probes").exists()

    post_order: list[str] = []
    post_provider = RecordedProvider(
        "probe",
        post_order,
        content=json.dumps({"choice": RAW_ANSWER}),
        error_at=3,
        error=LLMProviderError(ProviderErrorCode.HTTP_SERVER_ERROR, "provider failed"),
    )
    post_failure = experiment(tmp_path, run_id="post-failed", probe_provider=post_provider, call_order=post_order).run()
    assert post_failure.status is MeasurementExperimentStatus.POST_PROBE_FAILED
    assert post_failure.run_record is not None and post_failure.run_record.status is RunStatus.COMPLETED
    assert post_failure.measurement_record is not None
    assert post_failure.measurement_record.error_code == "post_probe_failed"
    assert post_failure.measurement_record.post_probe_status is MeasurementPhaseStatus.FAILED
    assert not (tmp_path / "results" / "post-failed" / "probes").exists()


def test_explicit_pairing_seed_and_existing_output_are_validated_before_calls(tmp_path) -> None:
    call_order: list[str] = []
    with pytest.raises(ValueError, match="probe_seed"):
        experiment(tmp_path, call_order=call_order, probe_seed=30)
    assert call_order == []

    runner = experiment(tmp_path, run_id="occupied")
    runner.output_directory.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.run()
