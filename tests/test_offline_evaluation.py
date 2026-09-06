"""Read-only manifest, matching, and report coverage for offline evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evicon.evaluation import (
    EvaluationInputError,
    EvaluationManifest,
    EvaluationRunEntry,
    OfflineEvaluator,
    write_evaluation_report,
)
from evicon.models import (
    AgentSpec,
    EvidenceCard,
    ProbeRunConfig,
    ProbeSet,
    ProtocolCondition,
    RunConfig,
    ScenarioSpec,
    ValueProbeItem,
)
from evicon.probe_results import write_probe_results
from evicon.probe_runner import ProbeRunner
from evicon.runner import ProtocolRunner


def evaluation_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="evaluation-scenario",
        title="Offline evaluation fixture",
        description="A local fixture for read-only evaluation tests.",
        agents=[
            AgentSpec(agent_id="agent-1", role="participant"),
            AgentSpec(agent_id="agent-2", role="participant"),
        ],
        evidence_cards=[
            EvidenceCard(
                evidence_id="evidence-public",
                claim="Local public evidence.",
                source="fixture",
                supports=["claim"],
                contradicts=[],
                introduced_round=0,
                visible_to=["*"],
                reliability=1.0,
            )
        ],
        max_rounds=1,
        metadata={},
    )


def evaluation_probe_set() -> ProbeSet:
    return ProbeSet(
        probe_set_id="evaluation-probes",
        dimensions=["fairness", "autonomy"],
        items=[
            ValueProbeItem(
                probe_id="fairness-standard",
                text="Fairness standard item.",
                dimension="fairness",
                response_scale=["1", "2", "3", "4", "5", "6", "7"],
                is_holdout=False,
            ),
            ValueProbeItem(
                probe_id="autonomy-standard",
                text="Autonomy standard item.",
                dimension="autonomy",
                response_scale=["1", "2", "3", "4", "5", "6", "7"],
                is_holdout=False,
            ),
            ValueProbeItem(
                probe_id="fairness-holdout",
                text="Fairness holdout item.",
                dimension="fairness",
                response_scale=["1", "2", "3", "4", "5", "6", "7"],
                is_holdout=True,
            ),
            ValueProbeItem(
                probe_id="autonomy-holdout",
                text="Autonomy holdout item.",
                dimension="autonomy",
                response_scale=["1", "2", "3", "4", "5", "6", "7"],
                is_holdout=True,
            ),
        ],
        version="1.0",
        metadata={},
    )


def create_artifacts(
    tmp_path: Path,
    *,
    run_id: str,
    protocol: ProtocolCondition,
    seed: int = 5,
    holdout: bool = False,
) -> tuple[Path, Path, Path]:
    scenario = evaluation_scenario()
    config = RunConfig(
        run_id=run_id,
        scenario_id=scenario.scenario_id,
        model_name="fake-llm",
        protocol=protocol,
        agent_count=2,
        max_rounds=1,
        seed=seed,
        intervention_budget=0.0,
        output_dir=str(tmp_path / "runs"),
    )
    ProtocolRunner(config, scenario=scenario).run()
    probe_results = ProbeRunner(
        evaluation_probe_set(),
        ProbeRunConfig(
            probe_set_id="evaluation-probes",
            round_id=1,
            agent_ids=["agent-1", "agent-2"],
            model_name="fake-llm",
            seed=seed,
            is_holdout=holdout,
        ),
    ).run()
    probe_path = write_probe_results(tmp_path / "probes", f"probe-{run_id}", probe_results)
    run_directory = Path(config.output_dir) / run_id
    return run_directory / "run_record.json", run_directory / "events.jsonl", probe_path


def entry(
    *,
    run_id: str,
    condition: str,
    protocol: ProtocolCondition,
    record_path: Path,
    events_path: Path,
    probe_path: Path,
    seed: int = 5,
    role: str = "final",
) -> EvaluationRunEntry:
    return EvaluationRunEntry(
        run_id=run_id,
        condition=condition,
        protocol=protocol,
        run_record_path=str(record_path),
        events_path=str(events_path),
        probe_result_path=str(probe_path),
        seed=seed,
        model_name="fake-llm",
        role=role,
        counterfactual_group_id="controlled-group",
    )


def manifest(entries: list[EvaluationRunEntry], suites: list[str]) -> EvaluationManifest:
    return EvaluationManifest(
        evaluation_id="offline-evaluation",
        scenario_id="evaluation-scenario",
        metric_suite=suites,
        runs=entries,
        probe_sets=["evaluation-probes"],
        metric_options={"coverage_threshold": 0.5, "minority_dimensions": ["autonomy"]},
        metadata={},
    )


def test_manifest_is_strict_and_json_round_trips() -> None:
    raw = {
        "evaluation_id": "evaluation",
        "scenario_id": "scenario",
        "metric_suite": ["audit"],
        "runs": [
            {
                "run_id": "run",
                "condition": "independent",
                "protocol": "independent",
                "run_record_path": "record.json",
                "events_path": "events.jsonl",
                "probe_result_path": None,
                "seed": 1,
                "model_name": "fake",
                "role": "final",
                "counterfactual_group_id": "group",
            }
        ],
        "probe_sets": [],
        "metadata": {},
    }
    parsed = EvaluationManifest.model_validate(raw)

    assert EvaluationManifest.model_validate_json(parsed.model_dump_json()) == parsed
    with pytest.raises(ValidationError, match="duplicate evaluation entry"):
        EvaluationManifest.model_validate({**raw, "runs": raw["runs"] * 2})
    with pytest.raises(ValidationError, match="Extra inputs"):
        EvaluationManifest.model_validate({**raw, "unexpected": True})


def test_evaluator_reads_completed_runs_and_writes_hashed_report(tmp_path: Path) -> None:
    evidence_record, evidence_events, evidence_probe = create_artifacts(
        tmp_path,
        run_id="evidence-only",
        protocol=ProtocolCondition.EVIDENCE_ONLY,
    )
    social_record, social_events, social_probe = create_artifacts(
        tmp_path,
        run_id="evidence-social",
        protocol=ProtocolCondition.EVIDENCE_SOCIAL,
    )
    evaluation = OfflineEvaluator(
        manifest(
            [
                entry(
                    run_id="evidence-only",
                    condition="evidence_only",
                    protocol=ProtocolCondition.EVIDENCE_ONLY,
                    record_path=evidence_record,
                    events_path=evidence_events,
                    probe_path=evidence_probe,
                ),
                entry(
                    run_id="evidence-social",
                    condition="evidence_social",
                    protocol=ProtocolCondition.EVIDENCE_SOCIAL,
                    record_path=social_record,
                    events_path=social_events,
                    probe_path=social_probe,
                ),
            ],
            ["baseline_diversity", "representation", "counterfactual", "audit"],
        ),
        manifest_directory=tmp_path,
    )

    report = evaluation.evaluate()
    names = [result.metric_name for result in report.metrics]

    assert "pairwise_diversity" in names
    assert "structural_diversity" in names
    assert "value_dimension_coverage" in names
    assert "social_influence_loss" in names
    assert "run_completeness" in names
    assert "evidence_exposure_consistency" in names
    assert len(report.input_audit) == 6
    assert all(len(item.sha256) == 64 for item in report.input_audit)
    assert any(pair.comparison == "evidence_social" for pair in report.matched_pairs)
    assert type(report).model_validate_json(report.model_dump_json()) == report

    report_path = write_evaluation_report(report, tmp_path / "reports" / "report.json")
    assert json.loads(report_path.read_text(encoding="utf-8"))["evaluation_id"] == "offline-evaluation"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_evaluation_report(report, report_path)


def test_matching_refuses_different_seed_even_with_shared_group(tmp_path: Path) -> None:
    left_record, left_events, left_probe = create_artifacts(
        tmp_path,
        run_id="evidence-only",
        protocol=ProtocolCondition.EVIDENCE_ONLY,
        seed=5,
    )
    right_record, right_events, right_probe = create_artifacts(
        tmp_path,
        run_id="evidence-social",
        protocol=ProtocolCondition.EVIDENCE_SOCIAL,
        seed=6,
    )
    report = OfflineEvaluator(
        manifest(
            [
                entry(
                    run_id="evidence-only",
                    condition="evidence_only",
                    protocol=ProtocolCondition.EVIDENCE_ONLY,
                    record_path=left_record,
                    events_path=left_events,
                    probe_path=left_probe,
                    seed=5,
                ),
                entry(
                    run_id="evidence-social",
                    condition="evidence_social",
                    protocol=ProtocolCondition.EVIDENCE_SOCIAL,
                    record_path=right_record,
                    events_path=right_events,
                    probe_path=right_probe,
                    seed=6,
                ),
            ],
            ["counterfactual"],
        ),
        manifest_directory=tmp_path,
    ).evaluate()

    assert not any(pair.comparison == "evidence_social" for pair in report.matched_pairs)
    assert "social_influence_loss" not in [metric.metric_name for metric in report.metrics]
    assert any("social_influence_loss: skipped" in warning for warning in report.warnings)


def test_holdout_initial_final_pair_produces_drift(tmp_path: Path) -> None:
    initial_record, initial_events, initial_probe = create_artifacts(
        tmp_path,
        run_id="initial",
        protocol=ProtocolCondition.EVIDENCE_SOCIAL,
        holdout=True,
    )
    final_record, final_events, final_probe = create_artifacts(
        tmp_path,
        run_id="final",
        protocol=ProtocolCondition.EVIDENCE_SOCIAL,
        holdout=True,
    )
    report = OfflineEvaluator(
        manifest(
            [
                entry(
                    run_id="initial",
                    condition="evidence_social",
                    protocol=ProtocolCondition.EVIDENCE_SOCIAL,
                    record_path=initial_record,
                    events_path=initial_events,
                    probe_path=initial_probe,
                    role="initial",
                ),
                entry(
                    run_id="final",
                    condition="evidence_social",
                    protocol=ProtocolCondition.EVIDENCE_SOCIAL,
                    record_path=final_record,
                    events_path=final_events,
                    probe_path=final_probe,
                    role="final",
                ),
            ],
            ["holdout"],
        ),
        manifest_directory=tmp_path,
    ).evaluate()

    assert "holdout_profile_drift" in [metric.metric_name for metric in report.metrics]
    assert any(
        pair.comparison == "initial_final" and pair.scope == "holdout"
        for pair in report.matched_pairs
    )


def test_loader_rejects_inconsistent_manifest_run_metadata(tmp_path: Path) -> None:
    record_path, events_path, probe_path = create_artifacts(
        tmp_path,
        run_id="completed",
        protocol=ProtocolCondition.INDEPENDENT,
    )
    invalid_entry = entry(
        run_id="completed",
        condition="independent",
        protocol=ProtocolCondition.INDEPENDENT,
        record_path=record_path,
        events_path=events_path,
        probe_path=probe_path,
        seed=99,
    )

    with pytest.raises(EvaluationInputError, match="seed mismatch"):
        OfflineEvaluator(manifest([invalid_entry], ["audit"]), manifest_directory=tmp_path).evaluate()


def test_failed_run_is_rejected_unless_explicitly_audit_only(tmp_path: Path) -> None:
    record_path, events_path, probe_path = create_artifacts(
        tmp_path,
        run_id="failed",
        protocol=ProtocolCondition.INDEPENDENT,
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["status"] = "failed"
    record["error_message"] = "fixture failure"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    rejected = entry(
        run_id="failed",
        condition="independent",
        protocol=ProtocolCondition.INDEPENDENT,
        record_path=record_path,
        events_path=events_path,
        probe_path=probe_path,
    )

    with pytest.raises(EvaluationInputError, match="not eligible"):
        OfflineEvaluator(manifest([rejected], ["audit"]), manifest_directory=tmp_path).evaluate()

    admitted = rejected.model_copy(update={"allow_failed_for_audit": True})
    report = OfflineEvaluator(manifest([admitted], ["audit"]), manifest_directory=tmp_path).evaluate()
    completeness = next(metric for metric in report.metrics if metric.metric_name == "run_completeness")
    assert not completeness.valid
    assert any("failed run was retained" in warning for warning in report.warnings)


def test_probe_set_must_be_explicitly_declared(tmp_path: Path) -> None:
    record_path, events_path, probe_path = create_artifacts(
        tmp_path,
        run_id="undeclared-probe",
        protocol=ProtocolCondition.INDEPENDENT,
    )
    run = entry(
        run_id="undeclared-probe",
        condition="independent",
        protocol=ProtocolCondition.INDEPENDENT,
        record_path=record_path,
        events_path=events_path,
        probe_path=probe_path,
    )
    invalid = manifest([run], ["audit"]).model_copy(update={"probe_sets": []})

    with pytest.raises(EvaluationInputError, match="absent from manifest.probe_sets"):
        OfflineEvaluator(invalid, manifest_directory=tmp_path).evaluate()
