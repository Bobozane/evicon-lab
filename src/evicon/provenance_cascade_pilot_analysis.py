"""Offline evaluator-only analysis for the completed H-D.2.1 Pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .cascade_agent_protocol_v21 import HD21RunSpec, load_hd21_config
from .cascade_agent_protocol_v21_pilot import (
    HD21PilotBatchRecord, HD21PilotReceipt, HD21PilotRunStatus, _load_scenarios,
)
from .cascade_intervention_application import CascadeScheduleStatus
from .cascade_outcome_replay import CascadeOutcomeReplayStatus, CascadeOutcomeReplayValidator
from .cascade_outcomes import ClaimStance
from .cascade_real_agent_runner import CascadeRealAgentRunRecord
from .provenance_cascade import (
    EvaluatorTruthLoader, GroundTruthLabel, SourceIndependenceLabel, validate_fixture_pair,
)
from .provenance_cascade_preregistration import (
    CascadeCondition, ProvenanceCascadePreregistrationLoader,
)
from .request_ledger import RequestLedger, RequestLedgerEntry, RequestLedgerStatus
from .provenance_cascade_pilot_analysis_plan import (
    ANALYSIS_VERSION, EXPECTED_CONDITIONS, EXPECTED_METRICS,
    ProvenanceCascadePilotAnalysisPlan, load_provenance_cascade_pilot_analysis_plan,
)

_ROOT = Path(__file__).resolve().parents[2]
_FORBIDDEN_PUBLIC_KEYS = {
    "ground_truth_label", "source_independence_label", "evaluator_annotation",
    "system_prompt", "user_prompt", "prompt", "provider_metadata", "api_key", "authorization",
}
_CONTRASTS = (
    (CascadeCondition.GENERIC_DISSENT, CascadeCondition.NO_INTERVENTION),
    (CascadeCondition.SOURCE_BLIND_CONTROLLER, CascadeCondition.NO_INTERVENTION),
    (CascadeCondition.PROVENANCE_AWARE_CONTROLLER, CascadeCondition.NO_INTERVENTION),
    (CascadeCondition.PROVENANCE_AWARE_CONTROLLER, CascadeCondition.SOURCE_BLIND_CONTROLLER),
)


class HFPilotAnalysisError(ValueError):
    """Stable, content-free analysis failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HFMetricStatus(str, Enum):
    VALUE = "value"
    NOT_APPLICABLE = "not_applicable"


class HFSafeInputHash(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HFRunMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    matched_group_id: str
    scenario_id: str
    seed: int
    condition: CascadeCondition
    metric_name: str
    status: HFMetricStatus
    value: float | None = None
    numerator: float | None = None
    denominator: int = Field(ge=0)
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def shape(self) -> "HFRunMetric":
        if self.status is HFMetricStatus.VALUE and self.value is None:
            raise ValueError("value metric requires value")
        if self.status is HFMetricStatus.NOT_APPLICABLE and self.value is not None:
            raise ValueError("not-applicable metric cannot contain value")
        return self


class HFRunOperations(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    scenario_id: str
    seed: int
    condition: CascadeCondition
    proposal_count: int = Field(ge=0)
    applied_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    cancelled_count: int = Field(ge=0)
    scheduled_unapplied_count: int = Field(ge=0)
    abstain_count: int = Field(ge=0)
    logical_request_count: int = Field(ge=0)
    transport_attempt_count: int = Field(ge=0)
    connection_failure_count: int = Field(ge=0)
    connection_recovery_count: int = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    mean_latency_ms: float | None = Field(default=None, ge=0)


class HFConditionMetricSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    condition: CascadeCondition
    metric_name: str
    run_count: int = Field(ge=0)
    applicable_count: int = Field(ge=0)
    not_applicable_count: int = Field(ge=0)
    denominator_sum: int = Field(ge=0)
    numerator_sum: float | None = None
    pooled_value: float | None = None
    mean: float | None = None
    median: float | None = None
    standard_deviation: float | None = None
    value_sum: float | None = None


class HFMatchedDifference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_id: str
    seed: int
    metric_name: str
    comparison_condition: CascadeCondition
    reference_condition: CascadeCondition
    status: HFMetricStatus
    comparison_value: float | None = None
    reference_value: float | None = None
    difference: float | None = None
    comparison_denominator: int = Field(ge=0)
    reference_denominator: int = Field(ge=0)
    warning: str | None = None


class HFConditionOperations(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    condition: CascadeCondition
    run_count: int = Field(ge=0)
    proposal_count: int = Field(ge=0)
    applied_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    abstain_count: int = Field(ge=0)
    logical_request_count: int = Field(ge=0)
    transport_attempt_count: int = Field(ge=0)
    connection_failure_count: int = Field(ge=0)
    connection_recovery_count: int = Field(ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    mean_latency_ms: float | None = Field(default=None, ge=0)


class HFAnalysisManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    analysis_id: str
    analysis_version: Literal["provenance_cascade_pilot_analysis.v1"]
    study_id: str
    input_hashes: tuple[HFSafeInputHash, ...]
    run_record_sha256: dict[str, str]
    scenario_count: Literal[4]
    seed_count: Literal[3]
    condition_count: Literal[4]
    run_count: Literal[48]
    matched_group_count: Literal[12]
    primary_metrics: tuple[str, ...]
    development_only: Literal[True]
    pilot_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    private_truth_exposed: Literal[False]
    network_used_for_analysis: Literal[False]


class HFAnalysisReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    analysis_id: str
    analysis_version: Literal["provenance_cascade_pilot_analysis.v1"]
    data_integrity_status: Literal["passed"]
    effect_evidence_status: Literal["descriptive_pilot_only"]
    decision_status: Literal[
        "go_for_confirmatory", "revise_and_repilot", "stop_for_nonseparation",
        "blocked_for_integrity", "thresholds_not_pre_registered",
    ]
    decision_reason_codes: tuple[str, ...]
    run_metrics: tuple[HFRunMetric, ...]
    condition_summaries: tuple[HFConditionMetricSummary, ...]
    matched_differences: tuple[HFMatchedDifference, ...]
    run_operations: tuple[HFRunOperations, ...]
    condition_operations: tuple[HFConditionOperations, ...]
    warnings: tuple[str, ...]
    development_only: Literal[True]
    pilot_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    private_truth_exposed: Literal[False]


class HFAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["completed", "validated", "blocked"]
    analysis_id: str
    run_count: int = Field(ge=0)
    matched_group_count: int = Field(ge=0)
    metric_count: int = Field(ge=0)
    decision_status: str | None = None
    output_dir: str | None = None
    blocking_reasons: tuple[str, ...] = ()
    network: Literal["disabled"] = "disabled"
    provider_constructed: Literal[False] = False
    private_truth_exposed: Literal[False] = False
    not_paper_result: Literal[True] = True


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (_ROOT / path).resolve()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_sha(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _require(value: bool, code: str) -> None:
    if not value:
        raise HFPilotAnalysisError(code)


def _assert_public_safe(value: object) -> None:
    if isinstance(value, dict):
        if _FORBIDDEN_PUBLIC_KEYS & {str(key).lower() for key in value}:
            raise HFPilotAnalysisError("private_truth_in_public_input")
        for item in value.values():
            _assert_public_safe(item)
    elif isinstance(value, list):
        for item in value:
            _assert_public_safe(item)


def _load_json_model(path: Path, model: type[BaseModel], code: str) -> BaseModel:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError, ValueError) as exc:
        raise HFPilotAnalysisError(code) from exc


def _metric(
    spec: HD21RunSpec, name: str, numerator: float | None, denominator: int,
    *, value: float | None = None, warnings: tuple[str, ...] = (),
) -> HFRunMetric:
    if denominator <= 0 or (numerator is None and value is None):
        return HFRunMetric(
            run_id=spec.run_id, matched_group_id=spec.matched_group_id,
            scenario_id=spec.scenario_id, seed=spec.seed, condition=spec.condition,
            metric_name=name, status=HFMetricStatus.NOT_APPLICABLE, denominator=0,
            warnings=warnings,
        )
    resolved = value if value is not None else float(numerator) / denominator
    return HFRunMetric(
        run_id=spec.run_id, matched_group_id=spec.matched_group_id,
        scenario_id=spec.scenario_id, seed=spec.seed, condition=spec.condition,
        metric_name=name, status=HFMetricStatus.VALUE, value=float(resolved),
        numerator=numerator, denominator=denominator, warnings=warnings,
    )


def _latest_and_history(record: CascadeRealAgentRunRecord) -> tuple[dict[tuple[str, str], Any], dict[tuple[str, str], list[Any]]]:
    history: dict[tuple[str, str], list[Any]] = defaultdict(list)
    assert record.outcome_ledger is not None
    for outcome in record.outcome_ledger.outcomes:
        history[(outcome.agent_id, outcome.claim_id)].append(outcome)
    for values in history.values():
        values.sort(key=lambda item: item.round_id)
    return {key: values[-1] for key, values in history.items()}, history


def _public_exposure(record: CascadeRealAgentRunRecord, graph: Any) -> tuple[dict[Any, set[str]], dict[Any, set[str]], dict[Any, set[str]], dict[Any, set[str]]]:
    assert record.exposure_ledger is not None
    content: dict[tuple[str, str], set[str]] = defaultdict(set)
    social: dict[tuple[str, str], set[str]] = defaultdict(set)
    evidence: dict[tuple[str, str], set[str]] = defaultdict(set)
    roots: dict[tuple[str, str], set[str]] = defaultdict(set)
    for event in record.exposure_ledger.events:
        key = (event.target_agent_id, event.claim_id)
        content[key].add(event.content_id)
        evidence[key].update(event.evidence_card_ids)
        roots[key].update(graph.root_sources_for_node(event.provenance_node_id))
        if event.source_actor_id != "system":
            social[key].add(event.content_id)
    return content, social, evidence, roots


def _terminal_entries(entries: list[RequestLedgerEntry]) -> list[RequestLedgerEntry]:
    return [entry for entry in entries if entry.status in {RequestLedgerStatus.COMPLETED, RequestLedgerStatus.FAILED}]


def _evaluate_run(
    spec: HD21RunSpec, record: CascadeRealAgentRunRecord, graph: Any, truth: Any,
    entries: list[RequestLedgerEntry],
) -> tuple[list[HFRunMetric], HFRunOperations]:
    truth_by_claim = {item.claim_id: item for item in truth.records}
    false_claims = {key for key, item in truth_by_claim.items() if item.ground_truth_label is GroundTruthLabel.FALSE}
    correction_claims = {
        key for key, item in truth_by_claim.items()
        if item.ground_truth_label is GroundTruthLabel.TRUE
        and item.source_independence_label is SourceIndependenceLabel.INDEPENDENT
    }
    latest, history = _latest_and_history(record)
    content, social, evidence, roots = _public_exposure(record, graph)
    cards = {card.evidence_id: card for card in graph.evidence_cards}

    false_eligible = [key for key, value in latest.items() if key[1] in false_claims and value.stance is not ClaimStance.NO_POSITION]
    false_endorse = sum(latest[key].stance is ClaimStance.ENDORSES for key in false_eligible)
    metrics = [_metric(spec, EXPECTED_METRICS[0], false_endorse, len(false_eligible))]

    correction_eligible = [key for key in latest if key[1] in correction_claims and content.get(key)]
    correction_retained = sum(latest[key].stance is ClaimStance.ENDORSES for key in correction_eligible)
    metrics.append(_metric(spec, EXPECTED_METRICS[1], correction_retained, len(correction_eligible), warnings=("no_supported_correction_exposure",) if not correction_eligible else ()))

    receptivity_den = 0
    receptivity_num = 0
    for agent_id in record.agent_ids:
        exposed_corrections = [claim_id for claim_id in correction_claims if content.get((agent_id, claim_id))]
        for false_claim in false_claims:
            values = history.get((agent_id, false_claim), [])
            if not values or values[0].stance is not ClaimStance.ENDORSES or not exposed_corrections:
                continue
            receptivity_den += 1
            rejects_false = latest[(agent_id, false_claim)].stance is ClaimStance.REJECTS
            endorses_correction = any(latest.get((agent_id, claim_id)) and latest[(agent_id, claim_id)].stance is ClaimStance.ENDORSES for claim_id in exposed_corrections)
            receptivity_num += rejects_false or endorses_correction
    metrics.append(_metric(spec, EXPECTED_METRICS[2], receptivity_num, receptivity_den, warnings=("no_exposure_linked_false_to_correction_transition",) if not receptivity_den else ()))

    harmful_den = 0
    harmful_num = 0
    for key, value in latest.items():
        if key[1] not in false_claims or not social.get(key):
            continue
        independently_supportive = any(key[1] in cards[evidence_id].supports for evidence_id in evidence.get(key, set()))
        if independently_supportive:
            continue
        harmful_den += 1
        harmful_num += value.stance is ClaimStance.ENDORSES
    metrics.append(_metric(spec, EXPECTED_METRICS[3], harmful_num, harmful_den, warnings=("no_socially_exposed_false_claim_unit",) if not harmful_den else ("operational_not_causal_conformity",)))

    assert record.application_ledger is not None
    applied = [item for item in record.application_ledger.schedules if item.status is CascadeScheduleStatus.APPLIED]
    false_positives = sum(item.claim_id in correction_claims for item in applied)
    metrics.append(_metric(spec, EXPECTED_METRICS[4], false_positives if applied else None, len(applied), warnings=("no_applied_intervention",) if not applied else ()))

    root_values = [len(value) for value in roots.values()]
    metrics.append(_metric(spec, EXPECTED_METRICS[5], None, len(root_values), value=statistics.mean(root_values) if root_values else None, warnings=("no_exposed_agent_claim_unit",) if not root_values else ()))
    metrics.append(_metric(spec, EXPECTED_METRICS[6], 1, 1))

    completed_entries = [item for item in _terminal_entries(entries) if item.status is RequestLedgerStatus.COMPLETED]
    token_values = [item.total_tokens for item in completed_entries if item.total_tokens is not None]
    latency_values = [item.latency_ms for item in completed_entries if item.latency_ms is not None]
    metrics.append(_metric(spec, EXPECTED_METRICS[7], None, len(token_values), value=float(sum(token_values)) if token_values and len(token_values) == len(completed_entries) else None, warnings=("provider_usage_missing",) if len(token_values) != len(completed_entries) else ()))
    metrics.append(_metric(spec, EXPECTED_METRICS[8], None, len(latency_values), value=float(statistics.mean(latency_values)) if latency_values and len(latency_values) == len(completed_entries) else None, warnings=("provider_latency_missing",) if len(latency_values) != len(completed_entries) else ()))

    schedules = list(record.application_ledger.schedules)
    counts = Counter(item.status for item in schedules)
    terminal = _terminal_entries(entries)
    failed_connection = {item.fingerprint for item in terminal if item.status is RequestLedgerStatus.FAILED and item.error_code == "connection_failure"}
    completed_fingerprints = {item.fingerprint for item in terminal if item.status is RequestLedgerStatus.COMPLETED}
    started = [item for item in entries if item.status is RequestLedgerStatus.STARTED]
    unique_started = {item.fingerprint for item in started}
    all_prompt_known = all(item.prompt_tokens is not None for item in completed_entries)
    all_completion_known = all(item.completion_tokens is not None for item in completed_entries)
    all_total_known = all(item.total_tokens is not None for item in completed_entries)
    operations = HFRunOperations(
        run_id=spec.run_id, scenario_id=spec.scenario_id, seed=spec.seed, condition=spec.condition,
        proposal_count=record.proposal_count,
        applied_count=counts[CascadeScheduleStatus.APPLIED],
        rejected_count=counts[CascadeScheduleStatus.REJECTED],
        cancelled_count=counts[CascadeScheduleStatus.CANCELLED],
        scheduled_unapplied_count=counts[CascadeScheduleStatus.SCHEDULED],
        abstain_count=max(0, record.proposal_count - len(schedules)),
        logical_request_count=len(unique_started),
        transport_attempt_count=len(started),
        connection_failure_count=len(failed_connection),
        connection_recovery_count=len(failed_connection & completed_fingerprints),
        prompt_tokens=sum(item.prompt_tokens for item in completed_entries if item.prompt_tokens is not None) if all_prompt_known else None,
        completion_tokens=sum(item.completion_tokens for item in completed_entries if item.completion_tokens is not None) if all_completion_known else None,
        total_tokens=sum(item.total_tokens for item in completed_entries if item.total_tokens is not None) if all_total_known else None,
        mean_latency_ms=float(statistics.mean(latency_values)) if latency_values else None,
    )
    return metrics, operations


def _condition_summaries(metrics: list[HFRunMetric]) -> tuple[HFConditionMetricSummary, ...]:
    output = []
    for condition in EXPECTED_CONDITIONS:
        for metric_name in EXPECTED_METRICS:
            rows = [item for item in metrics if item.condition is condition and item.metric_name == metric_name]
            values = [item.value for item in rows if item.status is HFMetricStatus.VALUE and item.value is not None]
            numerators = [item.numerator for item in rows if item.status is HFMetricStatus.VALUE and item.numerator is not None]
            denominators = [item.denominator for item in rows if item.status is HFMetricStatus.VALUE]
            numerator_sum = float(sum(numerators)) if numerators and len(numerators) == len(values) else None
            denominator_sum = sum(denominators)
            pooled = numerator_sum / denominator_sum if numerator_sum is not None and denominator_sum else None
            output.append(HFConditionMetricSummary(
                condition=condition, metric_name=metric_name, run_count=len(rows),
                applicable_count=len(values), not_applicable_count=len(rows) - len(values),
                denominator_sum=denominator_sum, numerator_sum=numerator_sum, pooled_value=pooled,
                mean=float(statistics.mean(values)) if values else None,
                median=float(statistics.median(values)) if values else None,
                standard_deviation=float(statistics.pstdev(values)) if values else None,
                value_sum=float(sum(values)) if values else None,
            ))
    return tuple(output)


def _matched_differences(metrics: list[HFRunMetric], scenario_ids: tuple[str, ...], seeds: tuple[int, ...]) -> tuple[HFMatchedDifference, ...]:
    by_key = {(item.scenario_id, item.seed, item.condition, item.metric_name): item for item in metrics}
    output = []
    for scenario_id in scenario_ids:
        for seed in seeds:
            for metric_name in EXPECTED_METRICS:
                for comparison, reference in _CONTRASTS:
                    left = by_key[(scenario_id, seed, comparison, metric_name)]
                    right = by_key[(scenario_id, seed, reference, metric_name)]
                    valid = left.status is HFMetricStatus.VALUE and right.status is HFMetricStatus.VALUE
                    output.append(HFMatchedDifference(
                        scenario_id=scenario_id, seed=seed, metric_name=metric_name,
                        comparison_condition=comparison, reference_condition=reference,
                        status=HFMetricStatus.VALUE if valid else HFMetricStatus.NOT_APPLICABLE,
                        comparison_value=left.value if valid else None,
                        reference_value=right.value if valid else None,
                        difference=(left.value - right.value) if valid and left.value is not None and right.value is not None else None,
                        comparison_denominator=left.denominator, reference_denominator=right.denominator,
                        warning=None if valid else "paired_metric_not_applicable",
                    ))
    return tuple(output)


def _condition_operations(rows: list[HFRunOperations]) -> tuple[HFConditionOperations, ...]:
    output = []
    for condition in EXPECTED_CONDITIONS:
        items = [item for item in rows if item.condition is condition]
        totals = [item.total_tokens for item in items if item.total_tokens is not None]
        latencies = [item.mean_latency_ms for item in items if item.mean_latency_ms is not None]
        output.append(HFConditionOperations(
            condition=condition, run_count=len(items), proposal_count=sum(item.proposal_count for item in items),
            applied_count=sum(item.applied_count for item in items), rejected_count=sum(item.rejected_count for item in items),
            abstain_count=sum(item.abstain_count for item in items), logical_request_count=sum(item.logical_request_count for item in items),
            transport_attempt_count=sum(item.transport_attempt_count for item in items),
            connection_failure_count=sum(item.connection_failure_count for item in items),
            connection_recovery_count=sum(item.connection_recovery_count for item in items),
            total_tokens=sum(totals) if len(totals) == len(items) else None,
            mean_latency_ms=float(statistics.mean(latencies)) if len(latencies) == len(items) and latencies else None,
        ))
    return tuple(output)


def _validate_hash(path: Path, expected: str, code: str) -> None:
    try:
        actual = _sha256_file(path)
    except OSError as exc:
        raise HFPilotAnalysisError(code) from exc
    if actual != expected:
        raise HFPilotAnalysisError(code)


def validate_hf_run_coordinates(
    specs: Sequence[HD21RunSpec],
    states: Sequence[Any],
) -> None:
    """Validate the complete public 48-run matched design without reading content."""
    spec_by_id = {item.run_id: item for item in specs}
    state_by_id = {item.run_id: item for item in states}
    _require(
        len(spec_by_id) == len(specs) == 48
        and len(state_by_id) == len(states) == 48
        and set(spec_by_id) == set(state_by_id),
        "run_set_incomplete",
    )
    groups: dict[str, set[CascadeCondition]] = defaultdict(set)
    for spec in spec_by_id.values():
        groups[spec.matched_group_id].add(spec.condition)
        state = state_by_id[spec.run_id]
        _require(
            state.scenario_id == spec.scenario_id
            and state.seed == spec.seed
            and state.condition == spec.condition.value,
            "run_coordinate_mismatch",
        )
        _require(
            state.status is HD21PilotRunStatus.COMPLETED and state.error_code is None,
            "run_not_completed",
        )
        _require(
            state.cascade_replay_status == "passed"
            and state.application_replay_status == "passed"
            and state.outcome_replay_status == "passed",
            "replay_not_passed",
        )
    _require(
        len(groups) == 12
        and all(value == set(EXPECTED_CONDITIONS) for value in groups.values()),
        "matched_group_incomplete",
    )


def analyze_provenance_cascade_pilot(
    plan: ProvenanceCascadePilotAnalysisPlan,
) -> tuple[HFAnalysisManifest, HFAnalysisReport]:
    config_path = _rooted(plan.config_path)
    protocol_path = _rooted(plan.protocol_path)
    template_path = _rooted(plan.template_path)
    receipt_path = _rooted(plan.pilot_receipt_path)
    batch_path = _rooted(plan.batch_record_path)
    prereg_path = _rooted(plan.preregistration_path)
    pilot_root = _rooted(plan.pilot_root)
    for path, expected, code in (
        (config_path, plan.config_sha256, "config_hash_mismatch"),
        (protocol_path, plan.protocol_sha256, "protocol_hash_mismatch"),
        (template_path, plan.template_sha256, "template_hash_mismatch"),
        (receipt_path, plan.pilot_receipt_sha256, "pilot_receipt_hash_mismatch"),
        (batch_path, plan.batch_record_sha256, "batch_record_hash_mismatch"),
        (prereg_path, plan.preregistration_sha256, "preregistration_hash_mismatch"),
    ):
        _validate_hash(path, expected, code)

    config, _ = load_hd21_config(config_path)
    receipt = _load_json_model(receipt_path, HD21PilotReceipt, "pilot_receipt_invalid")
    batch = _load_json_model(batch_path, HD21PilotBatchRecord, "batch_record_invalid")
    assert isinstance(receipt, HD21PilotReceipt) and isinstance(batch, HD21PilotBatchRecord)
    _require(receipt.status == "completed" and receipt.completed_run_count == 48 and receipt.failed_run_count == 0, "pilot_receipt_incomplete")
    _require(receipt.run_count == 48 and receipt.matched_group_count == 12 and receipt.replay_passed_count == 48, "pilot_receipt_incomplete")
    _require(receipt.config_sha256 == plan.config_sha256 and receipt.protocol_sha256 == plan.protocol_sha256 and receipt.template_sha256 == plan.template_sha256, "pilot_receipt_binding_mismatch")
    _require(receipt.logical_request_count == 864 and receipt.transport_attempt_count == 866, "pilot_request_audit_mismatch")
    _require(batch.status is HD21PilotRunStatus.COMPLETED and batch.failure_code is None and len(batch.runs) == 48, "batch_record_incomplete")
    _require(batch.config_sha256 == plan.config_sha256 and batch.protocol_sha256 == plan.protocol_sha256 and batch.template_sha256 == plan.template_sha256, "batch_record_binding_mismatch")
    _require(batch.study_id == plan.study_id and receipt.study_id == plan.study_id, "study_id_mismatch")
    _require(not receipt.old_pilot_touched and not receipt.private_truth_exposed and not receipt.parser_recovery_enabled, "pilot_safety_binding_mismatch")

    prereg = ProvenanceCascadePreregistrationLoader.load(prereg_path)
    _require(tuple(prereg.primary_metrics) == EXPECTED_METRICS, "preregistered_metric_suite_mismatch")
    _require(prereg.success_criterion.no_unregistered_thresholds, "preregistration_threshold_boundary_missing")
    scenarios = _load_scenarios(config_path, config)
    _require(tuple(config.scenario_ids) == plan.scenario_ids, "scenario_plan_mismatch")
    _require(tuple(config.conditions) == plan.conditions and tuple(config.seeds) == plan.seeds, "matched_plan_mismatch")
    validate_hf_run_coordinates(config.runs, batch.runs)
    states = {item.run_id: item for item in batch.runs}

    truth_by_scenario = {}
    safe_input_hashes = [
        HFSafeInputHash(kind="config", sha256=plan.config_sha256),
        HFSafeInputHash(kind="protocol", sha256=plan.protocol_sha256),
        HFSafeInputHash(kind="template", sha256=plan.template_sha256),
        HFSafeInputHash(kind="pilot_receipt", sha256=plan.pilot_receipt_sha256),
        HFSafeInputHash(kind="batch_record", sha256=plan.batch_record_sha256),
        HFSafeInputHash(kind="preregistration", sha256=plan.preregistration_sha256),
    ]
    for binding in plan.truth_fixtures:
        path = _rooted(binding.path)
        _validate_hash(path, binding.sha256, "evaluator_fixture_hash_mismatch")
        try:
            truth = EvaluatorTruthLoader.load(path)
            validate_fixture_pair(scenarios[binding.scenario_id].graph, truth)
        except Exception as exc:
            raise HFPilotAnalysisError("evaluator_fixture_binding_mismatch") from exc
        truth_by_scenario[binding.scenario_id] = truth
        safe_input_hashes.append(HFSafeInputHash(kind=f"private_fixture:{binding.scenario_id}", sha256=binding.sha256))

    run_metrics: list[HFRunMetric] = []
    run_operations: list[HFRunOperations] = []
    run_hashes: dict[str, str] = {}
    for spec in config.runs:
        state = states[spec.run_id]
        run_path = pilot_root / spec.run_id / "run_record.json"
        _validate_hash(run_path, state.run_record_sha256 or "", "run_record_hash_mismatch")
        try:
            raw = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HFPilotAnalysisError("run_record_invalid") from exc
        _assert_public_safe(raw)
        try:
            record = CascadeRealAgentRunRecord.model_validate(raw)
        except ValidationError as exc:
            raise HFPilotAnalysisError("run_record_invalid") from exc
        _require(record.run_id == spec.run_id and record.scenario_id == spec.scenario_id and record.seed == spec.seed and record.condition is spec.condition, "run_coordinate_mismatch")
        _require(record.status == "completed" and record.error_code is None and record.exposure_ledger is not None and record.application_ledger is not None and record.outcome_ledger is not None and record.replay is not None, "run_record_incomplete")
        _require(record.exposure_ledger_sha256 == _stable_sha(record.exposure_ledger.model_dump(mode="json")), "exposure_hash_mismatch")
        _require(record.application_ledger_sha256 == _stable_sha(record.application_ledger.model_dump(mode="json")), "application_hash_mismatch")
        _require(record.outcome_ledger_sha256 == record.outcome_ledger.sha256(), "outcome_hash_mismatch")
        scenario = scenarios[spec.scenario_id]
        replay = CascadeOutcomeReplayValidator.validate(scenario.graph, record.exposure_ledger, record.outcome_ledger, record.application_ledger, record.round_contexts)
        _require(replay.status is CascadeOutcomeReplayStatus.PASSED, "replay_not_passed")
        ledger_path = pilot_root / spec.run_id / "request_ledger.jsonl"
        entries = RequestLedger(ledger_path).entries()
        _require(len({item.fingerprint for item in entries if item.status is RequestLedgerStatus.STARTED}) == 18, "logical_request_count_mismatch")
        metrics, operations = _evaluate_run(spec, record, scenario.graph, truth_by_scenario[spec.scenario_id], entries)
        run_metrics.extend(metrics)
        run_operations.append(operations)
        run_hashes[spec.run_id] = state.run_record_sha256 or ""

    _require(len(run_metrics) == 48 * 9 and sum(item.logical_request_count for item in run_operations) == 864, "analysis_coordinate_incomplete")
    _require(sum(item.transport_attempt_count for item in run_operations) == 866, "transport_attempt_audit_mismatch")
    _require(sum(item.connection_recovery_count for item in run_operations) == 2, "connection_recovery_audit_mismatch")
    _require(sum(item.applied_count for item in run_operations) == receipt.directive_applied_count, "directive_count_mismatch")
    condition_summaries = _condition_summaries(run_metrics)
    differences = _matched_differences(run_metrics, plan.scenario_ids, plan.seeds)
    operations = _condition_operations(run_operations)
    warnings = {
        "three_seed_pilot_descriptive_only", "no_causal_conclusion",
        "thresholds_not_pre_registered", "manual_go_no_go_decision_required",
        "pilot_seed_amendment_differs_from_24a_planning_seeds",
    }
    for item in run_metrics:
        warnings.update(item.warnings)
    manifest = HFAnalysisManifest(
        analysis_id=plan.analysis_id, analysis_version=ANALYSIS_VERSION, study_id=plan.study_id,
        input_hashes=tuple(safe_input_hashes), run_record_sha256=run_hashes,
        scenario_count=4, seed_count=3, condition_count=4, run_count=48, matched_group_count=12,
        primary_metrics=EXPECTED_METRICS, development_only=True, pilot_only=True,
        not_paper_result=True, no_causal_conclusion=True, private_truth_exposed=False,
        network_used_for_analysis=False,
    )
    report = HFAnalysisReport(
        analysis_id=plan.analysis_id, analysis_version=ANALYSIS_VERSION,
        data_integrity_status="passed", effect_evidence_status="descriptive_pilot_only",
        decision_status="thresholds_not_pre_registered",
        decision_reason_codes=("thresholds_not_pre_registered", "manual_review_required"),
        run_metrics=tuple(run_metrics), condition_summaries=condition_summaries,
        matched_differences=differences, run_operations=tuple(run_operations),
        condition_operations=operations, warnings=tuple(sorted(warnings)),
        development_only=True, pilot_only=True, not_paper_result=True,
        no_causal_conclusion=True, private_truth_exposed=False,
    )
    return manifest, report


def _write_json(path: Path, value: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def run_hf_analysis(
    config_path: str | Path,
    *,
    write_outputs: bool,
    output_dir: str | Path | None = None,
) -> HFAnalysisResult:
    try:
        plan = load_provenance_cascade_pilot_analysis_plan(config_path)
        manifest, report = analyze_provenance_cascade_pilot(plan)
        destination = _rooted(output_dir or plan.output_dir)
        if write_outputs:
            if destination.exists():
                raise HFPilotAnalysisError("analysis_output_exists")
            destination.mkdir(parents=True)
            _write_json(destination / "analysis_manifest.json", manifest)
            _write_json(destination / "analysis_report.json", report)
        return HFAnalysisResult(
            status="completed" if write_outputs else "validated", analysis_id=plan.analysis_id,
            run_count=48, matched_group_count=12, metric_count=len(report.run_metrics),
            decision_status=report.decision_status,
            output_dir=str(destination) if write_outputs else None,
        )
    except Exception as exc:
        code = exc.code if isinstance(exc, HFPilotAnalysisError) else "pilot_analysis_failed"
        return HFAnalysisResult(
            status="blocked", analysis_id="provenance-cascade-hd21-pilot-analysis",
            run_count=0, matched_group_count=0, metric_count=0,
            blocking_reasons=(code,),
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline evaluator-only H-D.2.1 Pilot analysis")
    parser.add_argument("--config", default="configs/provenance_cascade/pilot/provenance_cascade_pilot_hd21_analysis.v1.toml")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    result = run_hf_analysis(args.config, write_outputs=not args.validate_only)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))
    return 0 if result.status != "blocked" else 1


__all__ = [
    "HFAnalysisManifest", "HFAnalysisReport", "HFAnalysisResult", "HFConditionMetricSummary",
    "HFConditionOperations", "HFMatchedDifference", "HFMetricStatus", "HFPilotAnalysisError",
    "HFRunMetric", "HFRunOperations", "analyze_provenance_cascade_pilot", "main", "run_hf_analysis",
    "validate_hf_run_coordinates",
]


if __name__ == "__main__":
    raise SystemExit(main())
