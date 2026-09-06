"""Evaluator-only, adoption-decision analysis for completed H-G.2.3.2."""
from __future__ import annotations

import argparse
import json
import statistics
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .cascade_agent_protocol_hg23 import AdoptionDecision, SharingDecision
from .cascade_intervention_application import CascadeScheduleStatus
from .cascade_real_agent_runner import CascadeRealAgentRunRecord
from .provenance_cascade import (
    EvaluatorTruthLoader, GroundTruthLabel, SourceIndependenceLabel, validate_fixture_pair,
)
from .provenance_cascade_hg1 import HG1PilotConfig
from .provenance_cascade_hg2 import sha256_file
from .provenance_cascade_hg21_calibration import HG21BatchRecord, load_decisions as _unused
from .provenance_cascade_hg232 import DEFAULT_CONFIG, load_config
from .provenance_cascade_hg232_calibration import load_hg232_decisions
from .provenance_cascade_hg232_integrity_audit import (
    AUDIT_RECEIPT_PATH, CALIBRATION_RECEIPT_PATH, PILOT_ROOT,
    HG232FinalIntegrityAuditReceipt, audit_hg232_integrity,
)
from .provenance_cascade_pilot_analysis import (
    HFConditionMetricSummary, HFConditionOperations, HFMetricStatus, HFRunMetric,
    HFRunOperations, HFSafeInputHash, _condition_operations, _condition_summaries,
    _metric, _public_exposure,
)
from .provenance_cascade_pilot_analysis_plan import EXPECTED_METRICS
from .provenance_cascade_preregistration import CascadeCondition
from .request_ledger import RequestLedger, RequestLedgerStatus

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = _ROOT / "results/analyses/provenance-cascade-hg232-adoption-identifiability-evaluator-v2"
ANALYSIS_VERSION = "provenance_cascade_hg232_adoption_decision_analysis.v2"
BASE_MATERIAL_CONFIG = _ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg1.v1.toml"
BATCH_PATH = PILOT_ROOT / "calibration_batch_record.json"
_FORBIDDEN = (
    "ground_truth_label", "source_independence_label", "system_prompt", "user_prompt",
    "prompt", "raw_response", "provider_metadata", "api_key", "authorization",
    "public_statement", "private_fixture", "evaluator_private",
)


class HG232AnalysisError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HG232ScenarioConditionMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_id: str
    condition: CascadeCondition
    metric_name: str
    status: HFMetricStatus
    value: float | None = None
    numerator: float | None = None
    denominator: int = Field(ge=0)
    eligible_unit: str
    warnings: tuple[str, ...] = ()


class HG232BehaviorSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    scenario_id: str
    condition: CascadeCondition
    observation_count: Literal[18]
    adoption_counts: dict[str, int]
    sharing_counts: dict[str, int]
    share_requested_count: int = Field(ge=0)
    round0_adopt_count: int = Field(ge=0)
    final_adopt_count: int = Field(ge=0)
    final_share_count: int = Field(ge=0)


class HG232ConditionBehaviorSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    condition: CascadeCondition
    run_count: Literal[4]
    observation_count: Literal[72]
    adoption_counts: dict[str, int]
    sharing_counts: dict[str, int]
    share_requested_count: int = Field(ge=0)
    round0_adopt_count: int = Field(ge=0)
    final_adopt_count: int = Field(ge=0)
    final_share_count: int = Field(ge=0)


class HG232EligibilitySummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    scenario_id: str
    condition: CascadeCondition
    false_adoption_eligible_count: int = Field(ge=0)
    supported_correction_eligible_count: int = Field(ge=0)
    correction_transition_eligible_count: int = Field(ge=0)
    harmful_conformity_eligible_count: int = Field(ge=0)
    intervention_false_positive_eligible_count: int = Field(ge=0)


class HG232RecoveryAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    unique_logical_request_count: Literal[288]
    transport_attempt_count: Literal[289]
    timeout_failed_attempt_count: Literal[1]
    timeout_recovered_fingerprint_count: Literal[1]
    duplicate_completed_fingerprint_count: Literal[0]
    parser_recovery_count: Literal[0]


class HG232AnalysisManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    analysis_id: Literal["provenance-cascade-hg232-adoption-identifiability-analysis"]
    analysis_version: Literal["provenance_cascade_hg232_adoption_decision_analysis.v2"]
    study_id: str
    input_hashes: tuple[HFSafeInputHash, ...]
    run_record_sha256: dict[str, str]
    run_count: Literal[16]
    matched_group_count: Literal[4]
    scenario_count: Literal[4]
    seed_count: Literal[1]
    condition_count: Literal[4]
    logical_request_count: Literal[288]
    transport_attempt_count: Literal[289]
    primary_metrics: tuple[str, ...]
    false_adoption_source: Literal["adoption_decision_only"]
    epistemic_stance_used_for_false_adoption: Literal[False]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    private_truth_exposed: Literal[False]
    network_used_for_analysis: Literal[False]


class HG232AnalysisReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    analysis_id: Literal["provenance-cascade-hg232-adoption-identifiability-analysis"]
    analysis_version: Literal["provenance_cascade_hg232_adoption_decision_analysis.v2"]
    data_integrity_status: Literal["passed"]
    analysis_scope: Literal["single_seed_calibration_descriptive_only"]
    effect_evidence_status: Literal["not_assessed"]
    decision_status: Literal["thresholds_not_pre_registered"]
    false_adoption_source: Literal["adoption_decision_only"]
    run_metrics: tuple[HFRunMetric, ...]
    scenario_condition_metrics: tuple[HG232ScenarioConditionMetric, ...]
    condition_summaries: tuple[HFConditionMetricSummary, ...]
    behavior_summaries: tuple[HG232BehaviorSummary, ...]
    condition_behavior_summaries: tuple[HG232ConditionBehaviorSummary, ...]
    eligibility_summaries: tuple[HG232EligibilitySummary, ...]
    run_operations: tuple[HFRunOperations, ...]
    condition_operations: tuple[HFConditionOperations, ...]
    recovery_audit: HG232RecoveryAudit
    not_applicable_reasons: dict[str, int]
    warnings: tuple[str, ...]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    private_truth_exposed: Literal[False]


class HG232AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["completed", "validated", "blocked"]
    run_count: int = Field(default=0, ge=0)
    matched_group_count: int = Field(default=0, ge=0)
    metric_count: int = Field(default=0, ge=0)
    output_dir: str | None = None
    blocking_reasons: tuple[str, ...] = ()
    network: Literal["disabled"] = "disabled"
    provider_constructed: Literal[False] = False
    private_truth_exposed: Literal[False] = False
    not_paper_result: Literal[True] = True
    no_causal_conclusion: Literal[True] = True


def _require(value: bool, code: str) -> None:
    if not value:
        raise HG232AnalysisError(code)


def _safe(value: object) -> None:
    def walk(item: object) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                _require(str(key).lower() not in _FORBIDDEN, "analysis_output_sensitive_field_detected")
                walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
    walk(value)


def _behavior(spec: Any, decisions: tuple[Any, ...]) -> HG232BehaviorSummary:
    adoption = Counter(item.adoption_decision for item in decisions)
    sharing = Counter(item.sharing_decision for item in decisions)
    round0 = [item for item in decisions if item.round_id == 0]
    final = [item for item in decisions if item.round_id == 2]
    return HG232BehaviorSummary(
        run_id=spec.run_id, scenario_id=spec.scenario_id, condition=spec.condition,
        observation_count=18,
        adoption_counts={item.value: adoption[item.value] for item in AdoptionDecision},
        sharing_counts={item.value: sharing[item.value] for item in SharingDecision},
        share_requested_count=sum(item.share_requested for item in decisions),
        round0_adopt_count=sum(item.adoption_decision == AdoptionDecision.ADOPT_CLAIM.value for item in round0),
        final_adopt_count=sum(item.adoption_decision == AdoptionDecision.ADOPT_CLAIM.value for item in final),
        final_share_count=sum(item.sharing_decision in {SharingDecision.SHARE.value, SharingDecision.SHARE_WITH_CAVEAT.value} for item in final),
    )


def _condition_behavior(rows: list[HG232BehaviorSummary]) -> tuple[HG232ConditionBehaviorSummary, ...]:
    output = []
    for condition in CascadeCondition:
        selected = [item for item in rows if item.condition is condition]
        adoption: Counter[str] = Counter(); sharing: Counter[str] = Counter()
        for item in selected:
            adoption.update(item.adoption_counts); sharing.update(item.sharing_counts)
        output.append(HG232ConditionBehaviorSummary(
            condition=condition, run_count=4, observation_count=72,
            adoption_counts=dict(adoption), sharing_counts=dict(sharing),
            share_requested_count=sum(item.share_requested_count for item in selected),
            round0_adopt_count=sum(item.round0_adopt_count for item in selected),
            final_adopt_count=sum(item.final_adopt_count for item in selected),
            final_share_count=sum(item.final_share_count for item in selected),
        ))
    return tuple(output)


def _decision_history(record: CascadeRealAgentRunRecord, decisions: tuple[Any, ...]) -> tuple[dict[tuple[str, str], Any], dict[tuple[str, str], list[Any]]]:
    assert record.outcome_ledger is not None
    coordinate_claim: dict[tuple[str, int], str] = {}
    for outcome in record.outcome_ledger.outcomes:
        coordinate = (outcome.agent_id, outcome.round_id)
        _require(coordinate not in coordinate_claim, "multiple_claim_outcomes_per_agent_round")
        coordinate_claim[coordinate] = outcome.claim_id
    _require(len(coordinate_claim) == 18, "outcome_coordinate_count_mismatch")
    history: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for item in sorted(decisions, key=lambda value: (value.round_id, value.agent_id)):
        claim_id = coordinate_claim.get((item.agent_id, item.round_id))
        _require(claim_id is not None, "decision_outcome_coordinate_mismatch")
        history[(item.agent_id, claim_id)].append(item)
    latest = {key: values[-1] for key, values in history.items()}
    return latest, history


def _public_exposure_hg232(record: CascadeRealAgentRunRecord, graph: Any) -> tuple[dict[Any, set[str]], dict[Any, set[str]], dict[Any, set[str]], dict[Any, set[str]]]:
    """Join public exposure to provenance semantics, not delivery actor alone.

    The frozen fixtures use the system as a deterministic delivery mechanism
    for public repost/recommendation content.  Such content remains social
    propagation when its channel or public graph edge says so.
    """
    assert record.exposure_ledger is not None
    content: dict[tuple[str, str], set[str]] = defaultdict(set)
    social: dict[tuple[str, str], set[str]] = defaultdict(set)
    evidence: dict[tuple[str, str], set[str]] = defaultdict(set)
    roots: dict[tuple[str, str], set[str]] = defaultdict(set)
    social_nodes = {
        edge.target_node_id for edge in graph.edges
        if edge.relation.value in {"repost", "reply"}
    }
    for event in record.exposure_ledger.events:
        key = (event.target_agent_id, event.claim_id)
        content[key].add(event.content_id)
        evidence[key].update(event.evidence_card_ids)
        roots[key].update(graph.root_sources_for_node(event.provenance_node_id))
        if (
            event.source_actor_id != "system"
            or event.channel.value in {"reply", "repost", "recommendation"}
            or event.provenance_node_id in social_nodes
        ):
            social[key].add(event.content_id)
    return content, social, evidence, roots


def _evaluate_adoption_run(spec: Any, record: CascadeRealAgentRunRecord, graph: Any, truth: Any,
                           decisions: tuple[Any, ...], entries: list[Any]) -> tuple[list[HFRunMetric], HFRunOperations]:
    false_claims = {item.claim_id for item in truth.records if item.ground_truth_label is GroundTruthLabel.FALSE}
    correction_claims = {item.claim_id for item in truth.records if item.ground_truth_label is GroundTruthLabel.TRUE and item.source_independence_label is SourceIndependenceLabel.INDEPENDENT}
    latest, history = _decision_history(record, decisions)
    content, social, evidence, roots = _public_exposure_hg232(record, graph)
    cards = {card.evidence_id: card for card in graph.evidence_cards}

    false_eligible = [key for key in latest if key[1] in false_claims]
    false_adopt = sum(latest[key].adoption_decision == AdoptionDecision.ADOPT_CLAIM.value for key in false_eligible)
    metrics = [_metric(spec, EXPECTED_METRICS[0], false_adopt, len(false_eligible), warnings=("adoption_decision_operationalization",))]
    correction_eligible = [key for key in latest if key[1] in correction_claims and content.get(key)]
    correction_adopt = sum(latest[key].adoption_decision == AdoptionDecision.ADOPT_CLAIM.value for key in correction_eligible)
    metrics.append(_metric(spec, EXPECTED_METRICS[1], correction_adopt, len(correction_eligible), warnings=("no_supported_correction_exposure",) if not correction_eligible else ()))
    receptivity_den = receptivity_num = 0
    for agent_id in record.agent_ids:
        exposed_corrections = [claim for claim in correction_claims if content.get((agent_id, claim))]
        for false_claim in false_claims:
            values = history.get((agent_id, false_claim), [])
            if not values or values[0].adoption_decision != AdoptionDecision.ADOPT_CLAIM.value or not exposed_corrections:
                continue
            receptivity_den += 1
            rejects_false = latest[(agent_id, false_claim)].adoption_decision == AdoptionDecision.REJECT_CLAIM.value
            adopts_correction = any(latest.get((agent_id, claim)) and latest[(agent_id, claim)].adoption_decision == AdoptionDecision.ADOPT_CLAIM.value for claim in exposed_corrections)
            receptivity_num += rejects_false or adopts_correction
    metrics.append(_metric(spec, EXPECTED_METRICS[2], receptivity_num, receptivity_den, warnings=("no_exposure_linked_adoption_transition",) if not receptivity_den else ()))
    harmful_den = harmful_num = 0
    for key, value in latest.items():
        if key[1] not in false_claims or not social.get(key):
            continue
        independently_supportive = any(key[1] in cards[evidence_id].supports for evidence_id in evidence.get(key, set()))
        if independently_supportive:
            continue
        harmful_den += 1; harmful_num += value.adoption_decision == AdoptionDecision.ADOPT_CLAIM.value
    metrics.append(_metric(spec, EXPECTED_METRICS[3], harmful_num, harmful_den, warnings=("no_socially_exposed_false_claim_unit",) if not harmful_den else ("operational_not_causal_conformity",)))
    assert record.application_ledger is not None
    applied = [item for item in record.application_ledger.schedules if item.status is CascadeScheduleStatus.APPLIED]
    false_positive = sum(item.claim_id in correction_claims for item in applied)
    metrics.append(_metric(spec, EXPECTED_METRICS[4], false_positive if applied else None, len(applied), warnings=("no_applied_intervention",) if not applied else ()))
    root_values = [len(value) for value in roots.values()]
    metrics.append(_metric(spec, EXPECTED_METRICS[5], None, len(root_values), value=statistics.mean(root_values) if root_values else None, warnings=("no_exposed_agent_claim_unit",) if not root_values else ()))
    metrics.append(_metric(spec, EXPECTED_METRICS[6], 1, 1))
    terminal = [item for item in entries if item.status is not RequestLedgerStatus.STARTED]
    completed = [item for item in terminal if item.status is RequestLedgerStatus.COMPLETED]
    token_values = [item.total_tokens for item in completed if item.total_tokens is not None]
    latency_values = [item.latency_ms for item in completed if item.latency_ms is not None]
    metrics.append(_metric(spec, EXPECTED_METRICS[7], None, len(token_values), value=float(sum(token_values)) if len(token_values) == len(completed) else None, warnings=("provider_usage_missing",) if len(token_values) != len(completed) else ()))
    metrics.append(_metric(spec, EXPECTED_METRICS[8], None, len(latency_values), value=float(statistics.mean(latency_values)) if len(latency_values) == len(completed) else None, warnings=("provider_latency_missing",) if len(latency_values) != len(completed) else ()))
    statuses = Counter(item.status for item in record.application_ledger.schedules)
    failures = {item.fingerprint for item in terminal if item.status is RequestLedgerStatus.FAILED}
    completions = {item.fingerprint for item in terminal if item.status is RequestLedgerStatus.COMPLETED}
    operations = HFRunOperations(
        run_id=spec.run_id, scenario_id=spec.scenario_id, seed=spec.seed, condition=spec.condition,
        proposal_count=record.proposal_count, applied_count=statuses[CascadeScheduleStatus.APPLIED],
        rejected_count=statuses[CascadeScheduleStatus.REJECTED], cancelled_count=statuses[CascadeScheduleStatus.CANCELLED],
        scheduled_unapplied_count=statuses[CascadeScheduleStatus.SCHEDULED],
        abstain_count=max(0, record.proposal_count - len(record.application_ledger.schedules)),
        logical_request_count=len({item.fingerprint for item in entries}),
        transport_attempt_count=sum(item.status is RequestLedgerStatus.STARTED for item in entries),
        connection_failure_count=sum(item.status is RequestLedgerStatus.FAILED and item.error_code in {"timeout", "connection_failure"} for item in terminal),
        connection_recovery_count=len(failures & completions),
        prompt_tokens=sum(item.prompt_tokens or 0 for item in completed) if all(item.prompt_tokens is not None for item in completed) else None,
        completion_tokens=sum(item.completion_tokens or 0 for item in completed) if all(item.completion_tokens is not None for item in completed) else None,
        total_tokens=sum(item.total_tokens or 0 for item in completed) if all(item.total_tokens is not None for item in completed) else None,
        mean_latency_ms=float(statistics.mean(latency_values)) if len(latency_values) == len(completed) else None,
    )
    return metrics, operations


def analyze_hg232() -> tuple[HG232AnalysisManifest, HG232AnalysisReport]:
    audit_disk = HG232FinalIntegrityAuditReceipt.model_validate_json(AUDIT_RECEIPT_PATH.read_text(encoding="utf-8"))
    _require(audit_hg232_integrity() == audit_disk, "integrity_audit_state_changed")
    config, config_path = load_config(DEFAULT_CONFIG, allow_existing_output=True)
    base = HG1PilotConfig.model_validate(tomllib.loads(BASE_MATERIAL_CONFIG.read_text(encoding="utf-8")))
    material_by = {item.scenario_id: item for item in base.scenario_materials}
    from .provenance_cascade_hg232_integrity_audit import _specs
    specs = _specs(tuple(item.scenario_id for item in base.scenario_materials), config.seed)
    batch = HG21BatchRecord.model_validate_json(BATCH_PATH.read_text(encoding="utf-8"))
    states = {item.run_id: item for item in batch.runs}
    public: list[tuple[Any, CascadeRealAgentRunRecord, tuple[Any, ...], list[Any]]] = []
    behaviors: list[HG232BehaviorSummary] = []
    run_hashes: dict[str, str] = {}
    # Validate all public inputs before loading evaluator-only annotations.
    for spec in specs:
        run_dir = PILOT_ROOT / spec.run_id
        record_path = run_dir / "run_record.json"
        _require(sha256_file(record_path) == states[spec.run_id].run_record_sha256 == audit_disk.run_record_sha256[spec.run_id], "run_record_hash_mismatch")
        record = CascadeRealAgentRunRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
        entries = RequestLedger(run_dir / "request_ledger.jsonl").entries()
        binding = _hash_binding(spec, config, sha256_file(config_path), batch.model_name)
        decisions = load_hg232_decisions(run_dir / "behavior_decisions.jsonl", binding)
        _require(len(decisions) == 18, "behavior_decision_count_mismatch")
        public.append((spec, record, decisions, entries)); behaviors.append(_behavior(spec, decisions)); run_hashes[spec.run_id] = sha256_file(record_path)

    truth_by: dict[str, Any] = {}
    truth_hashes: list[HFSafeInputHash] = []
    for scenario_id, material in material_by.items():
        truth_path = _ROOT / material.truth_path
        _require(sha256_file(truth_path) == material.truth_sha256, "evaluator_binding_hash_mismatch")
        truth = EvaluatorTruthLoader.load(truth_path)
        validate_fixture_pair(next(record for key, record in _scenario_graphs().items() if key == scenario_id), truth)
        truth_by[scenario_id] = truth
        truth_hashes.append(HFSafeInputHash(kind=f"evaluator_annotation:{scenario_id}", sha256=material.truth_sha256))

    metrics: list[HFRunMetric] = []; operations: list[HFRunOperations] = []; eligibility = []
    graphs = _scenario_graphs()
    for spec, record, decisions, entries in public:
        run_metrics, run_ops = _evaluate_adoption_run(spec, record, graphs[spec.scenario_id], truth_by[spec.scenario_id], decisions, entries)
        metrics.extend(run_metrics); operations.append(run_ops)
        den = {item.metric_name: item.denominator for item in run_metrics}
        eligibility.append(HG232EligibilitySummary(
            run_id=spec.run_id, scenario_id=spec.scenario_id, condition=spec.condition,
            false_adoption_eligible_count=den[EXPECTED_METRICS[0]], supported_correction_eligible_count=den[EXPECTED_METRICS[1]],
            correction_transition_eligible_count=den[EXPECTED_METRICS[2]], harmful_conformity_eligible_count=den[EXPECTED_METRICS[3]],
            intervention_false_positive_eligible_count=den[EXPECTED_METRICS[4]],
        ))
    scenario_metrics = tuple(HG232ScenarioConditionMetric(
        scenario_id=item.scenario_id, condition=item.condition, metric_name=item.metric_name,
        status=item.status, value=item.value, numerator=item.numerator, denominator=item.denominator,
        eligible_unit={
            EXPECTED_METRICS[0]:"final_agent_false_claim_decision", EXPECTED_METRICS[1]:"exposed_agent_supported_correction_decision",
            EXPECTED_METRICS[2]:"initial_adopter_exposed_to_correction", EXPECTED_METRICS[3]:"socially_exposed_false_claim_agent",
            EXPECTED_METRICS[4]:"applied_non_abstain_intervention", EXPECTED_METRICS[5]:"agent_claim_exposure",
            EXPECTED_METRICS[6]:"included_run", EXPECTED_METRICS[7]:"completed_provider_request",
            EXPECTED_METRICS[8]:"completed_provider_request",
        }[item.metric_name], warnings=item.warnings,
    ) for item in metrics)
    n_a = Counter(warning for item in metrics if item.status is HFMetricStatus.NOT_APPLICABLE for warning in item.warnings)
    warnings = {"single_seed_calibration", "descriptive_only", "thresholds_not_pre_registered", "no_significance_test", "no_causal_conclusion", "false_adoption_uses_adoption_decision_not_epistemic_stance", "social_exposure_join_uses_public_provenance_relation"}
    warnings.update(warning for item in metrics for warning in item.warnings)
    input_hashes = (
        HFSafeInputHash(kind="config", sha256=sha256_file(config_path)),
        HFSafeInputHash(kind="protocol_template", sha256=config.protocol_sha256),
        HFSafeInputHash(kind="v4_resume_runner", sha256=audit_disk.v4_resume_runner_sha256),
        HFSafeInputHash(kind="technical_amendment", sha256=audit_disk.technical_amendment_sha256),
        HFSafeInputHash(kind="final_integrity_audit", sha256=sha256_file(AUDIT_RECEIPT_PATH)),
        HFSafeInputHash(kind="final_batch_record", sha256=sha256_file(BATCH_PATH)),
        HFSafeInputHash(kind="calibration_receipt", sha256=sha256_file(CALIBRATION_RECEIPT_PATH)),
    ) + tuple(truth_hashes)
    manifest = HG232AnalysisManifest(
        analysis_id="provenance-cascade-hg232-adoption-identifiability-analysis", analysis_version=ANALYSIS_VERSION,
        study_id=config.study_id, input_hashes=input_hashes, run_record_sha256=run_hashes,
        run_count=16, matched_group_count=4, scenario_count=4, seed_count=1, condition_count=4,
        logical_request_count=288, transport_attempt_count=289, primary_metrics=EXPECTED_METRICS,
        false_adoption_source="adoption_decision_only", epistemic_stance_used_for_false_adoption=False,
        development_only=True, calibration_only=True, not_paper_result=True, no_causal_conclusion=True,
        private_truth_exposed=False, network_used_for_analysis=False,
    )
    report = HG232AnalysisReport(
        analysis_id=manifest.analysis_id, analysis_version=ANALYSIS_VERSION, data_integrity_status="passed",
        analysis_scope="single_seed_calibration_descriptive_only", effect_evidence_status="not_assessed",
        decision_status="thresholds_not_pre_registered", false_adoption_source="adoption_decision_only",
        run_metrics=tuple(metrics), scenario_condition_metrics=scenario_metrics,
        condition_summaries=_condition_summaries(metrics), behavior_summaries=tuple(behaviors),
        condition_behavior_summaries=_condition_behavior(behaviors), eligibility_summaries=tuple(eligibility),
        run_operations=tuple(operations), condition_operations=_condition_operations(operations),
        recovery_audit=HG232RecoveryAudit(unique_logical_request_count=288, transport_attempt_count=289,
            timeout_failed_attempt_count=1, timeout_recovered_fingerprint_count=1,
            duplicate_completed_fingerprint_count=0, parser_recovery_count=0),
        not_applicable_reasons=dict(sorted(n_a.items())), warnings=tuple(sorted(warnings)),
        development_only=True, calibration_only=True, not_paper_result=True,
        no_causal_conclusion=True, private_truth_exposed=False,
    )
    _safe({"manifest":manifest.model_dump(mode="json"),"report":report.model_dump(mode="json")})
    return manifest, report


def _scenario_graphs() -> dict[str, Any]:
    from .provenance_cascade_hg2 import load_hg2_config
    _, scenarios, _ = load_hg2_config(_ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg2_calibration.v1.toml")
    return {key: value.graph for key, value in scenarios.items()}


def _hash_binding(spec: Any, config: Any, config_sha: str, model_name: str) -> str:
    from .provenance_cascade_hg21_calibration import _hash_json
    return _hash_json({"config_sha256":config_sha,"protocol_sha256":config.protocol_sha256,
        "template_version":config.template_version,"scenario_id":spec.scenario_id,"seed":spec.seed,
        "condition":spec.condition.value,"run_id":spec.run_id,"model_name":model_name,
        "max_tokens":config.agent_max_tokens,"temperature":config.temperature})


def run_analysis(*, write_outputs: bool = False, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> HG232AnalysisResult:
    try:
        manifest, report = analyze_hg232(); destination = Path(output_dir)
        if write_outputs:
            if destination.exists():
                raise HG232AnalysisError("analysis_output_exists")
            destination.mkdir(parents=True, exist_ok=False)
            (destination / "analysis_manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
            (destination / "analysis_report.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return HG232AnalysisResult(status="completed" if write_outputs else "validated", run_count=16,
            matched_group_count=4, metric_count=len(report.run_metrics), output_dir=str(destination) if write_outputs else None)
    except HG232AnalysisError as exc:
        return HG232AnalysisResult(status="blocked", blocking_reasons=(exc.code,))
    except Exception:
        return HG232AnalysisResult(status="blocked", blocking_reasons=("hg232_analysis_failed",))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline evaluator-only H-G.2.3.2 analysis.")
    parser.add_argument("--write-outputs", action="store_true"); args = parser.parse_args(argv)
    result = run_analysis(write_outputs=args.write_outputs)
    print(result.model_dump_json()); return 0 if result.status != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
