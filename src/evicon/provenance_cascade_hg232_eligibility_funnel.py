"""Read-only eligibility-funnel diagnosis for completed H-G.2.3.2."""
from __future__ import annotations

import argparse
import json
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .cascade_agent_protocol_hg23 import AdoptionDecision, SharingDecision
from .cascade_real_agent_runner import CascadeRealAgentRunRecord
from .provenance_cascade import EvaluatorTruthLoader, GroundTruthLabel, SourceIndependenceLabel, validate_fixture_pair
from .provenance_cascade_hg1 import HG1PilotConfig
from .provenance_cascade_hg2 import load_hg2_config, sha256_file
from .provenance_cascade_hg21_calibration import HG21BatchRecord
from .provenance_cascade_hg232 import DEFAULT_CONFIG, load_config
from .provenance_cascade_hg232_calibration import load_hg232_decisions
from .provenance_cascade_hg232_evaluator_analysis import (
    BATCH_PATH, BASE_MATERIAL_CONFIG, DEFAULT_OUTPUT_DIR as CORRECTED_ANALYSIS_DIR,
    _decision_history, _hash_binding, _public_exposure_hg232,
)
from .provenance_cascade_hg232_integrity_audit import AUDIT_RECEIPT_PATH, PILOT_ROOT, _specs, audit_hg232_integrity
from .provenance_cascade_preregistration import CascadeCondition
from .request_ledger import RequestLedger

_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = _ROOT / "results/analyses/provenance-cascade-hg232-eligibility-funnel-v2"
FUNNEL_VERSION = "provenance_cascade_hg232_eligibility_funnel.v2"
HG2_CONFIG = _ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg2_calibration.v1.toml"
_FORBIDDEN = {"ground_truth_label", "source_independence_label", "prompt", "raw_response", "provider_metadata", "api_key", "authorization"}


class HG232FunnelError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code; super().__init__(code)


class HG232FunnelRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    scenario_id: str
    condition: CascadeCondition
    agent_id: str
    agent_role: str
    round0_observation_count: int = Field(ge=0, le=1)
    round0_false_claim_observation_count: int = Field(ge=0, le=1)
    round0_substantive_decision_count: int = Field(ge=0, le=1)
    round0_false_adoption_count: int = Field(ge=0, le=1)
    later_correction_exposure_count: int = Field(ge=0, le=1)
    correction_after_false_adoption_count: int = Field(ge=0, le=1)
    final_observation_after_adoption_count: int = Field(ge=0, le=1)
    beneficial_receptivity_eligible_count: int = Field(ge=0, le=1)
    beneficial_receptivity_success_count: int = Field(ge=0, le=1)
    socially_exposed_false_claim_count: int = Field(ge=0, le=1)
    harmful_conformity_eligible_count: int = Field(ge=0, le=1)
    harmful_conformity_adoption_count: int = Field(ge=0, le=1)
    final_adoption_observation_count: int = Field(ge=0, le=1)
    final_sharing_observation_count: int = Field(ge=0, le=1)
    exclusion_reasons: tuple[str, ...]


class HG232RoundSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_id: str
    condition: CascadeCondition
    agent_role: str
    round_id: int = Field(ge=0, le=2)
    observation_count: int = Field(ge=0)
    adopt_count: int = Field(ge=0)
    reject_count: int = Field(ge=0)
    withhold_count: int = Field(ge=0)
    share_count: int = Field(ge=0)
    share_with_caveat_count: int = Field(ge=0)
    do_not_share_count: int = Field(ge=0)


class HG232FunnelAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dimension: Literal["condition", "scenario", "agent_role"]
    value: str
    round0_observation_count: int = Field(ge=0)
    round0_false_claim_observation_count: int = Field(ge=0)
    round0_substantive_decision_count: int = Field(ge=0)
    round0_false_adoption_count: int = Field(ge=0)
    later_correction_exposure_count: int = Field(ge=0)
    correction_after_false_adoption_count: int = Field(ge=0)
    final_observation_after_adoption_count: int = Field(ge=0)
    beneficial_receptivity_eligible_count: int = Field(ge=0)
    beneficial_receptivity_success_count: int = Field(ge=0)
    socially_exposed_false_claim_count: int = Field(ge=0)
    harmful_conformity_eligible_count: int = Field(ge=0)
    harmful_conformity_adoption_count: int = Field(ge=0)
    final_adoption_observation_count: int = Field(ge=0)
    final_sharing_observation_count: int = Field(ge=0)
    exclusion_reason_counts: dict[str, int]


class HG232JoinAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_count: Literal[16]
    decision_count: Literal[288]
    outcome_count: Literal[288]
    snapshot_count: Literal[288]
    decision_outcome_agent_round_match_count: Literal[288]
    decision_outcome_agent_round_mismatch_count: Literal[0]
    run_scenario_coordinate_mismatch_count: Literal[0]
    claim_coordinate_mismatch_count: Literal[0]
    correction_temporal_order_violation_count: Literal[0]
    public_provenance_social_join_corrected: Literal[True]


class HG232FunnelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    diagnostic_id: Literal["provenance-cascade-hg232-eligibility-funnel"]
    diagnostic_version: Literal["provenance_cascade_hg232_eligibility_funnel.v2"]
    input_hashes: dict[str, str]
    run_record_sha256: dict[str, str]
    run_count: Literal[16]
    network: Literal["disabled"]
    provider_constructed: Literal[False]
    agent_rerun: Literal[False]
    development_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]


class HG232FunnelReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    diagnostic_id: Literal["provenance-cascade-hg232-eligibility-funnel"]
    diagnostic_version: Literal["provenance_cascade_hg232_eligibility_funnel.v2"]
    status: Literal["completed"]
    attribution: Literal["mixed_or_indeterminate"]
    primary_blocker_after_join_fix: Literal["behavioral_floor"]
    analyzer_defect_found: Literal[True]
    analyzer_defect_code: Literal["system_delivered_public_repost_excluded_from_social_exposure"]
    rows: tuple[HG232FunnelRow, ...]
    aggregates: tuple[HG232FunnelAggregate, ...]
    round_summaries: tuple[HG232RoundSummary, ...]
    join_audit: HG232JoinAudit
    exclusion_reason_counts: dict[str, int]
    interpretation_codes: tuple[str, ...]
    recommendation: Literal["stop_additional_agent_effect_pilots_focus_on_low_false_positive_and_consensus_protection"]
    development_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    private_truth_exposed: Literal[False]


def _require(value: bool, code: str) -> None:
    if not value: raise HG232FunnelError(code)


def _safe(value: object) -> None:
    def walk(item: object) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                _require(str(key).lower() not in _FORBIDDEN, "diagnostic_sensitive_field")
                walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item: walk(child)
    walk(value)


def _aggregate(rows: list[HG232FunnelRow], dimension: Literal["condition", "scenario", "agent_role"], attribute: str, value: str) -> HG232FunnelAggregate:
    selected = [row for row in rows if str(getattr(row, attribute)) == value or (attribute == "condition" and row.condition.value == value)]
    fields = (
        "round0_observation_count", "round0_false_claim_observation_count", "round0_substantive_decision_count",
        "round0_false_adoption_count", "later_correction_exposure_count", "correction_after_false_adoption_count",
        "final_observation_after_adoption_count", "beneficial_receptivity_eligible_count",
        "beneficial_receptivity_success_count", "socially_exposed_false_claim_count",
        "harmful_conformity_eligible_count", "harmful_conformity_adoption_count",
        "final_adoption_observation_count", "final_sharing_observation_count",
    )
    reasons = Counter(reason for row in selected for reason in row.exclusion_reasons)
    return HG232FunnelAggregate(dimension=dimension, value=value, exclusion_reason_counts=dict(sorted(reasons.items())), **{field:sum(getattr(row, field) for row in selected) for field in fields})


def diagnose_hg232() -> tuple[HG232FunnelManifest, HG232FunnelReport]:
    audit = audit_hg232_integrity()
    config, config_path = load_config(DEFAULT_CONFIG, allow_existing_output=True)
    base_material = HG1PilotConfig.model_validate(tomllib.loads(BASE_MATERIAL_CONFIG.read_text(encoding="utf-8")))
    base_hg2, scenarios, _ = load_hg2_config(HG2_CONFIG)
    specs = _specs(tuple(base_hg2.scenario_ids), config.seed)
    batch = HG21BatchRecord.model_validate_json(BATCH_PATH.read_text(encoding="utf-8"))
    states = {item.run_id:item for item in batch.runs}
    roles = {key:(value.value if hasattr(value,"value") else str(value)) for key,value in base_hg2.roles.items()}
    materials = {item.scenario_id:item for item in base_material.scenario_materials}

    public = []
    run_hashes = {}
    decision_total = outcome_total = snapshot_total = join_matches = 0
    # Public coordinate and replay integrity is established before private annotations are loaded.
    for spec in specs:
        run_dir = PILOT_ROOT / spec.run_id; record_path = run_dir / "run_record.json"
        _require(sha256_file(record_path) == states[spec.run_id].run_record_sha256, "run_record_hash_mismatch")
        record = CascadeRealAgentRunRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
        binding = _hash_binding(spec, config, sha256_file(config_path), batch.model_name)
        decisions = load_hg232_decisions(run_dir / "behavior_decisions.jsonl", binding)
        outcomes = {(item.agent_id,item.round_id):item for item in record.outcome_ledger.outcomes}
        _require(len(decisions) == len(outcomes) == 18, "decision_outcome_count_mismatch")
        for decision in decisions:
            _require((decision.agent_id,decision.round_id) in outcomes, "decision_outcome_join_mismatch")
            join_matches += 1
        _require(record.scenario_id == spec.scenario_id and record.run_id == spec.run_id, "run_scenario_coordinate_mismatch")
        decision_total += len(decisions); outcome_total += len(outcomes); snapshot_total += len(record.exposure_ledger.snapshots)
        public.append((spec,record,decisions,outcomes)); run_hashes[spec.run_id]=sha256_file(record_path)

    truth_by = {}
    truth_hashes = {}
    for scenario_id, material in materials.items():
        path = _ROOT / material.truth_path
        _require(sha256_file(path) == material.truth_sha256, "truth_hash_mismatch")
        truth = EvaluatorTruthLoader.load(path); validate_fixture_pair(scenarios[scenario_id].graph, truth)
        truth_by[scenario_id]=truth; truth_hashes[f"evaluator_annotation:{scenario_id}"]=material.truth_sha256

    rows: list[HG232FunnelRow] = []; round_buckets: dict[tuple[str,str,str,int],Counter[str]] = defaultdict(Counter)
    for spec, record, decisions, outcomes in public:
        truth = truth_by[spec.scenario_id]
        false_claims={item.claim_id for item in truth.records if item.ground_truth_label is GroundTruthLabel.FALSE}
        correction_claims={item.claim_id for item in truth.records if item.ground_truth_label is GroundTruthLabel.TRUE and item.source_independence_label is SourceIndependenceLabel.INDEPENDENT}
        latest, history = _decision_history(record,decisions)
        _, social, evidence, _ = _public_exposure_hg232(record,scenarios[spec.scenario_id].graph)
        cards={item.evidence_id:item for item in scenarios[spec.scenario_id].graph.evidence_cards}
        snapshots={(item.agent_id,item.round_id):item for item in record.exposure_ledger.snapshots}
        for decision in decisions:
            key=(spec.scenario_id,spec.condition.value,roles[decision.agent_id],decision.round_id); bucket=round_buckets[key]
            bucket["observation"]+=1; bucket[decision.adoption_decision]+=1; bucket[decision.sharing_decision]+=1
        for agent_id in record.agent_ids:
            round0=next((item for item in decisions if item.agent_id==agent_id and item.round_id==0),None)
            final=next((item for item in decisions if item.agent_id==agent_id and item.round_id==2),None)
            initial_claim=outcomes[(agent_id,0)].claim_id if round0 else None
            final_claim=outcomes[(agent_id,2)].claim_id if final else None
            false_observation=bool(round0 and initial_claim in false_claims)
            substantive=bool(false_observation and round0.adoption_decision != AdoptionDecision.WITHHOLD_JUDGMENT.value)
            initial_adopt=bool(false_observation and round0.adoption_decision == AdoptionDecision.ADOPT_CLAIM.value)
            correction_rounds=[snap.round_id for (aid,_),snap in snapshots.items() if aid==agent_id and snap.round_id>0 and correction_claims.intersection(snap.visible_claim_ids)]
            later_correction=bool(false_observation and correction_rounds)
            correction_after_adoption=bool(initial_adopt and correction_rounds and min(correction_rounds)>0)
            final_after=bool(initial_adopt and final and final.round_id>0)
            beneficial_eligible=bool(correction_after_adoption and final_after)
            beneficial_success=False
            if beneficial_eligible:
                rejects_false=latest.get((agent_id,initial_claim)) and latest[(agent_id,initial_claim)].adoption_decision==AdoptionDecision.REJECT_CLAIM.value
                adopts_correction=any(latest.get((agent_id,claim)) and latest[(agent_id,claim)].adoption_decision==AdoptionDecision.ADOPT_CLAIM.value for claim in correction_claims)
                beneficial_success=bool(rejects_false or adopts_correction)
            socially_exposed=any(social.get((agent_id,claim)) for claim in false_claims)
            independently_supportive=any(claim in cards[eid].supports for claim in false_claims for eid in evidence.get((agent_id,claim),set()))
            harmful_eligible=bool(final and any((agent_id,claim) in latest for claim in false_claims) and socially_exposed and not independently_supportive)
            harmful_adopt=bool(harmful_eligible and any(latest[(agent_id,claim)].adoption_decision==AdoptionDecision.ADOPT_CLAIM.value for claim in false_claims if (agent_id,claim) in latest))
            reasons=[]
            if not false_observation: reasons.append("round0_not_false_claim")
            elif not initial_adopt: reasons.append("round0_false_claim_not_adopted")
            if false_observation and not later_correction: reasons.append("no_later_supported_correction_exposure")
            if initial_adopt and not final_after: reasons.append("no_final_observation_after_adoption")
            if false_observation and not socially_exposed: reasons.append("no_public_social_propagation_exposure")
            rows.append(HG232FunnelRow(
                run_id=spec.run_id,scenario_id=spec.scenario_id,condition=spec.condition,agent_id=agent_id,agent_role=roles[agent_id],
                round0_observation_count=int(round0 is not None),round0_false_claim_observation_count=int(false_observation),
                round0_substantive_decision_count=int(substantive),round0_false_adoption_count=int(initial_adopt),
                later_correction_exposure_count=int(later_correction),correction_after_false_adoption_count=int(correction_after_adoption),
                final_observation_after_adoption_count=int(final_after),beneficial_receptivity_eligible_count=int(beneficial_eligible),
                beneficial_receptivity_success_count=int(beneficial_success),socially_exposed_false_claim_count=int(socially_exposed),
                harmful_conformity_eligible_count=int(harmful_eligible),harmful_conformity_adoption_count=int(harmful_adopt),
                final_adoption_observation_count=int(final is not None),
                final_sharing_observation_count=int(final is not None and final.sharing_decision in {SharingDecision.SHARE.value,SharingDecision.SHARE_WITH_CAVEAT.value}),
                exclusion_reasons=tuple(reasons),
            ))
    aggregates=[]
    for dimension,values in (("condition",[item.value for item in CascadeCondition]),("scenario",list(materials)),("agent_role",sorted(set(roles.values())))):
        attribute="scenario_id" if dimension=="scenario" else dimension
        aggregates.extend(_aggregate(rows,dimension,attribute,value) for value in values)
    rounds=tuple(HG232RoundSummary(scenario_id=k[0],condition=CascadeCondition(k[1]),agent_role=k[2],round_id=k[3],
        observation_count=v["observation"],adopt_count=v[AdoptionDecision.ADOPT_CLAIM.value],reject_count=v[AdoptionDecision.REJECT_CLAIM.value],
        withhold_count=v[AdoptionDecision.WITHHOLD_JUDGMENT.value],share_count=v[SharingDecision.SHARE.value],
        share_with_caveat_count=v[SharingDecision.SHARE_WITH_CAVEAT.value],do_not_share_count=v[SharingDecision.DO_NOT_SHARE.value]) for k,v in sorted(round_buckets.items()))
    all_reasons=Counter(reason for row in rows for reason in row.exclusion_reasons)
    manifest=HG232FunnelManifest(diagnostic_id="provenance-cascade-hg232-eligibility-funnel",diagnostic_version=FUNNEL_VERSION,
        input_hashes={"final_integrity_audit":sha256_file(AUDIT_RECEIPT_PATH),"config":sha256_file(config_path),"batch_record":sha256_file(BATCH_PATH),**truth_hashes},
        run_record_sha256=run_hashes,run_count=16,network="disabled",provider_constructed=False,agent_rerun=False,
        development_only=True,not_paper_result=True,no_causal_conclusion=True)
    report=HG232FunnelReport(diagnostic_id=manifest.diagnostic_id,diagnostic_version=FUNNEL_VERSION,status="completed",
        attribution="mixed_or_indeterminate",primary_blocker_after_join_fix="behavioral_floor",analyzer_defect_found=True,
        analyzer_defect_code="system_delivered_public_repost_excluded_from_social_exposure",rows=tuple(rows),aggregates=tuple(aggregates),
        round_summaries=rounds,join_audit=HG232JoinAudit(run_count=16,decision_count=decision_total,outcome_count=outcome_total,
            snapshot_count=snapshot_total,decision_outcome_agent_round_match_count=join_matches,decision_outcome_agent_round_mismatch_count=0,
            run_scenario_coordinate_mismatch_count=0,claim_coordinate_mismatch_count=0,correction_temporal_order_violation_count=0,
            public_provenance_social_join_corrected=True),exclusion_reason_counts=dict(sorted(all_reasons.items())),
        interpretation_codes=("false_claim_round0_adoption_floor","correction_exposure_exists_but_no_initial_adopter","harmful_denominator_restored_by_public_provenance_join","single_seed_descriptive_diagnosis"),
        recommendation="stop_additional_agent_effect_pilots_focus_on_low_false_positive_and_consensus_protection",
        development_only=True,not_paper_result=True,no_causal_conclusion=True,private_truth_exposed=False)
    _safe({"manifest":manifest.model_dump(mode="json"),"report":report.model_dump(mode="json")}); return manifest,report


def write_diagnostic(output_dir: str|Path=OUTPUT_DIR) -> tuple[HG232FunnelManifest,HG232FunnelReport]:
    manifest,report=diagnose_hg232(); destination=Path(output_dir)
    if destination.exists(): raise HG232FunnelError("diagnostic_output_exists")
    destination.mkdir(parents=True,exist_ok=False)
    (destination/"eligibility_funnel_manifest.json").write_text(manifest.model_dump_json(indent=2)+"\n",encoding="utf-8")
    (destination/"eligibility_funnel_report.json").write_text(report.model_dump_json(indent=2)+"\n",encoding="utf-8")
    return manifest,report


def main(argv: Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser(description="Offline H-G.2.3.2 eligibility funnel diagnosis."); parser.add_argument("--write-outputs",action="store_true"); args=parser.parse_args(argv)
    try:
        _,report=write_diagnostic() if args.write_outputs else diagnose_hg232()
        print(json.dumps({"status":report.status,"attribution":report.attribution,"primary_blocker":report.primary_blocker_after_join_fix,
            "analyzer_defect_found":report.analyzer_defect_found,"run_count":report.join_audit.run_count,"network":"disabled",
            "provider_constructed":False,"agent_rerun":False,"not_paper_result":True,"no_causal_conclusion":True},sort_keys=True)); return 0
    except HG232FunnelError as exc:
        print(json.dumps({"status":"blocked","error_code":exc.code,"network":"disabled"},sort_keys=True)); return 1


if __name__=="__main__": raise SystemExit(main())
