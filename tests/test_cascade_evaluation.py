from pathlib import Path
import pytest
from pydantic import ValidationError
from evicon.cascade_protocol import CascadeProtocolRunner, CascadeScenarioLoader
from evicon.cascade_outcomes import CascadeOutcomeLedger, ClaimStance, OutcomeSource, PublicClaimOutcome
from evicon.cascade_evaluation import CascadeOfflineEvaluator, CascadeEvaluationError, CascadeMatchedGroup, CascadeMatchedRun, CascadeMetricStatus
from evicon.provenance_cascade import EvaluatorTruthLoader
from evicon.provenance_cascade_preregistration import ProvenanceCascadePreregistrationLoader, CascadeCondition
from evicon.cascade_replay import CascadeReplayReport, CascadeReplayStatus

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'configs/provenance_cascade'
PREREG=ProvenanceCascadePreregistrationLoader.load(ROOT/'configs/studies/provenance_cascade_pilot_preregistration.toml')

def build(name='false_majority'):
    scenario=CascadeScenarioLoader.load(BASE/'scenarios'/f'{name}.toml')
    run=CascadeProtocolRunner().run_scenario(scenario, 20260901)
    by={}
    for e in run.ledger.events: by.setdefault((e.target_agent_id,e.claim_id),[]).append(e)
    claims=sorted({e.claim_id for e in run.ledger.events})
    out=[]
    for agent in run.agent_ids:
        for claim in claims:
            events=by.get((agent,claim),[])
            out.append(PublicClaimOutcome(scenario_id=run.scenario_id,agent_id=agent,claim_id=claim,round_id=2,stance=ClaimStance.ENDORSES if events else ClaimStance.NO_POSITION,content_ids=tuple(e.content_id for e in events),evidence_card_ids=tuple(x for e in events for x in e.evidence_card_ids),source=OutcomeSource.DETERMINISTIC_FIXTURE))
    outcomes=CascadeOutcomeLedger(scenario_id=run.scenario_id,agent_ids=run.agent_ids,outcomes=tuple(out))
    truth=EvaluatorTruthLoader.load(BASE/'fixtures'/f'{name}.evaluator_private.json')
    return scenario,run,outcomes,truth

def test_false_metrics_and_safe_report():
    s,r,o,t=build()
    report=CascadeOfflineEvaluator.evaluate(run=r,graph=s.graph,outcomes=o,truth=t,preregistration=PREREG)
    assert [m.metric_name for m in report.metrics]==list(PREREG.primary_metrics)
    assert report.metrics[0].status is CascadeMetricStatus.VALUE
    dumped=str(report.model_dump())
    assert 'ground_truth_label' not in dumped and 'source_independence_label' not in dumped

def test_public_outcome_cannot_contain_private_fields():
    with pytest.raises(ValidationError):
        PublicClaimOutcome.model_validate({'scenario_id':'x','agent_id':'a','claim_id':'c','round_id':0,'stance':'uncertain','source':'deterministic_fixture','ground_truth_label':'false'})

def test_unexposed_content_rejected():
    s,r,o,t=build()
    bad=o.outcomes[0].model_copy(update={'content_ids':('not-exposed',)})
    badledger=CascadeOutcomeLedger(scenario_id=o.scenario_id,agent_ids=o.agent_ids,outcomes=(bad,)+o.outcomes[1:])
    with pytest.raises(ValueError, match="outcome_public_validation_failed"):
        CascadeOfflineEvaluator.evaluate(run=r,graph=s.graph,outcomes=badledger,truth=t,preregistration=PREREG)

def test_outcome_time_reversal_rejected():
    a=PublicClaimOutcome(scenario_id='s',agent_id='a',claim_id='c',round_id=2,stance=ClaimStance.UNCERTAIN,source=OutcomeSource.DETERMINISTIC_FIXTURE)
    b=PublicClaimOutcome(scenario_id='s',agent_id='a',claim_id='c',round_id=1,stance=ClaimStance.UNCERTAIN,source=OutcomeSource.DETERMINISTIC_FIXTURE)
    with pytest.raises(ValidationError): CascadeOutcomeLedger(scenario_id='s',agent_ids=('a',),outcomes=(a,b))

def test_token_and_latency_are_not_applicable_without_usage():
    s,r,o,t=build('true_minority_correction')
    report=CascadeOfflineEvaluator.evaluate(run=r,graph=s.graph,outcomes=o,truth=t,preregistration=PREREG)
    assert report.metrics[-1].status is CascadeMetricStatus.NOT_APPLICABLE
    assert report.metrics[-2].status is CascadeMetricStatus.NOT_APPLICABLE

def test_replay_failure_blocks_evaluation():
    s,r,o,t=build()
    failed=r.model_copy(update={'replay':CascadeReplayReport(status=CascadeReplayStatus.BLOCKED,error_code='x',scenario_id=s.scenario_id,agent_count=6,exposure_count=0,snapshot_count=0,root_relation_count=0)})
    with pytest.raises(CascadeEvaluationError, match='replay_not_passed'): CascadeOfflineEvaluator.evaluate(run=failed,graph=s.graph,outcomes=o,truth=t,preregistration=PREREG)

def test_matched_group_requires_all_fixed_conditions_and_matching_config():
    base=dict(scenario_id='s',seed=1,agent_ids=('a',),round_count=3,config_hash='a'*64,outcome_contract_hash='b'*64)
    runs=tuple(CascadeMatchedRun(run_id=f'r{i}',condition=c,**base) for i,c in enumerate(CascadeCondition))
    group=CascadeMatchedGroup(matched_group_id='g',runs=runs)
    assert len(group.runs)==4
    with pytest.raises(ValidationError): CascadeMatchedGroup(matched_group_id='g',runs=runs[:3])

def test_supported_correction_and_independent_roots_are_safe():
    for name in ('true_minority_correction','independent_true_consensus','unresolved_disagreement'):
        s,r,o,t=build(name)
        report=CascadeOfflineEvaluator.evaluate(run=r,graph=s.graph,outcomes=o,truth=t,preregistration=PREREG)
        assert report.replay_status=='passed'
        assert report.development_only and report.no_causal_conclusion
