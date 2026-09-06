from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evicon.cascade_application_replay import CascadeApplicationReplayError, CascadeApplicationReplayValidator
from evicon.cascade_application_smoke import main as smoke_main, run_smoke
from evicon.cascade_controller import CascadeAction, CascadeControllerPolicyLoader, propose
from evicon.cascade_controller_smoke import fixture_view
from evicon.cascade_intervention_application import (
    CascadeApplicationLedger, CascadeApplicationReasonCode, CascadeApplicationStatus,
    CascadeDirectiveType, CascadeScheduleStatus, CascadeInterventionAuditEvent,
    ControlledRoundContext, ScheduledCascadeIntervention, StructuredCascadeDirective,
    snapshot_sha256,
)
from evicon.provenance_cascade_exposure import CascadeExposureSnapshot
from evicon.provenance_cascade_preregistration import CascadeCondition

ROOT=Path(__file__).resolve().parents[1]
POLICY=ROOT/'configs/provenance_cascade/cascade_controller_policy.v1.toml'
SCENARIOS=ROOT/'configs/provenance_cascade/scenarios'

def policy(): return CascadeControllerPolicyLoader.load(POLICY)
def current(): return fixture_view('false_majority', SCENARIOS)

def test_abstain_is_zero_effect_and_no_audit():
    view=current(); p=propose(CascadeCondition.NO_INTERVENTION,view,policy())
    ledger,result=CascadeApplicationLedger().schedule_proposal(p,view,snapshot_hash='a'*64,config=policy())
    assert result.status is CascadeApplicationStatus.NO_EFFECT
    assert result.schedule is None and result.audit_events == ()
    assert ledger.schedules == () and ledger.audit_events == ()

def test_false_majority_schedules_only_next_round_and_applies_sidecar():
    view=current(); p=propose(CascadeCondition.PROVENANCE_AWARE_CONTROLLER,view,policy())
    ledger,result=CascadeApplicationLedger().schedule_proposal(p,view,snapshot_hash='a'*64,config=policy())
    assert result.status is CascadeApplicationStatus.SCHEDULED
    schedule=result.schedule
    assert schedule and schedule.created_round_id == view.round_id and schedule.effective_round_id == view.round_id+1
    assert schedule.status is CascadeScheduleStatus.SCHEDULED
    assert result.directive is None
    next_view=view.model_copy(update={'round_id':view.round_id+1})
    applied_ledger,applied=ledger.apply_schedule(schedule.schedule_id,view=next_view,snapshot_hash='a'*64,current_round_id=view.round_id+1)
    assert applied.status is CascadeApplicationStatus.APPLIED
    assert applied.directive and applied.directive.directive_type is CascadeDirectiveType.VERIFICATION_REQUEST
    assert applied.directive.visible_evidence_ids == schedule.visible_evidence_ids
    assert ledger.schedules[0].status is CascadeScheduleStatus.SCHEDULED
    assert applied_ledger.schedules[0].status is CascadeScheduleStatus.APPLIED
    report=CascadeApplicationReplayValidator.validate(applied_ledger)
    assert report.status.value == 'passed' and report.applied_count == 1

def test_current_round_and_duplicate_application_rejected():
    view=current(); p=propose(CascadeCondition.PROVENANCE_AWARE_CONTROLLER,view,policy())
    ledger,result=CascadeApplicationLedger().schedule_proposal(p,view,snapshot_hash='b'*64,config=policy())
    schedule=result.schedule; assert schedule
    _,too_early=ledger.apply_schedule(schedule.schedule_id,view=view,snapshot_hash='b'*64,current_round_id=view.round_id)
    assert too_early.status is CascadeApplicationStatus.REJECTED
    next_view=view.model_copy(update={'round_id':view.round_id+1})
    applied_ledger,ok=ledger.apply_schedule(schedule.schedule_id,view=next_view,snapshot_hash='b'*64,current_round_id=view.round_id+1)
    assert ok.status is CascadeApplicationStatus.APPLIED
    _,again=applied_ledger.apply_schedule(schedule.schedule_id,view=next_view,snapshot_hash='b'*64,current_round_id=view.round_id+1)
    assert again.error_code == CascadeApplicationReasonCode.DUPLICATE_APPLICATION.value

def test_hash_and_visibility_tampering_rejected():
    view=current(); p=propose(CascadeCondition.PROVENANCE_AWARE_CONTROLLER,view,policy())
    ledger,result=CascadeApplicationLedger().schedule_proposal(p,view,snapshot_hash='c'*64,config=policy())
    schedule=result.schedule; assert schedule
    next_view=view.model_copy(update={'round_id':view.round_id+1})
    _,bad_hash=ledger.apply_schedule(schedule.schedule_id,view=next_view,snapshot_hash='d'*64,current_round_id=view.round_id+1)
    assert bad_hash.error_code == CascadeApplicationReasonCode.SNAPSHOT_HASH_MISMATCH.value
    hidden=next_view.model_copy(update={'provenance_nodes': tuple(next_view.provenance_nodes[:1]),'evidence_cards': ()})
    _,bad_visibility=ledger.apply_schedule(schedule.schedule_id,view=hidden,snapshot_hash='c'*64,current_round_id=view.round_id+1)
    assert bad_visibility.status is CascadeApplicationStatus.REJECTED

def test_invalid_proposal_is_rejected_not_scheduled():
    view=current(); p=propose(CascadeCondition.PROVENANCE_AWARE_CONTROLLER,view,policy()).model_copy(update={'round_id':99})
    ledger,result=CascadeApplicationLedger().schedule_proposal(p,view,snapshot_hash='e'*64,config=policy())
    assert result.status is CascadeApplicationStatus.REJECTED
    assert ledger.schedules and ledger.schedules[0].status is CascadeScheduleStatus.REJECTED and result.rejected_count == 1
    assert result.audit_events and result.audit_events[0].event_type == 'schedule_rejected'

def test_replay_rejects_unapplied_directive_and_expanded_ids():
    view=current(); p=propose(CascadeCondition.PROVENANCE_AWARE_CONTROLLER,view,policy())
    ledger,result=CascadeApplicationLedger().schedule_proposal(p,view,snapshot_hash='f'*64,config=policy())
    schedule=result.schedule; assert schedule
    directive=StructuredCascadeDirective(directive_id=schedule.schedule_id+'-directive',directive_type=CascadeDirectiveType.VERIFICATION_REQUEST,scenario_id=schedule.scenario_id,target_agent_id=schedule.target_agent_id,claim_id=schedule.claim_id,effective_round_id=schedule.effective_round_id,visible_content_ids=('not-visible',))
    snapshot=CascadeExposureSnapshot(snapshot_id='s',scenario_id=view.scenario_id,agent_id=view.agent_id,round_id=schedule.effective_round_id,captured_event_ids=(),visible_claim_ids=(),visible_content_ids=(),visible_evidence_ids=(),visible_provenance_node_ids=(),visible_root_relations=())
    context=ControlledRoundContext(snapshot=snapshot,directives=(directive,),transformation_sha256='1'*64)
    with pytest.raises(CascadeApplicationReplayError,match='directive_from_unapplied_schedule'):
        CascadeApplicationReplayValidator.validate(ledger,(context,))

def test_application_models_forbid_private_fields():
    with pytest.raises(ValidationError):
        StructuredCascadeDirective.model_validate({'directive_id':'d','directive_type':'verification_request','scenario_id':'s','target_agent_id':'a','claim_id':'c','effective_round_id':1,'ground_truth_label':'x'})

def test_four_scenario_smoke_is_offline_and_safe(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert smoke_main(['--policy',str(POLICY),'--scenario-dir',str(SCENARIOS)]) == 0
    output=json.loads(capsys.readouterr().out)
    assert output['status']=='offline_application_smoke' and len(output['scenarios'])==4
    indexed={item['scenario_id']:item for item in output['scenarios']}
    assert indexed['cascade-false-majority']['schedule_count']==1
    for name in ('cascade-true-minority-correction','cascade-independent-true-consensus','cascade-unresolved-disagreement'):
        assert indexed[name]['schedule_count']==0 and indexed[name]['action']=='abstain'
    assert 'ground_truth_label' not in json.dumps(output)
    assert list(tmp_path.iterdir()) == []

def test_smoke_helper_deterministic():
    assert run_smoke(POLICY,SCENARIOS)==run_smoke(POLICY,SCENARIOS)
