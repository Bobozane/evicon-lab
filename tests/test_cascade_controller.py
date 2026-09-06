from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evicon.cascade_controller import (
    CascadeAction,
    CascadeControllerPolicyConfig,
    CascadeControllerPolicyLoader,
    CascadeControllerValidationError,
    CascadeInterventionProposal,
    CascadeReasonCode,
    propose,
    validate_cascade_proposal,
)
from evicon.cascade_controller_smoke import fixture_view, main as smoke_main, run_smoke
from evicon.provenance_cascade_exposure import ControllerPublicView
from evicon.provenance_cascade_preregistration import CascadeCondition

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'configs/provenance_cascade/cascade_controller_policy.v1.toml'
SCENARIOS = ROOT / 'configs/provenance_cascade/scenarios'

@pytest.fixture
def policy() -> CascadeControllerPolicyConfig:
    return CascadeControllerPolicyLoader.load(POLICY)

def view(name: str) -> ControllerPublicView:
    return fixture_view(name, SCENARIOS)

def test_deterministic_and_no_mutation(policy: CascadeControllerPolicyConfig) -> None:
    current = view('false_majority')
    before = current.model_dump_json()
    a = [propose(c, current, policy) for c in policy.conditions]
    b = [propose(c, current, policy) for c in policy.conditions]
    assert a == b and current.model_dump_json() == before
    assert all(item.valid for item in a)

def test_false_majority_root_only_for_aware(policy: CascadeControllerPolicyConfig) -> None:
    current = view('false_majority')
    generic = propose(CascadeCondition.GENERIC_DISSENT, current, policy)
    blind = propose(CascadeCondition.SOURCE_BLIND_CONTROLLER, current, policy)
    aware = propose(CascadeCondition.PROVENANCE_AWARE_CONTROLLER, current, policy)
    assert generic.action is CascadeAction.REQUEST_EVIDENCE_BASED_REASONING
    assert blind.action is aware.action is CascadeAction.REQUEST_INDEPENDENT_SOURCE
    assert generic.used_source_root_ids == blind.used_source_root_ids == ()
    assert blind.reason_codes == (CascadeReasonCode.VISIBLE_UNVERIFIED_REPETITION,)
    assert aware.reason_codes == (CascadeReasonCode.VISIBLE_UNVERIFIED_SAME_ROOT_REPETITION,)
    assert aware.used_source_root_ids == ('root-fm',)

def test_source_blind_does_not_read_root_summary(policy: CascadeControllerPolicyConfig) -> None:
    current = view('false_majority').model_copy(update={'source_roots': (), 'root_relations': ()})
    blind = propose(CascadeCondition.SOURCE_BLIND_CONTROLLER, current, policy)
    assert blind.action is CascadeAction.REQUEST_INDEPENDENT_SOURCE
    with pytest.raises(CascadeControllerValidationError, match='controller_view_root_relations_invalid'):
        propose(CascadeCondition.PROVENANCE_AWARE_CONTROLLER, current, policy)

def test_supported_correction_is_not_suppressed(policy: CascadeControllerPolicyConfig) -> None:
    current = view('true_minority_correction')
    for condition in policy.conditions:
        item = propose(condition, current, policy)
        assert item.action is CascadeAction.ABSTAIN
        if condition is not CascadeCondition.NO_INTERVENTION:
            assert CascadeReasonCode.VISIBLE_SUPPORTED_CORRECTION in item.reason_codes
            assert CascadeReasonCode.POLICY_ABSTAIN_PROTECTION in item.reason_codes

def test_consensus_and_unresolved_are_abstentions(policy: CascadeControllerPolicyConfig) -> None:
    consensus = view('independent_true_consensus')
    for condition in policy.conditions:
        assert propose(condition, consensus, policy).action is CascadeAction.ABSTAIN
    aware = propose(CascadeCondition.PROVENANCE_AWARE_CONTROLLER, consensus, policy)
    assert len(aware.used_source_root_ids) == 2
    unresolved = view('unresolved_disagreement')
    for condition in policy.conditions:
        item = propose(condition, unresolved, policy)
        assert item.action is CascadeAction.ABSTAIN
        assert CascadeReasonCode.VISIBLE_REFUTED_CLAIM not in item.reason_codes
    enabled = policy.model_copy(update={'allow_unresolved_independent_source_request': True})
    assert propose(CascadeCondition.PROVENANCE_AWARE_CONTROLLER, unresolved, enabled).action is CascadeAction.REQUEST_INDEPENDENT_SOURCE

def test_no_intervention_is_empty(policy: CascadeControllerPolicyConfig) -> None:
    item = propose(CascadeCondition.NO_INTERVENTION, view('false_majority'), policy)
    assert item.action is CascadeAction.ABSTAIN
    assert not item.reason_codes and not item.used_content_ids and not item.used_evidence_ids and not item.used_source_root_ids

@pytest.mark.parametrize('field', ['ground_truth_label', 'source_independence_label', 'evaluator_truth_record'])
def test_private_fields_rejected(field: str) -> None:
    payload = view('false_majority').model_dump(mode='python')
    payload[field] = 'private'
    with pytest.raises(ValidationError):
        ControllerPublicView.model_validate(payload)

@pytest.mark.parametrize(('field','value'), [('action','hide'), ('reason_codes',['majority']), ('ledger_update',True), ('snapshot_update',True), ('graph_update',True)])
def test_forbidden_proposal_fields_rejected(field: str, value: object) -> None:
    item = propose(CascadeCondition.PROVENANCE_AWARE_CONTROLLER, view('false_majority'), CascadeControllerPolicyLoader.load(POLICY))
    payload = item.model_dump(mode='json')
    payload[field] = value
    with pytest.raises(ValidationError):
        CascadeInterventionProposal.model_validate(payload)

def test_source_blind_provenance_fields_rejected(policy: CascadeControllerPolicyConfig) -> None:
    current = view('false_majority')
    item = propose(CascadeCondition.SOURCE_BLIND_CONTROLLER, current, policy).model_copy(update={
        'used_source_root_ids': ('root-fm',),
        'reason_codes': (CascadeReasonCode.VISIBLE_UNVERIFIED_SAME_ROOT_REPETITION,),
    })
    with pytest.raises(CascadeControllerValidationError, match='source_blind_provenance_access'):
        validate_cascade_proposal(item, current, policy)

@pytest.mark.parametrize(('updates','code'), [
    ({'used_content_ids': ('unexposed-content',)}, 'proposal_content_not_visible'),
    ({'used_evidence_ids': ('unexposed-evidence',)}, 'proposal_evidence_not_visible'),
    ({'used_source_root_ids': ('unexposed-root',)}, 'proposal_root_not_visible'),
    ({'scenario_id': 'other'}, 'proposal_scenario_mismatch'),
    ({'round_id': 99}, 'proposal_round_mismatch'),
    ({'target_agent_id': 'network-agent-06'}, 'proposal_target_not_current_agent'),
])
def test_coordinates_must_be_current_public_view(policy: CascadeControllerPolicyConfig, updates: dict[str, object], code: str) -> None:
    current = view('false_majority')
    item = propose(CascadeCondition.PROVENANCE_AWARE_CONTROLLER, current, policy).model_copy(update=updates)
    with pytest.raises(CascadeControllerValidationError, match=code):
        validate_cascade_proposal(item, current, policy)

def test_future_node_rejected(policy: CascadeControllerPolicyConfig) -> None:
    current = view('false_majority')
    nodes = list(current.provenance_nodes)
    nodes[0] = nodes[0].model_copy(update={'round_id': current.round_id})
    future = current.model_copy(update={'provenance_nodes': tuple(nodes)})
    with pytest.raises(CascadeControllerValidationError, match='controller_view_future_or_cross_scenario_data'):
        propose(CascadeCondition.SOURCE_BLIND_CONTROLLER, future, policy)

def test_policy_shape_and_unknown_fields_rejected(policy: CascadeControllerPolicyConfig) -> None:
    payload = policy.model_dump(mode='python')
    payload['hidden_truth'] = True
    with pytest.raises(ValidationError):
        CascadeControllerPolicyConfig.model_validate(payload)
    payload = policy.model_dump(mode='python')
    payload['conditions'] = payload['conditions'][:-1]
    with pytest.raises(ValidationError):
        CascadeControllerPolicyConfig.model_validate(payload)

def test_smoke_safe_and_empty_side_effects(capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert smoke_main(['--policy', str(POLICY), '--scenario-dir', str(SCENARIOS)]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload['status'] == 'offline_proposal_smoke' and len(payload['proposals']) == 16
    assert all(x['status'] == 'proposal_only' and x['valid'] for x in payload['proposals'])
    for forbidden in ('ground_truth_label','source_independence_label','api_key','system_prompt','user_prompt','results/'):
        assert forbidden not in output.lower()
    assert list(tmp_path.iterdir()) == []

def test_smoke_expected_root_access() -> None:
    records = run_smoke(POLICY, SCENARIOS)
    indexed = {(x['scenario_id'], x['condition_id']): x for x in records}
    assert indexed[('cascade-false-majority','provenance_aware_controller')]['used_root_count'] == 1
    assert indexed[('cascade-false-majority','source_blind_controller')]['used_root_count'] == 0
    assert indexed[('cascade-independent-true-consensus','provenance_aware_controller')]['action'] == 'abstain'
