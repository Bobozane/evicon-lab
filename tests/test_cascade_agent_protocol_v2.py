from __future__ import annotations
import json
import pytest
from pydantic import ValidationError
from evicon.cascade_agent_protocol_v2 import (
    CascadeAgentProtocolV2Runtime, CascadeAgentV2Diagnostic, parse_cascade_agent_response_v2,
    render_cascade_agent_turn_v2, V2_TEMPLATE_VERSION, load_hd2_config,
)
from evicon.cascade_agent_prompts import CascadeAgentPromptContext, CascadeAgentRuntimeConfig
from evicon.cascade_agent_runtime import CascadeAgentRuntimeStatus
from evicon.cascade_agent_protocol_v2_preflight import preflight
from evicon.cascade_agent_protocol_v2_smoke import run_smoke
from evicon.cascade_agent_response import parse_cascade_agent_response
from evicon.cascade_agent_prompts import render_cascade_agent_turn
from evicon.cascade_intervention_application import CascadeDirectiveType, StructuredCascadeDirective
from evicon.cascade_outcomes import ClaimStance
from evicon.provenance_cascade import ProvenanceNode, SourceCategory, VerificationStatus
from evicon.provenance_cascade_exposure import ControllerClaimView, ControllerPublicView, ControllerSourceRootView, VisibleRootRelation
from evicon.models import EvidenceCard
from evicon.llm_contract import LLMResponse, LLMProviderError, ProviderErrorCode
from evicon.openai_provider import OpenAICompatibleProvider, ProviderConfig, ResponseFormatMode, TransportResponse

class Provider:
    def __init__(self, content, error=None): self.content=content; self.error=error; self.calls=0
    def complete(self, request):
        self.calls += 1
        if self.error: raise self.error
        return LLMResponse(request_id=request.request_id, model_name=request.model_name, content=self.content, finish_reason='stop', prompt_tokens=2, completion_tokens=3, total_tokens=5, latency_ms=1.0)

def context():
    view=ControllerPublicView(
        scenario_id='v2-test', agent_id='network-agent-01', round_id=1,
        claims=(ControllerClaimView(claim_id='claim-1', verification_status=VerificationStatus.UNVERIFIED, evidence_card_ids=('evidence-1',)),),
        evidence_cards=(EvidenceCard(evidence_id='evidence-1', claim='public evidence summary', source='public source', supports=['claim-1'], introduced_round=0, visible_to=['network-agent-01'], reliability=.8),),
        provenance_nodes=(ProvenanceNode(node_id='node-1', content_id='content-1', scenario_id='v2-test', claim_id='claim-1', source_root_id='root-1', round_id=0),),
        source_roots=(ControllerSourceRootView(source_root_id='root-1', public_source_category=SourceCategory.PRIMARY_RECORD),),
        root_relations=(VisibleRootRelation(provenance_node_id='node-1', source_root_ids=('root-1',)),),
    )
    return CascadeAgentPromptContext.from_public_view(view, claim_id='claim-1', directive=None, runtime_config=CascadeAgentRuntimeConfig(model_name='v2-fake', seed=7))

def payload(**overrides):
    data={'stance':'uncertain','content_ids_used':['content-1'],'evidence_ids_used':['evidence-1'],'share_content_id':'content-1'}; data.update(overrides); return json.dumps(data)

def test_v2_valid_and_strict_runtime_one_call():
    parsed=parse_cascade_agent_response_v2(payload(), context()); assert parsed.valid and parsed.stance is ClaimStance.UNCERTAIN
    provider=Provider(payload()); result=CascadeAgentProtocolV2Runtime().execute(context(), provider); assert result.status is CascadeAgentRuntimeStatus.COMPLETED; assert provider.calls == 1
    audit=result.audit.model_dump_json(); assert 'public evidence summary' not in audit and 'system_prompt' not in audit and 'user_prompt' not in audit

@pytest.mark.parametrize('body,code', [
    ('[]', CascadeAgentV2Diagnostic.TOP_LEVEL_TYPE),
    ('{}', CascadeAgentV2Diagnostic.MISSING_FIELD),
    (payload(action='abstain'), CascadeAgentV2Diagnostic.EXTRA_FIELD),
    (payload(content_ids_used='content-1'), CascadeAgentV2Diagnostic.FIELD_TYPE),
    (payload(stance='bogus'), CascadeAgentV2Diagnostic.INVALID_STANCE),
    (payload(content_ids_used=['not-visible']), CascadeAgentV2Diagnostic.UNAVAILABLE_CONTENT_ID),
    (payload(evidence_ids_used=['not-visible']), CascadeAgentV2Diagnostic.UNAVAILABLE_EVIDENCE_ID),
    (payload(share_content_id='not-visible'), CascadeAgentV2Diagnostic.SHARE_FIELD),
    ('not-json', CascadeAgentV2Diagnostic.MALFORMED_JSON),
    ('{"stance":"uncertain","content_ids_used":[],"content_ids_used":[],"evidence_ids_used":[],"share_content_id":null}', CascadeAgentV2Diagnostic.EXTRA_FIELD),
])
def test_v2_diagnostics_are_stable_and_content_free(body, code):
    parsed=parse_cascade_agent_response_v2(body, context()); assert not parsed.valid; assert parsed.validation_errors == (code,); assert parsed.model_dump_json().find('not-visible') < 0

def test_forbidden_private_or_controller_fields_are_rejected_without_values():
    parsed=parse_cascade_agent_response_v2(payload(ground_truth_label='private-value'), context()); assert not parsed.valid; assert parsed.validation_errors[0] in {CascadeAgentV2Diagnostic.EXTRA_FIELD}
    assert 'private-value' not in parsed.model_dump_json()

def test_v2_template_and_fingerprint_are_distinct():
    v2=render_cascade_agent_turn_v2(context()); v1=render_cascade_agent_turn(context()); assert v2.metadata['template_version']==V2_TEMPLATE_VERSION; assert v2.request_id != v1.request_id; assert 'response_format' not in v2.metadata

def test_provider_response_format_is_opt_in():
    from provider_cases import request, FakeTransport, response_body
    _, _, rendered=request()
    base=ProviderConfig(base_url='https://provider.example/v1', model_name='m', allow_network=True, max_tokens=16)
    p=OpenAICompatibleProvider(base, transport=FakeTransport([TransportResponse(200,response_body())]), environment={'EVICON_LLM_API_KEY':'x'})
    assert 'response_format' not in p._payload(rendered)
    schema=ProviderConfig(base_url='https://provider.example/v1', model_name='m', allow_network=True, max_tokens=16, response_format=ResponseFormatMode.JSON_SCHEMA, response_schema_name='cascade_agent_response_v2')
    p2=OpenAICompatibleProvider(schema, transport=FakeTransport([TransportResponse(200,response_body())]), environment={'EVICON_LLM_API_KEY':'x'})
    payload2=p2._payload(rendered); assert payload2['response_format']['type']=='json_schema'; assert payload2['response_format']['json_schema']['strict'] is True; assert payload2['response_format']['json_schema']['schema']['additionalProperties'] is False
    obj=ProviderConfig(base_url='https://provider.example/v1', model_name='m', allow_network=True, max_tokens=16, response_format=ResponseFormatMode.JSON_OBJECT)
    assert OpenAICompatibleProvider(obj, transport=FakeTransport([]), environment={'EVICON_LLM_API_KEY':'x'})._payload(rendered)['response_format']=={'type':'json_object'}
    with pytest.raises(ValidationError): ProviderConfig(response_schema_name='bad')

def test_v2_preflight_and_48_run_fake_smoke():
    path='configs/provenance_cascade/pilot/provenance_cascade_pilot_hd2.v1.toml'
    report=preflight(path); assert report['status']=='ready_offline'; assert report['network']=='disabled'; assert report['provider_constructed'] is False; assert report['transport_called'] is False
    smoke=run_smoke(path); assert smoke['status']=='passed'; assert smoke['run_count']==48; assert smoke['matched_group_count']==12; assert smoke['provider_call_count']==864; assert smoke['logical_request_count']==864; assert smoke['replay_passed_count']==48; assert smoke['completion_reservation_cap']==221184; assert smoke['results_written'] is False

def test_v2_config_scope_and_budget_are_new_but_unchanged():
    config,_=load_hd2_config('configs/provenance_cascade/pilot/provenance_cascade_pilot_hd2.v1.toml'); assert config.seeds==(20260911,20260912,20260913); assert config.output_root!='results/provenance-cascade-pilot-hd-v2'; assert config.request_cap==864; assert config.completion_reservation_cap==221184


def test_response_format_transport_retry_and_unsupported_status_are_bounded():
    from provider_cases import request, FakeTransport, response_body
    _, _, rendered = request()
    schema = ProviderConfig(
        base_url="https://provider.example/v1", model_name="m", allow_network=True,
        max_tokens=16, max_retries=1, response_format=ResponseFormatMode.JSON_SCHEMA,
        response_schema_name="cascade_agent_response_v2",
    )
    transport = FakeTransport([TransportResponse(500, "{}"), TransportResponse(200, response_body())])
    provider = OpenAICompatibleProvider(schema, transport=transport, environment={"EVICON_LLM_API_KEY": "x"})
    response = provider.complete(rendered)
    assert response.finish_reason == "stop" and len(transport.calls) == 2
    assert transport.calls[0]["payload"]["response_format"]["type"] == "json_schema"

    unsupported = FakeTransport([TransportResponse(400, "{}"), TransportResponse(200, response_body())])
    blocked = OpenAICompatibleProvider(schema, transport=unsupported, environment={"EVICON_LLM_API_KEY": "x"})
    with pytest.raises(LLMProviderError) as exc:
        blocked.complete(rendered)
    assert exc.value.code is ProviderErrorCode.HTTP_CLIENT_ERROR
    assert len(unsupported.calls) == 1
    assert "x" not in str(exc.value)
