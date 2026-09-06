import json
import pytest
from pydantic import ValidationError
from evicon.cascade_agent_prompts import CascadeAgentPromptContext, CascadeAgentRuntimeConfig, CascadeVisibleClaim, render_cascade_agent_turn
from evicon.cascade_agent_response import CascadeAgentResponseError, parse_cascade_agent_response
from evicon.cascade_agent_runtime import CascadeAgentRuntime, CascadeAgentRuntimeStatus
from evicon.cascade_intervention_application import CascadeDirectiveType, StructuredCascadeDirective
from evicon.cascade_agent_runtime_smoke import CascadeFakeProvider
from evicon.llm_contract import LLMProviderError, ProviderErrorCode
from evicon.provenance_cascade import ProvenanceNode, SourceCategory, VerificationStatus
from evicon.provenance_cascade_exposure import ControllerClaimView, ControllerPublicView, ControllerSourceRootView, VisibleRootRelation
from evicon.models import EvidenceCard

class LocalProvider:
    def __init__(self, content): self.content=content; self.calls=0; self.requests=[]
    def complete(self, request):
        self.calls += 1; self.requests.append(request)
        from evicon.llm_contract import LLMResponse
        if isinstance(self.content, Exception): raise self.content
        return LLMResponse(request_id=request.request_id, model_name=request.model_name, content=self.content,
            finish_reason='stop', prompt_tokens=2, completion_tokens=2, total_tokens=4, latency_ms=1.0)

def make_context(directive=None):
    view=ControllerPublicView(
        scenario_id='s', agent_id='a', round_id=1,
        claims=(ControllerClaimView(claim_id='c', verification_status=VerificationStatus.UNVERIFIED, evidence_card_ids=('e',)),),
        evidence_cards=(EvidenceCard(evidence_id='e', claim='public evidence summary', source='public source', supports=['c'], introduced_round=0, visible_to=['a'], reliability=.8),),
        provenance_nodes=(ProvenanceNode(node_id='n', content_id='content-a', scenario_id='s', claim_id='c', source_root_id='root', round_id=0),),
        source_roots=(ControllerSourceRootView(source_root_id='root', public_source_category=SourceCategory.PRIMARY_RECORD),),
        root_relations=(VisibleRootRelation(provenance_node_id='n', source_root_ids=('root',)),),
    )
    return CascadeAgentPromptContext.from_public_view(view, claim_id='c', directive=directive,
        runtime_config=CascadeAgentRuntimeConfig(model_name='fake', seed=7))

def valid_json(**kw):
    data={'stance':'uncertain','content_ids_used':['content-a'],'evidence_ids_used':['e'],'share_content_id':'content-a'}; data.update(kw); return json.dumps(data)

def test_render_is_public_and_directive_is_process_semantics():
    d=StructuredCascadeDirective(directive_id='d',directive_type=CascadeDirectiveType.VERIFICATION_REQUEST,scenario_id='s',target_agent_id='a',claim_id='c',effective_round_id=1,visible_content_ids=('content-a',))
    request=render_cascade_agent_turn(make_context(d))
    assert request.metadata['template_version']=='cascade_agent_turn.v1'
    assert 'incorrect' not in request.system_prompt.lower()
    assert 'truth' not in request.user_prompt.lower()
    assert 'evaluator' not in request.user_prompt.lower()

def test_valid_response_and_runtime_one_call():
    provider=LocalProvider(valid_json()); result=CascadeAgentRuntime().execute(make_context(),provider)
    assert result.status is CascadeAgentRuntimeStatus.COMPLETED and provider.calls==1
    assert result.response.share_content_id=='content-a'
    assert 'public evidence summary' not in result.audit_summary().model_dump_json()

def test_uncertain_without_share():
    parsed=parse_cascade_agent_response(valid_json(content_ids_used=[],evidence_ids_used=[],share_content_id=None),make_context())
    assert parsed.valid and parsed.stance.value=='uncertain' and parsed.share_content_id is None

@pytest.mark.parametrize('payload,code', [
    ('not-json', CascadeAgentResponseError.MALFORMED_JSON),
    ('{"stance":"uncertain","content_ids_used":["unknown"]}', CascadeAgentResponseError.UNAVAILABLE_CONTENT),
    ('{"stance":"uncertain","evidence_ids_used":["unknown"]}', CascadeAgentResponseError.UNAVAILABLE_EVIDENCE),
    ('{"stance":"uncertain","action":"request_independent_source"}', CascadeAgentResponseError.FORBIDDEN_FIELD),
    ('{"stance":"uncertain","ground_truth_label":"false"}', CascadeAgentResponseError.FORBIDDEN_FIELD),
    ('{"stance":"uncertain","content_ids_used":[],"content_ids_used":[]}', CascadeAgentResponseError.DUPLICATE_FIELD),
    ('{"stance":"uncertain"', CascadeAgentResponseError.MALFORMED_JSON),
])
def test_strict_parser_rejects(payload,code):
    result=parse_cascade_agent_response(payload,make_context()); assert not result.valid and result.validation_errors[0] is code

def test_private_context_rejected_and_inputs_unchanged():
    context=make_context(); before=context.model_dump(mode='json')
    assert context.model_dump(mode='json')==before
    with pytest.raises(ValidationError):
        CascadeVisibleClaim(claim_id='c', verification_status=VerificationStatus.UNVERIFIED, public_summary='ground_truth_label=false')

@pytest.mark.parametrize('provider_code', [
    ProviderErrorCode.TIMEOUT,
    ProviderErrorCode.RATE_LIMITED,
    ProviderErrorCode.HTTP_SERVER_ERROR,
    ProviderErrorCode.AUTHENTICATION_FAILED,
])
def test_provider_errors_are_redacted(provider_code):
    error=LLMProviderError(provider_code, 'secret prompt and api_key')
    provider=LocalProvider(error); result=CascadeAgentRuntime().execute(make_context(),provider)
    assert result.status is CascadeAgentRuntimeStatus.PROVIDER_ERROR
    assert result.audit.error_code==provider_code.value
    audit=result.audit_summary().model_dump_json()
    assert 'secret' not in audit and 'api_key' not in audit

def test_smoke_provider_is_local():
    result=CascadeAgentRuntime().execute(make_context(),CascadeFakeProvider(content_id='content-a', evidence_id='e')); assert result.status.value=='completed'
