"""Offline smoke for the isolated provenance-cascade Agent runtime."""
from __future__ import annotations
import json
from .cascade_agent_prompts import CascadeAgentPromptContext, CascadeAgentRuntimeConfig
from .cascade_agent_runtime import CascadeAgentRuntime
from .cascade_intervention_application import CascadeDirectiveType, StructuredCascadeDirective
from .llm_contract import LLMRequest, LLMResponse
from .provenance_cascade import ProvenanceNode, SourceCategory, VerificationStatus
from .provenance_cascade_exposure import ControllerClaimView, ControllerPublicView, ControllerSourceRootView, VisibleRootRelation
from .models import EvidenceCard

class CascadeFakeProvider:
    """Fixed local provider used only by this smoke command."""
    def __init__(self, *, content_id: str = "cascade-cal-content-1", evidence_id: str = "cascade-cal-evidence-1") -> None:
        self.content_id = content_id
        self.evidence_id = evidence_id

    def complete(self, request: LLMRequest) -> LLMResponse:
        content=json.dumps({
            "stance":"uncertain",
            "content_ids_used":[self.content_id],
            "evidence_ids_used":[self.evidence_id],
            "share_content_id":self.content_id,
        }, sort_keys=True, separators=(",", ":"))
        return LLMResponse(request_id=request.request_id, model_name=request.model_name, content=content,
                           finish_reason="stop", prompt_tokens=10, completion_tokens=10,
                           total_tokens=20, latency_ms=1.0,
                           provider_metadata={"provider":"cascade_fake.v1"})

def _context() -> CascadeAgentPromptContext:
    view=ControllerPublicView(
        scenario_id="cascade-agent-runtime-smoke", agent_id="network-agent-01", round_id=1,
        claims=(ControllerClaimView(claim_id="cascade-cal-claim-1", verification_status=VerificationStatus.UNVERIFIED,
                                    evidence_card_ids=("cascade-cal-evidence-1",)),),
        evidence_cards=(EvidenceCard(evidence_id="cascade-cal-evidence-1", claim="publicly available supporting evidence",
                                     source="public fixture source", supports=["cascade-cal-claim-1"], introduced_round=0,
                                     visible_to=["network-agent-01"], reliability=0.8),),
        provenance_nodes=(ProvenanceNode(node_id="cascade-cal-node-1", content_id="cascade-cal-content-1",
                                         scenario_id="cascade-agent-runtime-smoke", claim_id="cascade-cal-claim-1",
                                         source_root_id="cascade-cal-root-1", round_id=0),),
        source_roots=(ControllerSourceRootView(source_root_id="cascade-cal-root-1", public_source_category=SourceCategory.PRIMARY_RECORD),),
        root_relations=(VisibleRootRelation(provenance_node_id="cascade-cal-node-1", source_root_ids=("cascade-cal-root-1",)),),
    )
    directive=StructuredCascadeDirective(directive_id="cascade-directive-smoke", directive_type=CascadeDirectiveType.REASONING_REQUEST,
        scenario_id=view.scenario_id, target_agent_id=view.agent_id, claim_id="cascade-cal-claim-1", effective_round_id=1,
        visible_content_ids=("cascade-cal-content-1",), visible_evidence_ids=("cascade-cal-evidence-1",))
    return CascadeAgentPromptContext.from_public_view(view, claim_id="cascade-cal-claim-1", directive=directive,
        runtime_config=CascadeAgentRuntimeConfig(model_name="cascade-fake-v1", seed=24))

def main() -> None:
    result=CascadeAgentRuntime().execute(_context(), CascadeFakeProvider())
    audit=result.audit_summary().model_dump(mode="json")
    print(json.dumps({
        "template_version": audit["template_version"], "request_id": audit["request_id"],
        "provider_status": result.status.value, "parser_status": audit["parser_valid"],
        "stance": result.response.stance.value if result.response and result.response.valid else None,
        "content_ids_used_count": audit["content_ids_used_count"], "evidence_ids_used_count": audit["evidence_ids_used_count"],
        "share_requested": audit["share_requested"], "error_code": audit["error_code"], "network":"disabled",
        "results_written":False, "private_truth_exposed":False,
        "safety":{"development_only":True,"not_paper_result":True,"no_causal_conclusion":True},
    }, ensure_ascii=True, sort_keys=True))
if __name__ == "__main__":
    main()
__all__=["CascadeFakeProvider","main"]
