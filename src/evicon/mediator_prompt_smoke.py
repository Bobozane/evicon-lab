"""Local-only smoke coverage for rendered mediator requests and strict parsing."""

from __future__ import annotations

import json

from .executor import ExecutionContext, InterventionExecutor
from .fake_provider_adapter import FakeLLMProviderAdapter, FakeResponseMode
from .mediator_prompts import PromptContext, render_action_instruction
from .mediator_response import parse_mediator_response
from .models import InterventionAction
from .policy import InterventionDecision, PolicyReason


def _plan_for(action: InterventionAction):
    targets = [] if action is InterventionAction.NO_OP else ["agent-a"]
    context = ExecutionContext(
        run_id="mediator-smoke-run",
        scenario_id="mediator-smoke-scenario",
        round_id=1,
        known_agent_ids=["agent-a", "agent-b"],
        visible_evidence_ids=["evidence-0"],
        visible_evidence_introduced_rounds={"evidence-0": 0},
        candidate_peer_turn_ids=["turn-r0-agent-b"],
        candidate_peer_turn_rounds={"turn-r0-agent-b": 0},
        current_budget=1.0,
        max_target_count=2,
        intervention_history=[],
        last_action=None,
        metadata={},
    )
    decision = InterventionDecision(
        action=action,
        target_agent_ids=targets,
        reason_codes=[PolicyReason.HIGH_RISK],
        risk_score=0.8,
        estimated_cost=0.0 if action is InterventionAction.NO_OP else 0.2,
        round_id=1,
        policy_version="mediator-smoke-policy.v1",
        cooldown_until=2,
        valid=True,
    )
    plan = InterventionExecutor.plan(decision, context)
    if not plan.valid:
        raise RuntimeError("smoke fixture produced an invalid action plan")
    return plan


def _context_for(plan: object) -> PromptContext:
    return PromptContext(
        run_id=plan.run_id,
        scenario_id=plan.scenario_id,
        round_id=plan.round_id,
        action=plan.action,
        target_agent_ids=list(plan.target_agent_ids),
        public_dialogue_summary="A compact public summary for local contract testing.",
        visible_evidence_ids=["evidence-0"],
        requested_evidence_ids=["evidence-0"],
        preservation_constraints=[
            "Use public evidence only.",
            "Preserve any genuine remaining position.",
            "Separate factual claims from value preferences.",
        ],
        metadata={},
    )


def _result(action: InterventionAction, request: object, parsed: object) -> dict[str, object]:
    return {
        "action": action.value,
        "template_version": request.metadata.get("template_version") if request else None,
        "request_id": request.request_id if request else None,
        "parser_valid": parsed.valid if parsed is not None else None,
        "validation_errors": [error.value for error in parsed.validation_errors] if parsed else [],
        "evidence_ids_requested": parsed.evidence_ids_requested if parsed else [],
    }


def main() -> int:
    results: list[dict[str, object]] = []
    actions = [
        InterventionAction.REQUEST_EVIDENCE,
        InterventionAction.BLIND_EVIDENCE_REFLECTION,
        InterventionAction.SOLICIT_DISSENT,
        InterventionAction.ADAPTIVE_EXPOSURE,
        InterventionAction.MINORITY_REPORT,
        InterventionAction.RESTRUCTURE,
    ]
    for action in actions:
        plan = _plan_for(action)
        context = _context_for(plan)
        request = render_action_instruction(plan, context, seed=29)
        if request is None:
            raise RuntimeError("non-no_op action did not render a request")
        parsed = parse_mediator_response(FakeLLMProviderAdapter().complete(request).content, plan, context)
        results.append(_result(action, request, parsed))

    no_op = _plan_for(InterventionAction.NO_OP)
    results.append(_result(InterventionAction.NO_OP, render_action_instruction(no_op, _context_for(no_op)), None))

    request_evidence_plan = _plan_for(InterventionAction.REQUEST_EVIDENCE)
    request_evidence_context = _context_for(request_evidence_plan)
    request = render_action_instruction(request_evidence_plan, request_evidence_context, seed=29)
    if request is None:
        raise RuntimeError("request_evidence smoke fixture did not render a request")
    invalid_evidence = FakeLLMProviderAdapter(
        {request.request_id: FakeResponseMode.UNKNOWN_EVIDENCE}
    ).complete(request)
    results.append(_result(request_evidence_plan.action, request, parse_mediator_response(
        invalid_evidence.content, request_evidence_plan, request_evidence_context
    )))
    invalid_json = FakeLLMProviderAdapter(
        {request.request_id: FakeResponseMode.INVALID_JSON}
    ).complete(request)
    results.append(_result(request_evidence_plan.action, request, parse_mediator_response(
        invalid_json.content, request_evidence_plan, request_evidence_context
    )))
    hidden_leak = json.dumps(
        {
            "message": "hidden probe response",
            "evidence_ids_requested": [],
            "dissent_preserved": False,
            "factual_claims": [],
        },
        ensure_ascii=True,
    )
    results.append(_result(request_evidence_plan.action, request, parse_mediator_response(
        hidden_leak, request_evidence_plan, request_evidence_context
    )))
    print(json.dumps(results, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
