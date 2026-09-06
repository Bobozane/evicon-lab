"""Offline qualification smoke for Conformity Identification Study v1."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Sequence

from .conformity_identification import (
    DEFAULT_CONFIG,
    IdentificationCondition,
    IdentificationObservation,
    IdentificationStage,
    OrdinalJudgment,
    PublicAction,
    ShareDecision,
    load_identification_config,
    safe_json,
    validate_matched_inputs,
    visible_roles,
)
from .conformity_identification_protocol import (
    IdentificationPromptContext,
    parse_identification_response,
    render_identification_turn,
)
from .llm_contract import LLMResponse


class QualificationFakeProvider:
    """Condition-neutral fake used to validate observability, never effectiveness."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request) -> LLMResponse:
        self.calls += 1
        payload = json.loads(request.user_prompt)
        agent_number = int(str(payload["agent_id"]).rsplit("-", 1)[-1])
        judgment = {
            1: "lean_adopt", 2: "lean_adopt",
            3: "uncertain", 4: "uncertain",
            5: "lean_reject", 6: "lean_reject",
        }[agent_number]
        action = "adopt" if judgment == "lean_adopt" else "reject" if judgment == "lean_reject" else "withhold"
        share = "share_with_caveat" if payload["stage"] == "final_judgment" and action != "withhold" else "do_not_share"
        content_ids = [item["content_id"] for item in payload["visible_content"]]
        evidence_ids = list(payload["visible_evidence_ids"])
        body = json.dumps({
            "ordinal_judgment": judgment,
            "public_action": action,
            "share_decision": share,
            "confidence": 4,
            "content_ids_used": content_ids,
            "evidence_ids_used": evidence_ids,
        }, sort_keys=True)
        return LLMResponse(
            request_id=request.request_id,
            content=body,
            model_name=request.model_name,
            finish_reason="stop",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            latency_ms=0.0,
        )


def _context(scenario, condition, seed, agent_id, stage):
    roles = visible_roles(condition, stage)
    by_role = {item.role: item for item in scenario.stimuli}
    stimuli = [by_role[role] for role in roles]
    if condition is IdentificationCondition.SOURCE_FREE_REPETITION:
        roots: tuple[str, ...] = ()
    elif condition is IdentificationCondition.INDEPENDENT_ROOTS:
        roots = tuple(
            item.independent_source_root_id for item in stimuli
            if item.role in {"repeat_a", "repeat_b"} and item.independent_source_root_id is not None
        )
    elif condition is IdentificationCondition.SAME_ROOT_SOCIAL:
        roots = tuple(dict.fromkeys(
            item.source_root_id for item in stimuli
            if item.role in {"repeat_a", "repeat_b"} and item.source_root_id is not None
        ))
    else:
        roots = ()
    evidence_ids = tuple(item.evidence_id for item in stimuli if item.evidence_id is not None)
    return IdentificationPromptContext(
        scenario_id=scenario.scenario_id,
        agent_id=agent_id,
        seed=seed,
        stage=stage,
        target_claim_id=scenario.target_claim_id,
        decision_task=scenario.decision_task,
        visible_stimulus_ids=tuple(item.stimulus_id for item in stimuli),
        visible_public_summaries=tuple(item.public_summary for item in stimuli),
        visible_source_root_ids=roots,
        visible_evidence_ids=evidence_ids,
        reflection_only=condition is IdentificationCondition.SELF_REFLECTION,
    )


def run_fake_smoke(path: str = DEFAULT_CONFIG) -> dict[str, object]:
    config, scenarios = load_identification_config(path)
    validate_matched_inputs(config, scenarios)
    provider = QualificationFakeProvider()
    observations: list[IdentificationObservation] = []
    root_opportunities: Counter[str] = Counter()
    for scenario in scenarios:
        for seed in config.seeds:
            for condition in config.conditions:
                for agent_id in config.agent_ids:
                    for stage in config.stages:
                        context = _context(scenario, condition, seed, agent_id, stage)
                        response = parse_identification_response(
                            provider.complete(render_identification_turn(context)).content,
                            context,
                        )
                        observations.append(IdentificationObservation(
                            scenario_id=scenario.scenario_id,
                            seed=seed,
                            condition=condition,
                            agent_id=agent_id,
                            stage=stage,
                            judgment=response.ordinal_judgment,
                            action=response.public_action,
                            share=response.share_decision,
                            confidence=response.confidence,
                            visible_stimulus_ids=response.content_ids_used,
                        ))
                        if stage is IdentificationStage.SOCIAL_EXPOSURE:
                            root_opportunities[condition.value] += len(set(context.visible_source_root_ids))
    initial = [item for item in observations if item.stage is IdentificationStage.INITIAL_PRIVATE]
    substantive = [item for item in initial if item.judgment is not OrdinalJudgment.UNCERTAIN]
    initial_adopters = {
        (item.scenario_id, item.seed, item.agent_id)
        for item in initial if item.action is PublicAction.ADOPT
    }
    correction_eligible = sum(
        1 for item in observations
        if item.condition is IdentificationCondition.VERIFIED_EVIDENCE
        and item.stage is IdentificationStage.CORRECTION_EVIDENCE
        and (item.scenario_id, item.seed, item.agent_id) in initial_adopters
        and len(item.visible_stimulus_ids) == 3
    )
    matched_coordinates = {
        (item.scenario_id, item.seed, item.agent_id, item.stage)
        for item in observations
    }
    complete_coordinates = sum(
        1 for coordinate in matched_coordinates
        if {item.condition for item in observations if (item.scenario_id, item.seed, item.agent_id, item.stage) == coordinate}
        == set(IdentificationCondition)
    )
    return {
        "status": "qualification_smoke_passed",
        "study_id": config.study_id,
        "run_count": config.run_count,
        "matched_group_count": config.matched_group_count,
        "logical_request_count": len(observations),
        "provider_call_count": provider.calls,
        "observation_count": len(observations),
        "complete_six_condition_coordinate_count": complete_coordinates,
        "initial_observation_count": len(initial),
        "initial_substantive_count": len(substantive),
        "correction_transition_eligible_count": correction_eligible,
        "source_projection_root_totals": dict(sorted(root_opportunities.items())),
        "social_conformity_contrast_estimable": True,
        "evidence_receptivity_contrast_estimable": correction_eligible > 0,
        "effect_estimated": False,
        "network": "disabled",
        "provider": "qualification_fake",
        "results_written": False,
        "private_truth_exposed": False,
        "development_only": True,
        "calibration_only": True,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline conformity identification qualification smoke")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = run_fake_smoke(args.config)
    print(safe_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
