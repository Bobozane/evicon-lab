"""Offline shared-T0 branch smoke for Conformity Identification Study v1.1."""
from __future__ import annotations

import argparse
import hashlib
from typing import Sequence

from .conformity_identification import (
    IdentificationCondition,
    IdentificationObservation,
    IdentificationStage,
    OrdinalJudgment,
    PublicAction,
    load_identification_config,
    safe_json,
)
from .conformity_identification_protocol import parse_identification_response, render_identification_turn
from .conformity_identification_protocol_v11 import BranchPromptContext, parse_branch_response, render_branch_turn
from .conformity_identification_smoke import QualificationFakeProvider, _context
from .conformity_identification_v11 import DEFAULT_V11_CONFIG, load_v11_config, validate_branch_plan


def run_fake_smoke(path: str = DEFAULT_V11_CONFIG) -> dict[str, object]:
    config, _ = load_v11_config(path)
    validate_branch_plan(config)
    _, scenarios = load_identification_config(config.parent_config_path)
    scenario_map = {item.scenario_id: item for item in scenarios}
    provider = QualificationFakeProvider()
    initial: dict[tuple[str, int, str], tuple[object, str]] = {}
    request_ids: set[str] = set()
    observations: list[IdentificationObservation] = []

    for checkpoint in config.shared_t0_specs:
        scenario = scenario_map[checkpoint.scenario_id]
        for agent_id in config.agent_ids:
            context = _context(
                scenario, IdentificationCondition.PRIVATE_BASELINE,
                checkpoint.seed, agent_id, IdentificationStage.INITIAL_PRIVATE,
            )
            request = render_identification_turn(context)
            if request.request_id in request_ids:
                raise RuntimeError("duplicate_shared_t0_fingerprint")
            request_ids.add(request.request_id)
            response = parse_identification_response(provider.complete(request).content, context)
            response_hash = hashlib.sha256(response.model_dump_json().encode()).hexdigest()
            initial[(scenario.scenario_id, checkpoint.seed, agent_id)] = (response, response_hash)

    for branch in config.branch_runs:
        scenario = scenario_map[branch.scenario_id]
        for agent_id in config.agent_ids:
            initial_response, t0_hash = initial[(scenario.scenario_id, branch.seed, agent_id)]
            observations.append(IdentificationObservation(
                scenario_id=scenario.scenario_id,
                seed=branch.seed,
                condition=branch.condition,
                agent_id=agent_id,
                stage=IdentificationStage.INITIAL_PRIVATE,
                judgment=initial_response.ordinal_judgment,
                action=initial_response.public_action,
                share=initial_response.share_decision,
                confidence=initial_response.confidence,
                visible_stimulus_ids=initial_response.content_ids_used,
            ))
            for stage in config.continuation_stages:
                public = _context(scenario, branch.condition, branch.seed, agent_id, stage)
                branch_context = BranchPromptContext(
                    public_context=public,
                    matched_group_id=branch.matched_group_id,
                    branch_id=branch.run_id,
                    shared_t0_observation_sha256=t0_hash,
                )
                request = render_branch_turn(branch_context)
                if request.request_id in request_ids:
                    raise RuntimeError("duplicate_branch_fingerprint")
                request_ids.add(request.request_id)
                response = parse_branch_response(provider.complete(request).content, branch_context)
                observations.append(IdentificationObservation(
                    scenario_id=scenario.scenario_id,
                    seed=branch.seed,
                    condition=branch.condition,
                    agent_id=agent_id,
                    stage=stage,
                    judgment=response.ordinal_judgment,
                    action=response.public_action,
                    share=response.share_decision,
                    confidence=response.confidence,
                    visible_stimulus_ids=response.content_ids_used,
                ))

    shared_initial_count = len(initial)
    projected_initial_count = sum(item.stage is IdentificationStage.INITIAL_PRIVATE for item in observations)
    substantive_count = sum(
        response.ordinal_judgment is not OrdinalJudgment.UNCERTAIN
        for response, _ in initial.values()
    )
    initial_adopters = {
        key for key, (response, _) in initial.items()
        if response.public_action is PublicAction.ADOPT
    }
    correction_eligible = sum(
        1 for item in observations
        if item.condition is IdentificationCondition.VERIFIED_EVIDENCE
        and item.stage is IdentificationStage.CORRECTION_EVIDENCE
        and (item.scenario_id, item.seed, item.agent_id) in initial_adopters
        and len(item.visible_stimulus_ids) == 3
    )
    return {
        "status": "shared_t0_fake_smoke_passed",
        "study_id": config.study_id,
        "matched_group_count": config.matched_group_count,
        "branch_run_count": config.branch_run_count,
        "shared_t0_request_count": shared_initial_count,
        "projected_initial_observation_count": projected_initial_count,
        "continuation_request_count": len(observations) - projected_initial_count,
        "logical_request_count": provider.calls,
        "unique_request_fingerprint_count": len(request_ids),
        "initial_substantive_count": substantive_count,
        "correction_transition_eligible_count": correction_eligible,
        "shared_t0_reused_without_provider_replay": True,
        "condition_labels_rendered_to_agent": False,
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
    parser = argparse.ArgumentParser(description="Run the shared-T0 identification FakeProvider smoke")
    parser.add_argument("--config", default=DEFAULT_V11_CONFIG)
    args = parser.parse_args(argv)
    print(safe_json(run_fake_smoke(args.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
