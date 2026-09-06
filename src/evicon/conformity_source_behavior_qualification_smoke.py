"""Network-free smoke for the behavioral source-manipulation protocol gate."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Sequence

from .conformity_identification import load_identification_config, safe_json
from .conformity_identification_v11 import load_v11_config
from .conformity_source_behavior_qualification import (
    DEFAULT_CONFIG, AdoptionDecision, BehaviorQualificationContext, SharingDecision,
    SourceProjection, load_config, parse_behavior_response, render_behavior_request, safe_case_audit,
)
from .conformity_source_manipulation import RootAssignment
from .llm_contract import LLMResponse


class BehaviorQualificationFakeProvider:
    """Constant behavior avoids manufacturing a source effect in FakeProvider tests."""
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request: object) -> LLMResponse:
        self.calls += 1
        request_id = getattr(request, "request_id")
        model_name = getattr(request, "model_name")
        body = json.dumps({
            "adoption_decision": AdoptionDecision.WITHHOLD.value,
            "sharing_decision": SharingDecision.DO_NOT_SHARE.value,
            "content_ids_used": [], "share_content_id": None,
        }, sort_keys=True)
        return LLMResponse(request_id=request_id, content=body, model_name=model_name,
                           finish_reason="stop", prompt_tokens=None, completion_tokens=None,
                           total_tokens=None, latency_ms=0.0)


def build_cases(path: str = DEFAULT_CONFIG) -> tuple[BehaviorQualificationContext, ...]:
    config = load_config(path)
    # v1.1 is a shared-T0 branch config, so its public scenario corpus remains
    # in its immutable v1 parent rather than in the branch config itself.
    branch, _ = load_v11_config(config.study_config_path)
    _, study = load_identification_config(branch.parent_config_path)
    cases: list[BehaviorQualificationContext] = []
    for scenario in study:
        selected = tuple(item for item in scenario.stimuli if item.role in {"repeat_a", "repeat_b"})
        if len(selected) != 2:
            raise ValueError("source_behavior_repeat_pair_missing")
        for projection in SourceProjection:
            assignments: tuple[RootAssignment, RootAssignment]
            if projection is SourceProjection.SOURCE_FREE:
                assignments = tuple(RootAssignment(content_id=item.stimulus_id, source_root_id=None) for item in selected)  # type: ignore[assignment]
            elif projection is SourceProjection.SAME_ROOT:
                assignments = tuple(RootAssignment(content_id=item.stimulus_id, source_root_id=selected[0].source_root_id) for item in selected)  # type: ignore[assignment]
            else:
                assignments = tuple(RootAssignment(content_id=item.stimulus_id, source_root_id=item.independent_source_root_id) for item in selected)  # type: ignore[assignment]
            cases.append(BehaviorQualificationContext(
                case_id=(
                    "source-behavior-v2-"
                    f"{scenario.scenario_id.removeprefix('cascade-identification-')}"
                    f"-{projection.value.replace('_', '-')}"
                ),
                scenario_id=scenario.scenario_id, projection=projection,
                visible_content_ids=(selected[0].stimulus_id, selected[1].stimulus_id),
                public_summaries=(selected[0].public_summary, selected[1].public_summary),
                public_root_assignments=assignments,
            ))
    return tuple(cases)


def run_fake_smoke(path: str = DEFAULT_CONFIG) -> dict[str, object]:
    config = load_config(path)
    provider = BehaviorQualificationFakeProvider()
    audits = []
    for case in build_cases(path):
        response = parse_behavior_response(provider.complete(render_behavior_request(case)).content, case)
        audits.append(safe_case_audit(case, response))
    projections = Counter(item.projection.value for item in audits)
    decisions = Counter(item.adoption_decision.value for item in audits)
    return {
        "status": "source_behavior_fake_smoke_passed",
        "gate_id": config.gate_id, "case_count": len(audits), "logical_request_count": provider.calls,
        "unique_case_count": len({item.case_id for item in audits}), "projection_case_counts": dict(sorted(projections.items())),
        "adoption_decision_counts": dict(sorted(decisions.items())), "safe_case_audit_count": len(audits),
        "behavior_effect_estimated": False, "network": "disabled", "provider_constructed": False,
        "api_key_read": False, "results_written": False, "private_truth_exposed": False,
        "not_paper_result": True, "no_causal_conclusion": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline source-behavior qualification smoke")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    print(safe_json(run_fake_smoke(args.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
