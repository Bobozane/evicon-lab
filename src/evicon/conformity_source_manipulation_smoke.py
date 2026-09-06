"""FakeProvider smoke for the source-root manipulation gate."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Sequence

from .conformity_identification import load_identification_config, safe_json
from .conformity_identification_v11 import load_v11_config
from .conformity_source_manipulation import (
    DEFAULT_MANIPULATION_CONFIG,
    ManipulationContext,
    RootAssignment,
    SourceProjection,
    load_manipulation_config,
    parse_manipulation_response,
    render_manipulation_request,
)
from .llm_contract import LLMResponse


class ManipulationFakeProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request) -> LLMResponse:
        self.calls += 1
        payload = json.loads(request.user_prompt)
        assignments = payload["public_root_relations"]
        roots = {item["source_root_id"] for item in assignments if item["source_root_id"] is not None}
        content = json.dumps({
            "visible_message_count": len(payload["visible_content"]),
            "visible_source_root_count": len(roots),
            "root_assignments": assignments,
        }, sort_keys=True)
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content=content,
            finish_reason="stop",
            latency_ms=0.0,
        )


def build_cases(config_path: str = DEFAULT_MANIPULATION_CONFIG) -> tuple[ManipulationContext, ...]:
    gate = load_manipulation_config(config_path)
    study, _ = load_v11_config(gate.study_config_path)
    _, scenarios = load_identification_config(study.parent_config_path)
    cases: list[ManipulationContext] = []
    for scenario in scenarios:
        repeats = [item for item in scenario.stimuli if item.role in {"repeat_a", "repeat_b"}]
        for projection in gate.projections:
            if projection is SourceProjection.SOURCE_FREE:
                roots = (None, None)
            elif projection is SourceProjection.SAME_ROOT:
                roots = tuple(item.source_root_id for item in repeats)
            else:
                roots = tuple(item.independent_source_root_id for item in repeats)
            cases.append(ManipulationContext(
                case_id=f"source-check-{scenario.scenario_id}-{projection.value}",
                scenario_id=scenario.scenario_id,
                projection=projection,
                visible_content_ids=tuple(item.stimulus_id for item in repeats),
                public_summaries=tuple(item.public_summary for item in repeats),
                public_root_assignments=tuple(
                    RootAssignment(content_id=item.stimulus_id, source_root_id=root)
                    for item, root in zip(repeats, roots, strict=True)
                ),
            ))
    return tuple(cases)


def run_fake_smoke(config_path: str = DEFAULT_MANIPULATION_CONFIG) -> dict[str, object]:
    gate = load_manipulation_config(config_path)
    provider = ManipulationFakeProvider()
    root_counts: Counter[str] = Counter()
    passed = 0
    for context in build_cases(config_path):
        request = render_manipulation_request(context)
        response = parse_manipulation_response(provider.complete(request).content, context)
        root_counts[context.projection.value] += response.visible_source_root_count
        passed += 1
    return {
        "status": "manipulation_fake_smoke_passed",
        "case_count": gate.case_count,
        "passed_case_count": passed,
        "logical_request_count": provider.calls,
        "root_count_totals": dict(sorted(root_counts.items())),
        "behavior_effect_estimated": False,
        "joined_to_behavior_study": False,
        "network": "disabled",
        "provider": "manipulation_fake",
        "results_written": False,
        "private_truth_exposed": False,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline source-root manipulation smoke")
    parser.add_argument("--config", default=DEFAULT_MANIPULATION_CONFIG)
    args = parser.parse_args(argv)
    print(safe_json(run_fake_smoke(args.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
