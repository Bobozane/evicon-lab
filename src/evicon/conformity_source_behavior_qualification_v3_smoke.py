"""Network-free construction and sensitivity smoke for qualification v3."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Sequence

from .conformity_identification import safe_json
from .conformity_source_behavior_qualification_v3 import (
    DEFAULT_CONFIG,
    BehaviorQualificationContextV3,
    ContentRole,
    SafeBehaviorCaseAuditV3,
    VisibleContent,
    load_config,
    parse_behavior_response_v3,
    render_behavior_request_v3,
    safe_case_audit_v3,
    validate_case_design,
)
from .conformity_source_manipulation import RootAssignment, SourceProjection
from .llm_contract import LLMResponse


_ORDERS = (
    tuple(SourceProjection),
    (
        SourceProjection.SAME_ROOT,
        SourceProjection.INDEPENDENT_ROOTS,
        SourceProjection.SOURCE_FREE,
    ),
    (
        SourceProjection.INDEPENDENT_ROOTS,
        SourceProjection.SOURCE_FREE,
        SourceProjection.SAME_ROOT,
    ),
)


def _assignments(projection: SourceProjection) -> tuple[
    RootAssignment, RootAssignment, RootAssignment, RootAssignment
]:
    if projection is SourceProjection.SOURCE_FREE:
        roots = (None, None)
    elif projection is SourceProjection.SAME_ROOT:
        roots = ("root-01", "root-01")
    else:
        roots = ("root-01", "root-02")
    focal = tuple(
        RootAssignment(content_id=f"content-0{index}", source_root_id=root)
        for index, root in enumerate(roots, start=1)
    )
    fixed = (
        RootAssignment(content_id="content-03", source_root_id="root-03"),
        RootAssignment(content_id="content-04", source_root_id="root-04"),
    )
    return focal + fixed  # type: ignore[return-value]


def build_cases(
    path: str = DEFAULT_CONFIG,
) -> tuple[BehaviorQualificationContextV3, ...]:
    config, corpus = load_config(path)
    cases: list[BehaviorQualificationContextV3] = []
    sequence = 0
    for scenario in corpus.scenarios:
        content = tuple(
            VisibleContent(
                content_id=content_id,
                public_summary=summary,
                role=(ContentRole.FOCAL if index < 2 else ContentRole.FIXED_CONTEXT),
            )
            for index, (content_id, summary) in enumerate(
                zip(scenario.content_ids, scenario.public_summaries, strict=True)
            )
        )
        for seed, order in zip(config.seeds, _ORDERS, strict=True):
            for position, projection in enumerate(order, start=1):
                sequence += 1
                cases.append(BehaviorQualificationContextV3(
                    case_id=f"source-behavior-v3-q{sequence:03d}",
                    scenario_id=scenario.scenario_id,
                    projection=projection,
                    seed=seed,
                    order_position=position,
                    target_claim=scenario.target_claim,
                    decision_task=scenario.decision_task,
                    visible_content=content,  # type: ignore[arg-type]
                    public_root_assignments=_assignments(projection),
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                ))
    validate_case_design(cases)
    return tuple(cases)


class V3SensitivityFakeProvider:
    """Deterministic public-root responder used only to test measurement sensitivity."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request) -> LLMResponse:
        self.calls += 1
        payload = json.loads(request.user_prompt)
        roots = {
            item["source_root_id"]
            for item in payload["public_root_relations"][:2]
            if item["source_root_id"] is not None
        }
        if len(roots) == 0:
            response = {
                "ordinal_judgment": "uncertain",
                "public_action": "withhold",
                "share_decision": "do_not_share",
                "confidence": 3,
                "content_ids_used": ["content-01", "content-03", "content-04"],
                "share_content_id": None,
            }
        elif len(roots) == 1:
            response = {
                "ordinal_judgment": "lean_reject",
                "public_action": "reject",
                "share_decision": "do_not_share",
                "confidence": 4,
                "content_ids_used": [
                    "content-01", "content-02", "content-03", "content-04"
                ],
                "share_content_id": None,
            }
        else:
            response = {
                "ordinal_judgment": "lean_adopt",
                "public_action": "adopt",
                "share_decision": "share_with_caveat",
                "confidence": 5,
                "content_ids_used": [
                    "content-01", "content-02", "content-03", "content-04"
                ],
                "share_content_id": "content-01",
            }
        body = json.dumps(response, ensure_ascii=True, sort_keys=True)
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content=body,
            finish_reason="stop",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            latency_ms=0.0,
        )


def run_fake_smoke(path: str = DEFAULT_CONFIG) -> dict[str, object]:
    config, _ = load_config(path)
    provider = V3SensitivityFakeProvider()
    audits: list[SafeBehaviorCaseAuditV3] = []
    requests = []
    for case in build_cases(path):
        request = render_behavior_request_v3(case)
        requests.append(request)
        response = provider.complete(request)
        parsed = parse_behavior_response_v3(response.content, case)
        audits.append(safe_case_audit_v3(case, parsed))
    projection_counts = Counter(item.projection.value for item in audits)
    judgment_counts = Counter(item.ordinal_judgment.value for item in audits)
    return {
        "status": "source_behavior_v3_fake_sensitivity_smoke_passed",
        "gate_id": config.gate_id,
        "case_count": len(audits),
        "logical_request_count": provider.calls,
        "unique_case_count": len({item.case_id for item in audits}),
        "unique_request_count": len({item.request_id for item in requests}),
        "projection_case_counts": dict(sorted(projection_counts.items())),
        "judgment_counts": dict(sorted(judgment_counts.items())),
        "safe_case_audit_count": len(audits),
        "measurement_sensitivity_exercised": True,
        "behavior_effect_estimated": False,
        "network": "disabled",
        "provider_constructed": False,
        "api_key_read": False,
        "results_written": False,
        "private_truth_exposed": False,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the offline source-behavior v3 sensitivity smoke"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    arguments = parser.parse_args(argv)
    print(safe_json(run_fake_smoke(arguments.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["V3SensitivityFakeProvider", "build_cases", "run_fake_smoke"]
