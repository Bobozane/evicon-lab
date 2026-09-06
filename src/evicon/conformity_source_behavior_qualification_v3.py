"""Identifiable, offline-only source-behavior qualification protocol v3.

Version 3 is isolated from the completed v2 batch. It retains text-matched
source projections while exposing an explicit decision task and fixed public
counter/evidence material. The default path validates design bindings only;
this module has no Provider construction or network execution path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections import Counter
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Final, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .conformity_identification import (
    IdentificationError,
    OrdinalJudgment,
    PublicAction,
    ShareDecision,
    safe_json,
    sha256_file,
)
from .conformity_source_manipulation import RootAssignment, SourceProjection
from .llm_contract import LLMRequest

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG: Final[str] = (
    "configs/provenance_cascade/identification/"
    "conformity_source_behavior_qualification.v3.toml"
)
DEFAULT_APPROVAL: Final[str] = (
    "configs/provenance_cascade/identification/"
    "conformity_source_behavior_qualification_approval.v3.toml"
)
PROTOCOL_VERSION: Final[str] = "conformity_source_behavior_qualification.v3"
TEMPLATE_VERSION: Final[str] = "conformity_source_behavior_turn.v3"
SCHEMA_NAME: Final[str] = "conformity_source_behavior_response_v3"


class ContentRole(str, Enum):
    FOCAL = "focal"
    FIXED_CONTEXT = "fixed_context"


class V3Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(pattern=r"^source-behavior-v3-s0[1-4]$")
    target_claim: str = Field(min_length=24, max_length=240)
    decision_task: str = Field(min_length=24, max_length=280)
    content_ids: tuple[str, str, str, str]
    public_summaries: tuple[str, str, str, str]

    @model_validator(mode="after")
    def identifiable_public_material(self) -> "V3Scenario":
        if self.content_ids != ("content-01", "content-02", "content-03", "content-04"):
            raise ValueError("v3 content IDs must be opaque and fixed")
        if any(len(summary) < 48 or len(summary) > 360 for summary in self.public_summaries):
            raise ValueError("v3 public summary length is outside the fixed range")
        public_text = " ".join((self.target_claim, self.decision_task, *self.public_summaries)).lower()
        forbidden = (
            "ground_truth", "evaluator_private", "correct_answer",
            "false_majority", "true_minority", "independent_roots", "same_root",
        )
        if any(token in public_text for token in forbidden):
            raise ValueError("v3 public material leaks a condition or private label")
        return self


class V3ScenarioCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_id: Literal["evicon-conformity-source-behavior-scenarios-v3"]
    corpus_version: Literal["conformity_source_behavior_scenarios.v3"]
    status: Literal["development_only"]
    source_kind: Literal["author_original"]
    synthetic_disclaimer: Literal[True]
    evaluator_private_truth_included: Literal[False]
    scenarios: tuple[V3Scenario, V3Scenario, V3Scenario, V3Scenario]

    @model_validator(mode="after")
    def exact_scenario_set(self) -> "V3ScenarioCorpus":
        if tuple(item.scenario_id for item in self.scenarios) != tuple(
            f"source-behavior-v3-s0{index}" for index in range(1, 5)
        ):
            raise ValueError("v3 scenario order mismatch")
        return self


class VisibleContent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    content_id: str = Field(pattern=r"^content-0[1-4]$")
    public_summary: str = Field(min_length=48, max_length=360)
    role: ContentRole


class BehaviorQualificationContextV3(BaseModel):
    """Internal coordinates plus the exact public view for one request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^source-behavior-v3-q\d{3}$")
    scenario_id: str = Field(pattern=r"^source-behavior-v3-s0[1-4]$")
    projection: SourceProjection
    seed: Literal[20261201, 20261202, 20261203]
    order_position: int = Field(ge=1, le=3)
    target_claim: str
    decision_task: str
    visible_content: tuple[VisibleContent, VisibleContent, VisibleContent, VisibleContent]
    public_root_assignments: tuple[
        RootAssignment, RootAssignment, RootAssignment, RootAssignment
    ]
    model_name: str = "fake-source-behavior-v3-model"
    temperature: Literal[0.2] = 0.2
    max_tokens: Literal[256] = 256

    @model_validator(mode="after")
    def exact_public_view(self) -> "BehaviorQualificationContextV3":
        content_ids = tuple(item.content_id for item in self.visible_content)
        if content_ids != ("content-01", "content-02", "content-03", "content-04"):
            raise ValueError("v3 visible content order mismatch")
        if tuple(item.role for item in self.visible_content) != (
            ContentRole.FOCAL,
            ContentRole.FOCAL,
            ContentRole.FIXED_CONTEXT,
            ContentRole.FIXED_CONTEXT,
        ):
            raise ValueError("v3 focal/context material mismatch")
        if tuple(item.content_id for item in self.public_root_assignments) != content_ids:
            raise ValueError("v3 public root assignments must cover visible content")
        focal_assignments = self.public_root_assignments[:2]
        roots = {
            item.source_root_id
            for item in focal_assignments
            if item.source_root_id is not None
        }
        expected = {
            SourceProjection.SOURCE_FREE: 0,
            SourceProjection.SAME_ROOT: 1,
            SourceProjection.INDEPENDENT_ROOTS: 2,
        }[self.projection]
        if len(roots) != expected:
            raise ValueError("v3 projection/root count mismatch")
        allowed_roots = {None, "root-01", "root-02", "root-03", "root-04"}
        if any(item.source_root_id not in allowed_roots for item in self.public_root_assignments):
            raise ValueError("v3 root IDs must be opaque")
        if tuple(item.source_root_id for item in self.public_root_assignments[2:]) != (
            "root-03", "root-04"
        ):
            raise ValueError("v3 fixed context roots must remain distinct and fixed")
        return self


class BehaviorQualificationResponseV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ordinal_judgment: OrdinalJudgment
    public_action: PublicAction
    share_decision: ShareDecision
    confidence: int = Field(ge=1, le=7)
    content_ids_used: tuple[str, ...] = Field(min_length=1, max_length=4)
    share_content_id: str | None


RESPONSE_SCHEMA: Final[dict[str, object]] = BehaviorQualificationResponseV3.model_json_schema(
    mode="validation"
)
RESPONSE_SCHEMA_SHA256: Final[str] = hashlib.sha256(
    json.dumps(RESPONSE_SCHEMA, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


class SafeBehaviorCaseAuditV3(BaseModel):
    """Persistable case outcome with no prompt, text, raw reply, or private label."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^source-behavior-v3-q\d{3}$")
    scenario_id: str = Field(pattern=r"^source-behavior-v3-s0[1-4]$")
    projection: SourceProjection
    seed: Literal[20261201, 20261202, 20261203]
    order_position: int = Field(ge=1, le=3)
    parser_status: Literal["valid"]
    ordinal_judgment: OrdinalJudgment
    public_action: PublicAction
    share_decision: ShareDecision
    confidence: int = Field(ge=1, le=7)
    used_content_count: int = Field(ge=1, le=4)
    visible_focal_root_count: int = Field(ge=0, le=2)


class SourceBehaviorQualificationConfigV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gate_id: Literal["evicon-conformity-source-behavior-qualification-v3"]
    gate_version: Literal["conformity_source_behavior_qualification.v3"]
    status: Literal["offline_design_validation"]
    development_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    behavior_effect_estimated: Literal[False]
    v2_artifacts_frozen: Literal[True]
    old_results_excluded: Literal[True]
    corpus_path: str
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_path: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_version: Literal["conformity_source_behavior_qualification.v3"]
    template_version: Literal["conformity_source_behavior_turn.v3"]
    response_schema_name: Literal["conformity_source_behavior_response_v3"]
    projections: tuple[SourceProjection, SourceProjection, SourceProjection]
    seeds: tuple[Literal[20261201, 20261202, 20261203], ...]
    scenario_count: Literal[4]
    case_count: Literal[36]
    logical_request_cap: Literal[36]
    temperature: Literal[0.2]
    max_tokens: Literal[256]
    completion_reservation_cap: Literal[9216]
    max_retries: Literal[0]
    timeout_seconds: Literal[5]
    projection_order_counterbalanced: Literal[True]
    opaque_prompt_coordinates: Literal[True]
    decision_task_visible: Literal[True]
    fixed_counter_evidence_visible: Literal[True]
    public_root_relations_permitted: Literal[True]
    output_root: Literal["outputs/conformity-source-behavior-qualification-v3"]

    @model_validator(mode="after")
    def frozen_v3_design(self) -> "SourceBehaviorQualificationConfigV3":
        if self.projections != tuple(SourceProjection):
            raise ValueError("v3 projection set mismatch")
        if self.seeds != (20261201, 20261202, 20261203):
            raise ValueError("v3 seed set mismatch")
        if self.case_count != self.scenario_count * len(self.projections) * len(self.seeds):
            raise ValueError("v3 case count mismatch")
        if self.logical_request_cap != self.case_count:
            raise ValueError("v3 request cap mismatch")
        if self.completion_reservation_cap != self.case_count * self.max_tokens:
            raise ValueError("v3 completion reservation mismatch")
        return self


class SourceBehaviorQualificationApprovalV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: Literal["evicon-conformity-source-behavior-qualification-v3"]
    approval_version: Literal["conformity_source_behavior_qualification_approval.v3"]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    accepted_on: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirm_v2_frozen: bool
    confirm_opaque_prompt_coordinates: bool
    confirm_decision_task_visible: bool
    confirm_counter_evidence_visible: bool
    confirm_public_provenance_usable: bool
    confirm_three_seed_counterbalance: bool
    confirm_development_only: bool
    confirm_no_network: bool
    network_execution_authorized: Literal[False]

    @model_validator(mode="after")
    def accepted_fields_complete(self) -> "SourceBehaviorQualificationApprovalV3":
        if self.acceptance_status != "accepted":
            return self
        if not self.accepted_by.strip() or not self.accepted_on.strip():
            raise ValueError("accepted v3 approval requires reviewer and date")
        try:
            date.fromisoformat(self.accepted_on)
        except ValueError as exc:
            raise ValueError("accepted_on must be an ISO date") from exc
        confirmations = (
            self.confirm_v2_frozen,
            self.confirm_opaque_prompt_coordinates,
            self.confirm_decision_task_visible,
            self.confirm_counter_evidence_visible,
            self.confirm_public_provenance_usable,
            self.confirm_three_seed_counterbalance,
            self.confirm_development_only,
            self.confirm_no_network,
        )
        if not all(confirmations):
            raise ValueError("accepted v3 approval is incomplete")
        return self


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def load_config(
    path: str | Path = DEFAULT_CONFIG,
) -> tuple[SourceBehaviorQualificationConfigV3, V3ScenarioCorpus]:
    try:
        config = SourceBehaviorQualificationConfigV3.model_validate(
            tomllib.loads(_resolve(path).read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise IdentificationError("source_behavior_v3_config_invalid") from exc
    if sha256_file(config.protocol_path) != config.protocol_sha256:
        raise IdentificationError("source_behavior_v3_protocol_hash_mismatch")
    if sha256_file(config.corpus_path) != config.corpus_sha256:
        raise IdentificationError("source_behavior_v3_corpus_hash_mismatch")
    try:
        corpus = V3ScenarioCorpus.model_validate(
            tomllib.loads(_resolve(config.corpus_path).read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise IdentificationError("source_behavior_v3_corpus_invalid") from exc
    if len(corpus.scenarios) != config.scenario_count:
        raise IdentificationError("source_behavior_v3_scenario_count_mismatch")
    return config, corpus


def load_approval(
    path: str | Path = DEFAULT_APPROVAL,
) -> SourceBehaviorQualificationApprovalV3:
    config, _ = load_config(DEFAULT_CONFIG)
    try:
        approval = SourceBehaviorQualificationApprovalV3.model_validate(
            tomllib.loads(_resolve(path).read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise IdentificationError("source_behavior_v3_approval_invalid") from exc
    expected = {
        "config_sha256": sha256_file(DEFAULT_CONFIG),
        "corpus_sha256": config.corpus_sha256,
        "protocol_sha256": config.protocol_sha256,
        "response_schema_sha256": RESPONSE_SCHEMA_SHA256,
    }
    if {name: getattr(approval, name) for name in expected} != expected:
        raise IdentificationError("source_behavior_v3_approval_binding_mismatch")
    return approval


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate_field")
        output[key] = value
    return output


def parse_behavior_response_v3(
    raw: str,
    context: BehaviorQualificationContextV3,
) -> BehaviorQualificationResponseV3:
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_pairs)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("malformed_json") from exc
    if not isinstance(payload, dict):
        raise ValueError("top_level_type")
    try:
        response = BehaviorQualificationResponseV3.model_validate(payload)
    except Exception as exc:
        raise ValueError("invalid_schema") from exc
    if len(set(response.content_ids_used)) != len(response.content_ids_used):
        raise ValueError("duplicate_content_id")
    visible = {item.content_id for item in context.visible_content}
    if not set(response.content_ids_used).issubset(visible):
        raise ValueError("unavailable_content_id")
    expected_action = {
        OrdinalJudgment.STRONGLY_REJECT: PublicAction.REJECT,
        OrdinalJudgment.LEAN_REJECT: PublicAction.REJECT,
        OrdinalJudgment.UNCERTAIN: PublicAction.WITHHOLD,
        OrdinalJudgment.LEAN_ADOPT: PublicAction.ADOPT,
        OrdinalJudgment.STRONGLY_ADOPT: PublicAction.ADOPT,
    }[response.ordinal_judgment]
    if response.public_action is not expected_action:
        raise ValueError("judgment_action_mismatch")
    if response.share_decision is ShareDecision.DO_NOT_SHARE:
        if response.share_content_id is not None:
            raise ValueError("share_content_id_unexpected")
    elif (
        response.share_content_id not in visible
        or response.share_content_id not in response.content_ids_used
    ):
        raise ValueError("share_content_id_invalid")
    return response


def safe_case_audit_v3(
    context: BehaviorQualificationContextV3,
    response: BehaviorQualificationResponseV3,
) -> SafeBehaviorCaseAuditV3:
    return SafeBehaviorCaseAuditV3(
        case_id=context.case_id,
        scenario_id=context.scenario_id,
        projection=context.projection,
        seed=context.seed,
        order_position=context.order_position,
        parser_status="valid",
        ordinal_judgment=response.ordinal_judgment,
        public_action=response.public_action,
        share_decision=response.share_decision,
        confidence=response.confidence,
        used_content_count=len(response.content_ids_used),
        visible_focal_root_count=len({
            item.source_root_id
            for item in context.public_root_assignments[:2]
            if item.source_root_id is not None
        }),
    )


def render_behavior_request_v3(context: BehaviorQualificationContextV3) -> LLMRequest:
    payload = {
        "template_version": TEMPLATE_VERSION,
        "target_claim": context.target_claim,
        "decision_task": context.decision_task,
        "visible_content": [
            {
                "content_id": item.content_id,
                "public_summary": item.public_summary,
            }
            for item in context.visible_content
        ],
        "public_root_relations": [
            item.model_dump(mode="json") for item in context.public_root_assignments
        ],
    }
    user_prompt = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(
        f"{context.case_id}|{context.seed}|{user_prompt}".encode()
    ).hexdigest()[:24]
    return LLMRequest(
        request_id=f"source-behavior-v3-{digest}",
        model_name=context.model_name,
        system_prompt=(
            "Return exactly ordinal_judgment, public_action, share_decision, confidence, "
            "content_ids_used, and share_content_id as JSON. Assess the target claim for the "
            "stated reversible decision task using all visible public material. You may use "
            "supplied public source-root relations to distinguish repeated reporting from "
            "corroboration: equal root IDs mean the focal messages trace to one public root, "
            "distinct root IDs mean distinct public roots, and null means provenance is not "
            "shown rather than that no source exists. Root relations do not by themselves "
            "establish truth or source quality. Use only visible content IDs and do not invent "
            "sources, evidence, private labels, controller actions, or new content."
        ),
        user_prompt=user_prompt,
        temperature=context.temperature,
        max_tokens=context.max_tokens,
        seed=context.seed,
        metadata={"protocol_version": PROTOCOL_VERSION, "template_version": TEMPLATE_VERSION},
    )


def validate_case_design(cases: Sequence[BehaviorQualificationContextV3]) -> None:
    if len(cases) != 36 or len({item.case_id for item in cases}) != 36:
        raise IdentificationError("source_behavior_v3_case_coverage_mismatch")
    coordinates = Counter((item.scenario_id, item.seed, item.projection) for item in cases)
    if len(coordinates) != 36 or any(count != 1 for count in coordinates.values()):
        raise IdentificationError("source_behavior_v3_coordinate_coverage_mismatch")
    for scenario_id in sorted({item.scenario_id for item in cases}):
        scenario_cases = [item for item in cases if item.scenario_id == scenario_id]
        text_views = {
            (
                item.target_claim,
                item.decision_task,
                tuple(content.public_summary for content in item.visible_content),
            )
            for item in scenario_cases
        }
        if len(text_views) != 1:
            raise IdentificationError("source_behavior_v3_text_matching_failed")
        for position in (1, 2, 3):
            if Counter(
                item.projection for item in scenario_cases if item.order_position == position
            ) != Counter(SourceProjection):
                raise IdentificationError("source_behavior_v3_order_counterbalance_failed")


def safe_preflight(
    config_path: str | Path = DEFAULT_CONFIG,
    approval_path: str | Path = DEFAULT_APPROVAL,
) -> dict[str, object]:
    try:
        config, _ = load_config(config_path)
        approval = load_approval(approval_path)
    except Exception as exc:
        return {
            "status": "blocked",
            "blocking_reasons": [
                getattr(exc, "code", "source_behavior_v3_preflight_failed")
            ],
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
        }
    reasons = ["provider_compatibility_not_requested"]
    if approval.acceptance_status != "accepted":
        reasons.insert(0, "human_approval_required")
    return {
        "status": "offline_source_behavior_v3_gate_ready",
        "gate_id": config.gate_id,
        "approval_status": approval.acceptance_status,
        "case_count": config.case_count,
        "logical_request_cap": config.logical_request_cap,
        "completion_reservation_cap": config.completion_reservation_cap,
        "response_schema_sha256": RESPONSE_SCHEMA_SHA256,
        "blocking_reasons": reasons,
        "ready_for_network": False,
        "network": "disabled",
        "provider_constructed": False,
        "api_key_read": False,
        "results_written": False,
        "v2_artifacts_frozen": True,
        "private_truth_exposed": False,
        "behavior_effect_estimated": False,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the offline source-behavior qualification v3 design"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--approval", default=DEFAULT_APPROVAL)
    arguments = parser.parse_args(argv)
    report = safe_preflight(arguments.config, arguments.approval)
    print(safe_json(report))
    return 0 if report["status"] == "offline_source_behavior_v3_gate_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BehaviorQualificationContextV3",
    "BehaviorQualificationResponseV3",
    "ContentRole",
    "DEFAULT_APPROVAL",
    "DEFAULT_CONFIG",
    "PROTOCOL_VERSION",
    "RESPONSE_SCHEMA",
    "RESPONSE_SCHEMA_SHA256",
    "SCHEMA_NAME",
    "SafeBehaviorCaseAuditV3",
    "SourceBehaviorQualificationApprovalV3",
    "SourceBehaviorQualificationConfigV3",
    "TEMPLATE_VERSION",
    "V3Scenario",
    "V3ScenarioCorpus",
    "VisibleContent",
    "load_approval",
    "load_config",
    "parse_behavior_response_v3",
    "render_behavior_request_v3",
    "safe_case_audit_v3",
    "safe_preflight",
    "validate_case_design",
]
