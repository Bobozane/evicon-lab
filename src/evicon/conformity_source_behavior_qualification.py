"""Offline behavioral qualification contract for source-manipulation studies.

This is deliberately a *protocol* gate.  It verifies that a model response can
be strictly parsed into a public adoption/share decision under text-matched
source projections.  It does not estimate a source effect and it never loads
evaluator-private labels.
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from enum import Enum
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_identification import IdentificationError, sha256_file
from .conformity_identification_v11 import load_v11_config
from .conformity_source_manipulation import RootAssignment, SourceProjection
from .llm_contract import LLMRequest

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG: Final[str] = (
    "configs/provenance_cascade/identification/"
    "conformity_source_behavior_qualification.v2.toml"
)
PROTOCOL_VERSION: Final[str] = "conformity_source_behavior_qualification.v2"
TEMPLATE_VERSION: Final[str] = "conformity_source_behavior_turn.v2"
SCHEMA_NAME: Final[str] = "conformity_source_behavior_response_v2"


class AdoptionDecision(str, Enum):
    ADOPT_VISIBLE_CLAIM = "adopt_visible_claim"
    WITHHOLD = "withhold"


class SharingDecision(str, Enum):
    SHARE_VISIBLE_CONTENT = "share_visible_content"
    DO_NOT_SHARE = "do_not_share"


class BehaviorQualificationContext(BaseModel):
    """Only text and root relations already public in the local case."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str = Field(pattern=r"^source-behavior-v2-[a-z0-9-]+$")
    scenario_id: str
    projection: SourceProjection
    visible_content_ids: tuple[str, str]
    public_summaries: tuple[str, str]
    public_root_assignments: tuple[RootAssignment, RootAssignment]
    model_name: str = "fake-source-behavior-model"
    temperature: Literal[0.0] = 0.0
    max_tokens: Literal[128] = 128
    seed: int = 20261110

    @model_validator(mode="after")
    def public_inputs_match(self) -> "BehaviorQualificationContext":
        if len(set(self.visible_content_ids)) != 2:
            raise ValueError("visible content IDs must be distinct")
        if {item.content_id for item in self.public_root_assignments} != set(self.visible_content_ids):
            raise ValueError("root assignments must cover visible content")
        root_count = len({item.source_root_id for item in self.public_root_assignments if item.source_root_id})
        expected = {
            SourceProjection.SOURCE_FREE: 0,
            SourceProjection.SAME_ROOT: 1,
            SourceProjection.INDEPENDENT_ROOTS: 2,
        }[self.projection]
        if root_count != expected:
            raise ValueError("projection/root count mismatch")
        return self


class BehaviorQualificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    adoption_decision: AdoptionDecision
    sharing_decision: SharingDecision
    content_ids_used: tuple[str, ...]
    share_content_id: str | None


class SafeBehaviorCaseAudit(BaseModel):
    """The persistable semantic fact set; no prompt or raw response exists here."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str
    scenario_id: str
    projection: SourceProjection
    parser_status: Literal["valid"]
    adoption_decision: AdoptionDecision
    sharing_decision: SharingDecision
    used_content_count: int = Field(ge=0, le=2)
    visible_root_count: int = Field(ge=0, le=2)


class SourceBehaviorQualificationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    gate_id: Literal["evicon-conformity-source-behavior-qualification-v2"]
    gate_version: Literal["conformity_source_behavior_qualification.v2"]
    status: Literal["offline_design_validation"]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    behavior_effect_estimated: Literal[False]
    study_config_path: str
    study_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_path: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_version: Literal["conformity_source_behavior_qualification.v2"]
    template_version: Literal["conformity_source_behavior_turn.v2"]
    response_schema_name: Literal["conformity_source_behavior_response_v2"]
    projections: tuple[SourceProjection, ...]
    scenario_count: Literal[4]
    case_count: Literal[12]
    logical_request_cap: Literal[12]
    max_tokens: Literal[128]
    completion_reservation_cap: Literal[1536]
    max_retries: Literal[0]
    timeout_seconds: Literal[5]
    output_root: Literal["outputs/conformity-source-behavior-qualification-v2"]
    old_results_excluded: Literal[True]

    @model_validator(mode="after")
    def fixed_scope(self) -> "SourceBehaviorQualificationConfig":
        if self.projections != tuple(SourceProjection):
            raise ValueError("projection set mismatch")
        if self.case_count != self.scenario_count * len(self.projections):
            raise ValueError("case count mismatch")
        if self.logical_request_cap != self.case_count:
            raise ValueError("logical request cap mismatch")
        if self.completion_reservation_cap != self.logical_request_cap * self.max_tokens:
            raise ValueError("completion reservation mismatch")
        return self


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def load_config(path: str | Path = DEFAULT_CONFIG) -> SourceBehaviorQualificationConfig:
    resolved = _resolve(path)
    if not resolved.is_file():
        raise IdentificationError("source_behavior_config_missing")
    try:
        config = SourceBehaviorQualificationConfig.model_validate(tomllib.loads(resolved.read_text(encoding="utf-8")))
    except Exception as exc:
        raise IdentificationError("source_behavior_config_invalid") from exc
    if sha256_file(config.study_config_path) != config.study_config_sha256:
        raise IdentificationError("source_behavior_study_hash_mismatch")
    if sha256_file(config.protocol_path) != config.protocol_sha256:
        raise IdentificationError("source_behavior_protocol_hash_mismatch")
    study, _ = load_v11_config(config.study_config_path)
    if len(study.scenario_bindings) != config.scenario_count:
        raise IdentificationError("source_behavior_scenario_count_mismatch")
    return config


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate_field")
        output[key] = value
    return output


def parse_behavior_response(raw: str, context: BehaviorQualificationContext) -> BehaviorQualificationResponse:
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_pairs)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("malformed_json") from exc
    if not isinstance(payload, dict):
        raise ValueError("top_level_type")
    try:
        response = BehaviorQualificationResponse.model_validate(payload)
    except Exception as exc:
        raise ValueError("invalid_schema") from exc
    if len(set(response.content_ids_used)) != len(response.content_ids_used):
        raise ValueError("duplicate_content_id")
    visible = set(context.visible_content_ids)
    if not set(response.content_ids_used).issubset(visible):
        raise ValueError("unavailable_content_id")
    if response.sharing_decision is SharingDecision.SHARE_VISIBLE_CONTENT:
        if response.share_content_id not in visible or response.share_content_id not in response.content_ids_used:
            raise ValueError("share_content_id_invalid")
    elif response.share_content_id is not None:
        raise ValueError("share_content_id_unexpected")
    return response


def safe_case_audit(
    context: BehaviorQualificationContext, response: BehaviorQualificationResponse,
) -> SafeBehaviorCaseAudit:
    return SafeBehaviorCaseAudit(
        case_id=context.case_id,
        scenario_id=context.scenario_id,
        projection=context.projection,
        parser_status="valid",
        adoption_decision=response.adoption_decision,
        sharing_decision=response.sharing_decision,
        used_content_count=len(response.content_ids_used),
        visible_root_count=len({item.source_root_id for item in context.public_root_assignments if item.source_root_id}),
    )


def render_behavior_request(context: BehaviorQualificationContext) -> LLMRequest:
    payload = {
        "template_version": TEMPLATE_VERSION,
        "case_id": context.case_id,
        "visible_content": [
            {"content_id": content_id, "public_summary": summary}
            for content_id, summary in zip(context.visible_content_ids, context.public_summaries, strict=True)
        ],
        "public_root_relations": [item.model_dump(mode="json") for item in context.public_root_assignments],
        "task": "Choose a public adoption and sharing decision using only the visible public material.",
    }
    user_prompt = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{context.case_id}|{user_prompt}".encode()).hexdigest()[:24]
    return LLMRequest(
        request_id=f"source-behavior-{digest}",
        model_name=context.model_name,
        system_prompt=(
            "Return exactly adoption_decision, sharing_decision, content_ids_used, and share_content_id as JSON. "
            "A decision is not a truth judgment. Use only visible IDs. Do not infer source quality, private "
            "independence, truth labels, evidence, controller actions, or new content."
        ),
        user_prompt=user_prompt,
        temperature=context.temperature,
        max_tokens=context.max_tokens,
        seed=context.seed,
        metadata={"protocol_version": PROTOCOL_VERSION, "template_version": TEMPLATE_VERSION},
    )


def safe_preflight(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    try:
        config = load_config(path)
        return {
            "status": "offline_behavior_gate_ready",
            "gate_id": config.gate_id,
            "case_count": config.case_count,
            "logical_request_cap": config.logical_request_cap,
            "completion_reservation_cap": config.completion_reservation_cap,
            "ready_for_network": False,
            "blocking_reasons": ["human_approval_required", "provider_compatibility_not_requested"],
            "network": "disabled", "provider_constructed": False, "api_key_read": False,
            "results_written": False, "private_truth_exposed": False, "behavior_effect_estimated": False,
            "not_paper_result": True, "no_causal_conclusion": True,
        }
    except Exception as exc:
        return {
            "status": "blocked", "blocking_reasons": [getattr(exc, "code", "source_behavior_preflight_failed")],
            "ready_for_network": False, "network": "disabled", "provider_constructed": False,
            "api_key_read": False, "results_written": False, "private_truth_exposed": False,
        }


__all__ = [
    "AdoptionDecision", "BehaviorQualificationContext", "BehaviorQualificationResponse", "DEFAULT_CONFIG",
    "PROTOCOL_VERSION", "SCHEMA_NAME", "SafeBehaviorCaseAudit", "SharingDecision", "SourceBehaviorQualificationConfig",
    "TEMPLATE_VERSION", "load_config", "parse_behavior_response", "render_behavior_request", "safe_case_audit",
    "safe_preflight",
]
