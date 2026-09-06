"""Structural source-root comprehension gate for the identification study."""
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
from .llm_contract import LLMRequest

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIPULATION_CONFIG = "configs/provenance_cascade/identification/conformity_source_manipulation_gate.v1.toml"
PROTOCOL_VERSION: Final[str] = "conformity_source_manipulation.v1"
TEMPLATE_VERSION: Final[str] = "conformity_source_structure_check.v1"
SCHEMA_NAME: Final[str] = "conformity_source_structure_response_v1"


class SourceProjection(str, Enum):
    SOURCE_FREE = "source_free"
    SAME_ROOT = "same_root"
    INDEPENDENT_ROOTS = "independent_roots"


class RootAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    content_id: str
    source_root_id: str | None


class ManipulationContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str
    scenario_id: str
    projection: SourceProjection
    visible_content_ids: tuple[str, ...] = Field(min_length=2, max_length=2)
    public_summaries: tuple[str, ...] = Field(min_length=2, max_length=2)
    public_root_assignments: tuple[RootAssignment, ...] = Field(min_length=2, max_length=2)
    model_name: str = "fake-source-manipulation-model"
    temperature: Literal[0.0] = 0.0
    max_tokens: Literal[128] = 128
    seed: int = 20261100

    @model_validator(mode="after")
    def visible_structure(self) -> "ManipulationContext":
        if len(set(self.visible_content_ids)) != 2:
            raise ValueError("content IDs must be distinct")
        if {item.content_id for item in self.public_root_assignments} != set(self.visible_content_ids):
            raise ValueError("root assignments must cover visible content")
        roots = {item.source_root_id for item in self.public_root_assignments if item.source_root_id is not None}
        expected = {
            SourceProjection.SOURCE_FREE: 0,
            SourceProjection.SAME_ROOT: 1,
            SourceProjection.INDEPENDENT_ROOTS: 2,
        }[self.projection]
        if len(roots) != expected:
            raise ValueError("projection/root count mismatch")
        return self


class ManipulationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    visible_message_count: Literal[2]
    visible_source_root_count: Literal[0, 1, 2]
    root_assignments: tuple[RootAssignment, ...] = Field(min_length=2, max_length=2)


class ManipulationGateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    gate_id: Literal["evicon-conformity-source-manipulation-v1"]
    gate_version: Literal["conformity_source_manipulation_gate.v1"]
    status: Literal["offline_validation"]
    development_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    study_config_path: str
    study_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_path: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_version: Literal["conformity_source_manipulation.v1"]
    template_version: Literal["conformity_source_structure_check.v1"]
    response_schema_name: Literal["conformity_source_structure_response_v1"]
    projections: tuple[SourceProjection, ...]
    scenario_count: Literal[4]
    case_count: Literal[12]
    logical_request_cap: Literal[12]
    max_tokens: Literal[128]
    completion_reservation_cap: Literal[1536]
    max_retries: Literal[0]
    timeout_seconds: Literal[5]
    results_joined_to_behavior_study: Literal[False]
    output_root: Literal["outputs/conformity-source-manipulation-v1"]

    @model_validator(mode="after")
    def fixed_gate(self) -> "ManipulationGateConfig":
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


def load_manipulation_config(path: str | Path = DEFAULT_MANIPULATION_CONFIG) -> ManipulationGateConfig:
    resolved = _resolve(path)
    if not resolved.is_file():
        raise IdentificationError("manipulation_config_missing")
    try:
        config = ManipulationGateConfig.model_validate(tomllib.loads(resolved.read_text(encoding="utf-8")))
    except Exception as exc:
        raise IdentificationError("manipulation_config_invalid") from exc
    if sha256_file(config.study_config_path) != config.study_config_sha256:
        raise IdentificationError("manipulation_study_hash_mismatch")
    if sha256_file(config.protocol_path) != config.protocol_sha256:
        raise IdentificationError("manipulation_protocol_hash_mismatch")
    study, _ = load_v11_config(config.study_config_path)
    if len(study.scenario_bindings) != config.scenario_count:
        raise IdentificationError("manipulation_scenario_count_mismatch")
    return config


def parse_manipulation_response(raw: str, context: ManipulationContext) -> ManipulationResponse:
    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_field")
            result[key] = value
        return result

    try:
        data = json.loads(raw, object_pairs_hook=unique_pairs)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("malformed_json") from exc
    if not isinstance(data, dict):
        raise ValueError("top_level_type")
    try:
        response = ManipulationResponse.model_validate(data)
    except Exception as exc:
        raise ValueError("invalid_schema") from exc
    if {item.content_id for item in response.root_assignments} != set(context.visible_content_ids):
        raise ValueError("assignment_content_mismatch")
    expected_assignments = {item.content_id: item.source_root_id for item in context.public_root_assignments}
    actual_assignments = {item.content_id: item.source_root_id for item in response.root_assignments}
    if actual_assignments != expected_assignments:
        raise ValueError("root_assignment_mismatch")
    expected_root_count = len({value for value in expected_assignments.values() if value is not None})
    if response.visible_source_root_count != expected_root_count:
        raise ValueError("root_count_mismatch")
    return response


def render_manipulation_request(context: ManipulationContext) -> LLMRequest:
    payload = {
        "template_version": TEMPLATE_VERSION,
        "case_id": context.case_id,
        "visible_content": [
            {"content_id": content_id, "public_summary": summary}
            for content_id, summary in zip(context.visible_content_ids, context.public_summaries, strict=True)
        ],
        "public_root_relations": [item.model_dump(mode="json") for item in context.public_root_assignments],
        "task": "Count visible messages and distinct public source roots. Do not assess truth or reliability.",
    }
    user_prompt = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{context.case_id}|{user_prompt}".encode()).hexdigest()[:24]
    return LLMRequest(
        request_id=f"source-check-{digest}",
        model_name=context.model_name,
        system_prompt=(
            "Return exactly visible_message_count, visible_source_root_count, and root_assignments as JSON. "
            "Use only the supplied public relations. A missing relation means no visible source root. "
            "Do not infer truth, reliability, evidence quality, or private source independence."
        ),
        user_prompt=user_prompt,
        temperature=context.temperature,
        max_tokens=context.max_tokens,
        seed=context.seed,
        metadata={"protocol_version": PROTOCOL_VERSION, "template_version": TEMPLATE_VERSION},
    )


def safe_preflight(path: str | Path = DEFAULT_MANIPULATION_CONFIG) -> dict[str, object]:
    try:
        config = load_manipulation_config(path)
        return {
            "status": "offline_gate_ready",
            "ready_for_network": False,
            "blocking_reasons": ["human_approval_required", "provider_compatibility_not_requested"],
            "gate_id": config.gate_id,
            "study_config_sha256": config.study_config_sha256,
            "case_count": config.case_count,
            "logical_request_cap": config.logical_request_cap,
            "completion_reservation_cap": config.completion_reservation_cap,
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "private_truth_exposed": False,
            "behavior_effect_estimated": False,
            "not_paper_result": True,
            "no_causal_conclusion": True,
        }
    except Exception as exc:
        return {
            "status": "blocked",
            "blocking_reasons": [getattr(exc, "code", "manipulation_preflight_failed")],
            "ready_for_network": False,
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "private_truth_exposed": False,
        }


__all__ = [
    "DEFAULT_MANIPULATION_CONFIG", "ManipulationContext", "ManipulationGateConfig",
    "ManipulationResponse", "RootAssignment", "SourceProjection", "load_manipulation_config",
    "parse_manipulation_response", "render_manipulation_request", "safe_preflight",
]
