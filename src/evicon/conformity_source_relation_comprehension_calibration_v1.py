"""Offline-only source-relation comprehension calibration for the new route.

This independent protocol verifies whether a model can read the *publicly
shown* relationship between two social reports without treating source roots
or an evidence-card label as a truth signal.  It is a construct-calibration
gate, not a behavior, intervention, or network experiment.  The historical
source-manipulation and factorization artifacts are intentionally not read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Final, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .conformity_identification import safe_json, sha256_file
from .llm_contract import LLMRequest, LLMResponse


_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG: Final[str] = (
    "configs/provenance_cascade/identification/"
    "conformity_source_relation_comprehension_calibration.v1.toml"
)
CALIBRATION_ID: Final[str] = "evicon-source-relation-comprehension-calibration-v1"
PROTOCOL_VERSION: Final[str] = "conformity_source_relation_comprehension_calibration.v1"
TEMPLATE_VERSION: Final[str] = "conformity_source_relation_comprehension_turn.v1"
SCHEMA_NAME: Final[str] = "conformity_source_relation_comprehension_response_v1"
_SEED: Final[int] = 20261401
_ROOT_A: Final[str] = "lineage-k7"
_ROOT_B: Final[str] = "lineage-p4"
_EVIDENCE_CARD_ID: Final[str] = "evidence-card-01"


class SourceRelationCalibrationError(ValueError):
    """Stable, non-sensitive error boundary for the calibration protocol."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class SourceRelation(str, Enum):
    """Only the relation visible in the current public view."""

    NOT_SHOWN = "not_shown"
    SAME_SHOWN_ROOT = "same_shown_root"
    DISTINCT_SHOWN_ROOTS = "distinct_shown_roots"


class EvidenceState(str, Enum):
    ABSENT = "absent"
    PRESENT = "present"


class PresentationCounterbalance(str, Enum):
    FORWARD = "forward"
    REVERSED = "reversed"


class SupportKind(str, Enum):
    SOCIAL_REPORTS_ONLY = "social_reports_only"
    DIRECT_EVIDENCE_PRESENT = "direct_evidence_present"


class MaterialType(str, Enum):
    SOCIAL_REPORT = "social_report"
    DIRECT_EVIDENCE_CARD = "direct_evidence_card"


class PublicRootAssignment(BaseModel):
    """A public relation, distinct from any latent provenance graph."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content_id: Literal["message-a", "message-b"]
    source_root_id: str | None

    @field_validator("source_root_id")
    @classmethod
    def root_is_opaque_public_code(cls, value: str | None) -> str | None:
        if value not in {None, _ROOT_A, _ROOT_B}:
            raise ValueError("source_relation_calibration_unknown_public_root")
        return value


class VisibleMaterial(BaseModel):
    """Minimal public synthetic material supplied to the structural task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content_id: Literal["message-a", "message-b", "evidence-card-01"]
    material_type: MaterialType
    public_summary: str = Field(min_length=24, max_length=260)


class CalibrationScenario(BaseModel):
    """New author-written public material with no evaluator answer label."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: Literal[
        "source-relation-v1-s01", "source-relation-v1-s02"
    ]
    message_a_summary: str = Field(min_length=32, max_length=260)
    message_b_summary: str = Field(min_length=32, max_length=260)
    evidence_card_summary: str = Field(min_length=32, max_length=260)

    @model_validator(mode="after")
    def material_is_public_and_structurally_neutral(self) -> "CalibrationScenario":
        material = " ".join(self.model_dump().values()).lower()
        forbidden = (
            "ground_truth",
            "evaluator_private",
            "correct_answer",
            "wvs",
            "pilot",
            "same_root",
            "independent_roots",
        )
        if any(token in material for token in forbidden):
            raise ValueError("source_relation_calibration_public_material_leak")
        summaries = (
            self.message_a_summary,
            self.message_b_summary,
            self.evidence_card_summary,
        )
        if len(set(summaries)) != len(summaries):
            raise ValueError("source_relation_calibration_public_summaries_must_be_distinct")
        return self


class SourceRelationComprehensionResponse(BaseModel):
    """Strict structural response contract; it contains no adoption outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    visible_social_message_count: Literal[2]
    source_relation: SourceRelation
    visible_social_root_count: Literal[0, 1, 2]
    social_root_assignments: tuple[PublicRootAssignment, PublicRootAssignment]
    direct_evidence_card_ids: tuple[Literal["evidence-card-01"], ...] = Field(max_length=1)
    support_kind: SupportKind
    source_structure_alone_decides_truth: Literal[False]


RESPONSE_SCHEMA: Final[dict[str, object]] = SourceRelationComprehensionResponse.model_json_schema(
    mode="validation"
)
RESPONSE_SCHEMA_SHA256: Final[str] = hashlib.sha256(
    json.dumps(
        RESPONSE_SCHEMA, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
).hexdigest()


class SourceRelationCalibrationContext(BaseModel):
    """One static public structural-reading case.

    The source relation and counterbalance are internal construction fields.
    Only visible material and public assignments enter a rendered request.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^source-relation-v1-c\d{2}$")
    scenario_id: Literal[
        "source-relation-v1-s01", "source-relation-v1-s02"
    ]
    source_relation: SourceRelation
    evidence_state: EvidenceState
    presentation_counterbalance: PresentationCounterbalance
    visible_material: tuple[VisibleMaterial, ...] = Field(min_length=2, max_length=3)
    public_root_assignments: tuple[PublicRootAssignment, PublicRootAssignment]
    model_name: str = "offline-source-relation-calibration-fake-model"
    temperature: Literal[0.0] = 0.0
    max_tokens: Literal[128] = 128
    seed: Literal[20261401] = _SEED

    @property
    def social_material(self) -> tuple[VisibleMaterial, VisibleMaterial]:
        social = tuple(
            item
            for item in self.visible_material
            if item.material_type is MaterialType.SOCIAL_REPORT
        )
        if len(social) != 2:
            raise SourceRelationCalibrationError(
                "source_relation_calibration_social_material_count_invalid"
            )
        return social  # type: ignore[return-value]

    @property
    def social_content_ids(self) -> tuple[str, str]:
        return tuple(item.content_id for item in self.social_material)  # type: ignore[return-value]

    @property
    def visible_social_root_count(self) -> int:
        return len({
            assignment.source_root_id
            for assignment in self.public_root_assignments
            if assignment.source_root_id is not None
        })

    @property
    def direct_evidence_card_ids(self) -> tuple[str, ...]:
        return tuple(
            item.content_id
            for item in self.visible_material
            if item.material_type is MaterialType.DIRECT_EVIDENCE_CARD
        )

    @property
    def support_kind(self) -> SupportKind:
        return (
            SupportKind.DIRECT_EVIDENCE_PRESENT
            if self.direct_evidence_card_ids
            else SupportKind.SOCIAL_REPORTS_ONLY
        )

    @model_validator(mode="after")
    def exact_public_structure(self) -> "SourceRelationCalibrationContext":
        material_ids = tuple(item.content_id for item in self.visible_material)
        if len(set(material_ids)) != len(material_ids):
            raise ValueError("source_relation_calibration_duplicate_visible_material")
        if len(self.social_material) != 2:
            raise ValueError("source_relation_calibration_social_material_count_invalid")
        if tuple(item.content_id for item in self.public_root_assignments) != self.social_content_ids:
            raise ValueError("source_relation_calibration_assignment_order_mismatch")
        expected_evidence_ids = (
            (_EVIDENCE_CARD_ID,)
            if self.evidence_state is EvidenceState.PRESENT
            else ()
        )
        if self.direct_evidence_card_ids != expected_evidence_ids:
            raise ValueError("source_relation_calibration_evidence_visibility_mismatch")
        expected_root_count = {
            SourceRelation.NOT_SHOWN: 0,
            SourceRelation.SAME_SHOWN_ROOT: 1,
            SourceRelation.DISTINCT_SHOWN_ROOTS: 2,
        }[self.source_relation]
        if self.visible_social_root_count != expected_root_count:
            raise ValueError("source_relation_calibration_root_count_mismatch")
        return self


class SafeSourceRelationCalibrationAudit(BaseModel):
    """Persistable surface if a future authorized runner is added.

    It deliberately excludes summaries, prompt text, response text, root codes,
    headers, provider metadata, credentials, and truth labels.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^source-relation-v1-c\d{2}$")
    scenario_id: Literal[
        "source-relation-v1-s01", "source-relation-v1-s02"
    ]
    source_relation: SourceRelation
    evidence_state: EvidenceState
    presentation_counterbalance: PresentationCounterbalance
    parser_status: Literal["valid"]
    visible_social_message_count: Literal[2]
    visible_social_root_count: Literal[0, 1, 2]
    direct_evidence_card_count: Literal[0, 1]
    support_kind: SupportKind
    source_structure_alone_decides_truth: Literal[False]


class SourceRelationCalibrationConfig(BaseModel):
    """Hash-bound, offline-only scope lock for this calibration gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    calibration_id: Literal["evicon-source-relation-comprehension-calibration-v1"]
    calibration_version: Literal["conformity_source_relation_comprehension_calibration.v1"]
    status: Literal["offline_validation"]
    development_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    network_enabled: Literal[False]
    real_provider_permitted: Literal[False]
    api_key_read_permitted: Literal[False]
    results_write_permitted: Literal[False]
    evaluator_private_truth_available: Literal[False]
    historical_results_excluded: Literal[True]
    historical_protocols_imported: Literal[False]
    protocol_path: Literal[
        "src/evicon/conformity_source_relation_comprehension_calibration_v1.py"
    ]
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_version: Literal["conformity_source_relation_comprehension_calibration.v1"]
    template_version: Literal["conformity_source_relation_comprehension_turn.v1"]
    response_schema_name: Literal["conformity_source_relation_comprehension_response_v1"]
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_count: Literal[2]
    source_relation_count: Literal[3]
    evidence_state_count: Literal[2]
    presentation_counterbalance_count: Literal[2]
    case_count: Literal[24]
    logical_request_cap: Literal[24]
    completion_reservation_cap: Literal[3072]
    max_tokens: Literal[128]
    max_retries: Literal[0]
    timeout_seconds: Literal[5]

    @model_validator(mode="after")
    def fixed_scope(self) -> "SourceRelationCalibrationConfig":
        expected_case_count = (
            self.scenario_count
            * self.source_relation_count
            * self.evidence_state_count
            * self.presentation_counterbalance_count
        )
        if self.case_count != expected_case_count:
            raise ValueError("source_relation_calibration_case_count_mismatch")
        if self.logical_request_cap != self.case_count:
            raise ValueError("source_relation_calibration_request_cap_mismatch")
        if self.completion_reservation_cap != self.logical_request_cap * self.max_tokens:
            raise ValueError("source_relation_calibration_reservation_cap_mismatch")
        return self


_SCENARIOS: Final[tuple[CalibrationScenario, CalibrationScenario]] = (
    CalibrationScenario(
        scenario_id="source-relation-v1-s01",
        message_a_summary=(
            "A fictional HarborLink community update says a late Friday ferry trip may be added."
        ),
        message_b_summary=(
            "A fictional HarborLink travel notice reports a possible additional ferry trip on Friday evening."
        ),
        evidence_card_summary=(
            "A fictional HarborLink operating record lists a late Friday ferry trip for the current timetable review."
        ),
    ),
    CalibrationScenario(
        scenario_id="source-relation-v1-s02",
        message_a_summary=(
            "A fictional CedarWorks library update says Saturday opening hours may be extended."
        ),
        message_b_summary=(
            "A fictional CedarWorks neighborhood notice reports possible extra opening hours this Saturday."
        ),
        evidence_card_summary=(
            "A fictional CedarWorks scheduling record lists extended Saturday opening hours for the current review."
        ),
    ),
)


class FakeResponsePattern(str, Enum):
    CORRECT = "correct"
    ORDER_SHORTCUT = "order_shortcut"
    EVIDENCE_CONFUSION = "evidence_confusion"
    TRUTH_INFERENCE = "truth_inference"


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def load_config(
    path: str | Path = DEFAULT_CONFIG,
) -> SourceRelationCalibrationConfig:
    resolved = _resolve(path)
    if not resolved.is_file():
        raise SourceRelationCalibrationError("source_relation_calibration_config_missing")
    try:
        config = SourceRelationCalibrationConfig.model_validate(
            tomllib.loads(resolved.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise SourceRelationCalibrationError("source_relation_calibration_config_invalid") from exc
    if sha256_file(config.protocol_path) != config.protocol_sha256:
        raise SourceRelationCalibrationError("source_relation_calibration_protocol_hash_mismatch")
    if config.response_schema_sha256 != RESPONSE_SCHEMA_SHA256:
        raise SourceRelationCalibrationError("source_relation_calibration_schema_hash_mismatch")
    return config


def _visible_material(
    scenario: CalibrationScenario,
    evidence_state: EvidenceState,
    counterbalance: PresentationCounterbalance,
) -> tuple[VisibleMaterial, ...]:
    messages = {
        "message-a": VisibleMaterial(
            content_id="message-a",
            material_type=MaterialType.SOCIAL_REPORT,
            public_summary=scenario.message_a_summary,
        ),
        "message-b": VisibleMaterial(
            content_id="message-b",
            material_type=MaterialType.SOCIAL_REPORT,
            public_summary=scenario.message_b_summary,
        ),
    }
    order = (
        ("message-a", "message-b")
        if counterbalance is PresentationCounterbalance.FORWARD
        else ("message-b", "message-a")
    )
    visible: list[VisibleMaterial] = [messages[item] for item in order]
    if evidence_state is EvidenceState.PRESENT:
        visible.append(VisibleMaterial(
            content_id=_EVIDENCE_CARD_ID,
            material_type=MaterialType.DIRECT_EVIDENCE_CARD,
            public_summary=scenario.evidence_card_summary,
        ))
    return tuple(visible)


def _public_root_assignments(
    source_relation: SourceRelation,
    counterbalance: PresentationCounterbalance,
    social_content_ids: tuple[str, str],
) -> tuple[PublicRootAssignment, PublicRootAssignment]:
    if source_relation is SourceRelation.NOT_SHOWN:
        roots: dict[str, str | None] = {"message-a": None, "message-b": None}
    elif source_relation is SourceRelation.SAME_SHOWN_ROOT:
        root = (
            _ROOT_A
            if counterbalance is PresentationCounterbalance.FORWARD
            else _ROOT_B
        )
        roots = {"message-a": root, "message-b": root}
    else:
        roots = (
            {"message-a": _ROOT_A, "message-b": _ROOT_B}
            if counterbalance is PresentationCounterbalance.FORWARD
            else {"message-a": _ROOT_B, "message-b": _ROOT_A}
        )
    return tuple(
        PublicRootAssignment(
            content_id=content_id,  # type: ignore[arg-type]
            source_root_id=roots[content_id],
        )
        for content_id in social_content_ids
    )  # type: ignore[return-value]


def build_cases(
    path: str | Path = DEFAULT_CONFIG,
) -> tuple[SourceRelationCalibrationContext, ...]:
    """Build the 2 x 3 x 2 x 2 public structural-reading matrix."""

    config = load_config(path)
    cases: list[SourceRelationCalibrationContext] = []
    case_number = 0
    for scenario in _SCENARIOS:
        for evidence_state in EvidenceState:
            for counterbalance in PresentationCounterbalance:
                visible_material = _visible_material(scenario, evidence_state, counterbalance)
                social_ids = tuple(
                    item.content_id
                    for item in visible_material
                    if item.material_type is MaterialType.SOCIAL_REPORT
                )
                for source_relation in SourceRelation:
                    case_number += 1
                    cases.append(SourceRelationCalibrationContext(
                        case_id=f"source-relation-v1-c{case_number:02d}",
                        scenario_id=scenario.scenario_id,
                        source_relation=source_relation,
                        evidence_state=evidence_state,
                        presentation_counterbalance=counterbalance,
                        visible_material=visible_material,
                        public_root_assignments=_public_root_assignments(
                            source_relation,
                            counterbalance,
                            social_ids,  # type: ignore[arg-type]
                        ),
                        max_tokens=config.max_tokens,
                    ))
    validate_case_matrix(cases, config)
    return tuple(cases)


def _public_payload(context: SourceRelationCalibrationContext) -> dict[str, object]:
    return {
        "template_version": TEMPLATE_VERSION,
        "visible_material": [item.model_dump(mode="json") for item in context.visible_material],
        "public_root_relations": [
            item.model_dump(mode="json") for item in context.public_root_assignments
        ],
        "task_scope": "Classify only visible source relations and visible material types.",
    }


SYSTEM_PROMPT: Final[str] = (
    "Return exactly one JSON object with these seven fields and no others: "
    "visible_social_message_count, source_relation, visible_social_root_count, "
    "social_root_assignments, direct_evidence_card_ids, support_kind, and "
    "source_structure_alone_decides_truth. Use only the supplied visible public "
    "material. Count only social_report items as social messages. For their public "
    "source_root_id values: two null values mean source_relation=not_shown and "
    "visible_social_root_count=0; equal non-null values mean "
    "source_relation=same_shown_root and visible_social_root_count=1; distinct "
    "non-null values mean source_relation=distinct_shown_roots and "
    "visible_social_root_count=2. Reproduce the supplied social root assignments "
    "in the visible social-message order. Count only directly visible "
    "direct_evidence_card items in direct_evidence_card_ids. Two social roots are "
    "not a direct evidence card. Source structure and material type do not decide "
    "truth or reliability, so source_structure_alone_decides_truth must be false. "
    "Do not infer hidden roots, truth, reliability, source quality, or private information."
)


def render_request(context: SourceRelationCalibrationContext) -> LLMRequest:
    """Render only public material; construction coordinates stay outside prompts."""

    user_prompt = json.dumps(
        _public_payload(context), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(
        f"{context.case_id}|{context.seed}|{user_prompt}".encode("utf-8")
    ).hexdigest()[:24]
    return LLMRequest(
        request_id=f"source-relation-calibration-v1-{digest}",
        model_name=context.model_name,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=context.temperature,
        max_tokens=context.max_tokens,
        seed=context.seed,
        metadata={"protocol_version": PROTOCOL_VERSION, "template_version": TEMPLATE_VERSION},
    )


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate_field")
        payload[key] = value
    return payload


def parse_response(
    raw: str,
    context: SourceRelationCalibrationContext,
) -> SourceRelationComprehensionResponse:
    """Parse and verify every structural answer against the public context."""

    try:
        payload = json.loads(raw, object_pairs_hook=_unique_pairs)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("malformed_json") from exc
    if not isinstance(payload, dict):
        raise ValueError("top_level_type")
    try:
        response = SourceRelationComprehensionResponse.model_validate(payload)
    except Exception as exc:
        raise ValueError("invalid_schema") from exc

    expected_ids = context.social_content_ids
    actual_ids = tuple(item.content_id for item in response.social_root_assignments)
    if any(content_id not in expected_ids for content_id in actual_ids):
        raise ValueError("unavailable_content_id")
    if actual_ids != expected_ids:
        raise ValueError("root_assignment_order_mismatch")
    if response.social_root_assignments != context.public_root_assignments:
        raise ValueError("root_assignment_mismatch")
    if response.visible_social_message_count != len(context.social_material):
        raise ValueError("social_message_count_mismatch")
    if response.visible_social_root_count != context.visible_social_root_count:
        raise ValueError("root_count_mismatch")
    if response.source_relation is not context.source_relation:
        raise ValueError("source_relation_mismatch")
    if response.direct_evidence_card_ids != context.direct_evidence_card_ids:
        raise ValueError("direct_evidence_card_mismatch")
    if response.support_kind is not context.support_kind:
        raise ValueError("support_kind_mismatch")
    if response.source_structure_alone_decides_truth is not False:
        raise ValueError("source_structure_truth_inference")
    return response


def safe_case_audit(
    context: SourceRelationCalibrationContext,
    response: SourceRelationComprehensionResponse,
) -> SafeSourceRelationCalibrationAudit:
    """Convert a valid response to a text-free, root-code-free audit record."""

    return SafeSourceRelationCalibrationAudit(
        case_id=context.case_id,
        scenario_id=context.scenario_id,
        source_relation=context.source_relation,
        evidence_state=context.evidence_state,
        presentation_counterbalance=context.presentation_counterbalance,
        parser_status="valid",
        visible_social_message_count=response.visible_social_message_count,
        visible_social_root_count=response.visible_social_root_count,
        direct_evidence_card_count=len(response.direct_evidence_card_ids),
        support_kind=response.support_kind,
        source_structure_alone_decides_truth=response.source_structure_alone_decides_truth,
    )


def validate_case_matrix(
    cases: Sequence[SourceRelationCalibrationContext],
    config: SourceRelationCalibrationConfig | None = None,
) -> None:
    """Verify text matching, construct separation, and counterbalancing."""

    expected_case_count = config.case_count if config is not None else 24
    if len(cases) != expected_case_count or len({item.case_id for item in cases}) != len(cases):
        raise SourceRelationCalibrationError("source_relation_calibration_case_identity_mismatch")

    grouped: dict[tuple[object, ...], list[SourceRelationCalibrationContext]] = {}
    for context in cases:
        key = (
            context.scenario_id,
            context.evidence_state,
            context.presentation_counterbalance,
        )
        grouped.setdefault(key, []).append(context)
    for group in grouped.values():
        if {item.source_relation for item in group} != set(SourceRelation):
            raise SourceRelationCalibrationError(
                "source_relation_calibration_projection_coverage_mismatch"
            )
        reference_material = tuple(item.model_dump(mode="json") for item in group[0].visible_material)
        if any(
            tuple(item.model_dump(mode="json") for item in context.visible_material)
            != reference_material
            for context in group[1:]
        ):
            raise SourceRelationCalibrationError(
                "source_relation_calibration_text_matching_mismatch"
            )
        by_relation = {item.source_relation: item for item in group}
        if any(
            assignment.source_root_id is not None
            for assignment in by_relation[SourceRelation.NOT_SHOWN].public_root_assignments
        ):
            raise SourceRelationCalibrationError(
                "source_relation_calibration_hidden_root_exposed"
            )
        if by_relation[SourceRelation.SAME_SHOWN_ROOT].visible_social_root_count != 1:
            raise SourceRelationCalibrationError(
                "source_relation_calibration_same_root_count_invalid"
            )
        if by_relation[SourceRelation.DISTINCT_SHOWN_ROOTS].visible_social_root_count != 2:
            raise SourceRelationCalibrationError(
                "source_relation_calibration_distinct_root_count_invalid"
            )

    paired: dict[tuple[object, ...], list[SourceRelationCalibrationContext]] = {}
    for context in cases:
        key = (context.scenario_id, context.evidence_state, context.source_relation)
        paired.setdefault(key, []).append(context)
    for pair in paired.values():
        if {item.presentation_counterbalance for item in pair} != set(PresentationCounterbalance):
            raise SourceRelationCalibrationError(
                "source_relation_calibration_counterbalance_coverage_mismatch"
            )
        by_counterbalance = {item.presentation_counterbalance: item for item in pair}
        forward = by_counterbalance[PresentationCounterbalance.FORWARD]
        reversed_case = by_counterbalance[PresentationCounterbalance.REVERSED]
        if tuple(item.content_id for item in forward.social_material) == tuple(
            item.content_id for item in reversed_case.social_material
        ):
            raise SourceRelationCalibrationError(
                "source_relation_calibration_message_order_not_counterbalanced"
            )
        forward_text = {
            item.content_id: item.public_summary for item in forward.visible_material
        }
        reversed_text = {
            item.content_id: item.public_summary for item in reversed_case.visible_material
        }
        if forward_text != reversed_text:
            raise SourceRelationCalibrationError(
                "source_relation_calibration_counterbalance_text_mismatch"
            )
        if (
            forward.source_relation is SourceRelation.SAME_SHOWN_ROOT
            and forward.public_root_assignments[0].source_root_id
            == reversed_case.public_root_assignments[0].source_root_id
        ):
            raise SourceRelationCalibrationError(
                "source_relation_calibration_root_label_not_counterbalanced"
            )
        if forward.source_relation is SourceRelation.DISTINCT_SHOWN_ROOTS:
            forward_map = {
                item.content_id: item.source_root_id for item in forward.public_root_assignments
            }
            reversed_map = {
                item.content_id: item.source_root_id
                for item in reversed_case.public_root_assignments
            }
            if forward_map == reversed_map:
                raise SourceRelationCalibrationError(
                    "source_relation_calibration_distinct_labels_not_counterbalanced"
                )

    for context in cases:
        request = render_request(context)
        prompt = request.system_prompt + request.user_prompt
        hidden_coordinates = (
            context.case_id,
            context.scenario_id,
            context.evidence_state.value,
            context.presentation_counterbalance.value,
            str(context.seed),
            "latent",
            "evaluator_private",
            "ground_truth",
        )
        if any(item in prompt for item in hidden_coordinates):
            raise SourceRelationCalibrationError(
                "source_relation_calibration_internal_coordinate_leaked"
            )
        if set(request.metadata) != {"protocol_version", "template_version"}:
            raise SourceRelationCalibrationError(
                "source_relation_calibration_request_metadata_invalid"
            )


def _response_payload_from_public_request(request: LLMRequest) -> dict[str, object]:
    """Derive a fake structural response from public request fields only."""

    payload = json.loads(request.user_prompt)
    material = payload["visible_material"]
    relations = payload["public_root_relations"]
    social_ids = [
        item["content_id"]
        for item in material
        if item["material_type"] == MaterialType.SOCIAL_REPORT.value
    ]
    roots = [item["source_root_id"] for item in relations]
    distinct_roots = {root for root in roots if root is not None}
    relation = {
        0: SourceRelation.NOT_SHOWN.value,
        1: SourceRelation.SAME_SHOWN_ROOT.value,
        2: SourceRelation.DISTINCT_SHOWN_ROOTS.value,
    }[len(distinct_roots)]
    evidence_ids = [
        item["content_id"]
        for item in material
        if item["material_type"] == MaterialType.DIRECT_EVIDENCE_CARD.value
    ]
    return {
        "visible_social_message_count": len(social_ids),
        "source_relation": relation,
        "visible_social_root_count": len(distinct_roots),
        "social_root_assignments": relations,
        "direct_evidence_card_ids": evidence_ids,
        "support_kind": (
            SupportKind.DIRECT_EVIDENCE_PRESENT.value
            if evidence_ids
            else SupportKind.SOCIAL_REPORTS_ONLY.value
        ),
        "source_structure_alone_decides_truth": False,
    }


class SourceRelationCalibrationFakeProvider:
    """Scripted public-request responder used only by the offline smoke."""

    def __init__(self, pattern: FakeResponsePattern) -> None:
        self.pattern = pattern
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        payload = _response_payload_from_public_request(request)
        if self.pattern is FakeResponsePattern.ORDER_SHORTCUT:
            payload["social_root_assignments"] = sorted(
                payload["social_root_assignments"], key=lambda item: item["content_id"]
            )
        elif self.pattern is FakeResponsePattern.EVIDENCE_CONFUSION:
            payload["direct_evidence_card_ids"] = [_EVIDENCE_CARD_ID]
            payload["support_kind"] = SupportKind.DIRECT_EVIDENCE_PRESENT.value
        elif self.pattern is FakeResponsePattern.TRUTH_INFERENCE:
            payload["source_structure_alone_decides_truth"] = True
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content=json.dumps(payload, ensure_ascii=True, sort_keys=True),
            finish_reason="stop",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            latency_ms=0.0,
        )


def run_offline_smoke(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    """Exercise a correct and an order-shortcut fake without writing any files."""

    config = load_config(path)
    cases = build_cases(path)
    correct_provider = SourceRelationCalibrationFakeProvider(FakeResponsePattern.CORRECT)
    audits = []
    request_ids: set[str] = set()
    for context in cases:
        request = render_request(context)
        request_ids.add(request.request_id)
        response = correct_provider.complete(request)
        parsed = parse_response(response.content, context)
        audits.append(safe_case_audit(context, parsed))

    shortcut_provider = SourceRelationCalibrationFakeProvider(
        FakeResponsePattern.ORDER_SHORTCUT
    )
    shortcut_rejections = 0
    for context in cases:
        request = render_request(context)
        response = shortcut_provider.complete(request)
        try:
            parse_response(response.content, context)
        except ValueError as exc:
            if str(exc) != "root_assignment_order_mismatch":
                raise SourceRelationCalibrationError(
                    "source_relation_calibration_shortcut_wrong_failure"
                ) from exc
            shortcut_rejections += 1

    expected_shortcut_rejections = sum(
        context.presentation_counterbalance is PresentationCounterbalance.REVERSED
        for context in cases
    )
    if shortcut_rejections != expected_shortcut_rejections:
        raise SourceRelationCalibrationError(
            "source_relation_calibration_shortcut_not_detected"
        )
    relation_counts = Counter(item.source_relation.value for item in audits)
    evidence_counts = Counter(item.direct_evidence_card_count for item in audits)
    counterbalance_counts = Counter(item.presentation_counterbalance.value for item in audits)
    return {
        "status": "source_relation_comprehension_calibration_offline_smoke_passed",
        "calibration_id": config.calibration_id,
        "case_count": len(cases),
        "correct_fake_request_count": correct_provider.calls,
        "order_shortcut_request_count": shortcut_provider.calls,
        "unique_request_count": len(request_ids),
        "safe_case_audit_count": len(audits),
        "source_relation_case_counts": dict(sorted(relation_counts.items())),
        "direct_evidence_card_count_cases": {
            str(key): value for key, value in sorted(evidence_counts.items())
        },
        "presentation_counterbalance_case_counts": dict(sorted(counterbalance_counts.items())),
        "order_shortcut_rejection_count": shortcut_rejections,
        "measurement_contract_exercised": True,
        "safety": {
            "network": "disabled",
            "real_provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "prompt_saved": False,
            "full_response_saved": False,
            "historical_results_used": False,
            "evaluator_private_truth_exposed": False,
            "behavior_effect_estimated": False,
            "not_paper_result": True,
            "no_causal_conclusion": True,
        },
    }


def safe_preflight(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    """Validate the static gate without reading environment variables or writing."""

    try:
        config = load_config(path)
        cases = build_cases(path)
        return {
            "status": "source_relation_comprehension_calibration_ready",
            "ready_for_network": False,
            "blocking_reasons": [
                "offline_calibration_only",
                "new_human_authorization_required",
                "behavior_protocol_not_implemented",
            ],
            "calibration_id": config.calibration_id,
            "case_count": len(cases),
            "logical_request_cap": config.logical_request_cap,
            "completion_reservation_cap": config.completion_reservation_cap,
            "network": "disabled",
            "real_provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "evaluator_private_truth_exposed": False,
            "behavior_effect_estimated": False,
            "not_paper_result": True,
            "no_causal_conclusion": True,
        }
    except Exception as exc:
        return {
            "status": "blocked",
            "blocking_reasons": [
                getattr(exc, "code", "source_relation_calibration_preflight_failed")
            ],
            "ready_for_network": False,
            "network": "disabled",
            "real_provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "evaluator_private_truth_exposed": False,
        }


def protocol_self_check(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    """Expose only static, non-sensitive structural checks for tests and docs."""

    contexts = build_cases(path)
    requests = tuple(render_request(context) for context in contexts)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "template_version": TEMPLATE_VERSION,
        "schema_name": SCHEMA_NAME,
        "schema_sha256": RESPONSE_SCHEMA_SHA256,
        "case_count": len(contexts),
        "all_request_ids_unique": len({item.request_id for item in requests}) == len(requests),
        "source_structure_does_not_decide_truth_rule": (
            "source_structure_alone_decides_truth must be false" in SYSTEM_PROMPT
        ),
        "direct_evidence_is_distinct_from_social_roots_rule": (
            "Two social roots are not a direct evidence card" in SYSTEM_PROMPT
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run offline source-relation comprehension calibration checks"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--offline-smoke",
        action="store_true",
        help="Exercise only deterministic fake responses; never uses a provider or network.",
    )
    arguments = parser.parse_args(argv)
    report = (
        run_offline_smoke(arguments.config)
        if arguments.offline_smoke
        else safe_preflight(arguments.config)
    )
    print(safe_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CALIBRATION_ID",
    "DEFAULT_CONFIG",
    "EvidenceState",
    "FakeResponsePattern",
    "MaterialType",
    "PresentationCounterbalance",
    "PROTOCOL_VERSION",
    "PublicRootAssignment",
    "RESPONSE_SCHEMA",
    "RESPONSE_SCHEMA_SHA256",
    "SCHEMA_NAME",
    "SYSTEM_PROMPT",
    "SafeSourceRelationCalibrationAudit",
    "SourceRelation",
    "SourceRelationCalibrationConfig",
    "SourceRelationCalibrationContext",
    "SourceRelationCalibrationError",
    "SourceRelationCalibrationFakeProvider",
    "SourceRelationComprehensionResponse",
    "SupportKind",
    "TEMPLATE_VERSION",
    "build_cases",
    "load_config",
    "parse_response",
    "protocol_self_check",
    "render_request",
    "run_offline_smoke",
    "safe_case_audit",
    "safe_preflight",
    "validate_case_matrix",
]
