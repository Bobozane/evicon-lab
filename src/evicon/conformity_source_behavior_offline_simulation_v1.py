"""Offline-only factorized source-behavior simulation for the new research route.

This module deliberately does not execute a model or a network experiment.  It
checks the new eight-condition design with scripted fake responses, including
the hidden-source placebo conditions that the frozen v3 qualification protocol
does not contain.  No API key, provider configuration, historic result, or
evaluator-private truth is read on this path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_identification import (
    OrdinalJudgment,
    PublicAction,
    ShareDecision,
    safe_json,
    sha256_file,
)
from .conformity_source_manipulation import RootAssignment
from .llm_contract import LLMRequest, LLMResponse


_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG: Final[str] = (
    "configs/provenance_cascade/identification/"
    "conformity_source_behavior_offline_simulation.v1.toml"
)
PROTOCOL_VERSION: Final[str] = "conformity_source_behavior_offline_simulation.v1"
TEMPLATE_VERSION: Final[str] = "conformity_source_behavior_factorized_turn.v1"
SCHEMA_NAME: Final[str] = "conformity_source_behavior_factorized_response_v1"
_SEEDS: Final[tuple[int, int, int]] = (20261301, 20261302, 20261303)
_INITIAL_ID: Final[str] = "content-initial"
_REPEAT_A_ID: Final[str] = "content-repeat-a"
_REPEAT_B_ID: Final[str] = "content-repeat-b"
_EVIDENCE_ID: Final[str] = "content-evidence"
_EVIDENCE_CARD_ID: Final[str] = "evidence-card-01"


class OfflineSimulationError(ValueError):
    """Stable, non-sensitive error boundary for the offline simulator."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class FactorCondition(str, Enum):
    PRIVATE_INITIAL = "private_initial"
    SELF_REFLECTION = "self_reflection"
    SOURCE_FREE_REPETITION = "source_free_repetition"
    SAME_ROOT_HIDDEN = "same_root_hidden"
    SAME_ROOT_SHOWN = "same_root_shown"
    INDEPENDENT_ROOTS_HIDDEN = "independent_roots_hidden"
    INDEPENDENT_ROOTS_SHOWN = "independent_roots_shown"
    EXTERNAL_EVIDENCE = "external_evidence"


class FactorContrastId(str, Enum):
    NATURAL_INSTABILITY = "self_reflection_minus_private_initial"
    TEXT_REPETITION = "source_free_repetition_minus_self_reflection"
    SAME_ROOT_DISCLOSURE = "same_root_shown_minus_same_root_hidden"
    INDEPENDENT_ROOT_DISCLOSURE = "independent_roots_shown_minus_independent_roots_hidden"
    HIDDEN_SOURCE_PLACEBO = "independent_roots_hidden_minus_same_root_hidden"
    ROOT_INDEPENDENCE_INCREMENT = "independent_roots_shown_minus_same_root_shown"
    EVIDENCE_INCREMENT = "external_evidence_minus_self_reflection"


class ContentRole(str, Enum):
    INITIAL = "initial"
    SOCIAL_REPEAT = "social_repeat"
    EXTERNAL_EVIDENCE = "external_evidence"


class FakeResponsePattern(str, Enum):
    KNOWN_DIFFERENCE = "known_difference"
    CONSTANT = "constant"


class FactorScenario(BaseModel):
    """Public, author-written material used solely in the offline simulator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(pattern=r"^source-factor-v1-s0[1-4]$")
    target_claim: str = Field(min_length=24, max_length=240)
    decision_task: str = Field(min_length=24, max_length=280)
    initial_summary: str = Field(min_length=48, max_length=360)
    repeat_a_summary: str = Field(min_length=48, max_length=360)
    repeat_b_summary: str = Field(min_length=48, max_length=360)
    evidence_summary: str = Field(min_length=48, max_length=360)

    @model_validator(mode="after")
    def public_material_is_nonleaking(self) -> "FactorScenario":
        material = " ".join(self.model_dump().values()).lower()
        forbidden = (
            "ground_truth",
            "evaluator_private",
            "correct_answer",
            "same_root",
            "independent_roots",
            "wvs",
            "pilot",
        )
        if any(token in material for token in forbidden):
            raise ValueError("factorization_public_material_leaks_internal_label")
        summaries = (
            self.initial_summary,
            self.repeat_a_summary,
            self.repeat_b_summary,
            self.evidence_summary,
        )
        if len(set(summaries)) != len(summaries):
            raise ValueError("factorization_public_summaries_must_be_distinct")
        return self


class VisibleContent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    content_id: str = Field(
        pattern=r"^content-(initial|repeat-a|repeat-b|evidence)$"
    )
    public_summary: str = Field(min_length=48, max_length=360)
    role: ContentRole


class FactorizationResponse(BaseModel):
    """Same public adoption/share surface used by the behavior qualification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ordinal_judgment: OrdinalJudgment
    public_action: PublicAction
    share_decision: ShareDecision
    confidence: int = Field(ge=1, le=7)
    content_ids_used: tuple[str, ...] = Field(min_length=1, max_length=4)
    share_content_id: str | None


RESPONSE_SCHEMA: Final[dict[str, object]] = FactorizationResponse.model_json_schema(
    mode="validation"
)
RESPONSE_SCHEMA_SHA256: Final[str] = hashlib.sha256(
    json.dumps(RESPONSE_SCHEMA, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


class FactorCase(BaseModel):
    """One branch in a shared-T0 factorized design.

    ``latent_focal_root_ids`` exists only to verify the hidden-source placebo
    construction.  It never enters a rendered request or persistable audit.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^source-factor-v1-g\d{2}-b\d{2}$")
    group_id: str = Field(pattern=r"^source-factor-v1-g\d{2}$")
    scenario_id: str = Field(pattern=r"^source-factor-v1-s0[1-4]$")
    seed: Literal[20261301, 20261302, 20261303]
    condition: FactorCondition
    target_claim: str
    decision_task: str
    visible_content: tuple[VisibleContent, ...] = Field(min_length=1, max_length=3)
    public_root_assignments: tuple[RootAssignment, ...] = Field(min_length=1, max_length=3)
    visible_evidence_ids: tuple[str, ...] = Field(max_length=1)
    latent_focal_root_ids: tuple[str | None, str | None]
    reflection_only: bool
    model_name: str = "offline-factorization-fake-model"
    temperature: Literal[0.0] = 0.0
    max_tokens: Literal[256] = 256

    @property
    def visible_content_ids(self) -> tuple[str, ...]:
        return tuple(item.content_id for item in self.visible_content)

    @property
    def visible_focal_root_count(self) -> int:
        focal = {_REPEAT_A_ID, _REPEAT_B_ID}
        return len({
            item.source_root_id
            for item in self.public_root_assignments
            if item.content_id in focal and item.source_root_id is not None
        })

    @model_validator(mode="after")
    def exact_condition_shape(self) -> "FactorCase":
        visible_ids = self.visible_content_ids
        if len(set(visible_ids)) != len(visible_ids):
            raise ValueError("factorization_duplicate_visible_content")
        if tuple(item.content_id for item in self.public_root_assignments) != visible_ids:
            raise ValueError("factorization_public_root_assignment_order_mismatch")
        if any(
            item.source_root_id not in {None, "root-01", "root-02"}
            for item in self.public_root_assignments
        ):
            raise ValueError("factorization_nonopaque_public_root")
        if any(
            item.source_root_id is not None and item.content_id not in {_REPEAT_A_ID, _REPEAT_B_ID}
            for item in self.public_root_assignments
        ):
            raise ValueError("factorization_non_social_content_has_public_root")

        expected_content = {
            FactorCondition.PRIVATE_INITIAL: (_INITIAL_ID,),
            FactorCondition.SELF_REFLECTION: (_INITIAL_ID,),
            FactorCondition.SOURCE_FREE_REPETITION: (_INITIAL_ID, _REPEAT_A_ID, _REPEAT_B_ID),
            FactorCondition.SAME_ROOT_HIDDEN: (_INITIAL_ID, _REPEAT_A_ID, _REPEAT_B_ID),
            FactorCondition.SAME_ROOT_SHOWN: (_INITIAL_ID, _REPEAT_A_ID, _REPEAT_B_ID),
            FactorCondition.INDEPENDENT_ROOTS_HIDDEN: (_INITIAL_ID, _REPEAT_A_ID, _REPEAT_B_ID),
            FactorCondition.INDEPENDENT_ROOTS_SHOWN: (_INITIAL_ID, _REPEAT_A_ID, _REPEAT_B_ID),
            FactorCondition.EXTERNAL_EVIDENCE: (_INITIAL_ID, _EVIDENCE_ID),
        }[self.condition]
        if visible_ids != expected_content:
            raise ValueError("factorization_visible_content_condition_mismatch")

        expected_root_count = {
            FactorCondition.PRIVATE_INITIAL: 0,
            FactorCondition.SELF_REFLECTION: 0,
            FactorCondition.SOURCE_FREE_REPETITION: 0,
            FactorCondition.SAME_ROOT_HIDDEN: 0,
            FactorCondition.SAME_ROOT_SHOWN: 1,
            FactorCondition.INDEPENDENT_ROOTS_HIDDEN: 0,
            FactorCondition.INDEPENDENT_ROOTS_SHOWN: 2,
            FactorCondition.EXTERNAL_EVIDENCE: 0,
        }[self.condition]
        if self.visible_focal_root_count != expected_root_count:
            raise ValueError("factorization_public_root_count_condition_mismatch")

        expected_latent = {
            FactorCondition.SAME_ROOT_HIDDEN: ("root-01", "root-01"),
            FactorCondition.SAME_ROOT_SHOWN: ("root-01", "root-01"),
            FactorCondition.INDEPENDENT_ROOTS_HIDDEN: ("root-01", "root-02"),
            FactorCondition.INDEPENDENT_ROOTS_SHOWN: ("root-01", "root-02"),
        }.get(self.condition, (None, None))
        if self.latent_focal_root_ids != expected_latent:
            raise ValueError("factorization_latent_root_condition_mismatch")
        if self.reflection_only is not (self.condition is FactorCondition.SELF_REFLECTION):
            raise ValueError("factorization_reflection_condition_mismatch")
        expected_evidence = (
            (_EVIDENCE_CARD_ID,) if self.condition is FactorCondition.EXTERNAL_EVIDENCE else ()
        )
        if self.visible_evidence_ids != expected_evidence:
            raise ValueError("factorization_evidence_condition_mismatch")
        return self


class SafeFactorObservation(BaseModel):
    """Safe audit surface: no material text, prompt, raw response, or truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^source-factor-v1-g\d{2}-b\d{2}$")
    group_id: str = Field(pattern=r"^source-factor-v1-g\d{2}$")
    scenario_id: str = Field(pattern=r"^source-factor-v1-s0[1-4]$")
    seed: Literal[20261301, 20261302, 20261303]
    condition: FactorCondition
    parser_status: Literal["valid"]
    ordinal_judgment: OrdinalJudgment
    public_action: PublicAction
    share_decision: ShareDecision
    confidence: int = Field(ge=1, le=7)
    used_content_count: int = Field(ge=1, le=3)
    visible_focal_root_count: int = Field(ge=0, le=2)
    visible_evidence_count: int = Field(ge=0, le=1)
    shared_t0_observation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ContrastSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contrast_id: FactorContrastId
    minuend: FactorCondition
    subtrahend: FactorCondition
    paired_group_count: int = Field(ge=0)
    positive_difference_count: int = Field(ge=0)
    zero_difference_count: int = Field(ge=0)
    negative_difference_count: int = Field(ge=0)
    mean_ordinal_difference: float


class OfflineSimulationConfig(BaseModel):
    """Versioned safety and scope lock for this test-only simulator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    simulation_id: Literal["evicon-conformity-source-behavior-offline-simulation-v1"]
    simulation_version: Literal["conformity_source_behavior_offline_simulation.v1"]
    status: Literal["offline_design_validation"]
    development_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    network_enabled: Literal[False]
    real_provider_permitted: Literal[False]
    api_key_read_permitted: Literal[False]
    results_write_permitted: Literal[False]
    evaluator_private_truth_available: Literal[False]
    historical_results_excluded: Literal[True]
    protocol_path: Literal["src/evicon/conformity_source_behavior_offline_simulation_v1.py"]
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_version: Literal["conformity_source_behavior_offline_simulation.v1"]
    template_version: Literal["conformity_source_behavior_factorized_turn.v1"]
    response_schema_name: Literal["conformity_source_behavior_factorized_response_v1"]
    conditions: tuple[FactorCondition, ...]
    seeds: tuple[Literal[20261301], Literal[20261302], Literal[20261303]]
    scenario_count: Literal[4]
    shared_t0_group_count: Literal[12]
    case_count: Literal[96]
    fake_pattern_count: Literal[2]
    fake_logical_request_cap: Literal[192]
    max_tokens: Literal[256]
    output_root: Literal["outputs/conformity-source-behavior-offline-simulation-v1"]

    @model_validator(mode="after")
    def fixed_scope(self) -> "OfflineSimulationConfig":
        if self.conditions != tuple(FactorCondition):
            raise ValueError("factorization_condition_order_mismatch")
        if self.seeds != _SEEDS:
            raise ValueError("factorization_seed_set_mismatch")
        expected_groups = self.scenario_count * len(self.seeds)
        expected_cases = expected_groups * len(self.conditions)
        if self.shared_t0_group_count != expected_groups or self.case_count != expected_cases:
            raise ValueError("factorization_case_count_mismatch")
        if self.fake_logical_request_cap != self.case_count * self.fake_pattern_count:
            raise ValueError("factorization_fake_request_cap_mismatch")
        return self


_SCENARIOS: Final[tuple[FactorScenario, FactorScenario, FactorScenario, FactorScenario]] = (
    FactorScenario(
        scenario_id="source-factor-v1-s01",
        target_claim="The fictional HarborLink ferry will add one late trip on Friday.",
        decision_task="Decide whether the claim is reliable enough for a reversible public travel notice.",
        initial_summary="A public service bulletin reports that HarborLink may add one late ferry trip on Friday.",
        repeat_a_summary="A community travel update says HarborLink is expected to run one late Friday ferry trip.",
        repeat_b_summary="A second public notice repeats that a late Friday HarborLink ferry trip is planned.",
        evidence_summary="A fictional operations record lists one late Friday HarborLink trip in the working timetable.",
    ),
    FactorScenario(
        scenario_id="source-factor-v1-s02",
        target_claim="The fictional CedarWorks library will extend its Saturday opening hours.",
        decision_task="Decide whether the claim is reliable enough for a reversible neighborhood access notice.",
        initial_summary="A public library update reports that CedarWorks may extend its opening hours this Saturday.",
        repeat_a_summary="A community calendar says CedarWorks is expected to remain open for extra Saturday hours.",
        repeat_b_summary="A second public update repeats that CedarWorks plans longer opening hours on Saturday.",
        evidence_summary="A fictional scheduling record lists the extended Saturday CedarWorks opening period.",
    ),
    FactorScenario(
        scenario_id="source-factor-v1-s03",
        target_claim="The fictional Northstar clinic will open an additional evening appointment block.",
        decision_task="Decide whether the claim is reliable enough for a reversible appointment planning notice.",
        initial_summary="A public clinic message reports that Northstar may add an evening appointment block next week.",
        repeat_a_summary="A neighborhood health update says Northstar is expected to offer additional evening appointments.",
        repeat_b_summary="A second public notice repeats that Northstar plans an extra evening appointment block.",
        evidence_summary="A fictional clinic schedule records an additional Northstar evening appointment block.",
    ),
    FactorScenario(
        scenario_id="source-factor-v1-s04",
        target_claim="The fictional Meadowline workshop will move to the east hall.",
        decision_task="Decide whether the claim is reliable enough for a reversible attendee planning notice.",
        initial_summary="A public workshop message reports that the Meadowline event may move to the east hall.",
        repeat_a_summary="An attendee update says Meadowline is expected to use the east hall for the workshop.",
        repeat_b_summary="A second public note repeats that the Meadowline workshop plans to use the east hall.",
        evidence_summary="A fictional venue allocation record lists the Meadowline workshop in the east hall.",
    ),
)

_CONTRASTS: Final[tuple[tuple[FactorContrastId, FactorCondition, FactorCondition], ...]] = (
    (
        FactorContrastId.NATURAL_INSTABILITY,
        FactorCondition.SELF_REFLECTION,
        FactorCondition.PRIVATE_INITIAL,
    ),
    (
        FactorContrastId.TEXT_REPETITION,
        FactorCondition.SOURCE_FREE_REPETITION,
        FactorCondition.SELF_REFLECTION,
    ),
    (
        FactorContrastId.SAME_ROOT_DISCLOSURE,
        FactorCondition.SAME_ROOT_SHOWN,
        FactorCondition.SAME_ROOT_HIDDEN,
    ),
    (
        FactorContrastId.INDEPENDENT_ROOT_DISCLOSURE,
        FactorCondition.INDEPENDENT_ROOTS_SHOWN,
        FactorCondition.INDEPENDENT_ROOTS_HIDDEN,
    ),
    (
        FactorContrastId.HIDDEN_SOURCE_PLACEBO,
        FactorCondition.INDEPENDENT_ROOTS_HIDDEN,
        FactorCondition.SAME_ROOT_HIDDEN,
    ),
    (
        FactorContrastId.ROOT_INDEPENDENCE_INCREMENT,
        FactorCondition.INDEPENDENT_ROOTS_SHOWN,
        FactorCondition.SAME_ROOT_SHOWN,
    ),
    (
        FactorContrastId.EVIDENCE_INCREMENT,
        FactorCondition.EXTERNAL_EVIDENCE,
        FactorCondition.SELF_REFLECTION,
    ),
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def load_config(path: str | Path = DEFAULT_CONFIG) -> OfflineSimulationConfig:
    try:
        config = OfflineSimulationConfig.model_validate(
            tomllib.loads(_resolve(path).read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise OfflineSimulationError("offline_simulation_config_invalid") from exc
    if sha256_file(config.protocol_path) != config.protocol_sha256:
        raise OfflineSimulationError("offline_simulation_protocol_hash_mismatch")
    return config


def _content_for_condition(
    scenario: FactorScenario,
    condition: FactorCondition,
) -> tuple[VisibleContent, ...]:
    all_content = {
        _INITIAL_ID: VisibleContent(
            content_id=_INITIAL_ID,
            public_summary=scenario.initial_summary,
            role=ContentRole.INITIAL,
        ),
        _REPEAT_A_ID: VisibleContent(
            content_id=_REPEAT_A_ID,
            public_summary=scenario.repeat_a_summary,
            role=ContentRole.SOCIAL_REPEAT,
        ),
        _REPEAT_B_ID: VisibleContent(
            content_id=_REPEAT_B_ID,
            public_summary=scenario.repeat_b_summary,
            role=ContentRole.SOCIAL_REPEAT,
        ),
        _EVIDENCE_ID: VisibleContent(
            content_id=_EVIDENCE_ID,
            public_summary=scenario.evidence_summary,
            role=ContentRole.EXTERNAL_EVIDENCE,
        ),
    }
    ids = {
        FactorCondition.PRIVATE_INITIAL: (_INITIAL_ID,),
        FactorCondition.SELF_REFLECTION: (_INITIAL_ID,),
        FactorCondition.SOURCE_FREE_REPETITION: (_INITIAL_ID, _REPEAT_A_ID, _REPEAT_B_ID),
        FactorCondition.SAME_ROOT_HIDDEN: (_INITIAL_ID, _REPEAT_A_ID, _REPEAT_B_ID),
        FactorCondition.SAME_ROOT_SHOWN: (_INITIAL_ID, _REPEAT_A_ID, _REPEAT_B_ID),
        FactorCondition.INDEPENDENT_ROOTS_HIDDEN: (_INITIAL_ID, _REPEAT_A_ID, _REPEAT_B_ID),
        FactorCondition.INDEPENDENT_ROOTS_SHOWN: (_INITIAL_ID, _REPEAT_A_ID, _REPEAT_B_ID),
        FactorCondition.EXTERNAL_EVIDENCE: (_INITIAL_ID, _EVIDENCE_ID),
    }[condition]
    return tuple(all_content[content_id] for content_id in ids)


def _public_roots(
    condition: FactorCondition,
    visible_content: tuple[VisibleContent, ...],
) -> tuple[RootAssignment, ...]:
    roots = {
        FactorCondition.SAME_ROOT_SHOWN: {
            _REPEAT_A_ID: "root-01", _REPEAT_B_ID: "root-01",
        },
        FactorCondition.INDEPENDENT_ROOTS_SHOWN: {
            _REPEAT_A_ID: "root-01", _REPEAT_B_ID: "root-02",
        },
    }.get(condition, {})
    return tuple(
        RootAssignment(content_id=item.content_id, source_root_id=roots.get(item.content_id))
        for item in visible_content
    )


def _latent_focal_roots(condition: FactorCondition) -> tuple[str | None, str | None]:
    return {
        FactorCondition.SAME_ROOT_HIDDEN: ("root-01", "root-01"),
        FactorCondition.SAME_ROOT_SHOWN: ("root-01", "root-01"),
        FactorCondition.INDEPENDENT_ROOTS_HIDDEN: ("root-01", "root-02"),
        FactorCondition.INDEPENDENT_ROOTS_SHOWN: ("root-01", "root-02"),
    }.get(condition, (None, None))


def build_cases(path: str | Path = DEFAULT_CONFIG) -> tuple[FactorCase, ...]:
    """Build the 4 scenario x 3 seed x 8 condition offline matrix."""

    config = load_config(path)
    cases: list[FactorCase] = []
    group_number = 0
    for scenario in _SCENARIOS:
        for seed in config.seeds:
            group_number += 1
            group_id = f"source-factor-v1-g{group_number:02d}"
            for branch_number, condition in enumerate(config.conditions, start=1):
                visible_content = _content_for_condition(scenario, condition)
                cases.append(FactorCase(
                    case_id=f"{group_id}-b{branch_number:02d}",
                    group_id=group_id,
                    scenario_id=scenario.scenario_id,
                    seed=seed,
                    condition=condition,
                    target_claim=scenario.target_claim,
                    decision_task=scenario.decision_task,
                    visible_content=visible_content,
                    public_root_assignments=_public_roots(condition, visible_content),
                    visible_evidence_ids=(
                        (_EVIDENCE_CARD_ID,)
                        if condition is FactorCondition.EXTERNAL_EVIDENCE
                        else ()
                    ),
                    latent_focal_root_ids=_latent_focal_roots(condition),
                    reflection_only=condition is FactorCondition.SELF_REFLECTION,
                    max_tokens=config.max_tokens,
                ))
    validate_case_matrix(cases, config)
    return tuple(cases)


def _public_payload(case: FactorCase) -> dict[str, object]:
    """The whole public prompt payload, deliberately excluding all internal labels."""

    return {
        "template_version": TEMPLATE_VERSION,
        "target_claim": case.target_claim,
        "decision_task": case.decision_task,
        "visible_content": [
            {"content_id": item.content_id, "public_summary": item.public_summary}
            for item in case.visible_content
        ],
        "public_root_relations": [
            item.model_dump(mode="json") for item in case.public_root_assignments
        ],
        "visible_evidence_ids": list(case.visible_evidence_ids),
        "reflection_only": case.reflection_only,
    }


def render_factor_request(case: FactorCase) -> LLMRequest:
    """Render a branch without exposing scenario, seed, condition, or latent roots."""

    user_prompt = json.dumps(
        _public_payload(case), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(
        f"{case.case_id}|{case.seed}|{user_prompt}".encode()
    ).hexdigest()[:24]
    return LLMRequest(
        request_id=f"source-factor-v1-{digest}",
        model_name=case.model_name,
        system_prompt=(
            "Return exactly ordinal_judgment, public_action, share_decision, confidence, "
            "content_ids_used, and share_content_id as JSON. Assess the target claim only "
            "from the visible public material. Equal public root IDs mean that messages trace "
            "to one shown root; distinct IDs mean distinct shown roots; null means a root is "
            "not shown. An evidence ID identifies a directly visible evidence card. Do not "
            "invent sources, evidence, truth labels, or hidden information."
        ),
        user_prompt=user_prompt,
        temperature=case.temperature,
        max_tokens=case.max_tokens,
        seed=case.seed,
        metadata={"protocol_version": PROTOCOL_VERSION, "template_version": TEMPLATE_VERSION},
    )


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate_field")
        payload[key] = value
    return payload


def parse_factor_response(raw: str, case: FactorCase) -> FactorizationResponse:
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_pairs)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("malformed_json") from exc
    if not isinstance(payload, dict):
        raise ValueError("top_level_type")
    try:
        response = FactorizationResponse.model_validate(payload)
    except Exception as exc:
        raise ValueError("invalid_schema") from exc
    if len(set(response.content_ids_used)) != len(response.content_ids_used):
        raise ValueError("duplicate_content_id")
    visible = set(case.visible_content_ids)
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


def safe_factor_observation(
    case: FactorCase,
    response: FactorizationResponse,
    shared_t0_observation_sha256: str,
) -> SafeFactorObservation:
    return SafeFactorObservation(
        case_id=case.case_id,
        group_id=case.group_id,
        scenario_id=case.scenario_id,
        seed=case.seed,
        condition=case.condition,
        parser_status="valid",
        ordinal_judgment=response.ordinal_judgment,
        public_action=response.public_action,
        share_decision=response.share_decision,
        confidence=response.confidence,
        used_content_count=len(response.content_ids_used),
        visible_focal_root_count=case.visible_focal_root_count,
        visible_evidence_count=len(case.visible_evidence_ids),
        shared_t0_observation_sha256=shared_t0_observation_sha256,
    )


def validate_case_matrix(
    cases: Sequence[FactorCase],
    config: OfflineSimulationConfig | None = None,
) -> None:
    """Validate orthogonality and the public-prompt placebo construction."""

    expected_case_count = config.case_count if config is not None else 96
    if len(cases) != expected_case_count or len({item.case_id for item in cases}) != len(cases):
        raise OfflineSimulationError("factorization_case_identity_mismatch")
    grouped: dict[str, list[FactorCase]] = {}
    for case in cases:
        grouped.setdefault(case.group_id, []).append(case)
    expected_conditions = set(FactorCondition)
    for group_cases in grouped.values():
        if {item.condition for item in group_cases} != expected_conditions:
            raise OfflineSimulationError("factorization_incomplete_condition_group")
        by_condition = {item.condition: item for item in group_cases}
        hidden_same = by_condition[FactorCondition.SAME_ROOT_HIDDEN]
        hidden_independent = by_condition[FactorCondition.INDEPENDENT_ROOTS_HIDDEN]
        source_free = by_condition[FactorCondition.SOURCE_FREE_REPETITION]
        for pair in ((source_free, hidden_same), (source_free, hidden_independent)):
            first, second = pair
            if _public_payload(first) != _public_payload(second):
                raise OfflineSimulationError("factorization_hidden_source_public_payload_mismatch")
        if hidden_same.latent_focal_root_ids == hidden_independent.latent_focal_root_ids:
            raise OfflineSimulationError("factorization_hidden_source_placebo_not_distinct")
        shown_same = by_condition[FactorCondition.SAME_ROOT_SHOWN]
        shown_independent = by_condition[FactorCondition.INDEPENDENT_ROOTS_SHOWN]
        if (
            tuple(item.public_summary for item in shown_same.visible_content)
            != tuple(item.public_summary for item in shown_independent.visible_content)
        ):
            raise OfflineSimulationError("factorization_shown_source_text_mismatch")
        if shown_same.visible_focal_root_count != 1 or shown_independent.visible_focal_root_count != 2:
            raise OfflineSimulationError("factorization_shown_source_root_mismatch")
        initial = by_condition[FactorCondition.PRIVATE_INITIAL]
        reflection = by_condition[FactorCondition.SELF_REFLECTION]
        if initial.target_claim != reflection.target_claim or initial.decision_task != reflection.decision_task:
            raise OfflineSimulationError("factorization_shared_t0_task_mismatch")
        for case in group_cases:
            request = render_factor_request(case)
            prompt = request.system_prompt + request.user_prompt
            for hidden in (
                case.case_id,
                case.group_id,
                case.scenario_id,
                case.condition.value,
                str(case.seed),
                "latent_focal_root_ids",
            ):
                if hidden in prompt:
                    raise OfflineSimulationError("factorization_internal_coordinate_leaked")


class FactorizationFakeProvider:
    """Scripted, prompt-only provider used to test pipeline identifiability.

    It sees only the rendered public request.  The known-difference pattern is
    intentionally synthetic; it validates analysis wiring rather than claiming
    that a real model exhibits any source or evidence effect.
    """

    def __init__(self, pattern: FakeResponsePattern) -> None:
        self.pattern = pattern
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        payload = json.loads(request.user_prompt)
        content_ids = [item["content_id"] for item in payload["visible_content"]]
        roots = {
            item["source_root_id"]
            for item in payload["public_root_relations"]
            if item["content_id"] in {_REPEAT_A_ID, _REPEAT_B_ID}
            and item["source_root_id"] is not None
        }
        has_evidence = bool(payload["visible_evidence_ids"])
        if self.pattern is FakeResponsePattern.CONSTANT:
            judgment = OrdinalJudgment.UNCERTAIN
            action = PublicAction.WITHHOLD
            share = ShareDecision.DO_NOT_SHARE
            confidence = 3
        elif has_evidence:
            judgment = OrdinalJudgment.STRONGLY_ADOPT
            action = PublicAction.ADOPT
            share = ShareDecision.SHARE_WITH_CAVEAT
            confidence = 6
        elif len(roots) == 2:
            judgment = OrdinalJudgment.LEAN_ADOPT
            action = PublicAction.ADOPT
            share = ShareDecision.SHARE_WITH_CAVEAT
            confidence = 5
        else:
            judgment = OrdinalJudgment.UNCERTAIN
            action = PublicAction.WITHHOLD
            share = ShareDecision.DO_NOT_SHARE
            confidence = 3
        response = {
            "ordinal_judgment": judgment.value,
            "public_action": action.value,
            "share_decision": share.value,
            "confidence": confidence,
            "content_ids_used": content_ids,
            "share_content_id": content_ids[0] if share is not ShareDecision.DO_NOT_SHARE else None,
        }
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content=json.dumps(response, ensure_ascii=True, sort_keys=True),
            finish_reason="stop",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            latency_ms=0.0,
        )


_ORDINAL_SCORE: Final[Mapping[OrdinalJudgment, int]] = {
    OrdinalJudgment.STRONGLY_REJECT: -2,
    OrdinalJudgment.LEAN_REJECT: -1,
    OrdinalJudgment.UNCERTAIN: 0,
    OrdinalJudgment.LEAN_ADOPT: 1,
    OrdinalJudgment.STRONGLY_ADOPT: 2,
}


def summarize_contrasts(
    observations: Sequence[SafeFactorObservation],
) -> tuple[ContrastSummary, ...]:
    """Recover predeclared paired ordinal contrasts from safe observations."""

    by_group: dict[str, dict[FactorCondition, SafeFactorObservation]] = {}
    for observation in observations:
        group = by_group.setdefault(observation.group_id, {})
        if observation.condition in group:
            raise OfflineSimulationError("factorization_duplicate_group_condition_observation")
        group[observation.condition] = observation
    summaries: list[ContrastSummary] = []
    for contrast_id, minuend, subtrahend in _CONTRASTS:
        differences: list[int] = []
        for group in by_group.values():
            if minuend not in group or subtrahend not in group:
                raise OfflineSimulationError("factorization_missing_paired_observation")
            differences.append(
                _ORDINAL_SCORE[group[minuend].ordinal_judgment]
                - _ORDINAL_SCORE[group[subtrahend].ordinal_judgment]
            )
        summaries.append(ContrastSummary(
            contrast_id=contrast_id,
            minuend=minuend,
            subtrahend=subtrahend,
            paired_group_count=len(differences),
            positive_difference_count=sum(value > 0 for value in differences),
            zero_difference_count=sum(value == 0 for value in differences),
            negative_difference_count=sum(value < 0 for value in differences),
            mean_ordinal_difference=(sum(differences) / len(differences)) if differences else 0.0,
        ))
    return tuple(summaries)


@dataclass(frozen=True)
class _OfflineRun:
    observations: tuple[SafeFactorObservation, ...]
    request_count: int
    unique_request_count: int
    contrasts: tuple[ContrastSummary, ...]
    shared_t0_group_count: int


def _run_pattern(
    pattern: FakeResponsePattern,
    cases: Sequence[FactorCase],
) -> _OfflineRun:
    provider = FactorizationFakeProvider(pattern)
    grouped: dict[str, list[FactorCase]] = {}
    for case in cases:
        grouped.setdefault(case.group_id, []).append(case)
    observations: list[SafeFactorObservation] = []
    request_ids: set[str] = set()
    for group_cases in grouped.values():
        by_condition = {item.condition: item for item in group_cases}
        initial_case = by_condition[FactorCondition.PRIVATE_INITIAL]
        initial_request = render_factor_request(initial_case)
        request_ids.add(initial_request.request_id)
        initial_response = parse_factor_response(provider.complete(initial_request).content, initial_case)
        initial_hash = hashlib.sha256(
            initial_response.model_dump_json().encode()
        ).hexdigest()
        observations.append(safe_factor_observation(initial_case, initial_response, initial_hash))
        for condition in FactorCondition:
            if condition is FactorCondition.PRIVATE_INITIAL:
                continue
            branch_case = by_condition[condition]
            request = render_factor_request(branch_case)
            if request.request_id in request_ids:
                raise OfflineSimulationError("factorization_duplicate_request_fingerprint")
            request_ids.add(request.request_id)
            response = parse_factor_response(provider.complete(request).content, branch_case)
            observations.append(safe_factor_observation(branch_case, response, initial_hash))
    if provider.calls != len(cases):
        raise OfflineSimulationError("factorization_shared_t0_replay_detected")
    if len(request_ids) != len(cases):
        raise OfflineSimulationError("factorization_request_fingerprint_count_mismatch")
    for group_id in grouped:
        hashes = {
            item.shared_t0_observation_sha256
            for item in observations
            if item.group_id == group_id
        }
        if len(hashes) != 1:
            raise OfflineSimulationError("factorization_shared_t0_hash_not_reused")
    return _OfflineRun(
        observations=tuple(observations),
        request_count=provider.calls,
        unique_request_count=len(request_ids),
        contrasts=summarize_contrasts(observations),
        shared_t0_group_count=len(grouped),
    )


def _contrast_map(summaries: Sequence[ContrastSummary]) -> dict[FactorContrastId, ContrastSummary]:
    return {item.contrast_id: item for item in summaries}


def _known_difference_recovered(summaries: Sequence[ContrastSummary]) -> bool:
    values = _contrast_map(summaries)
    return (
        values[FactorContrastId.HIDDEN_SOURCE_PLACEBO].zero_difference_count == 12
        and values[FactorContrastId.SAME_ROOT_DISCLOSURE].zero_difference_count == 12
        and values[FactorContrastId.INDEPENDENT_ROOT_DISCLOSURE].positive_difference_count == 12
        and values[FactorContrastId.ROOT_INDEPENDENCE_INCREMENT].positive_difference_count == 12
        and values[FactorContrastId.EVIDENCE_INCREMENT].positive_difference_count == 12
    )


def _known_no_difference_recovered(summaries: Sequence[ContrastSummary]) -> bool:
    return all(
        item.zero_difference_count == 12
        and item.positive_difference_count == 0
        and item.negative_difference_count == 0
        for item in summaries
    )


def run_offline_smoke(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    """Run both scripted patterns without reading environment or writing files."""

    config = load_config(path)
    cases = build_cases(path)
    difference_run = _run_pattern(FakeResponsePattern.KNOWN_DIFFERENCE, cases)
    constant_run = _run_pattern(FakeResponsePattern.CONSTANT, cases)
    if not _known_difference_recovered(difference_run.contrasts):
        raise OfflineSimulationError("factorization_known_difference_not_recovered")
    if not _known_no_difference_recovered(constant_run.contrasts):
        raise OfflineSimulationError("factorization_known_no_difference_not_recovered")
    condition_counts = Counter(item.condition.value for item in difference_run.observations)
    return {
        "status": "offline_factorization_smoke_passed",
        "simulation_id": config.simulation_id,
        "scenario_count": config.scenario_count,
        "seed_count": len(config.seeds),
        "condition_count": len(config.conditions),
        "case_count_per_pattern": config.case_count,
        "fake_pattern_count": config.fake_pattern_count,
        "logical_request_count": difference_run.request_count + constant_run.request_count,
        "known_difference_request_count": difference_run.request_count,
        "constant_request_count": constant_run.request_count,
        "unique_request_count_per_pattern": difference_run.unique_request_count,
        "shared_t0_group_count": difference_run.shared_t0_group_count,
        "shared_t0_reused_without_provider_replay": True,
        "condition_case_counts": dict(sorted(condition_counts.items())),
        "known_difference_recovered": True,
        "known_no_difference_recovered": True,
        "known_difference_contrasts": [
            item.model_dump(mode="json") for item in difference_run.contrasts
        ],
        "constant_contrasts": [
            item.model_dump(mode="json") for item in constant_run.contrasts
        ],
        "schema": {
            "schema_name": SCHEMA_NAME,
            "schema_sha256": RESPONSE_SCHEMA_SHA256,
            "template_version": TEMPLATE_VERSION,
        },
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
    """Validate the offline lock without rendering provider requests or writing output."""

    try:
        config = load_config(path)
        return {
            "status": "offline_factorization_design_ready",
            "ready_for_network": False,
            "blocking_reasons": ["real_model_authorization_not_requested"],
            "simulation_id": config.simulation_id,
            "case_count": config.case_count,
            "fake_logical_request_cap": config.fake_logical_request_cap,
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
            "ready_for_network": False,
            "blocking_reasons": [getattr(exc, "code", "offline_factorization_preflight_failed")],
            "network": "disabled",
            "real_provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "evaluator_private_truth_exposed": False,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the source-behavior factorization offline FakeProvider simulation"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate the offline configuration without rendering requests.",
    )
    args = parser.parse_args(argv)
    report = safe_preflight(args.config) if args.preflight else run_offline_smoke(args.config)
    print(safe_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ContrastSummary",
    "DEFAULT_CONFIG",
    "FactorCase",
    "FactorCondition",
    "FactorContrastId",
    "FactorScenario",
    "FactorizationFakeProvider",
    "FactorizationResponse",
    "OfflineSimulationConfig",
    "OfflineSimulationError",
    "RESPONSE_SCHEMA",
    "RESPONSE_SCHEMA_SHA256",
    "SCHEMA_NAME",
    "SafeFactorObservation",
    "TEMPLATE_VERSION",
    "build_cases",
    "load_config",
    "parse_factor_response",
    "render_factor_request",
    "run_offline_smoke",
    "safe_factor_observation",
    "safe_preflight",
    "summarize_contrasts",
    "validate_case_matrix",
]
