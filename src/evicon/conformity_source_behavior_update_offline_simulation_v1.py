"""Offline design validation for a stateful source-dependent update protocol.

The previous development pilot issued condition-specific requests independently.
This module intentionally does something narrower and more defensible: it
exercises a J0 -> J1 state machine with scripted providers.  Each J1 request
contains the exact public, structured J0 response for its own logical group.

It never reads provider configuration, API keys, historical outputs, evaluator
truth, or any network resource.  The synthetic patterns are test fixtures, not
behavioral findings about a real model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections import Counter, defaultdict
from enum import Enum
from pathlib import Path
from typing import Final, Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .conformity_identification import (
    OrdinalJudgment,
    PublicAction,
    ShareDecision,
    safe_json,
    sha256_file,
)
from .llm_contract import LLMRequest, LLMResponse


_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG: Final[str] = (
    "configs/provenance_cascade/identification/"
    "conformity_source_behavior_update_offline_simulation.v1.toml"
)
SIMULATION_ID: Final[str] = "evicon-source-dependent-update-offline-simulation-v1"
PROTOCOL_VERSION: Final[str] = "conformity_source_behavior_update_offline_simulation.v1"
TEMPLATE_VERSION: Final[str] = "conformity_source_behavior_update_turn.v1"
SCHEMA_NAME: Final[str] = "conformity_source_behavior_update_response_v1"
_SEEDS: Final[tuple[int, int]] = (20261421, 20261422)
_INITIAL_ID: Final[str] = "content-initial"
_MESSAGE_A_ID: Final[str] = "message-a"
_MESSAGE_B_ID: Final[str] = "message-b"
_EVIDENCE_ID: Final[str] = "evidence-card-01"
_ROOT_A: Final[str] = "lineage-k7"
_ROOT_B: Final[str] = "lineage-p4"


class StatefulUpdateError(ValueError):
    """A stable, non-sensitive failure boundary for the offline protocol."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class UpdatePhase(str, Enum):
    INITIAL = "initial"
    UPDATE = "update"


class UpdateCondition(str, Enum):
    SELF_REFLECTION = "self_reflection"
    SOURCE_UNSHOWN_REPETITION = "source_unshown_repetition"
    SAME_ROOT_SHOWN = "same_root_shown"
    INDEPENDENT_ROOTS_SHOWN = "independent_roots_shown"
    DIRECT_EVIDENCE = "direct_evidence"


class PresentationCounterbalance(str, Enum):
    FORWARD = "forward"
    REVERSED = "reversed"


class ContentKind(str, Enum):
    INITIAL_CONTEXT = "initial_context"
    SOCIAL_REPORT = "social_report"
    DIRECT_EVIDENCE_CARD = "direct_evidence_card"


class FakeResponsePattern(str, Enum):
    """Scripted patterns used only to validate analysis wiring."""

    CONSTANT = "constant"
    KNOWN_DIFFERENCE = "known_difference"
    ORDER_SHORTCUT = "order_shortcut"


class UpdateContrastId(str, Enum):
    NATURAL_INSTABILITY = "self_reflection_minus_initial"
    SOCIAL_TEXT_INCREMENT = "source_unshown_repetition_minus_self_reflection"
    SAME_ROOT_DISCLOSURE = "same_root_shown_minus_source_unshown_repetition"
    ROOT_INDEPENDENCE_INCREMENT = "independent_roots_shown_minus_same_root_shown"
    DIRECT_EVIDENCE_INCREMENT = "direct_evidence_minus_self_reflection"


class VisibleContent(BaseModel):
    """Public synthetic content.  No evaluator-only source graph is present."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content_id: Literal[
        "content-initial", "message-a", "message-b", "evidence-card-01"
    ]
    content_kind: ContentKind
    public_summary: str = Field(min_length=36, max_length=360)


class PublicRootAssignment(BaseModel):
    """A relation visible to the Agent, rather than latent provenance data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content_id: Literal["message-a", "message-b"]
    source_root_id: str | None

    @field_validator("source_root_id")
    @classmethod
    def opaque_public_root_only(cls, value: str | None) -> str | None:
        if value not in {None, _ROOT_A, _ROOT_B}:
            raise ValueError("stateful_update_unknown_public_root")
        return value


class StatefulUpdateResponse(BaseModel):
    """The only public response surface permitted for J0 and J1."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ordinal_judgment: OrdinalJudgment
    public_action: PublicAction
    share_decision: ShareDecision
    confidence: int = Field(ge=1, le=7)
    content_ids_used: tuple[str, ...] = Field(min_length=1, max_length=3)
    share_content_id: str | None


class PublicPriorJudgment(BaseModel):
    """The de-identified, structured J0 state handed to every J1 branch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ordinal_judgment: OrdinalJudgment
    public_action: PublicAction
    share_decision: ShareDecision
    confidence: int = Field(ge=1, le=7)
    content_ids_used: tuple[Literal["content-initial"], ...] = Field(
        min_length=1, max_length=1
    )
    share_content_id: Literal["content-initial"] | None

    @classmethod
    def from_initial_response(
        cls, response: StatefulUpdateResponse
    ) -> "PublicPriorJudgment":
        return cls.model_validate(response.model_dump(mode="json"))


RESPONSE_SCHEMA: Final[dict[str, object]] = StatefulUpdateResponse.model_json_schema(
    mode="validation"
)
RESPONSE_SCHEMA_SHA256: Final[str] = hashlib.sha256(
    json.dumps(
        RESPONSE_SCHEMA, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
).hexdigest()


class UpdateScenario(BaseModel):
    """Author-written, public material that does not encode truth labels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: Literal[
        "stateful-update-v1-s01",
        "stateful-update-v1-s02",
        "stateful-update-v1-s03",
        "stateful-update-v1-s04",
    ]
    target_claim: str = Field(min_length=30, max_length=240)
    decision_task: str = Field(min_length=30, max_length=280)
    initial_summary: str = Field(min_length=44, max_length=360)
    message_a_summary: str = Field(min_length=44, max_length=360)
    message_b_summary: str = Field(min_length=44, max_length=360)
    evidence_summary: str = Field(min_length=44, max_length=360)

    @model_validator(mode="after")
    def public_material_does_not_leak_study_labels(self) -> "UpdateScenario":
        material = " ".join(self.model_dump().values()).lower()
        forbidden = (
            "ground_truth",
            "evaluator_private",
            "correct_answer",
            "same_root",
            "independent_roots",
            "source_unshown",
            "wvs",
            "pilot",
        )
        if any(token in material for token in forbidden):
            raise ValueError("stateful_update_public_material_leak")
        summaries = (
            self.initial_summary,
            self.message_a_summary,
            self.message_b_summary,
            self.evidence_summary,
        )
        if len(set(summaries)) != len(summaries):
            raise ValueError("stateful_update_public_summaries_must_be_distinct")
        return self


class StatefulUpdateCase(BaseModel):
    """One actual J0 request or one J1 branch of a shared public J0 state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^stateful-update-v1-g\d{2}-(j0|j1-b\d)$")
    group_id: str = Field(pattern=r"^stateful-update-v1-g\d{2}$")
    scenario_id: Literal[
        "stateful-update-v1-s01",
        "stateful-update-v1-s02",
        "stateful-update-v1-s03",
        "stateful-update-v1-s04",
    ]
    seed: Literal[20261421, 20261422]
    presentation_counterbalance: PresentationCounterbalance
    phase: UpdatePhase
    condition: UpdateCondition | None
    target_claim: str
    decision_task: str
    visible_content: tuple[VisibleContent, ...] = Field(min_length=1, max_length=3)
    public_root_assignments: tuple[PublicRootAssignment, ...] = Field(max_length=2)
    visible_evidence_card_ids: tuple[Literal["evidence-card-01"], ...] = Field(max_length=1)
    visible_public_prior_judgment: PublicPriorJudgment | None
    public_prior_judgment_sha256: str | None
    reflection_only: bool
    model_name: str = "offline-stateful-update-fake-model"
    temperature: Literal[0.0] = 0.0
    max_tokens: Literal[256] = 256

    @property
    def visible_content_ids(self) -> tuple[str, ...]:
        return tuple(item.content_id for item in self.visible_content)

    @property
    def visible_social_root_count(self) -> int:
        return len({
            item.source_root_id
            for item in self.public_root_assignments
            if item.source_root_id is not None
        })

    @property
    def social_content_ids(self) -> tuple[str, ...]:
        return tuple(
            item.content_id
            for item in self.visible_content
            if item.content_kind is ContentKind.SOCIAL_REPORT
        )

    @model_validator(mode="after")
    def exact_phase_and_condition_shape(self) -> "StatefulUpdateCase":
        content_ids = self.visible_content_ids
        if len(set(content_ids)) != len(content_ids):
            raise ValueError("stateful_update_duplicate_visible_content")
        if tuple(item.content_id for item in self.public_root_assignments) != self.social_content_ids:
            raise ValueError("stateful_update_root_assignment_order_mismatch")

        if self.phase is UpdatePhase.INITIAL:
            if (
                self.condition is not None
                or content_ids != (_INITIAL_ID,)
                or self.public_root_assignments
                or self.visible_evidence_card_ids
                or self.visible_public_prior_judgment is not None
                or self.public_prior_judgment_sha256 is not None
                or self.reflection_only
            ):
                raise ValueError("stateful_update_initial_shape_invalid")
            return self

        if self.condition is None:
            raise ValueError("stateful_update_condition_missing")
        if self.visible_public_prior_judgment is None:
            raise ValueError("stateful_update_prior_missing")
        if not self.public_prior_judgment_sha256 or len(self.public_prior_judgment_sha256) != 64:
            raise ValueError("stateful_update_prior_hash_missing")
        if self.public_prior_judgment_sha256 != public_prior_sha256(
            self.visible_public_prior_judgment
        ):
            raise ValueError("stateful_update_prior_hash_mismatch")

        expected_content = {
            UpdateCondition.SELF_REFLECTION: (_INITIAL_ID,),
            UpdateCondition.SOURCE_UNSHOWN_REPETITION: (
                _INITIAL_ID, _MESSAGE_A_ID, _MESSAGE_B_ID
            ),
            UpdateCondition.SAME_ROOT_SHOWN: (
                _INITIAL_ID, _MESSAGE_A_ID, _MESSAGE_B_ID
            ),
            UpdateCondition.INDEPENDENT_ROOTS_SHOWN: (
                _INITIAL_ID, _MESSAGE_A_ID, _MESSAGE_B_ID
            ),
            UpdateCondition.DIRECT_EVIDENCE: (_INITIAL_ID, _EVIDENCE_ID),
        }[self.condition]
        expected_social = {
            UpdateCondition.SELF_REFLECTION: (),
            UpdateCondition.SOURCE_UNSHOWN_REPETITION: (_MESSAGE_A_ID, _MESSAGE_B_ID),
            UpdateCondition.SAME_ROOT_SHOWN: (_MESSAGE_A_ID, _MESSAGE_B_ID),
            UpdateCondition.INDEPENDENT_ROOTS_SHOWN: (_MESSAGE_A_ID, _MESSAGE_B_ID),
            UpdateCondition.DIRECT_EVIDENCE: (),
        }[self.condition]
        if self.presentation_counterbalance is PresentationCounterbalance.REVERSED:
            expected_content = (
                (_INITIAL_ID, _MESSAGE_B_ID, _MESSAGE_A_ID)
                if len(expected_content) == 3
                else expected_content
            )
            expected_social = tuple(reversed(expected_social))
        if content_ids != expected_content or self.social_content_ids != expected_social:
            raise ValueError("stateful_update_visible_content_condition_mismatch")
        expected_roots = {
            UpdateCondition.SELF_REFLECTION: 0,
            UpdateCondition.SOURCE_UNSHOWN_REPETITION: 0,
            UpdateCondition.SAME_ROOT_SHOWN: 1,
            UpdateCondition.INDEPENDENT_ROOTS_SHOWN: 2,
            UpdateCondition.DIRECT_EVIDENCE: 0,
        }[self.condition]
        if self.visible_social_root_count != expected_roots:
            raise ValueError("stateful_update_visible_root_count_mismatch")
        expected_evidence = (
            (_EVIDENCE_ID,) if self.condition is UpdateCondition.DIRECT_EVIDENCE else ()
        )
        if self.visible_evidence_card_ids != expected_evidence:
            raise ValueError("stateful_update_evidence_visibility_mismatch")
        if self.reflection_only is not (self.condition is UpdateCondition.SELF_REFLECTION):
            raise ValueError("stateful_update_reflection_condition_mismatch")
        return self


class SafeUpdateObservation(BaseModel):
    """Persistable metadata only; material, prompts, roots, and raw output stay out."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^stateful-update-v1-g\d{2}-(j0|j1-b\d)$")
    group_id: str = Field(pattern=r"^stateful-update-v1-g\d{2}$")
    scenario_id: str = Field(pattern=r"^stateful-update-v1-s0[1-4]$")
    seed: Literal[20261421, 20261422]
    presentation_counterbalance: PresentationCounterbalance
    phase: UpdatePhase
    condition: UpdateCondition | None
    parser_status: Literal["valid"]
    ordinal_judgment: OrdinalJudgment
    public_action: PublicAction
    share_decision: ShareDecision
    confidence: int = Field(ge=1, le=7)
    used_content_count: int = Field(ge=1, le=3)
    share_content_id_present: bool
    visible_social_root_count: int = Field(ge=0, le=2)
    visible_direct_evidence_count: int = Field(ge=0, le=1)
    public_prior_judgment_sha256: str | None


class ContrastSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contrast_id: UpdateContrastId
    minuend: str
    subtrahend: str
    paired_group_count: int = Field(ge=0)
    positive_difference_count: int = Field(ge=0)
    zero_difference_count: int = Field(ge=0)
    negative_difference_count: int = Field(ge=0)
    mean_ordinal_difference: float


class OrderShortcutSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: UpdateCondition
    paired_counterbalance_group_count: int = Field(ge=0)
    differing_ordinal_pair_count: int = Field(ge=0)


class OfflineSimulationConfig(BaseModel):
    """Versioned scope lock for this offline-only state-machine validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    simulation_id: Literal["evicon-source-dependent-update-offline-simulation-v1"]
    simulation_version: Literal["conformity_source_behavior_update_offline_simulation.v1"]
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
    protocol_path: Literal[
        "src/evicon/conformity_source_behavior_update_offline_simulation_v1.py"
    ]
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_version: Literal["conformity_source_behavior_update_offline_simulation.v1"]
    template_version: Literal["conformity_source_behavior_update_turn.v1"]
    response_schema_name: Literal["conformity_source_behavior_update_response_v1"]
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    conditions: tuple[UpdateCondition, ...]
    seeds: tuple[Literal[20261421], Literal[20261422]]
    scenario_count: Literal[4]
    counterbalance_count: Literal[2]
    group_count: Literal[16]
    branches_per_group: Literal[5]
    logical_request_count_per_pattern: Literal[96]
    fake_pattern_count: Literal[3]
    fake_logical_request_cap: Literal[288]
    max_tokens: Literal[256]
    output_root: Literal["outputs/conformity-source-behavior-update-offline-simulation-v1"]

    @model_validator(mode="after")
    def fixed_scope(self) -> "OfflineSimulationConfig":
        if self.conditions != tuple(UpdateCondition):
            raise ValueError("stateful_update_condition_order_mismatch")
        if self.seeds != _SEEDS:
            raise ValueError("stateful_update_seed_set_mismatch")
        expected_groups = self.scenario_count * len(self.seeds) * self.counterbalance_count
        if self.group_count != expected_groups:
            raise ValueError("stateful_update_group_count_mismatch")
        expected_requests = self.group_count * (1 + self.branches_per_group)
        if self.logical_request_count_per_pattern != expected_requests:
            raise ValueError("stateful_update_request_count_mismatch")
        if self.fake_logical_request_cap != (
            self.logical_request_count_per_pattern * self.fake_pattern_count
        ):
            raise ValueError("stateful_update_fake_request_cap_mismatch")
        return self


_SCENARIOS: Final[tuple[UpdateScenario, UpdateScenario, UpdateScenario, UpdateScenario]] = (
    UpdateScenario(
        scenario_id="stateful-update-v1-s01",
        target_claim="The fictional RiverLoop shuttle will add a late weekday departure.",
        decision_task="Decide whether the claim is reliable enough for a reversible public planning notice.",
        initial_summary="A short public notice says RiverLoop may add one late weekday shuttle departure.",
        message_a_summary="A neighborhood update says RiverLoop is expected to run one late weekday departure.",
        message_b_summary="A separate community notice repeats that a late RiverLoop weekday departure is planned.",
        evidence_summary="A fictional operating timetable lists one late weekday RiverLoop departure for the coming week.",
    ),
    UpdateScenario(
        scenario_id="stateful-update-v1-s02",
        target_claim="The fictional ArborDesk library will provide an extra Sunday service hour.",
        decision_task="Decide whether the claim is reliable enough for a reversible access planning notice.",
        initial_summary="A short public notice says ArborDesk may provide one additional Sunday service hour.",
        message_a_summary="A neighborhood update says ArborDesk is expected to remain available for an extra Sunday hour.",
        message_b_summary="A separate community notice repeats that ArborDesk plans an added Sunday service hour.",
        evidence_summary="A fictional weekly schedule lists the additional ArborDesk Sunday service hour.",
    ),
    UpdateScenario(
        scenario_id="stateful-update-v1-s03",
        target_claim="The fictional Greenway clinic will open an extra evening booking period.",
        decision_task="Decide whether the claim is reliable enough for a reversible appointment planning notice.",
        initial_summary="A short public notice says Greenway may open an extra evening booking period next week.",
        message_a_summary="A neighborhood update says Greenway is expected to offer extra evening appointment availability.",
        message_b_summary="A separate community notice repeats that Greenway plans another evening booking period.",
        evidence_summary="A fictional clinic schedule records an extra Greenway evening booking period next week.",
    ),
    UpdateScenario(
        scenario_id="stateful-update-v1-s04",
        target_claim="The fictional Lakeside workshop will meet in the north studio rather than the central room.",
        decision_task="Decide whether the claim is reliable enough for a reversible attendee planning notice.",
        initial_summary="A short public notice says the Lakeside workshop may move to the north studio.",
        message_a_summary="An attendee update says the Lakeside workshop is expected to use the north studio.",
        message_b_summary="A separate community notice repeats that the Lakeside workshop plans to meet in the north studio.",
        evidence_summary="A fictional venue allocation record lists the Lakeside workshop in the north studio.",
    ),
)

_CONTRASTS: Final[tuple[tuple[UpdateContrastId, str, str], ...]] = (
    (UpdateContrastId.NATURAL_INSTABILITY, UpdateCondition.SELF_REFLECTION.value, "initial"),
    (
        UpdateContrastId.SOCIAL_TEXT_INCREMENT,
        UpdateCondition.SOURCE_UNSHOWN_REPETITION.value,
        UpdateCondition.SELF_REFLECTION.value,
    ),
    (
        UpdateContrastId.SAME_ROOT_DISCLOSURE,
        UpdateCondition.SAME_ROOT_SHOWN.value,
        UpdateCondition.SOURCE_UNSHOWN_REPETITION.value,
    ),
    (
        UpdateContrastId.ROOT_INDEPENDENCE_INCREMENT,
        UpdateCondition.INDEPENDENT_ROOTS_SHOWN.value,
        UpdateCondition.SAME_ROOT_SHOWN.value,
    ),
    (
        UpdateContrastId.DIRECT_EVIDENCE_INCREMENT,
        UpdateCondition.DIRECT_EVIDENCE.value,
        UpdateCondition.SELF_REFLECTION.value,
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
        raise StatefulUpdateError("stateful_update_config_invalid") from exc
    if sha256_file(config.protocol_path) != config.protocol_sha256:
        raise StatefulUpdateError("stateful_update_protocol_hash_mismatch")
    if config.response_schema_sha256 != RESPONSE_SCHEMA_SHA256:
        raise StatefulUpdateError("stateful_update_schema_hash_mismatch")
    return config


def public_prior_sha256(prior: PublicPriorJudgment) -> str:
    """Hash the exact structured J0 state, never an opaque model transcript."""

    return hashlib.sha256(
        json.dumps(
            prior.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _all_content(scenario: UpdateScenario) -> dict[str, VisibleContent]:
    return {
        _INITIAL_ID: VisibleContent(
            content_id=_INITIAL_ID,
            content_kind=ContentKind.INITIAL_CONTEXT,
            public_summary=scenario.initial_summary,
        ),
        _MESSAGE_A_ID: VisibleContent(
            content_id=_MESSAGE_A_ID,
            content_kind=ContentKind.SOCIAL_REPORT,
            public_summary=scenario.message_a_summary,
        ),
        _MESSAGE_B_ID: VisibleContent(
            content_id=_MESSAGE_B_ID,
            content_kind=ContentKind.SOCIAL_REPORT,
            public_summary=scenario.message_b_summary,
        ),
        _EVIDENCE_ID: VisibleContent(
            content_id=_EVIDENCE_ID,
            content_kind=ContentKind.DIRECT_EVIDENCE_CARD,
            public_summary=scenario.evidence_summary,
        ),
    }


def _content_for_condition(
    scenario: UpdateScenario,
    condition: UpdateCondition | None,
    presentation: PresentationCounterbalance,
) -> tuple[VisibleContent, ...]:
    content = _all_content(scenario)
    if condition is None or condition is UpdateCondition.SELF_REFLECTION:
        ids = (_INITIAL_ID,)
    elif condition is UpdateCondition.DIRECT_EVIDENCE:
        ids = (_INITIAL_ID, _EVIDENCE_ID)
    elif presentation is PresentationCounterbalance.FORWARD:
        ids = (_INITIAL_ID, _MESSAGE_A_ID, _MESSAGE_B_ID)
    else:
        ids = (_INITIAL_ID, _MESSAGE_B_ID, _MESSAGE_A_ID)
    return tuple(content[content_id] for content_id in ids)


def _public_roots(
    condition: UpdateCondition,
    social_ids: tuple[str, ...],
    presentation: PresentationCounterbalance,
) -> tuple[PublicRootAssignment, ...]:
    if not social_ids:
        return ()
    if condition is UpdateCondition.SOURCE_UNSHOWN_REPETITION:
        roots: dict[str, str | None] = {_MESSAGE_A_ID: None, _MESSAGE_B_ID: None}
    elif condition is UpdateCondition.SAME_ROOT_SHOWN:
        root = _ROOT_A if presentation is PresentationCounterbalance.FORWARD else _ROOT_B
        roots = {_MESSAGE_A_ID: root, _MESSAGE_B_ID: root}
    elif condition is UpdateCondition.INDEPENDENT_ROOTS_SHOWN:
        if presentation is PresentationCounterbalance.FORWARD:
            roots = {_MESSAGE_A_ID: _ROOT_A, _MESSAGE_B_ID: _ROOT_B}
        else:
            roots = {_MESSAGE_A_ID: _ROOT_B, _MESSAGE_B_ID: _ROOT_A}
    else:
        raise StatefulUpdateError("stateful_update_social_roots_for_non_social_condition")
    return tuple(
        PublicRootAssignment(content_id=content_id, source_root_id=roots[content_id])
        for content_id in social_ids
    )  # type: ignore[arg-type]


def build_initial_cases(path: str | Path = DEFAULT_CONFIG) -> tuple[StatefulUpdateCase, ...]:
    """Create one J0 request for each scenario, seed, and counterbalance group."""

    config = load_config(path)
    cases: list[StatefulUpdateCase] = []
    group_number = 0
    for scenario in _SCENARIOS:
        for seed in config.seeds:
            for presentation in PresentationCounterbalance:
                group_number += 1
                group_id = f"stateful-update-v1-g{group_number:02d}"
                cases.append(StatefulUpdateCase(
                    case_id=f"{group_id}-j0",
                    group_id=group_id,
                    scenario_id=scenario.scenario_id,
                    seed=seed,
                    presentation_counterbalance=presentation,
                    phase=UpdatePhase.INITIAL,
                    condition=None,
                    target_claim=scenario.target_claim,
                    decision_task=scenario.decision_task,
                    visible_content=_content_for_condition(scenario, None, presentation),
                    public_root_assignments=(),
                    visible_evidence_card_ids=(),
                    visible_public_prior_judgment=None,
                    public_prior_judgment_sha256=None,
                    reflection_only=False,
                    max_tokens=config.max_tokens,
                ))
    if len(cases) != config.group_count:
        raise StatefulUpdateError("stateful_update_initial_group_count_mismatch")
    return tuple(cases)


def _scenario_for(case: StatefulUpdateCase) -> UpdateScenario:
    return next(item for item in _SCENARIOS if item.scenario_id == case.scenario_id)


def build_update_cases(
    initial_case: StatefulUpdateCase,
    initial_response: StatefulUpdateResponse,
    path: str | Path = DEFAULT_CONFIG,
) -> tuple[StatefulUpdateCase, ...]:
    """Build all J1 branches, bound to one actual parsed J0 response."""

    config = load_config(path)
    if initial_case.phase is not UpdatePhase.INITIAL:
        raise StatefulUpdateError("stateful_update_initial_case_required")
    prior = PublicPriorJudgment.from_initial_response(initial_response)
    prior_hash = public_prior_sha256(prior)
    scenario = _scenario_for(initial_case)
    cases: list[StatefulUpdateCase] = []
    for branch_number, condition in enumerate(config.conditions, start=1):
        visible_content = _content_for_condition(
            scenario, condition, initial_case.presentation_counterbalance
        )
        social_ids = tuple(
            item.content_id
            for item in visible_content
            if item.content_kind is ContentKind.SOCIAL_REPORT
        )
        cases.append(StatefulUpdateCase(
            case_id=f"{initial_case.group_id}-j1-b{branch_number}",
            group_id=initial_case.group_id,
            scenario_id=initial_case.scenario_id,
            seed=initial_case.seed,
            presentation_counterbalance=initial_case.presentation_counterbalance,
            phase=UpdatePhase.UPDATE,
            condition=condition,
            target_claim=initial_case.target_claim,
            decision_task=initial_case.decision_task,
            visible_content=visible_content,
            public_root_assignments=_public_roots(
                condition, social_ids, initial_case.presentation_counterbalance
            ),
            visible_evidence_card_ids=(
                (_EVIDENCE_ID,) if condition is UpdateCondition.DIRECT_EVIDENCE else ()
            ),
            visible_public_prior_judgment=prior,
            public_prior_judgment_sha256=prior_hash,
            reflection_only=condition is UpdateCondition.SELF_REFLECTION,
            max_tokens=config.max_tokens,
        ))
    if len(cases) != config.branches_per_group:
        raise StatefulUpdateError("stateful_update_branch_count_mismatch")
    return tuple(cases)


def public_payload(case: StatefulUpdateCase) -> dict[str, object]:
    """The exact public request body, excluding condition and private coordinates."""

    payload: dict[str, object] = {
        "template_version": TEMPLATE_VERSION,
        "phase": case.phase.value,
        "target_claim": case.target_claim,
        "decision_task": case.decision_task,
        "visible_content": [
            {
                "content_id": item.content_id,
                "content_kind": item.content_kind.value,
                "public_summary": item.public_summary,
            }
            for item in case.visible_content
        ],
        "public_root_relations": [
            item.model_dump(mode="json") for item in case.public_root_assignments
        ],
        "visible_evidence_card_ids": list(case.visible_evidence_card_ids),
    }
    if case.phase is UpdatePhase.UPDATE:
        assert case.visible_public_prior_judgment is not None
        assert case.public_prior_judgment_sha256 is not None
        payload["visible_public_prior_judgment"] = case.visible_public_prior_judgment.model_dump(
            mode="json"
        )
        payload["public_prior_judgment_sha256"] = case.public_prior_judgment_sha256
        payload["reflection_only"] = case.reflection_only
    return payload


_SYSTEM_PROMPT: Final[str] = (
    "Return exactly one JSON object with these six fields and no others: "
    "ordinal_judgment, public_action, share_decision, confidence, "
    "content_ids_used, and share_content_id. ordinal_judgment is one of "
    "strongly_reject, lean_reject, uncertain, lean_adopt, strongly_adopt. "
    "public_action is respectively reject for either reject judgment, withhold "
    "for uncertain, and adopt for either adopt judgment. share_decision is one "
    "of share, share_with_caveat, do_not_share. confidence is an integer 1 to 7. "
    "Use only IDs in visible_content. If do_not_share, set share_content_id to "
    "null; otherwise it must be a used visible ID. visible_public_prior_judgment "
    "is this Agent's prior public output, not new evidence. Equal visible root "
    "IDs mean the two shown social reports trace to one shown root; distinct IDs "
    "mean distinct shown roots; absent root IDs mean this relation is not shown. "
    "A root relation alone does not establish truth or source quality. A directly "
    "visible evidence card is a different material type, not a truth label. Do not "
    "invent sources, evidence, hidden information, truth labels, or IDs."
)


def render_request(case: StatefulUpdateCase) -> LLMRequest:
    """Render an initial or update request without internal study labels."""

    user_prompt = json.dumps(
        public_payload(case), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(
        f"{case.case_id}|{case.seed}|{user_prompt}".encode("utf-8")
    ).hexdigest()[:24]
    return LLMRequest(
        request_id=f"stateful-update-v1-{digest}",
        model_name=case.model_name,
        system_prompt=_SYSTEM_PROMPT,
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


def parse_response(raw: str, case: StatefulUpdateCase) -> StatefulUpdateResponse:
    """Strictly parse output against the concrete J0/J1 public visibility state."""

    try:
        payload = json.loads(raw, object_pairs_hook=_unique_pairs)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("malformed_json") from exc
    if not isinstance(payload, dict):
        raise ValueError("top_level_type")
    try:
        response = StatefulUpdateResponse.model_validate(payload)
    except Exception as exc:
        raise ValueError("invalid_schema") from exc
    if len(set(response.content_ids_used)) != len(response.content_ids_used):
        raise ValueError("duplicate_content_id")
    visible_ids = set(case.visible_content_ids)
    if not set(response.content_ids_used).issubset(visible_ids):
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
        response.share_content_id not in visible_ids
        or response.share_content_id not in response.content_ids_used
    ):
        raise ValueError("share_content_id_invalid")
    if case.phase is UpdatePhase.INITIAL:
        try:
            PublicPriorJudgment.from_initial_response(response)
        except Exception as exc:
            raise ValueError("initial_response_not_shareable_public_prior") from exc
    return response


def safe_observation(
    case: StatefulUpdateCase, response: StatefulUpdateResponse
) -> SafeUpdateObservation:
    return SafeUpdateObservation(
        case_id=case.case_id,
        group_id=case.group_id,
        scenario_id=case.scenario_id,
        seed=case.seed,
        presentation_counterbalance=case.presentation_counterbalance,
        phase=case.phase,
        condition=case.condition,
        parser_status="valid",
        ordinal_judgment=response.ordinal_judgment,
        public_action=response.public_action,
        share_decision=response.share_decision,
        confidence=response.confidence,
        used_content_count=len(response.content_ids_used),
        share_content_id_present=response.share_content_id is not None,
        visible_social_root_count=case.visible_social_root_count,
        visible_direct_evidence_count=len(case.visible_evidence_card_ids),
        public_prior_judgment_sha256=case.public_prior_judgment_sha256,
    )


class CompletionProvider(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...


class StatefulUpdateFakeProvider:
    """A prompt-only fake used to show that the analysis detects failures too."""

    def __init__(self, pattern: FakeResponsePattern) -> None:
        self.pattern = pattern
        self.calls = 0
        self.request_ids: list[str] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        self.request_ids.append(request.request_id)
        payload = json.loads(request.user_prompt)
        phase = payload["phase"]
        content_ids = [item["content_id"] for item in payload["visible_content"]]
        social_ids = [
            item["content_id"]
            for item in payload["visible_content"]
            if item["content_kind"] == ContentKind.SOCIAL_REPORT.value
        ]
        roots = {
            item["source_root_id"]
            for item in payload["public_root_relations"]
            if item["source_root_id"] is not None
        }
        has_evidence = bool(payload["visible_evidence_card_ids"])

        judgment = OrdinalJudgment.UNCERTAIN
        confidence = 3
        if self.pattern is FakeResponsePattern.KNOWN_DIFFERENCE and phase == UpdatePhase.UPDATE.value:
            if has_evidence:
                judgment, confidence = OrdinalJudgment.STRONGLY_ADOPT, 6
            elif len(roots) == 2:
                judgment, confidence = OrdinalJudgment.LEAN_ADOPT, 5
            elif social_ids and not roots:
                judgment, confidence = OrdinalJudgment.LEAN_ADOPT, 4
            elif social_ids and len(roots) == 1:
                judgment, confidence = OrdinalJudgment.UNCERTAIN, 3
        elif self.pattern is FakeResponsePattern.ORDER_SHORTCUT and social_ids:
            judgment, confidence = (
                (OrdinalJudgment.LEAN_ADOPT, 4)
                if social_ids[0] == _MESSAGE_A_ID
                else (OrdinalJudgment.LEAN_REJECT, 4)
            )

        action = {
            OrdinalJudgment.STRONGLY_REJECT: PublicAction.REJECT,
            OrdinalJudgment.LEAN_REJECT: PublicAction.REJECT,
            OrdinalJudgment.UNCERTAIN: PublicAction.WITHHOLD,
            OrdinalJudgment.LEAN_ADOPT: PublicAction.ADOPT,
            OrdinalJudgment.STRONGLY_ADOPT: PublicAction.ADOPT,
        }[judgment]
        share = (
            ShareDecision.DO_NOT_SHARE
            if judgment is OrdinalJudgment.UNCERTAIN
            else ShareDecision.SHARE_WITH_CAVEAT
        )
        response = {
            "ordinal_judgment": judgment.value,
            "public_action": action.value,
            "share_decision": share.value,
            "confidence": confidence,
            "content_ids_used": content_ids,
            "share_content_id": (
                content_ids[0] if share is not ShareDecision.DO_NOT_SHARE else None
            ),
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
            provider_metadata={},
        )


def run_pattern(
    pattern: FakeResponsePattern,
    path: str | Path = DEFAULT_CONFIG,
) -> tuple[tuple[SafeUpdateObservation, ...], int]:
    """Execute a full in-memory J0 -> five J1 branches matrix against a fake."""

    config = load_config(path)
    provider = StatefulUpdateFakeProvider(pattern)
    observations: list[SafeUpdateObservation] = []
    for initial_case in build_initial_cases(path):
        initial_response = parse_response(
            provider.complete(render_request(initial_case)).content, initial_case
        )
        observations.append(safe_observation(initial_case, initial_response))
        for update_case in build_update_cases(initial_case, initial_response, path):
            update_response = parse_response(
                provider.complete(render_request(update_case)).content, update_case
            )
            observations.append(safe_observation(update_case, update_response))
    if provider.calls != config.logical_request_count_per_pattern:
        raise StatefulUpdateError("stateful_update_fake_request_count_mismatch")
    if len(set(provider.request_ids)) != provider.calls:
        raise StatefulUpdateError("stateful_update_fake_request_identity_collision")
    validate_execution(observations, config)
    return tuple(observations), provider.calls


def _ordinal_value(judgment: OrdinalJudgment) -> int:
    return {
        OrdinalJudgment.STRONGLY_REJECT: -2,
        OrdinalJudgment.LEAN_REJECT: -1,
        OrdinalJudgment.UNCERTAIN: 0,
        OrdinalJudgment.LEAN_ADOPT: 1,
        OrdinalJudgment.STRONGLY_ADOPT: 2,
    }[judgment]


def contrast_summaries(
    observations: Sequence[SafeUpdateObservation],
) -> tuple[ContrastSummary, ...]:
    """Pair conditions within the same J0 group; never treat branches as iid."""

    grouped: dict[str, dict[str, SafeUpdateObservation]] = defaultdict(dict)
    for item in observations:
        label = "initial" if item.phase is UpdatePhase.INITIAL else item.condition.value  # type: ignore[union-attr]
        grouped[item.group_id][label] = item
    summaries: list[ContrastSummary] = []
    for contrast_id, minuend, subtrahend in _CONTRASTS:
        differences = [
            _ordinal_value(group[minuend].ordinal_judgment)
            - _ordinal_value(group[subtrahend].ordinal_judgment)
            for group in grouped.values()
            if minuend in group and subtrahend in group
        ]
        summaries.append(ContrastSummary(
            contrast_id=contrast_id,
            minuend=minuend,
            subtrahend=subtrahend,
            paired_group_count=len(differences),
            positive_difference_count=sum(item > 0 for item in differences),
            zero_difference_count=sum(item == 0 for item in differences),
            negative_difference_count=sum(item < 0 for item in differences),
            mean_ordinal_difference=(sum(differences) / len(differences) if differences else 0.0),
        ))
    return tuple(summaries)


def order_shortcut_summaries(
    observations: Sequence[SafeUpdateObservation],
) -> tuple[OrderShortcutSummary, ...]:
    """Detect output shifts caused only by the pre-fixed presentation variant."""

    paired: dict[tuple[str, int, UpdateCondition], dict[PresentationCounterbalance, SafeUpdateObservation]] = defaultdict(dict)
    for item in observations:
        if item.phase is UpdatePhase.UPDATE and item.condition is not None:
            paired[(item.scenario_id, item.seed, item.condition)][
                item.presentation_counterbalance
            ] = item
    summaries: list[OrderShortcutSummary] = []
    for condition in UpdateCondition:
        pairs = [
            value for key, value in paired.items() if key[2] is condition
            and set(value) == set(PresentationCounterbalance)
        ]
        summaries.append(OrderShortcutSummary(
            condition=condition,
            paired_counterbalance_group_count=len(pairs),
            differing_ordinal_pair_count=sum(
                _ordinal_value(pair[PresentationCounterbalance.FORWARD].ordinal_judgment)
                != _ordinal_value(pair[PresentationCounterbalance.REVERSED].ordinal_judgment)
                for pair in pairs
            ),
        ))
    return tuple(summaries)


def validate_execution(
    observations: Sequence[SafeUpdateObservation],
    config: OfflineSimulationConfig | None = None,
) -> None:
    """Check the state machine and paired design before any summary is trusted."""

    config = config or load_config()
    if len(observations) != config.logical_request_count_per_pattern:
        raise StatefulUpdateError("stateful_update_observation_count_mismatch")
    if len({item.case_id for item in observations}) != len(observations):
        raise StatefulUpdateError("stateful_update_duplicate_observation")
    grouped: dict[str, list[SafeUpdateObservation]] = defaultdict(list)
    for item in observations:
        grouped[item.group_id].append(item)
    if len(grouped) != config.group_count:
        raise StatefulUpdateError("stateful_update_group_coverage_mismatch")
    for group in grouped.values():
        initial = [item for item in group if item.phase is UpdatePhase.INITIAL]
        updates = [item for item in group if item.phase is UpdatePhase.UPDATE]
        if len(initial) != 1 or len(updates) != config.branches_per_group:
            raise StatefulUpdateError("stateful_update_j0_j1_count_mismatch")
        if {item.condition for item in updates} != set(UpdateCondition):
            raise StatefulUpdateError("stateful_update_condition_coverage_mismatch")
        expected_prior = public_prior_sha256(PublicPriorJudgment(
            ordinal_judgment=initial[0].ordinal_judgment,
            public_action=initial[0].public_action,
            share_decision=initial[0].share_decision,
            confidence=initial[0].confidence,
            content_ids_used=("content-initial",),
            share_content_id=(
                "content-initial"
                if initial[0].share_content_id_present
                else None
            ),
        ))
        if any(item.public_prior_judgment_sha256 != expected_prior for item in updates):
            raise StatefulUpdateError("stateful_update_prior_not_shared")


def _known_difference_recovered(summaries: Sequence[ContrastSummary]) -> bool:
    expected = {
        UpdateContrastId.NATURAL_INSTABILITY: 0.0,
        UpdateContrastId.SOCIAL_TEXT_INCREMENT: 1.0,
        UpdateContrastId.SAME_ROOT_DISCLOSURE: -1.0,
        UpdateContrastId.ROOT_INDEPENDENCE_INCREMENT: 1.0,
        UpdateContrastId.DIRECT_EVIDENCE_INCREMENT: 2.0,
    }
    return all(
        summary.mean_ordinal_difference == expected[summary.contrast_id]
        for summary in summaries
    )


def _constant_pattern_recovered(summaries: Sequence[ContrastSummary]) -> bool:
    return all(summary.mean_ordinal_difference == 0.0 for summary in summaries)


def run_offline_smoke(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    """Exercise zero effect, known effect, and presentation-shortcut outcomes locally."""

    config = load_config(path)
    outputs: dict[FakeResponsePattern, tuple[tuple[SafeUpdateObservation, ...], int]] = {
        pattern: run_pattern(pattern, path) for pattern in FakeResponsePattern
    }
    constant_summaries = contrast_summaries(outputs[FakeResponsePattern.CONSTANT][0])
    known_summaries = contrast_summaries(outputs[FakeResponsePattern.KNOWN_DIFFERENCE][0])
    shortcut_summaries = order_shortcut_summaries(outputs[FakeResponsePattern.ORDER_SHORTCUT][0])
    if not _constant_pattern_recovered(constant_summaries):
        raise StatefulUpdateError("stateful_update_constant_pattern_not_recovered")
    if not _known_difference_recovered(known_summaries):
        raise StatefulUpdateError("stateful_update_known_difference_not_recovered")
    if not any(item.differing_ordinal_pair_count for item in shortcut_summaries):
        raise StatefulUpdateError("stateful_update_order_shortcut_not_detected")
    return {
        "simulation_id": SIMULATION_ID,
        "status": "stateful_update_offline_smoke_passed",
        "group_count": config.group_count,
        "branches_per_group": config.branches_per_group,
        "logical_request_count_per_pattern": config.logical_request_count_per_pattern,
        "fake_pattern_count": config.fake_pattern_count,
        "fake_logical_request_count": sum(item[1] for item in outputs.values()),
        "constant_pattern_recovered": True,
        "known_difference_recovered": True,
        "order_shortcut_detected": True,
        "known_difference_contrasts": [
            item.model_dump(mode="json") for item in known_summaries
        ],
        "order_shortcut_summaries": [
            item.model_dump(mode="json") for item in shortcut_summaries
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


def protocol_self_check(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    """Return static, offline facts for preflight and tests."""

    config = load_config(path)
    initial_cases = build_initial_cases(path)
    initial_case = initial_cases[0]
    fake = StatefulUpdateFakeProvider(FakeResponsePattern.CONSTANT)
    initial_response = parse_response(fake.complete(render_request(initial_case)).content, initial_case)
    update_cases = build_update_cases(initial_case, initial_response, path)
    social = [
        item for item in update_cases
        if item.condition in {
            UpdateCondition.SOURCE_UNSHOWN_REPETITION,
            UpdateCondition.SAME_ROOT_SHOWN,
            UpdateCondition.INDEPENDENT_ROOTS_SHOWN,
        }
    ]
    public_social_payloads = [
        [(entry.content_id, entry.public_summary) for entry in item.visible_content]
        for item in social
    ]
    return {
        "group_count": config.group_count,
        "logical_request_count_per_pattern": config.logical_request_count_per_pattern,
        "all_initial_request_ids_unique": len({
            render_request(item).request_id for item in initial_cases
        }) == len(initial_cases),
        "j1_uses_exact_structured_j0": all(
            item.public_prior_judgment_sha256 == public_prior_sha256(
                item.visible_public_prior_judgment  # type: ignore[arg-type]
            ) for item in update_cases
        ),
        "social_text_and_order_matched_within_group": all(
            item == public_social_payloads[0] for item in public_social_payloads[1:]
        ),
        "same_root_and_independent_root_counts": [
            item.visible_social_root_count for item in social
        ],
        "schema_name": SCHEMA_NAME,
        "schema_sha256": RESPONSE_SCHEMA_SHA256,
    }


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the offline-only stateful source-dependent update smoke test."
    )
    parser.add_argument("--offline-smoke", action="store_true")
    args = parser.parse_args()
    if args.offline_smoke:
        print(safe_json(run_offline_smoke()))
    else:
        print(safe_json({
            "status": "offline_only",
            "network": "disabled",
            "api_key_read": False,
            "results_written": False,
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
