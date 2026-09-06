"""Private English-core WVS transcription template generation and checks.

Only authoring-plan metadata is copied into the template.  The generated file
is intentionally incomplete and belongs under the ignored
``private_wvs_transcriptions/`` directory.  Researchers fill its text and
labels locally; this module never reads source PDFs or creates a ProbeSet.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .wvs7_authoring import (
    AcceptanceStatus,
    AuthoringPlanStatus,
    WVS7AuthoringAcceptance,
    WVS7AuthoringError,
    WVS7AuthoringPlan,
    WVS7AuthoringPlanItem,
    WVS7LanguageArm,
    WVS7ManualTranscriptionLoader,
    WVS7ManualTranscriptionSet,
    validate_manual_transcriptions,
)


ENGLISH_CORE_VARIABLE_IDS = (
    "Q106", "Q107", "Q108", "Q109", "Q110", "Q111", "Q149", "Q150",
    "Q158", "Q159", "Q160", "Q161", "Q162", "Q163", "Q196", "Q197",
    "Q198", "Q241", "Q243", "Q246", "Q247", "Q248", "Q249",
)


class WVS7BlankTranscriptionItem(BaseModel):
    """A blank local row; full text fields are deliberately not valid yet."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variable_id: str = Field(min_length=1)
    language: WVS7LanguageArm
    source_review_id: str = Field(min_length=1)
    source_questionnaire_file: str = Field(min_length=1)
    source_codebook_file: str = Field(min_length=1)
    source_questionnaire_pages: list[int] = Field(min_length=1)
    source_codebook_pages: list[int] = Field(min_length=1)
    source_sha256_prefixes: dict[str, str] = Field(min_length=2)
    question_text: str = "TODO_MANUAL_ENTRY"
    response_options: list[dict[str, str]] = Field(default_factory=list)
    scale_min: int
    scale_max: int
    high_score_meaning: str = Field(min_length=1)
    reverse_scored: bool
    endpoint_labels: dict[str, str] = Field(default_factory=dict)
    special_response_codes: dict[str, str] = Field(default_factory=dict)
    manual_transcriber_id: str = "TODO_MANUAL_ENTRY"
    transcribed_on: str = "TODO_MANUAL_ENTRY"

    @model_validator(mode="after")
    def validate_metadata(self) -> "WVS7BlankTranscriptionItem":
        if self.language is not WVS7LanguageArm.ENGLISH_CORE:
            raise ValueError("private English template cannot contain a Chinese language arm")
        if self.scale_min >= self.scale_max:
            raise ValueError("template scale_min must be lower than scale_max")
        return self


class WVS7BlankTranscriptionTemplate(BaseModel):
    """Serializable empty template, not a completed transcription set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: WVS7LanguageArm
    source_review_id: str = Field(min_length=1)
    synthetic_fixture: bool = False
    items: list[WVS7BlankTranscriptionItem] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_template_shape(self) -> "WVS7BlankTranscriptionTemplate":
        if self.language is not WVS7LanguageArm.ENGLISH_CORE:
            raise ValueError("transcription template must be English core")
        ids = [item.variable_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("transcription template variable IDs must be unique")
        if tuple(ids) != ENGLISH_CORE_VARIABLE_IDS:
            raise ValueError("transcription template must contain exactly the 23 included English variables in stable order")
        if any(item.language is not self.language for item in self.items):
            raise ValueError("transcription template item language mismatch")
        if any(item.question_text not in {"", "TODO_MANUAL_ENTRY"} for item in self.items):
            raise ValueError("template question_text must remain blank until manual entry")
        return self


class WVS7TranscriptionTemplateError(ValueError):
    """Stable local template failure without source content in the message."""


def build_english_core_template(plan: WVS7AuthoringPlan, acceptance: WVS7AuthoringAcceptance) -> WVS7BlankTranscriptionTemplate:
    """Build a 23-row blank template from an accepted English metadata plan."""
    _validate_plan_acceptance(plan, acceptance)
    if tuple(item.variable_id for item in plan.items) != ENGLISH_CORE_VARIABLE_IDS:
        raise WVS7TranscriptionTemplateError("English authoring plan does not contain the exact 23 included variables")
    rows = [
        WVS7BlankTranscriptionItem(
            variable_id=item.variable_id,
            language=WVS7LanguageArm.ENGLISH_CORE,
            source_review_id=plan.source_review_id,
            source_questionnaire_file=item.source_questionnaire_file,
            source_codebook_file=item.source_codebook_file,
            source_questionnaire_pages=list(item.source_questionnaire_pages),
            source_codebook_pages=list(item.source_codebook_pages),
            source_sha256_prefixes=dict(item.source_sha256_prefixes),
            scale_min=item.scale_min,
            scale_max=item.scale_max,
            high_score_meaning=item.high_score_meaning,
            reverse_scored=item.reverse_scored,
            special_response_codes=dict(item.special_response_codes),
        )
        for item in plan.items
    ]
    return WVS7BlankTranscriptionTemplate(
        language=WVS7LanguageArm.ENGLISH_CORE,
        source_review_id=plan.source_review_id,
        synthetic_fixture=False,
        items=rows,
    )


def write_english_core_template(
    plan: WVS7AuthoringPlan,
    acceptance: WVS7AuthoringAcceptance,
    output_path: str | Path,
) -> WVS7BlankTranscriptionTemplate:
    """Write a new ignored local template and refuse overwrites."""
    path = Path(output_path)
    if "private_wvs_transcriptions" not in path.parts:
        raise WVS7TranscriptionTemplateError("template output must be under private_wvs_transcriptions")
    if path.exists():
        raise WVS7TranscriptionTemplateError("private transcription template already exists")
    template = build_english_core_template(plan, acceptance)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_template_toml(template), encoding="utf-8")
    return template


def load_blank_template(path: str | Path) -> WVS7BlankTranscriptionTemplate:
    """Load and validate the safe blank-template shape without PDFs."""
    local_path = Path(path)
    try:
        payload = tomllib.loads(local_path.read_text(encoding="utf-8"))
        return WVS7BlankTranscriptionTemplate.model_validate(payload)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise WVS7TranscriptionTemplateError("private transcription template is malformed") from exc


def validate_completed_english_transcription(
    plan: WVS7AuthoringPlan,
    acceptance: WVS7AuthoringAcceptance,
    transcription_path: str | Path,
) -> int:
    """Validate a researcher-completed private file against the English plan."""
    _validate_plan_acceptance(plan, acceptance)
    try:
        transcriptions = WVS7ManualTranscriptionLoader.load(transcription_path)
        return validate_manual_transcriptions(plan, transcriptions)
    except (WVS7AuthoringError, ValidationError, ValueError) as exc:
        raise WVS7TranscriptionTemplateError("manual transcription is invalid or incomplete") from exc


def make_template_from_files(
    candidate_path: str | Path,
    review_path: str | Path,
    acceptance_path: str | Path,
    output_path: str | Path,
) -> WVS7BlankTranscriptionTemplate:
    """Build the English plan from local metadata, then write its blank input."""
    from .wvs7_authoring import WVS7AuthoringAcceptanceLoader, build_wvs7_authoring_plan
    from .wvs7_candidates import WVS7CandidateLoader
    from .wvs7_review import WVS7ReviewLoader

    candidate = WVS7CandidateLoader.load(candidate_path)
    review = WVS7ReviewLoader.load(review_path)
    acceptance = WVS7AuthoringAcceptanceLoader.load(acceptance_path)
    plan = build_wvs7_authoring_plan(candidate, review, acceptance, WVS7LanguageArm.ENGLISH_CORE)
    return write_english_core_template(plan, acceptance, output_path)


def safe_template_summary(template: WVS7BlankTranscriptionTemplate, output_path: str | Path | None = None) -> dict[str, object]:
    """Return an audit-safe summary without text, option labels, or answers."""
    return {
        "status": "needs_manual_text",
        "language": template.language.value,
        "candidate_count": 24,
        "included_count": len(template.items),
        "transcribed_count": 0,
        "blocking_candidate_ids": [item.variable_id for item in template.items],
        "missing_manual_field_count": 5,
        "output_path": str(output_path) if output_path is not None else None,
    }


def _validate_plan_acceptance(plan: WVS7AuthoringPlan, acceptance: WVS7AuthoringAcceptance) -> None:
    if plan.language_arm is not WVS7LanguageArm.ENGLISH_CORE:
        raise WVS7TranscriptionTemplateError("only the English core plan may generate this template")
    if acceptance.acceptance_status is not AcceptanceStatus.ACCEPTED or WVS7LanguageArm.ENGLISH_CORE not in acceptance.acceptance_scope:
        raise WVS7TranscriptionTemplateError("researcher acceptance does not cover English core authoring")
    if plan.acceptance_id != acceptance.acceptance_id:
        raise WVS7TranscriptionTemplateError("authoring plan acceptance_id does not match acceptance receipt")
    if plan.status is not AuthoringPlanStatus.READY_FOR_AUTHORING:
        raise WVS7TranscriptionTemplateError("authoring plan is not ready_for_authoring")
    if plan.candidate_count != 24 or plan.included_count != 23 or plan.excluded_count != 1:
        raise WVS7TranscriptionTemplateError("English authoring plan counts are not 24/23/1")
    if any(item.variable_id == "Q48" for item in plan.items):
        raise WVS7TranscriptionTemplateError("Q48 is excluded and cannot enter the template")
    try:
        for item in plan.items:
            WVS7AuthoringPlanItem.model_validate(item.model_dump(mode="python"))
    except ValidationError as exc:
        raise WVS7TranscriptionTemplateError("authoring plan special-code contract is invalid") from exc


def _template_toml(template: WVS7BlankTranscriptionTemplate) -> str:
    lines = [
        f"language = {_toml_string(template.language.value)}",
        f"source_review_id = {_toml_string(template.source_review_id)}",
        "synthetic_fixture = false",
        "",
    ]
    for item in template.items:
        lines.extend(
            [
                "[[items]]",
                f"variable_id = {_toml_string(item.variable_id)}",
                f"language = {_toml_string(item.language.value)}",
                f"source_review_id = {_toml_string(item.source_review_id)}",
                f"source_questionnaire_file = {_toml_string(item.source_questionnaire_file)}",
                f"source_codebook_file = {_toml_string(item.source_codebook_file)}",
                f"source_questionnaire_pages = {_toml_array(item.source_questionnaire_pages)}",
                f"source_codebook_pages = {_toml_array(item.source_codebook_pages)}",
                f"source_sha256_prefixes = {_toml_mapping(item.source_sha256_prefixes)}",
                "question_text = \"TODO_MANUAL_ENTRY\"",
                "response_options = []",
                f"scale_min = {item.scale_min}",
                f"scale_max = {item.scale_max}",
                f"high_score_meaning = {_toml_string(item.high_score_meaning)}",
                f"reverse_scored = {str(item.reverse_scored).lower()}",
                "endpoint_labels = {}",
                f"special_response_codes = {_toml_mapping(item.special_response_codes)}",
                "manual_transcriber_id = \"TODO_MANUAL_ENTRY\"",
                "transcribed_on = \"TODO_MANUAL_ENTRY\"",
                "",
            ]
        )
    return "\n".join(lines)


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _toml_array(values: list[int]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"


def _toml_mapping(values: dict[str, str]) -> str:
    return "{" + ", ".join(f"{_toml_string(key)} = {_toml_string(value)}" for key, value in sorted(values.items())) + "}"


__all__ = [
    "ENGLISH_CORE_VARIABLE_IDS",
    "WVS7BlankTranscriptionItem",
    "WVS7BlankTranscriptionTemplate",
    "WVS7TranscriptionTemplateError",
    "build_english_core_template",
    "load_blank_template",
    "make_template_from_files",
    "safe_template_summary",
    "validate_completed_english_transcription",
    "write_english_core_template",
]
