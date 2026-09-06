"""Metadata-only WVS Wave 7 authoring gate.

The module never reads source PDFs. Full question text is accepted only from an
explicit, ignored local transcription set and can be written only for a marked
synthetic fixture.
"""

from __future__ import annotations

import tomllib
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import ProbeSet, ValueProbeItem
from .wvs7_candidates import InstrumentDomain, WVS7CandidateManifest, validate_wvs7_candidates
from .wvs7_review import AuthoringStatus, LanguageArmStatus, WVS7ReviewError, WVS7ReviewManifest, validate_wvs7_review


class WVS7LanguageArm(str, Enum):
    ENGLISH_CORE = "english_core"
    CHINESE_APPLIED = "chinese_applied"


class AcceptanceStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"


class AuthoringPlanStatus(str, Enum):
    BLOCKED = "blocked"
    READY_FOR_AUTHORING = "ready_for_authoring"


class AuthoringGateStatus(str, Enum):
    BLOCKED = "blocked"
    READY_FOR_FREEZE = "ready_for_freeze"


class WVS7AuthoringAcceptance(BaseModel):
    """A researcher acceptance receipt, separate from Codex's technical review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    acceptance_id: str = Field(min_length=1)
    acceptance_status: AcceptanceStatus
    review_id: str = Field(min_length=1)
    accepted_by: str | None = None
    accepted_on: date | None = None
    acceptance_scope: list[WVS7LanguageArm] = Field(min_length=1)
    confirm_preliminary_decisions: bool = False
    confirm_q48_exclusion: bool = False
    confirm_special_code_handling: bool = False
    confirm_cross_language_independence: bool = False
    notes: str | None = None

    @field_validator("acceptance_id", "review_id", "accepted_by", "notes", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("acceptance text must be a string")
        value = value.strip()
        return value or None

    @field_validator("accepted_on", mode="before")
    @classmethod
    def normalize_date(cls, value: object) -> object:
        return None if value == "" else value

    @model_validator(mode="after")
    def require_accepted_fields(self) -> "WVS7AuthoringAcceptance":
        if len(self.acceptance_scope) != len(set(self.acceptance_scope)):
            raise ValueError("acceptance_scope must not repeat a language arm")
        if self.acceptance_status is AcceptanceStatus.ACCEPTED:
            if self.accepted_by is None or self.accepted_on is None:
                raise ValueError("accepted authoring requires accepted_by and accepted_on")
            if not all(
                (
                    self.confirm_preliminary_decisions,
                    self.confirm_q48_exclusion,
                    self.confirm_special_code_handling,
                    self.confirm_cross_language_independence,
                )
            ):
                raise ValueError("accepted authoring requires every confirmation")
        return self


class WVS7AuthoringPlanItem(BaseModel):
    """A metadata row; it intentionally has no question or option text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variable_id: str = Field(min_length=1)
    instrument_domain: InstrumentDomain
    source_questionnaire_file: str = Field(min_length=1)
    source_codebook_file: str = Field(min_length=1)
    source_questionnaire_pages: list[int] = Field(min_length=1)
    source_codebook_pages: list[int] = Field(min_length=1)
    source_sha256_prefixes: dict[str, str] = Field(min_length=2)
    scale_min: int
    scale_max: int
    high_score_meaning: str = Field(min_length=1)
    reverse_scored: bool
    requires_manual_transcription: Literal[True] = True
    response_option_codes: list[str] = Field(min_length=2)
    endpoint_codes: list[str] = Field(min_length=2, max_length=2)
    scoring_mode: str = Field(min_length=1)
    special_response_codes: dict[str, str] = Field(default_factory=dict)
    authoring_notes: list[str] = Field(min_length=1)

    @field_validator("source_questionnaire_pages", "source_codebook_pages")
    @classmethod
    def positive_pages(cls, value: list[int]) -> list[int]:
        if any(page < 1 for page in value) or len(value) != len(set(value)):
            raise ValueError("page locators must be unique positive integers")
        return value

    @field_validator("source_sha256_prefixes")
    @classmethod
    def hash_prefixes(cls, value: dict[str, str]) -> dict[str, str]:
        for filename, prefix in value.items():
            prefix = prefix.lower().strip()
            if not filename or len(prefix) != 12 or any(c not in "0123456789abcdef" for c in prefix):
                raise ValueError("source_sha256_prefixes must use audited 12-character SHA-256 prefixes")
            value[filename] = prefix
        return value

    @model_validator(mode="after")
    def consistent_codes(self) -> "WVS7AuthoringPlanItem":
        if self.scale_min >= self.scale_max:
            raise ValueError("scale_min must be lower than scale_max")
        if len(self.response_option_codes) != len(set(self.response_option_codes)):
            raise ValueError("response_option_codes must not repeat")
        if self.endpoint_codes != [str(self.scale_min), str(self.scale_max)]:
            raise ValueError("endpoint_codes must match scale endpoints")
        if self.variable_id in {"Q149", "Q150"}:
            if self.scoring_mode != "binary_categorical_not_continuous" or self.special_response_codes.get("1|2") != "binary_categorical_not_continuous_or_equidistant":
                raise ValueError("Q149 and Q150 must remain binary categorical, not continuous")
        if self.variable_id == "Q111" and self.special_response_codes.get("3") != "volunteered_other_exclude_from_ordinary_ordinal_scoring":
            raise ValueError("Q111 must declare volunteered code 3 handling")
        if self.variable_id in _DEMOCRACY_SPECIAL_CODE_IDS and self.special_response_codes.get("0") != "volunteered_against_democracy_exclude_from_ordinary_1_to_10_distance":
            raise ValueError("democracy items must declare volunteered code 0 handling")
        if "evicon_operational_grouping_only_not_wvs_official_factor" not in self.authoring_notes:
            raise ValueError("authoring notes must preserve the operational-grouping boundary")
        return self


class WVS7AuthoringPlan(BaseModel):
    """One language-specific, pre-freeze authoring plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str = Field(min_length=1)
    language_arm: WVS7LanguageArm
    source_review_id: str = Field(min_length=1)
    acceptance_id: str = Field(min_length=1)
    status: AuthoringPlanStatus
    candidate_count: int = Field(ge=0)
    included_count: int = Field(ge=0)
    excluded_count: int = Field(ge=0)
    items: list[WVS7AuthoringPlanItem] = Field(min_length=1)

    @model_validator(mode="after")
    def reconcile_counts(self) -> "WVS7AuthoringPlan":
        ids = [item.variable_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("authoring plan variable IDs must be unique")
        if self.included_count != len(ids) or self.candidate_count != self.included_count + self.excluded_count:
            raise ValueError("authoring plan counts do not reconcile")
        return self


class WVS7ManualResponseOption(BaseModel):
    """A full option label allowed only in ignored local transcription input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    label: str = Field(min_length=1)


class WVS7ManualTranscriptionItem(BaseModel):
    """One full, researcher-entered local transcription."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variable_id: str = Field(min_length=1)
    language: WVS7LanguageArm
    source_review_id: str = Field(min_length=1)
    source_questionnaire_file: str = Field(min_length=1)
    source_codebook_file: str = Field(min_length=1)
    source_questionnaire_pages: list[int] = Field(min_length=1)
    source_codebook_pages: list[int] = Field(min_length=1)
    source_sha256_prefixes: dict[str, str] = Field(min_length=2)
    question_text: str = Field(min_length=1)
    response_options: list[WVS7ManualResponseOption] = Field(default_factory=list)
    scale_min: int
    scale_max: int
    high_score_meaning: str | None = None
    reverse_scored: bool | None = None
    endpoint_labels: dict[str, str] = Field(default_factory=dict)
    special_response_codes: dict[str, str] = Field(default_factory=dict)
    manual_transcriber_id: str = Field(min_length=1)
    transcribed_on: date
    transcription_origin: str | None = None
    human_verified: bool | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("question_text", "manual_transcriber_id")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("manual transcription text must not be blank")
        return value

    @field_validator("source_questionnaire_pages", "source_codebook_pages")
    @classmethod
    def valid_pages(cls, value: list[int]) -> list[int]:
        if any(page < 1 for page in value) or len(value) != len(set(value)):
            raise ValueError("manual transcription pages must be unique positive integers")
        return value

    @field_validator("source_sha256_prefixes")
    @classmethod
    def valid_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        return WVS7AuthoringPlanItem.hash_prefixes(value)

    @model_validator(mode="after")
    def valid_transcription(self) -> "WVS7ManualTranscriptionItem":
        if self.scale_min >= self.scale_max:
            raise ValueError("manual transcription scale_min must be lower than scale_max")
        draft = self.transcription_origin == "local_pdf_text_extraction_draft" and self.human_verified is False
        if self.transcription_origin is not None and self.transcription_origin not in {
            "local_pdf_text_extraction_draft",
            "manual_researcher_transcription",
        }:
            raise ValueError("transcription_origin is not an approved local origin")
        if self.transcription_origin == "local_pdf_text_extraction_draft" and self.human_verified is not False:
            raise ValueError("PDF extraction drafts must remain human_verified=false")
        codes = [option.code for option in self.response_options]
        if len(codes) != len(set(codes)):
            raise ValueError("manual response option codes must be unique")
        if not draft and len(self.response_options) < 2:
            raise ValueError("response_options must contain at least two entries")
        if not draft and set(self.endpoint_labels) != {str(self.scale_min), str(self.scale_max)}:
            raise ValueError("endpoint_labels must cover exactly both scale endpoints")
        return self


class WVS7ManualTranscriptionSet(BaseModel):
    """Full local input for one arm, normally under ignored private storage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: WVS7LanguageArm
    source_review_id: str = Field(min_length=1)
    synthetic_fixture: bool = False
    items: list[WVS7ManualTranscriptionItem] = Field(min_length=1)

    @model_validator(mode="after")
    def aligned_items(self) -> "WVS7ManualTranscriptionSet":
        ids = [item.variable_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("manual transcription IDs must be unique")
        if any(item.language is not self.language for item in self.items):
            raise ValueError("manual transcription language mismatch")
        if any(item.source_review_id != self.source_review_id for item in self.items):
            raise ValueError("manual transcription review mismatch")
        expected_questionnaire_marker = "USA_2017_English" if self.language is WVS7LanguageArm.ENGLISH_CORE else "China_2018_Mandarin"
        if any(expected_questionnaire_marker not in item.source_questionnaire_file for item in self.items):
            raise ValueError("manual transcription source questionnaire does not match language arm")
        return self


class WVS7AuthoringGateReport(BaseModel):
    """Safe CLI envelope with no full WVS content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language_arm: WVS7LanguageArm
    status: AuthoringGateStatus
    candidate_count: int = Field(ge=0)
    included_count: int = Field(ge=0)
    excluded_count: int = Field(ge=0)
    transcribed_count: int = Field(ge=0)
    blocking_reasons: list[str]
    blocking_candidate_ids: list[str]
    source_review_id: str
    acceptance_status: AcceptanceStatus


class WVS7AuthoringError(ValueError):
    """Stable local validation error."""


class WVS7AuthoringAcceptanceLoader:
    @staticmethod
    def load(path: str | Path) -> WVS7AuthoringAcceptance:
        payload = _load_toml(path, "authoring acceptance")
        try:
            return WVS7AuthoringAcceptance.model_validate(payload)
        except ValueError as exc:
            raise WVS7AuthoringError("authoring acceptance violates its contract") from exc


class WVS7ManualTranscriptionLoader:
    @staticmethod
    def load(path: str | Path) -> WVS7ManualTranscriptionSet:
        payload = _load_toml(path, "manual transcription")
        try:
            return WVS7ManualTranscriptionSet.model_validate(payload)
        except ValueError as exc:
            raise WVS7AuthoringError("manual transcription violates its contract") from exc


def build_wvs7_authoring_plan(
    candidate_manifest: WVS7CandidateManifest,
    review_manifest: WVS7ReviewManifest,
    acceptance: WVS7AuthoringAcceptance,
    language_arm: WVS7LanguageArm,
) -> WVS7AuthoringPlan:
    """Build one metadata-only plan from the preliminary review."""
    _validate_inputs(candidate_manifest, review_manifest, acceptance)
    candidates = {candidate.variable_id: candidate for candidate in candidate_manifest.candidates}
    reviews = {item.variable_id: item for item in review_manifest.items}
    rows: list[WVS7AuthoringPlanItem] = []
    for variable_id in sorted(candidates, key=_variable_sort_key):
        review_item = reviews[variable_id]
        arm = getattr(review_item, language_arm.value)
        if variable_id == "Q48":
            if arm.status is not LanguageArmStatus.EXCLUDED:
                raise WVS7AuthoringError("Q48 must remain excluded")
            continue
        if arm.status is not LanguageArmStatus.INCLUDED:
            raise WVS7AuthoringError("all non-Q48 candidates must be included by preliminary review")
        source = candidates[variable_id].usa if language_arm is WVS7LanguageArm.ENGLISH_CORE else candidates[variable_id].china
        scale_min = _required_int(arm.scale_min, variable_id)
        scale_max = _required_int(arm.scale_max, variable_id)
        rows.append(
            WVS7AuthoringPlanItem(
                variable_id=variable_id,
                instrument_domain=review_item.instrument_domain,
                source_questionnaire_file=source.questionnaire_file,
                source_codebook_file=source.codebook_file,
                source_questionnaire_pages=list(arm.questionnaire_pages),
                source_codebook_pages=list(arm.codebook_pages),
                source_sha256_prefixes={
                    source.questionnaire_file: _source_prefix(candidate_manifest, source.questionnaire_file),
                    source.codebook_file: _source_prefix(candidate_manifest, source.codebook_file),
                },
                scale_min=scale_min,
                scale_max=scale_max,
                high_score_meaning=_required_text(arm.high_score_meaning, variable_id),
                reverse_scored=_required_bool(arm.reverse_scored, variable_id),
                requires_manual_transcription=True,
                response_option_codes=_response_codes(variable_id, scale_min, scale_max),
                endpoint_codes=[str(scale_min), str(scale_max)],
                scoring_mode=_scoring_mode(variable_id),
                special_response_codes=_special_codes(variable_id),
                authoring_notes=_authoring_notes(variable_id),
            )
        )
    if len(rows) != 23 or {row.variable_id for row in rows} != set(candidates) - {"Q48"}:
        raise WVS7AuthoringError("authoring plan must contain exactly 23 non-Q48 candidates")
    status = AuthoringPlanStatus.READY_FOR_AUTHORING if (
        acceptance.acceptance_status is AcceptanceStatus.ACCEPTED and language_arm in acceptance.acceptance_scope
    ) else AuthoringPlanStatus.BLOCKED
    return WVS7AuthoringPlan(
        plan_id=f"{review_manifest.review_id}:{language_arm.value}:pre-freeze",
        language_arm=language_arm,
        source_review_id=review_manifest.review_id,
        acceptance_id=acceptance.acceptance_id,
        status=status,
        candidate_count=24,
        included_count=23,
        excluded_count=1,
        items=rows,
    )


def validate_manual_transcriptions(plan: WVS7AuthoringPlan, transcriptions: WVS7ManualTranscriptionSet) -> int:
    """Validate explicit local transcription metadata against one plan."""
    if transcriptions.language is not plan.language_arm or transcriptions.source_review_id != plan.source_review_id:
        raise WVS7AuthoringError("manual transcription language or review mismatch")
    expected = {item.variable_id: item for item in plan.items}
    actual = {item.variable_id: item for item in transcriptions.items}
    if set(expected) != set(actual):
        raise WVS7AuthoringError("manual transcription must exactly cover the 23 included variables")
    for variable_id, expected_item in expected.items():
        actual_item = actual[variable_id]
        if actual_item.human_verified is False:
            if actual_item.transcription_origin != "local_pdf_text_extraction_draft":
                raise WVS7AuthoringError("human verification required for local transcription draft")
            raise WVS7AuthoringError("human verification required")
        if actual_item.question_text.strip() in {"", "TODO_MANUAL_ENTRY"}:
            raise WVS7AuthoringError("manual transcription question_text is still TODO_MANUAL_ENTRY")
        if actual_item.manual_transcriber_id.strip() in {"", "TODO_MANUAL_ENTRY"}:
            raise WVS7AuthoringError("manual transcription manual_transcriber_id is still TODO_MANUAL_ENTRY")
        if actual_item.transcribed_on is None:
            raise WVS7AuthoringError("manual transcription transcribed_on is missing")
        checks = (
            (actual_item.source_questionnaire_file == expected_item.source_questionnaire_file, "questionnaire file"),
            (actual_item.source_codebook_file == expected_item.source_codebook_file, "codebook file"),
            (actual_item.source_questionnaire_pages == expected_item.source_questionnaire_pages, "questionnaire pages"),
            (actual_item.source_codebook_pages == expected_item.source_codebook_pages, "codebook pages"),
            (actual_item.source_sha256_prefixes == expected_item.source_sha256_prefixes, "source hashes"),
            ((actual_item.scale_min, actual_item.scale_max) == (expected_item.scale_min, expected_item.scale_max), "scale"),
            (actual_item.high_score_meaning is None or actual_item.high_score_meaning == expected_item.high_score_meaning, "high-score meaning"),
            (actual_item.reverse_scored is None or actual_item.reverse_scored == expected_item.reverse_scored, "reverse-score decision"),
            (set(actual_item.endpoint_labels) == set(expected_item.endpoint_codes), "endpoints"),
            ([option.code for option in actual_item.response_options] == expected_item.response_option_codes, "option codes"),
            (actual_item.special_response_codes == expected_item.special_response_codes, "special response codes"),
        )
        for valid, label in checks:
            if not valid:
                raise WVS7AuthoringError(f"manual transcription {label} does not match authoring plan")
    return len(actual)


def evaluate_wvs7_authoring_gate(
    candidate_manifest: WVS7CandidateManifest,
    review_manifest: WVS7ReviewManifest,
    acceptance: WVS7AuthoringAcceptance,
    language_arm: WVS7LanguageArm,
    transcriptions: WVS7ManualTranscriptionSet | None = None,
) -> tuple[WVS7AuthoringPlan, WVS7AuthoringGateReport]:
    """Return readiness only; no ProbeSet or result path is written."""
    plan = build_wvs7_authoring_plan(candidate_manifest, review_manifest, acceptance, language_arm)
    reasons: list[str] = []
    blocking_ids: list[str] = []
    transcribed_count = 0
    if acceptance.acceptance_status is AcceptanceStatus.PENDING:
        reasons.append("acceptance_pending")
    elif language_arm not in acceptance.acceptance_scope:
        reasons.append("acceptance_scope_missing_language_arm")
    if transcriptions is None:
        reasons.append("manual_transcription_required")
        blocking_ids = [item.variable_id for item in plan.items]
    else:
        transcribed_count = validate_manual_transcriptions(plan, transcriptions)
    ready = not reasons and transcribed_count == plan.included_count
    return plan, WVS7AuthoringGateReport(
        language_arm=language_arm,
        status=AuthoringGateStatus.READY_FOR_FREEZE if ready else AuthoringGateStatus.BLOCKED,
        candidate_count=plan.candidate_count,
        included_count=plan.included_count,
        excluded_count=plan.excluded_count,
        transcribed_count=transcribed_count,
        blocking_reasons=reasons,
        blocking_candidate_ids=sorted(blocking_ids, key=_variable_sort_key),
        source_review_id=plan.source_review_id,
        acceptance_status=acceptance.acceptance_status,
    )


def freeze_synthetic_probe_set(
    plan: WVS7AuthoringPlan,
    gate_report: WVS7AuthoringGateReport,
    transcriptions: WVS7ManualTranscriptionSet,
    output_path: str | Path,
    *,
    confirm_freeze: bool,
) -> ProbeSet:
    """Write only a marked synthetic fixture; formal WVS freezing is not enabled."""
    if not confirm_freeze:
        raise WVS7AuthoringError("confirm_freeze is required")
    if gate_report.status is not AuthoringGateStatus.READY_FOR_FREEZE:
        raise WVS7AuthoringError("authoring gate is not ready_for_freeze")
    if not transcriptions.synthetic_fixture:
        raise WVS7AuthoringError("only synthetic_fixture input may generate a ProbeSet")
    validate_manual_transcriptions(plan, transcriptions)
    path = Path(output_path)
    if path.exists():
        raise WVS7AuthoringError("synthetic ProbeSet output already exists")
    values = {item.variable_id: item for item in transcriptions.items}
    probe_items = [
        ValueProbeItem(
            probe_id=f"synthetic-{plan.language_arm.value}-{item.variable_id}",
            text=values[item.variable_id].question_text,
            dimension=item.instrument_domain.value,
            response_scale=[option.label for option in values[item.variable_id].response_options],
            is_holdout=False,
            reverse_scored=item.reverse_scored,
        )
        for item in plan.items
    ]
    probe_set = ProbeSet(
        probe_set_id=f"synthetic-{plan.plan_id}",
        dimensions=list(dict.fromkeys(item.instrument_domain.value for item in plan.items)),
        items=probe_items,
        version="synthetic-test-only.v1",
        metadata={"synthetic_fixture": True, "not_wvs_probe_set": True},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(probe_set.model_dump_json(indent=2), encoding="utf-8")
    return probe_set


def load_and_evaluate_wvs7_authoring_gate(candidate_path: str | Path, review_path: str | Path, acceptance_path: str | Path, language_arm: WVS7LanguageArm, transcription_path: str | Path | None = None) -> tuple[WVS7AuthoringPlan, WVS7AuthoringGateReport]:
    from .wvs7_candidates import WVS7CandidateLoader
    from .wvs7_review import WVS7ReviewLoader

    transcriptions = None if transcription_path is None else WVS7ManualTranscriptionLoader.load(transcription_path)
    return evaluate_wvs7_authoring_gate(
        WVS7CandidateLoader.load(candidate_path),
        WVS7ReviewLoader.load(review_path),
        WVS7AuthoringAcceptanceLoader.load(acceptance_path),
        language_arm,
        transcriptions,
    )


def safe_gate_summary(report: WVS7AuthoringGateReport) -> dict[str, object]:
    return {
        "language_arm": report.language_arm.value,
        "status": report.status.value,
        "candidate_count": report.candidate_count,
        "included_count": report.included_count,
        "excluded_count": report.excluded_count,
        "transcribed_count": report.transcribed_count,
        "blocking_reasons": report.blocking_reasons,
        "blocking_candidate_ids": report.blocking_candidate_ids,
        "source_review_id": report.source_review_id,
        "acceptance_status": report.acceptance_status.value,
    }


def _load_toml(path: str | Path, label: str) -> dict[str, object]:
    try:
        raw = Path(path).read_bytes()
        payload = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise WVS7AuthoringError(f"cannot load local {label}") from exc
    if not isinstance(payload, dict):
        raise WVS7AuthoringError(f"local {label} root must be an object")
    return payload


def _validate_inputs(candidate: WVS7CandidateManifest, review: WVS7ReviewManifest, acceptance: WVS7AuthoringAcceptance) -> None:
    try:
        validate_wvs7_candidates(candidate)
        report = validate_wvs7_review(candidate, review)
    except (WVS7ReviewError, ValueError) as exc:
        raise WVS7AuthoringError("candidate or preliminary review is invalid") from exc
    if report.status is not AuthoringStatus.READY_FOR_AUTHORING:
        raise WVS7AuthoringError("preliminary review is not ready_for_authoring")
    if acceptance.review_id != review.review_id:
        raise WVS7AuthoringError("acceptance review_id does not match preliminary review")


def _source_prefix(candidate: WVS7CandidateManifest, filename: str) -> str:
    for source in candidate.source_files:
        if source.filename == filename:
            return source.sha256[:12]
    raise WVS7AuthoringError("source filename is absent from candidate receipt")


def _required_int(value: int | None, variable_id: str) -> int:
    if value is None:
        raise WVS7AuthoringError(f"scale metadata missing for {variable_id}")
    return value


def _required_bool(value: bool | None, variable_id: str) -> bool:
    if value is None:
        raise WVS7AuthoringError(f"reverse-score metadata missing for {variable_id}")
    return value


def _required_text(value: str | None, variable_id: str) -> str:
    if value is None or not value.strip():
        raise WVS7AuthoringError(f"direction metadata missing for {variable_id}")
    return value


def _response_codes(variable_id: str, scale_min: int, scale_max: int) -> list[str]:
    values = [str(value) for value in range(scale_min, scale_max + 1)]
    return ["0", *values] if variable_id in _DEMOCRACY_SPECIAL_CODE_IDS else values


def _scoring_mode(variable_id: str) -> str:
    if variable_id in {"Q149", "Q150"}:
        return "binary_categorical_not_continuous"
    if variable_id == "Q111":
        return "ordered_substantive_codes_with_volunteered_other"
    if variable_id in _DEMOCRACY_SPECIAL_CODE_IDS:
        return "ordered_1_to_10_with_volunteered_non_substantive_code"
    return "ordered_as_reviewed"


def _special_codes(variable_id: str) -> dict[str, str]:
    if variable_id == "Q111":
        return {"3": "volunteered_other_exclude_from_ordinary_ordinal_scoring"}
    if variable_id in {"Q149", "Q150"}:
        return {
            "1|2": "binary_categorical_not_continuous_or_equidistant",
            "_scoring_mode": "binary_categorical_not_continuous",
        }
    if variable_id in _DEMOCRACY_SPECIAL_CODE_IDS:
        return {"0": "volunteered_against_democracy_exclude_from_ordinary_1_to_10_distance"}
    if variable_id in {"Q158", "Q159", "Q160", "Q161", "Q162", "Q163"}:
        return {"_analysis_note": "secondary_science_technology_attitude_analysis_not_pure_value_factor"}
    return {}


def _authoring_notes(variable_id: str) -> list[str]:
    notes = ["evicon_operational_grouping_only_not_wvs_official_factor"]
    if variable_id in {"Q158", "Q159", "Q160", "Q161", "Q162", "Q163"}:
        notes.append("secondary_science_technology_attitude_analysis_not_pure_value_factor")
    if variable_id == "Q246":
        notes.append("english_chinese_semantic_scope_requires_manual_cross_language_review")
    return notes


def _variable_sort_key(variable_id: str) -> int:
    return int(variable_id[1:])


_DEMOCRACY_SPECIAL_CODE_IDS = frozenset({"Q241", "Q243", "Q246", "Q247", "Q248", "Q249"})

# Short aliases keep the contract discoverable without creating a second model
# hierarchy for callers that use the generic authoring terminology.
AuthoringAcceptance = WVS7AuthoringAcceptance
AuthoringPlan = WVS7AuthoringPlan
AuthoringPlanItem = WVS7AuthoringPlanItem
ManualTranscriptionItem = WVS7ManualTranscriptionItem
ManualTranscriptionSet = WVS7ManualTranscriptionSet
build_authoring_plan = build_wvs7_authoring_plan
evaluate_authoring_gate = evaluate_wvs7_authoring_gate


__all__ = [
    "AcceptanceStatus", "AuthoringGateStatus", "AuthoringPlanStatus", "WVS7AuthoringAcceptance",
    "WVS7AuthoringAcceptanceLoader", "WVS7AuthoringError", "WVS7AuthoringGateReport", "WVS7AuthoringPlan",
    "WVS7AuthoringPlanItem", "WVS7LanguageArm", "WVS7ManualResponseOption", "WVS7ManualTranscriptionItem",
    "WVS7ManualTranscriptionLoader", "WVS7ManualTranscriptionSet", "build_wvs7_authoring_plan",
    "evaluate_wvs7_authoring_gate", "freeze_synthetic_probe_set", "load_and_evaluate_wvs7_authoring_gate",
    "safe_gate_summary", "validate_manual_transcriptions",
    "AuthoringAcceptance", "AuthoringPlan", "AuthoringPlanItem", "ManualTranscriptionItem",
    "ManualTranscriptionSet", "build_authoring_plan", "evaluate_authoring_gate",
]
