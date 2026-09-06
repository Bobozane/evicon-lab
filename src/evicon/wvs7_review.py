"""Draft-only manual signoff contracts for WVS Wave 7 candidate mappings."""

from __future__ import annotations

import tomllib
from collections import Counter
from datetime import date
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .wvs7_candidates import (
    InstrumentDomain,
    WVS7CandidateError,
    WVS7CandidateManifest,
    validate_wvs7_candidates,
)


_LANGUAGE_FIELDS = ("english_core", "chinese_applied")


class ReviewArtifactStatus(str, Enum):
    DRAFT = "draft"


class LanguageArmStatus(str, Enum):
    PENDING = "pending"
    INCLUDED = "included"
    EXCLUDED = "excluded"


class AuthoringStatus(str, Enum):
    BLOCKED = "blocked"
    READY_FOR_AUTHORING = "ready_for_authoring"


class WVS7LanguageReviewArm(BaseModel):
    """One independently reviewed language arm; no item text is permitted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: LanguageArmStatus
    reviewer_id: str | None = None
    reviewed_on: date | None = None
    codebook_pages: list[int] = Field(default_factory=list)
    questionnaire_pages: list[int] = Field(min_length=1)
    scale_min: int | None = None
    scale_max: int | None = None
    high_score_meaning: str | None = None
    reverse_scored: bool | None = None
    rationale: str | None = None

    @field_validator("reviewer_id", "high_score_meaning", "rationale", mode="before")
    @classmethod
    def empty_text_is_pending(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("review text fields must be strings")
        normalized = value.strip()
        return normalized or None

    @field_validator("reviewed_on", mode="before")
    @classmethod
    def empty_date_is_pending(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("scale_min", "scale_max", "reverse_scored", mode="before")
    @classmethod
    def empty_value_is_pending(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("codebook_pages", "questionnaire_pages")
    @classmethod
    def validate_page_numbers(cls, value: list[int]) -> list[int]:
        if any(page < 1 for page in value):
            raise ValueError("review page locators must be positive PDF page numbers")
        if len(value) != len(set(value)):
            raise ValueError("review page locators must not repeat a page number")
        return value

    @model_validator(mode="after")
    def validate_status_requirements(self) -> "WVS7LanguageReviewArm":
        if self.status is LanguageArmStatus.PENDING:
            return self
        if self.reviewer_id is None or self.reviewed_on is None or self.rationale is None:
            raise ValueError("included and excluded language arms require reviewer_id, reviewed_on, and rationale")
        if not self.codebook_pages:
            raise ValueError("included and excluded language arms require codebook_pages")
        if self.status is LanguageArmStatus.INCLUDED:
            if self.scale_min is None or self.scale_max is None or self.scale_min >= self.scale_max:
                raise ValueError("included language arms require a valid scale_min and scale_max range")
            if self.high_score_meaning is None:
                raise ValueError("included language arms require high_score_meaning")
            if self.reverse_scored is None:
                raise ValueError("included language arms require a reverse_scored decision")
        return self


class WVS7ReviewItem(BaseModel):
    """Review rows mirror candidate IDs and operational groups only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variable_id: str = Field(min_length=2)
    instrument_domain: InstrumentDomain
    english_core: WVS7LanguageReviewArm
    chinese_applied: WVS7LanguageReviewArm


class WVS7ReviewMetadata(BaseModel):
    """Fixed non-content metadata that prevents question-text leakage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    purpose: str = Field(min_length=1)
    operational_grouping_only: bool
    wvs_official_factor_structure: bool
    pdf_runtime_dependency: bool
    network_access: bool
    creates_probe_set: bool


class WVS7ReviewManifest(BaseModel):
    """A signoff template and completed review artifact, never a ProbeSet."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    review_id: str = Field(min_length=1)
    review_version: str = Field(min_length=1)
    artifact_status: ReviewArtifactStatus
    candidate_study_id: str = Field(min_length=1)
    candidate_manifest_version: str = Field(min_length=1)
    operational_groups: list[InstrumentDomain] = Field(min_length=5, max_length=5)
    source_hashes: dict[str, str] = Field(min_length=4, max_length=4)
    items: list[WVS7ReviewItem] = Field(min_length=1)
    metadata: WVS7ReviewMetadata

    @field_validator("source_hashes")
    @classmethod
    def validate_hash_values(cls, value: dict[str, str]) -> dict[str, str]:
        for filename, digest in value.items():
            if not filename or not isinstance(digest, str):
                raise ValueError("source_hashes must use filenames and string digests")
            normalized = digest.lower().strip()
            if len(normalized) not in {12, 64} or any(character not in "0123456789abcdef" for character in normalized):
                raise ValueError("source_hashes must use audited 12-character prefixes or full SHA-256 digests")
            value[filename] = normalized
        return value

    @model_validator(mode="after")
    def validate_review_shape(self) -> "WVS7ReviewManifest":
        if len(self.operational_groups) != len(set(self.operational_groups)):
            raise ValueError("operational_groups must not contain duplicates")
        if set(self.operational_groups) != set(InstrumentDomain):
            raise ValueError("operational_groups must preserve the five EviCon groups")
        if self.metadata.creates_probe_set:
            raise ValueError("WVS review artifacts cannot create a ProbeSet")
        return self


class LanguageReviewSummary(BaseModel):
    """Safe per-language readiness result without question content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language_arm: str
    blocked: bool
    ready_for_authoring: bool
    included_count: int = Field(ge=0)
    excluded_count: int = Field(ge=0)
    pending_count: int = Field(ge=0)
    blocking_reasons: list[str]
    blocking_candidate_ids: list[str]


class WVS7ReviewValidationReport(BaseModel):
    """Safe review report; readiness never creates a ProbeSet or experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    review_id: str
    artifact_status: ReviewArtifactStatus
    candidate_count: int = Field(ge=0)
    status: AuthoringStatus
    english_core: LanguageReviewSummary
    chinese_applied: LanguageReviewSummary
    next_step: str


class WVS7ReviewError(ValueError):
    """Stable local review-signoff validation failure."""


class WVS7ReviewLoader:
    """Load a local review TOML file without accessing WVS PDF sources."""

    @staticmethod
    def load(path: str | Path) -> WVS7ReviewManifest:
        review_path = Path(path)
        try:
            raw = review_path.read_bytes()
        except OSError as exc:
            raise WVS7ReviewError("cannot read local WVS review manifest") from exc
        try:
            payload = tomllib.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise WVS7ReviewError("cannot parse local WVS review manifest") from exc
        if not isinstance(payload, dict):
            raise WVS7ReviewError("WVS review manifest root must be an object")
        try:
            return WVS7ReviewManifest.model_validate(payload)
        except ValueError as exc:
            raise WVS7ReviewError("WVS review manifest violates a required contract") from exc


def validate_wvs7_review(
    candidate_manifest: WVS7CandidateManifest,
    review_manifest: WVS7ReviewManifest,
) -> WVS7ReviewValidationReport:
    """Validate review completeness using only the candidate manifest and review TOML."""
    try:
        validate_wvs7_candidates(candidate_manifest)
    except WVS7CandidateError as exc:
        raise WVS7ReviewError("candidate manifest is not a valid draft review input") from exc

    if review_manifest.candidate_study_id != candidate_manifest.study_id:
        raise WVS7ReviewError("review candidate_study_id does not match candidate manifest")
    if review_manifest.candidate_manifest_version != candidate_manifest.manifest_version:
        raise WVS7ReviewError("review candidate_manifest_version does not match candidate manifest")
    if review_manifest.operational_groups != candidate_manifest.instrument_domains:
        raise WVS7ReviewError("review operational groups do not match candidate manifest")
    _validate_source_hashes(candidate_manifest, review_manifest)

    candidate_by_id = {candidate.variable_id: candidate for candidate in candidate_manifest.candidates}
    review_ids = [item.variable_id for item in review_manifest.items]
    if len(review_ids) != len(set(review_ids)):
        raise WVS7ReviewError("review manifest must not contain duplicate variable_id values")
    if set(review_ids) != set(candidate_by_id):
        raise WVS7ReviewError("review manifest must cover exactly the candidate manifest variable IDs")

    for review_item in review_manifest.items:
        candidate = candidate_by_id[review_item.variable_id]
        if review_item.instrument_domain is not candidate.instrument_domain:
            raise WVS7ReviewError("review item operational group does not match candidate manifest")
        _validate_questionnaire_pages(review_item.english_core, candidate.usa.questionnaire_pages)
        _validate_questionnaire_pages(review_item.chinese_applied, candidate.china.questionnaire_pages)

    english_core = _summarize_language(review_manifest.items, "english_core")
    chinese_applied = _summarize_language(review_manifest.items, "chinese_applied")
    status = (
        AuthoringStatus.BLOCKED
        if english_core.blocked or chinese_applied.blocked
        else AuthoringStatus.READY_FOR_AUTHORING
    )
    next_step = (
        "manual_signoff_required_before_authoring"
        if status is AuthoringStatus.BLOCKED
        else "eligible_for_next_stage_human_approval_before_authoring"
    )
    return WVS7ReviewValidationReport(
        review_id=review_manifest.review_id,
        artifact_status=review_manifest.artifact_status,
        candidate_count=len(review_manifest.items),
        status=status,
        english_core=english_core,
        chinese_applied=chinese_applied,
        next_step=next_step,
    )


def load_and_validate_wvs7_review(
    candidate_path: str | Path,
    review_path: str | Path,
) -> WVS7ReviewValidationReport:
    """Convenience entry point for the review CLI; no files are written."""
    from .wvs7_candidates import WVS7CandidateLoader

    return validate_wvs7_review(WVS7CandidateLoader.load(candidate_path), WVS7ReviewLoader.load(review_path))


def _validate_source_hashes(
    candidate_manifest: WVS7CandidateManifest,
    review_manifest: WVS7ReviewManifest,
) -> None:
    candidate_hashes = {source.filename: source.sha256 for source in candidate_manifest.source_files}
    if set(review_manifest.source_hashes) != set(candidate_hashes):
        raise WVS7ReviewError("review source hashes do not cover the candidate source receipt")
    for filename, candidate_hash in candidate_hashes.items():
        review_hash = review_manifest.source_hashes[filename]
        if review_hash != candidate_hash and review_hash != candidate_hash[:12]:
            raise WVS7ReviewError("review source hash does not match candidate source receipt")


def _validate_questionnaire_pages(arm: WVS7LanguageReviewArm, expected_pages: list[int]) -> None:
    if arm.questionnaire_pages != expected_pages:
        raise WVS7ReviewError("review questionnaire pages do not match candidate locator")


def _summarize_language(
    items: list[WVS7ReviewItem],
    language_arm: str,
) -> LanguageReviewSummary:
    if language_arm not in _LANGUAGE_FIELDS:
        raise WVS7ReviewError("unknown review language arm")
    statuses = [getattr(item, language_arm).status for item in items]
    counts = Counter(statuses)
    pending_ids = [item.variable_id for item in items if getattr(item, language_arm).status is LanguageArmStatus.PENDING]
    blocked = bool(pending_ids)
    return LanguageReviewSummary(
        language_arm=language_arm,
        blocked=blocked,
        ready_for_authoring=not blocked,
        included_count=counts[LanguageArmStatus.INCLUDED],
        excluded_count=counts[LanguageArmStatus.EXCLUDED],
        pending_count=counts[LanguageArmStatus.PENDING],
        blocking_reasons=["pending_manual_review"] if blocked else [],
        blocking_candidate_ids=pending_ids,
    )


__all__ = [
    "AuthoringStatus",
    "LanguageArmStatus",
    "LanguageReviewSummary",
    "ReviewArtifactStatus",
    "WVS7LanguageReviewArm",
    "WVS7ReviewError",
    "WVS7ReviewItem",
    "WVS7ReviewLoader",
    "WVS7ReviewManifest",
    "WVS7ReviewMetadata",
    "WVS7ReviewValidationReport",
    "load_and_validate_wvs7_review",
    "validate_wvs7_review",
]
