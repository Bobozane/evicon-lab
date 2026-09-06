"""Draft-only, local contracts for manual WVS Wave 7 candidate review."""

from __future__ import annotations

import re
import tomllib
from collections import Counter
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_VARIABLE_PATTERN = re.compile(r"^Q(?:[1-9][0-9]{0,2})$")
_EXPECTED_VARIABLE_IDS = frozenset(
    {
        "Q48",
        "Q106",
        "Q107",
        "Q108",
        "Q109",
        "Q110",
        "Q111",
        "Q149",
        "Q150",
        "Q158",
        "Q159",
        "Q160",
        "Q161",
        "Q162",
        "Q163",
        "Q196",
        "Q197",
        "Q198",
        "Q241",
        "Q243",
        "Q246",
        "Q247",
        "Q248",
        "Q249",
    }
)
_EXPECTED_SOURCE_HASHES = {
    "F00008646-WVS7_Questionnaire_USA_2017_English.pdf": "57557e9b01af1ab98ab965697b92b941b44b1f5a183bd3f1b074754788e2fccc",
    "F00010562-World_Values_Survey_Wave_7_2017-2020_United_States_v3.0.pdf": "2b4225895f1491ca5ab304656302b683cd0d7002617371f12a369762a74a3395",
    "F00011251-WVS7_Questionnaire_China_2018_Mandarin.pdf": "5f912050ddbe4d46f990f4762651313cab474210df878e467936ead8272fe1f1",
    "F00011911-World_Values_Survey_Wave_7_2017-2020_China_v3.0.pdf": "0cc7db876c59a2874ab927e96a843843ef67ff51e57b484515bf9bd9f4ae4708",
}
_USA_QUESTIONNAIRE = "F00008646-WVS7_Questionnaire_USA_2017_English.pdf"
_USA_CODEBOOK = "F00010562-World_Values_Survey_Wave_7_2017-2020_United_States_v3.0.pdf"
_CHINA_QUESTIONNAIRE = "F00011251-WVS7_Questionnaire_China_2018_Mandarin.pdf"
_CHINA_CODEBOOK = "F00011911-World_Values_Survey_Wave_7_2017-2020_China_v3.0.pdf"


class CandidateMaterialStatus(str, Enum):
    DRAFT = "draft"
    PILOT = "pilot"
    FROZEN = "frozen"


class InstrumentDomain(str, Enum):
    AUTONOMY_SECURITY = "autonomy_security"
    ECONOMIC_DISTRIBUTION = "economic_distribution"
    SCIENCE_KNOWLEDGE = "science_knowledge"
    SURVEILLANCE_PRIVACY = "surveillance_privacy"
    DEMOCRATIC_ORDER = "democratic_order"


class ReviewStatus(str, Enum):
    PENDING = "pending"


class WVS7SourceFile(BaseModel):
    """A checked local source receipt; PDF bytes are not a validator dependency."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        if value not in _EXPECTED_SOURCE_HASHES:
            raise ValueError("filename is not an approved local WVS Wave 7 source")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_hash_shape(cls, value: str) -> str:
        normalized = value.lower().strip()
        if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        return normalized


class CountrySourcePresence(BaseModel):
    """Country-specific local questionnaire locator and paired source filename."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    present: bool
    questionnaire_file: str = Field(min_length=1)
    questionnaire_pages: list[int] = Field(min_length=1)
    codebook_file: str = Field(min_length=1)

    @field_validator("questionnaire_pages")
    @classmethod
    def validate_pages(cls, value: list[int]) -> list[int]:
        if any(page < 1 for page in value):
            raise ValueError("questionnaire_pages must contain positive PDF page numbers")
        return value


class WVS7CandidateItem(BaseModel):
    """One short, non-item-text candidate mapping awaiting manual review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variable_id: str = Field(min_length=2)
    instrument_domain: InstrumentDomain
    topic: str = Field(min_length=1, max_length=120)
    response_scale_type: str = Field(min_length=1, max_length=80)
    reverse_scoring_review: ReviewStatus
    manual_review_status: ReviewStatus
    usa: CountrySourcePresence
    china: CountrySourcePresence

    @field_validator("variable_id")
    @classmethod
    def validate_variable_id(cls, value: str) -> str:
        if not _VARIABLE_PATTERN.fullmatch(value):
            raise ValueError("variable_id must use the WVS Q-number form")
        return value

    @field_validator("topic", "response_scale_type")
    @classmethod
    def validate_short_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("candidate descriptor must not be blank")
        return normalized


class WVS7CandidateMetadata(BaseModel):
    """Fixed non-content metadata for a draft-only candidate checklist."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    purpose: str = Field(min_length=1)
    operational_grouping_only: bool
    wvs_official_factor_structure: bool
    pdf_runtime_dependency: bool
    network_access: bool


class WVS7CandidateManifest(BaseModel):
    """A draft mapping list, deliberately separate from ProbeSet contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: str = Field(min_length=1)
    manifest_version: str = Field(min_length=1)
    material_status: CandidateMaterialStatus
    final_benchmark: bool
    instrument_domains: list[InstrumentDomain] = Field(min_length=5, max_length=5)
    source_files: list[WVS7SourceFile] = Field(min_length=4, max_length=4)
    candidates: list[WVS7CandidateItem] = Field(min_length=1)
    metadata: WVS7CandidateMetadata

    @model_validator(mode="after")
    def validate_manifest_shape(self) -> "WVS7CandidateManifest":
        if len(self.instrument_domains) != len(set(self.instrument_domains)):
            raise ValueError("instrument_domains must not contain duplicates")
        if set(self.instrument_domains) != set(InstrumentDomain):
            raise ValueError("instrument_domains must list exactly the five declared EviCon domains")
        filenames = [source.filename for source in self.source_files]
        if len(filenames) != len(set(filenames)):
            raise ValueError("source_files must not contain duplicate filenames")
        if self.final_benchmark:
            raise ValueError("WVS candidate manifest cannot be a final benchmark")
        return self


class WVS7CandidateValidationReport(BaseModel):
    """Safe report that excludes WVS item text, options, and responses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: str
    manifest_version: str
    material_status: CandidateMaterialStatus
    candidate_count: int = Field(ge=0)
    domain_counts: dict[str, int]
    source_hash_prefixes: dict[str, str]


class WVS7CandidateError(ValueError):
    """Stable local candidate-manifest validation failure."""


class WVS7CandidateLoader:
    """Load a local TOML candidate manifest without reading source PDFs."""

    @staticmethod
    def load(path: str | Path) -> WVS7CandidateManifest:
        manifest_path = Path(path)
        try:
            raw = manifest_path.read_bytes()
        except OSError as exc:
            raise WVS7CandidateError("cannot read local WVS candidate manifest") from exc
        try:
            payload = tomllib.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise WVS7CandidateError("cannot parse local WVS candidate manifest") from exc
        if not isinstance(payload, dict):
            raise WVS7CandidateError("WVS candidate manifest root must be an object")
        try:
            return WVS7CandidateManifest.model_validate(payload)
        except ValueError as exc:
            raise WVS7CandidateError("WVS candidate manifest violates a required contract") from exc


def validate_wvs7_candidates(manifest: WVS7CandidateManifest) -> WVS7CandidateValidationReport:
    """Validate the static draft contract without reading local source PDFs."""
    if manifest.material_status is not CandidateMaterialStatus.DRAFT:
        raise WVS7CandidateError("WVS candidate manifest status must be draft")
    if len(manifest.candidates) != 24:
        raise WVS7CandidateError("WVS candidate manifest must contain exactly 24 candidates")

    variable_ids = [candidate.variable_id for candidate in manifest.candidates]
    if len(variable_ids) != len(set(variable_ids)):
        raise WVS7CandidateError("WVS candidate manifest must not contain duplicate variable_id values")
    if set(variable_ids) != _EXPECTED_VARIABLE_IDS:
        raise WVS7CandidateError("WVS candidate manifest must contain exactly the approved variable IDs")
    if any(not candidate.usa.present or not candidate.china.present for candidate in manifest.candidates):
        raise WVS7CandidateError("every candidate must declare USA and China source presence")
    for candidate in manifest.candidates:
        _validate_country_filenames(candidate.usa, _USA_QUESTIONNAIRE, _USA_CODEBOOK)
        _validate_country_filenames(candidate.china, _CHINA_QUESTIONNAIRE, _CHINA_CODEBOOK)

    sources = {source.filename: source.sha256 for source in manifest.source_files}
    if sources != _EXPECTED_SOURCE_HASHES:
        raise WVS7CandidateError("WVS candidate source hash receipt does not match the local audited sources")

    counts = Counter(candidate.instrument_domain.value for candidate in manifest.candidates)
    return WVS7CandidateValidationReport(
        study_id=manifest.study_id,
        manifest_version=manifest.manifest_version,
        material_status=manifest.material_status,
        candidate_count=len(manifest.candidates),
        domain_counts=dict(sorted(counts.items())),
        source_hash_prefixes={name: digest[:12] for name, digest in sorted(sources.items())},
    )


def load_and_validate_wvs7_candidates(path: str | Path) -> WVS7CandidateValidationReport:
    """Convenience entry point for the safe local-only CLI."""
    return validate_wvs7_candidates(WVS7CandidateLoader.load(path))


def _validate_country_filenames(
    source: CountrySourcePresence,
    questionnaire_file: str,
    codebook_file: str,
) -> None:
    if source.questionnaire_file != questionnaire_file or source.codebook_file != codebook_file:
        raise WVS7CandidateError("candidate country source filenames do not match the audited local source set")


__all__ = [
    "CandidateMaterialStatus",
    "CountrySourcePresence",
    "InstrumentDomain",
    "ReviewStatus",
    "WVS7CandidateError",
    "WVS7CandidateItem",
    "WVS7CandidateLoader",
    "WVS7CandidateManifest",
    "WVS7CandidateMetadata",
    "WVS7CandidateValidationReport",
    "WVS7SourceFile",
    "load_and_validate_wvs7_candidates",
    "validate_wvs7_candidates",
]
