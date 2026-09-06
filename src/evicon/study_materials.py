"""Local-only provenance and validation for auditable study materials."""

from __future__ import annotations

import hashlib
import json
import tomllib
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from .models import ProbeSet, ScenarioSpec
from .models._validation import Metadata, identifier_list, normalized_text
from .probe_loader import ProbeLoadError, ProbeSetLoader
from .scenario_loader import ScenarioLoadError, ScenarioLoader


PILOT_DIMENSIONS: tuple[str, ...] = (
    "harm_prevention",
    "fairness",
    "autonomy",
    "truth_evidence",
    "privacy_dignity",
    "social_solidarity",
)
_SCALE_1_TO_7 = ["1", "2", "3", "4", "5", "6", "7"]
_UNSAFE_METADATA = ("prompt", "api_key", "apikey", "authorization", "secret", "hidden", "raw_response")


class MaterialStatus(str, Enum):
    DRAFT = "draft"
    PILOT = "pilot"
    FROZEN = "frozen"


class SourceKind(str, Enum):
    AUTHOR_ORIGINAL = "author_original"
    OFFICIAL_EXTERNAL = "official_external"


class ProbeSourceProvenance(BaseModel):
    """Origin, permitted use, and local hash for a probe or scenario file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_kind: SourceKind
    source_name: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    citation_url: str | None = None
    license_note: str = Field(min_length=1)
    usage_note: str = Field(min_length=1)
    content_sha256: str = Field(min_length=64, max_length=64)
    language: str = Field(min_length=2, max_length=16)
    material_status: MaterialStatus
    classification: str = Field(min_length=1)

    @field_validator(
        "source_name",
        "source_version",
        "license_note",
        "usage_note",
        "language",
        "classification",
    )
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("citation_url")
    @classmethod
    def validate_optional_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if not normalized.startswith(("https://", "http://")):
            raise ValueError("citation_url must be a stable http(s) URL")
        return normalized

    @field_validator("content_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        normalized = value.lower().strip()
        if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
        return normalized

    @model_validator(mode="after")
    def validate_source_requirements(self) -> "ProbeSourceProvenance":
        if self.source_kind is SourceKind.OFFICIAL_EXTERNAL:
            if self.citation_url is None:
                raise ValueError("official_external provenance requires citation_url")
        elif self.material_status is not MaterialStatus.PILOT or self.classification != "not_wvs_pilot":
            raise ValueError("author_original provenance must be marked not_wvs_pilot and pilot")
        return self


class ScenarioMaterialReference(BaseModel):
    """One local ScenarioSpec file allocated to an explicit study split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    provenance: ProbeSourceProvenance

    @field_validator("scenario_id", "path")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))


class ScenarioSplit(BaseModel):
    """An explicit development or test allocation; paths are never inferred."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split: Literal["development", "test"]
    scenarios: list[ScenarioMaterialReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_scenarios(self) -> "ScenarioSplit":
        identifiers = [item.scenario_id for item in self.scenarios]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("ScenarioSplit must not contain duplicate scenario_id values")
        paths = [item.path for item in self.scenarios]
        if len(paths) != len(set(paths)):
            raise ValueError("ScenarioSplit must not contain duplicate paths")
        return self


class StudyMaterialsManifest(BaseModel):
    """Versioned, local-only map of material sources, splits, and hashes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: str = Field(min_length=1)
    material_version: str = Field(min_length=1)
    primary_language: str = Field(min_length=2, max_length=24)
    value_dimensions: list[str] = Field(min_length=6, max_length=6)
    probe_set_path: str = Field(min_length=1)
    probe_source: ProbeSourceProvenance
    scenario_splits: list[ScenarioSplit] = Field(min_length=2, max_length=2)
    material_status: MaterialStatus
    license_note: str = Field(min_length=1)
    version: str = Field(min_length=1)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator(
        "study_id",
        "material_version",
        "primary_language",
        "probe_set_path",
        "license_note",
        "version",
    )
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("value_dimensions")
    @classmethod
    def validate_dimensions(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "value_dimensions")

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: Metadata) -> Metadata:
        _reject_unsafe_metadata(value)
        return value

    @model_validator(mode="after")
    def validate_study_structure(self) -> "StudyMaterialsManifest":
        if tuple(self.value_dimensions) != PILOT_DIMENSIONS:
            raise ValueError("value_dimensions must list the six declared pilot dimensions in stable order")
        split_names = [item.split for item in self.scenario_splits]
        if set(split_names) != {"development", "test"} or len(split_names) != len(set(split_names)):
            raise ValueError("scenario_splits must contain one development split and one test split")
        scenarios = [scenario for split in self.scenario_splits for scenario in split.scenarios]
        ids = [scenario.scenario_id for scenario in scenarios]
        paths = [scenario.path for scenario in scenarios]
        if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
            raise ValueError("a scenario may not appear in both development and test splits")
        if self.material_status is MaterialStatus.FROZEN:
            if self.probe_source.content_sha256 is None or not self.probe_source.source_name:
                raise ValueError("frozen materials require complete probe provenance and hashes")
            if any(not scenario.provenance.source_name for scenario in scenarios):
                raise ValueError("frozen materials require complete scenario provenance")
        return self


class StudyMaterialsValidationReport(BaseModel):
    """Safe validation result that deliberately omits material text and answers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: str
    material_version: str
    material_status: MaterialStatus
    primary_language: str
    probe_count: int = Field(ge=0)
    scenario_count: int = Field(ge=0)
    development_scenario_count: int = Field(ge=0)
    test_scenario_count: int = Field(ge=0)
    probe_hash_prefix: str = Field(min_length=12, max_length=12)
    scenario_hash_prefixes: dict[str, str]


class StudyMaterialsError(ValueError):
    """Safe local validation error that never embeds material text."""


class StudyMaterialsLoader:
    """Load only a local TOML manifest; this class never downloads sources."""

    @staticmethod
    def load(path: str | Path) -> StudyMaterialsManifest:
        manifest_path = Path(path)
        try:
            raw = manifest_path.read_bytes()
        except OSError as exc:
            raise StudyMaterialsError(f"cannot read local materials manifest: {manifest_path}") from exc
        try:
            payload = tomllib.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise StudyMaterialsError(f"cannot parse local materials manifest: {manifest_path}") from exc
        if not isinstance(payload, dict):
            raise StudyMaterialsError("materials manifest root must be an object")
        try:
            return StudyMaterialsManifest.model_validate(payload)
        except ValueError as exc:
            raise StudyMaterialsError("materials manifest violates a required contract") from exc


def validate_study_materials(
    manifest: StudyMaterialsManifest,
    *,
    manifest_path: str | Path,
) -> StudyMaterialsValidationReport:
    """Validate hashes, local contracts, splits, and source labels without network access."""
    base_directory = Path(manifest_path).resolve().parent
    probe_path = _resolve_local(base_directory, manifest.probe_set_path, "probe_set_path")
    probe_hash = _sha256_file(probe_path, "probe_set_path")
    if probe_hash != manifest.probe_source.content_sha256:
        raise StudyMaterialsError("probe_set_path content hash does not match manifest provenance")
    probe_set = _load_probe_set(probe_path)
    _validate_probe_set(probe_set, manifest)

    scenario_hashes: dict[str, str] = {}
    split_counts = {"development": 0, "test": 0}
    seen_scenario_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for split in manifest.scenario_splits:
        for reference in split.scenarios:
            scenario_path = _resolve_local(base_directory, reference.path, "scenario path")
            if reference.scenario_id in seen_scenario_ids or scenario_path in seen_paths:
                raise StudyMaterialsError("scenario is allocated to more than one split")
            seen_scenario_ids.add(reference.scenario_id)
            seen_paths.add(scenario_path)
            content_hash = _sha256_file(scenario_path, "scenario path")
            if content_hash != reference.provenance.content_sha256:
                raise StudyMaterialsError("scenario content hash does not match manifest provenance")
            scenario = _load_scenario(scenario_path)
            _validate_scenario(scenario, reference, split, manifest)
            scenario_hashes[reference.scenario_id] = content_hash[:12]
            split_counts[split.split] += 1

    return StudyMaterialsValidationReport(
        study_id=manifest.study_id,
        material_version=manifest.material_version,
        material_status=manifest.material_status,
        primary_language=manifest.primary_language,
        probe_count=len(probe_set.items),
        scenario_count=sum(split_counts.values()),
        development_scenario_count=split_counts["development"],
        test_scenario_count=split_counts["test"],
        probe_hash_prefix=probe_hash[:12],
        scenario_hash_prefixes=scenario_hashes,
    )


def load_and_validate_study_materials(path: str | Path) -> StudyMaterialsValidationReport:
    """Convenience entry point for the local CLI and offline callers."""
    manifest = StudyMaterialsLoader.load(path)
    return validate_study_materials(manifest, manifest_path=path)


def _validate_probe_set(probe_set: ProbeSet, manifest: StudyMaterialsManifest) -> None:
    if probe_set.dimensions != manifest.value_dimensions:
        raise StudyMaterialsError("probe dimensions do not match manifest value_dimensions")
    if any(item.dimension not in manifest.value_dimensions for item in probe_set.items):
        raise StudyMaterialsError("probe item dimension is absent from manifest value_dimensions")
    if any(item.response_scale != _SCALE_1_TO_7 for item in probe_set.items):
        raise StudyMaterialsError("pilot probe response scales must be the declared 1-7 scale")
    for dimension in manifest.value_dimensions:
        dimension_items = [item for item in probe_set.items if item.dimension == dimension]
        if len(dimension_items) != 2:
            raise StudyMaterialsError("each pilot dimension must contain exactly two probe items")
        if sum(item.reverse_scored for item in dimension_items) != 1:
            raise StudyMaterialsError("each pilot dimension must contain exactly one reverse-scored item")
    metadata = probe_set.metadata
    if metadata.get("language") != manifest.probe_source.language:
        raise StudyMaterialsError("probe language does not match manifest provenance")
    if manifest.probe_source.source_kind is SourceKind.AUTHOR_ORIGINAL:
        expected = {
            "source_kind": "author_original",
            "material_status": "pilot",
            "classification": "not_wvs_pilot",
            "applies_to_all_items": True,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise StudyMaterialsError("author_original probe material must be marked non-WVS pilot")
    if manifest.material_status is MaterialStatus.FROZEN and manifest.probe_source.material_status is not MaterialStatus.FROZEN:
        raise StudyMaterialsError("frozen manifest requires frozen probe provenance")


def _validate_scenario(
    scenario: ScenarioSpec,
    reference: ScenarioMaterialReference,
    split: ScenarioSplit,
    manifest: StudyMaterialsManifest,
) -> None:
    if scenario.scenario_id != reference.scenario_id:
        raise StudyMaterialsError("scenario ID does not match its manifest reference")
    if scenario.metadata.get("language") != reference.provenance.language:
        raise StudyMaterialsError("scenario language does not match manifest provenance")
    if reference.provenance.source_kind is SourceKind.AUTHOR_ORIGINAL:
        expected = {
            "source_kind": "author_original",
            "material_status": "pilot",
            "classification": "not_wvs_pilot",
            "synthetic": True,
        }
        if any(scenario.metadata.get(key) != value for key, value in expected.items()):
            raise StudyMaterialsError("author_original scenario must be marked synthetic non-WVS pilot")
    _require_nonempty_metadata_list(scenario.metadata, "decision_options")
    _require_nonempty_metadata_list(scenario.metadata, "intended_output")
    if len(scenario.agents) < 4:
        raise StudyMaterialsError("pilot scenario requires at least four agents")
    if not any(card.supports for card in scenario.evidence_cards):
        raise StudyMaterialsError("pilot scenario requires at least one supporting EvidenceCard")
    if not any(card.contradicts for card in scenario.evidence_cards):
        raise StudyMaterialsError("pilot scenario requires at least one limiting EvidenceCard")
    if manifest.material_status is MaterialStatus.FROZEN and reference.provenance.material_status is not MaterialStatus.FROZEN:
        raise StudyMaterialsError("frozen manifest requires frozen scenario provenance")
    if split.split == "test" and reference.provenance.material_status is MaterialStatus.PILOT:
        raise StudyMaterialsError("pilot scenarios must not be allocated to the test split")


def _require_nonempty_metadata_list(metadata: Metadata, field_name: str) -> None:
    value = metadata.get(field_name)
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item.strip() for item in value):
        raise StudyMaterialsError(f"pilot scenario metadata requires non-empty {field_name}")


def _resolve_local(base_directory: Path, reference: str, field_name: str) -> Path:
    candidate = (base_directory / reference).resolve()
    if not candidate.is_file():
        raise StudyMaterialsError(f"{field_name} does not exist locally")
    return candidate


def _sha256_file(path: Path, field_name: str) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise StudyMaterialsError(f"cannot hash local {field_name}") from exc


def _load_probe_set(path: Path) -> ProbeSet:
    try:
        return ProbeSetLoader.load(path)
    except ProbeLoadError as exc:
        raise StudyMaterialsError("probe set failed local schema validation") from exc


def _load_scenario(path: Path) -> ScenarioSpec:
    try:
        return ScenarioLoader.load(path)
    except ScenarioLoadError as exc:
        raise StudyMaterialsError("scenario failed local schema validation") from exc


def _reject_unsafe_metadata(value: Metadata) -> None:
    def visit(item: JsonValue, path: str) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                if any(token in key.lower() for token in _UNSAFE_METADATA):
                    raise ValueError(f"metadata contains unsafe field at {path}")
                visit(nested, f"{path}.{key}")
        elif isinstance(item, list):
            for index, nested in enumerate(item):
                visit(nested, f"{path}[{index}]")
        elif isinstance(item, str) and any(token in item.lower() for token in _UNSAFE_METADATA):
            raise ValueError(f"metadata contains unsafe content at {path}")

    visit(value, "metadata")


__all__ = [
    "MaterialStatus",
    "PILOT_DIMENSIONS",
    "ProbeSourceProvenance",
    "ScenarioMaterialReference",
    "ScenarioSplit",
    "SourceKind",
    "StudyMaterialsError",
    "StudyMaterialsLoader",
    "StudyMaterialsManifest",
    "StudyMaterialsValidationReport",
    "load_and_validate_study_materials",
    "validate_study_materials",
]
