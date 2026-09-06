"""Local-only study-material provenance and pilot validation tests."""

from __future__ import annotations

import hashlib
import json
import socket
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from evicon import validate_study_materials as materials_cli
from evicon.probe_loader import ProbeSetLoader
from evicon.scenario_loader import ScenarioLoader
from evicon.study_materials import (
    PILOT_DIMENSIONS,
    MaterialStatus,
    ProbeSourceProvenance,
    SourceKind,
    StudyMaterialsError,
    StudyMaterialsLoader,
    StudyMaterialsManifest,
    validate_study_materials,
)


def _manifest_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs/studies/evicon_pilot_materials.toml"


def _manifest():
    return StudyMaterialsLoader.load(_manifest_path())


def _material_path(reference: str) -> Path:
    return (_manifest_path().parent / reference).resolve()


def test_checked_in_original_pilot_materials_validate_with_expected_counts() -> None:
    manifest = _manifest()
    report = validate_study_materials(manifest, manifest_path=_manifest_path())

    assert report.study_id == "evicon-pilot-materials"
    assert report.material_status is MaterialStatus.PILOT
    assert report.probe_count == 12
    assert report.scenario_count == 2
    assert report.development_scenario_count == 2
    assert report.test_scenario_count == 0
    assert len(report.probe_hash_prefix) == 12


def test_pilot_pairs_and_scenario_decision_contracts_are_explicit() -> None:
    manifest = _manifest()
    probe_set = ProbeSetLoader.load(_material_path(manifest.probe_set_path))
    by_dimension = Counter(item.dimension for item in probe_set.items)

    assert tuple(probe_set.dimensions) == PILOT_DIMENSIONS
    assert by_dimension == Counter({dimension: 2 for dimension in PILOT_DIMENSIONS})
    for dimension in PILOT_DIMENSIONS:
        items = [item for item in probe_set.items if item.dimension == dimension]
        assert sum(item.reverse_scored for item in items) == 1

    development = next(split for split in manifest.scenario_splits if split.split == "development")
    for reference in development.scenarios:
        scenario = ScenarioLoader.load(_material_path(reference.path))
        assert isinstance(scenario.metadata["decision_options"], list)
        assert scenario.metadata["decision_options"]
        assert isinstance(scenario.metadata["intended_output"], list)
        assert scenario.metadata["intended_output"]
        assert any(card.supports for card in scenario.evidence_cards)
        assert any(card.contradicts for card in scenario.evidence_cards)


def test_validator_rejects_hash_mismatch_language_mismatch_and_missing_file(tmp_path: Path) -> None:
    manifest = _manifest()
    bad_hash = manifest.model_copy(
        update={"probe_source": manifest.probe_source.model_copy(update={"content_sha256": "0" * 64})}
    )
    with pytest.raises(StudyMaterialsError, match="content hash"):
        validate_study_materials(bad_hash, manifest_path=_manifest_path())

    bad_language = manifest.model_copy(
        update={"probe_source": manifest.probe_source.model_copy(update={"language": "zh"})}
    )
    with pytest.raises(StudyMaterialsError, match="language"):
        validate_study_materials(bad_language, manifest_path=_manifest_path())

    missing = manifest.model_copy(update={"probe_set_path": str(tmp_path / "not-present.toml")})
    with pytest.raises(StudyMaterialsError, match="does not exist"):
        validate_study_materials(missing, manifest_path=_manifest_path())


def test_validator_rejects_duplicate_probe_ids_unbalanced_pairs_and_split_overlap(tmp_path: Path) -> None:
    manifest = _manifest()
    original_path = _material_path(manifest.probe_set_path)
    duplicate_path = tmp_path / "duplicate-ids.toml"
    duplicate_path.write_text(
        original_path.read_text(encoding="utf-8").replace(
            'probe_id = "pilot-harm-01"',
            'probe_id = "pilot-harm-02"',
            1,
        ),
        encoding="utf-8",
    )
    duplicate_source = manifest.probe_source.model_copy(
        update={"content_sha256": hashlib.sha256(duplicate_path.read_bytes()).hexdigest()}
    )
    duplicate_manifest = manifest.model_copy(
        update={"probe_set_path": str(duplicate_path), "probe_source": duplicate_source}
    )
    with pytest.raises(StudyMaterialsError, match="schema validation"):
        validate_study_materials(duplicate_manifest, manifest_path=_manifest_path())

    unbalanced_path = tmp_path / "unbalanced-pairs.toml"
    unbalanced_path.write_text(
        original_path.read_text(encoding="utf-8").replace("reverse_scored = true", "reverse_scored = false", 1),
        encoding="utf-8",
    )
    unbalanced_source = manifest.probe_source.model_copy(
        update={"content_sha256": hashlib.sha256(unbalanced_path.read_bytes()).hexdigest()}
    )
    unbalanced_manifest = manifest.model_copy(
        update={"probe_set_path": str(unbalanced_path), "probe_source": unbalanced_source}
    )
    with pytest.raises(StudyMaterialsError, match="exactly one reverse-scored"):
        validate_study_materials(unbalanced_manifest, manifest_path=_manifest_path())

    development = next(split for split in manifest.scenario_splits if split.split == "development")
    test = next(split for split in manifest.scenario_splits if split.split == "test")
    overlap = test.model_copy(update={"scenarios": [development.scenarios[0]]})
    overlapping_manifest = manifest.model_copy(update={"scenario_splits": [development, overlap]})
    with pytest.raises(StudyMaterialsError, match="more than one split"):
        validate_study_materials(overlapping_manifest, manifest_path=_manifest_path())


def test_validator_requires_nonempty_decision_metadata(tmp_path: Path) -> None:
    manifest = _manifest()
    development = next(split for split in manifest.scenario_splits if split.split == "development")
    original_reference = development.scenarios[0]
    original_path = _material_path(original_reference.path)
    incomplete_path = tmp_path / "incomplete-decision-metadata.toml"
    incomplete_path.write_text(
        original_path.read_text(encoding="utf-8").replace("decision_options = ", "removed_options = ", 1),
        encoding="utf-8",
    )
    incomplete_reference = original_reference.model_copy(
        update={
            "path": str(incomplete_path),
            "provenance": original_reference.provenance.model_copy(
                update={"content_sha256": hashlib.sha256(incomplete_path.read_bytes()).hexdigest()}
            ),
        }
    )
    incomplete_development = development.model_copy(
        update={"scenarios": [incomplete_reference, development.scenarios[1]]}
    )
    test = next(split for split in manifest.scenario_splits if split.split == "test")
    incomplete_manifest = manifest.model_copy(
        update={"scenario_splits": [incomplete_development, test]}
    )
    with pytest.raises(StudyMaterialsError, match="decision_options"):
        validate_study_materials(incomplete_manifest, manifest_path=_manifest_path())


def test_source_contract_rejects_missing_official_provenance_wrong_wvs_marker_and_incomplete_frozen_source() -> None:
    with pytest.raises(ValidationError, match="citation_url"):
        ProbeSourceProvenance(
            source_kind=SourceKind.OFFICIAL_EXTERNAL,
            source_name="official source",
            source_version="v1",
            license_note="reviewed use",
            usage_note="local-only validation",
            content_sha256="0" * 64,
            language="en",
            material_status=MaterialStatus.FROZEN,
            classification="external_reference",
        )

    data = _manifest().model_dump(mode="python")
    data["probe_source"]["classification"] = "wvs"
    with pytest.raises(ValidationError, match="not_wvs_pilot"):
        StudyMaterialsManifest.model_validate(data)

    data = _manifest().model_dump(mode="python")
    data["material_status"] = "frozen"
    data["probe_source"]["material_status"] = "frozen"
    del data["probe_source"]["source_name"]
    with pytest.raises(ValidationError, match="source_name"):
        StudyMaterialsManifest.model_validate(data)


def test_cli_output_is_safe_and_never_connects(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def reject_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("study-material validation attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", reject_connection)
    assert materials_cli.main(["--manifest", str(_manifest_path())]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["study_id"] == "evicon-pilot-materials"
    assert output["probe_count"] == 12
    assert output["scenario_count"] == 2
    assert set(output) == {
        "study_id",
        "material_version",
        "material_status",
        "primary_language",
        "probe_count",
        "scenario_count",
        "development_scenario_count",
        "test_scenario_count",
        "hash_summary",
    }
