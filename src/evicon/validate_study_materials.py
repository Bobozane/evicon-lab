"""CLI for safe, local-only validation of a study-materials manifest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .study_materials import StudyMaterialsError, load_and_validate_study_materials


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate local EviCon-Lab study materials without network access.")
    parser.add_argument("--manifest", required=True, help="Path to a local TOML study-materials manifest.")
    arguments = parser.parse_args(argv)
    try:
        report = load_and_validate_study_materials(arguments.manifest)
    except StudyMaterialsError as error:
        print(json.dumps({"status": "invalid", "error_code": _error_code(error)}, ensure_ascii=True, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "study_id": report.study_id,
                "material_version": report.material_version,
                "material_status": report.material_status.value,
                "primary_language": report.primary_language,
                "probe_count": report.probe_count,
                "scenario_count": report.scenario_count,
                "development_scenario_count": report.development_scenario_count,
                "test_scenario_count": report.test_scenario_count,
                "hash_summary": {
                    "probe": report.probe_hash_prefix,
                    "scenarios": report.scenario_hash_prefixes,
                },
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


def _error_code(error: StudyMaterialsError) -> str:
    message = str(error)
    if "hash" in message:
        return "hash_mismatch"
    if "does not exist" in message:
        return "file_missing"
    return "validation_failed"


if __name__ == "__main__":
    raise SystemExit(main())
