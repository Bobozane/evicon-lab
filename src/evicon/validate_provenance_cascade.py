"""Offline validator for public provenance graphs and private truth fixtures."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .provenance_cascade import (
    EvaluatorTruthLoader,
    ProvenanceFixtureError,
    PublicScenarioLoader,
    file_sha256,
    validate_fixture_pair,
)


def _validate_pair(public_path: Path, private_path: Path) -> dict[str, object]:
    public = PublicScenarioLoader.load(public_path)
    private = EvaluatorTruthLoader.load(private_path)
    summary = validate_fixture_pair(public, private)
    summary["public_sha256"] = file_sha256(public_path)
    summary["private_sha256"] = file_sha256(private_path)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate local provenance-cascade fixtures without providers or network access.")
    parser.add_argument("--public", type=Path)
    parser.add_argument("--private", type=Path)
    parser.add_argument("--fixture-dir", type=Path)
    arguments = parser.parse_args(argv)
    try:
        summaries: list[dict[str, object]] = []
        if arguments.public is not None or arguments.private is not None:
            if arguments.public is None or arguments.private is None:
                raise ProvenanceFixtureError("public_and_private_paths_required_together")
            summaries.append(_validate_pair(arguments.public, arguments.private))
        else:
            fixture_dir = arguments.fixture_dir or Path("configs/provenance_cascade/fixtures")
            public_paths = sorted(fixture_dir.glob("*.public.json"))
            if not public_paths:
                raise ProvenanceFixtureError("no_public_fixtures_found")
            for public_path in public_paths:
                private_path = public_path.with_name(public_path.name.replace(".public.json", ".evaluator_private.json"))
                summaries.append(_validate_pair(public_path, private_path))
        output = {"status": "validated", "fixture_count": len(summaries), "fixtures": summaries}
    except (OSError, ValueError, ProvenanceFixtureError):
        print(json.dumps({"status": "blocked", "blocking_reasons": ["provenance_fixture_validation_failed"]}, sort_keys=True))
        return 1
    print(json.dumps(output, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
