"""Local-only audit summary for the synthetic cascade exposure fixtures."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .cascade_replay import CascadeReplayValidationError, CascadeReplayValidator, safe_file_sha256
from .provenance_cascade import PublicScenarioLoader


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay local provenance-cascade exposure fixtures.")
    parser.add_argument("--fixture-dir", type=Path, default=Path("configs/provenance_cascade/fixtures"))
    arguments = parser.parse_args(argv)
    try:
        reports: list[dict[str, object]] = []
        for public_path in sorted(arguments.fixture_dir.glob("*.public.json")):
            ledger_path = public_path.with_name(public_path.name.replace(".public.json", ".exposure_ledger.json"))
            graph = PublicScenarioLoader.load(public_path)
            report = CascadeReplayValidator.validate_paths(public_path, ledger_path)
            reports.append({
                "scenario_id": graph.scenario_id,
                "scenario_type": graph.scenario_type.value,
                "agent_count": report.agent_count,
                "exposure_count": report.exposure_count,
                "snapshot_count": report.snapshot_count,
                "root_relation_count": report.root_relation_count,
                "public_sha256": safe_file_sha256(public_path),
                "ledger_sha256": safe_file_sha256(ledger_path),
                "replay_status": report.status.value,
            })
        if not reports:
            raise CascadeReplayValidationError("no_exposure_fixtures_found")
    except (OSError, ValueError, CascadeReplayValidationError):
        print(json.dumps({"status": "blocked", "error_code": "cascade_exposure_smoke_failed"}, sort_keys=True))
        return 1
    print(json.dumps({"status": "passed", "fixture_count": len(reports), "fixtures": reports}, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
