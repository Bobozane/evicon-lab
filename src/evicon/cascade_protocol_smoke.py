"""Offline smoke CLI for the deterministic provenance-cascade runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cascade_protocol import CascadeProtocolError, CascadeProtocolRunner, CascadeScenarioLoader

DEFAULT_SCENARIO_DIR = Path("configs/provenance_cascade/scenarios")
DEFAULT_SEEDS = (20260901, 20260902, 20260903)


def run_smoke(scenario_dir: Path = DEFAULT_SCENARIO_DIR, seeds: tuple[int, ...] = DEFAULT_SEEDS) -> list[dict[str, object]]:
    runner = CascadeProtocolRunner()
    summaries: list[dict[str, object]] = []
    for path in sorted(scenario_dir.glob("*.toml")):
        scenario = CascadeScenarioLoader.load(path)
        for seed in seeds:
            try:
                record = runner.run_scenario(scenario, seed)
                root_count = sum(record.independent_root_count_by_claim.values())
                summaries.append(
                    {
                        "scenario_id": record.scenario_id,
                        "scenario_type": scenario.scenario_type.value,
                        "seed": seed,
                        "agent_count": len(record.agent_ids),
                        "round_count": record.round_count,
                        "content_event_count": record.content_event_count,
                        "exposure_event_count": record.exposure_event_count,
                        "snapshot_count": record.snapshot_count,
                        "independent_root_count": root_count,
                        "replay_status": record.replay.status.value,
                        "trajectory_sha256": record.trajectory_sha256,
                    }
                )
            except CascadeProtocolError as exc:
                summaries.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "scenario_type": scenario.scenario_type.value,
                        "seed": seed,
                        "replay_status": "blocked",
                        "error_code": exc.code,
                    }
                )
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a local provenance-cascade protocol smoke")
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    args = parser.parse_args(argv)
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    summaries = run_smoke(args.scenario_dir, seeds)
    print(json.dumps({"status": "offline_smoke", "scenarios": summaries}, ensure_ascii=True, sort_keys=True))
    return 0 if summaries and all(item["replay_status"] == "passed" for item in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
