"""Offline smoke for the opt-in public cascade outcome runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cascade_outcome_runner import CascadeOutcomeRunner
from .cascade_protocol import CascadeScenarioLoader
from .provenance_cascade_preregistration import (
    CascadeCondition,
    ProvenanceCascadePreregistrationLoader,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREREG = ROOT / "configs" / "studies" / "provenance_cascade_pilot_preregistration.toml"
DEFAULT_SCENARIO_DIR = ROOT / "configs" / "provenance_cascade" / "scenarios"
SCENARIO_NAMES = (
    "false_majority",
    "true_minority_correction",
    "independent_true_consensus",
    "unresolved_disagreement",
)


def run_smoke(
    preregistration_path: Path = DEFAULT_PREREG,
    scenario_dir: Path = DEFAULT_SCENARIO_DIR,
) -> list[dict[str, object]]:
    prereg = ProvenanceCascadePreregistrationLoader.load(preregistration_path)
    runner = CascadeOutcomeRunner()
    rows: list[dict[str, object]] = []
    for scenario_name in SCENARIO_NAMES:
        scenario = CascadeScenarioLoader.load(scenario_dir / f"{scenario_name}.toml")
        for seed in prereg.seeds:
            for condition in CascadeCondition:
                record = runner.run_scenario(scenario, seed, condition)
                rows.append(
                    {
                        "scenario_id": record.scenario_id,
                        "seed": seed,
                        "condition": condition.value,
                        "agent_count": len(record.agent_ids),
                        "round_count": record.round_count,
                        "event_count": record.exposure_event_count,
                        "outcome_count": record.outcome_count,
                        "directive_applied_count": record.directive_applied_count,
                        "proposal_count": record.proposal_count,
                        "trajectory_hash": record.outcome_ledger_sha256,
                        "replay_status": record.replay.status.value,
                        "application_replay_status": record.replay.application_replay.status.value,
                    }
                )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline public cascade outcome smoke")
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    args = parser.parse_args(argv)
    rows = run_smoke(args.preregistration, args.scenario_dir)
    print(
        json.dumps(
            {
                "status": "offline_outcome_smoke",
                "network": "disabled",
                "results_written": False,
                "private_truth_exposed": False,
                "development_only": True,
                "not_paper_result": True,
                "no_causal_conclusion": True,
                "run_count": len(rows),
                "runs": rows,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
