"""Offline proposal-to-next-round application smoke."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cascade_application_replay import CascadeApplicationReplayValidator
from .cascade_controller import CascadeControllerPolicyLoader, propose
from .cascade_controller_smoke import fixture_view
from .cascade_intervention_application import CascadeApplicationLedger, CascadeApplicationStatus
from .provenance_cascade_preregistration import CascadeCondition

POLICY = Path("configs/provenance_cascade/cascade_controller_policy.v1.toml")
SCENARIOS = Path("configs/provenance_cascade/scenarios")


def run_smoke(policy_path: Path = POLICY, scenario_dir: Path = SCENARIOS) -> list[dict[str, object]]:
    config = CascadeControllerPolicyLoader.load(policy_path)
    records=[]
    for name in ("false_majority", "true_minority_correction", "independent_true_consensus", "unresolved_disagreement"):
        current = fixture_view(name, scenario_dir)
        proposal = propose(CascadeCondition.PROVENANCE_AWARE_CONTROLLER, current, config)
        ledger = CascadeApplicationLedger()
        ledger, scheduled = ledger.schedule_proposal(proposal, current, snapshot_hash="a"*64, config=config)
        if scheduled.schedule is not None:
            next_view = current.model_copy(update={"round_id": current.round_id + 1})
            ledger, applied = ledger.apply_schedule(
                scheduled.schedule.schedule_id,
                view=next_view,
                snapshot_hash="a"*64,
                current_round_id=scheduled.schedule.effective_round_id,
            )
            replay = CascadeApplicationReplayValidator.validate(ledger)
            status = applied.status.value
            applied_count = applied.applied_count
            schedule_count = len(ledger.schedules)
            replay_status = replay.status.value
        else:
            status = scheduled.status.value
            applied_count = 0
            schedule_count = 0
            replay_status = "passed"
        records.append({
            "scenario_id": current.scenario_id,
            "created_round_id": current.round_id,
            "effective_round_id": current.round_id + 1,
            "action": proposal.action.value,
            "schedule_status": status,
            "schedule_count": schedule_count,
            "applied_count": applied_count,
            "rejected_count": scheduled.rejected_count,
            "replay_status": replay_status,
            "proposal_id": proposal.proposal_id,
        })
    return records


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description="Offline next-round cascade application smoke")
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--scenario-dir", type=Path, default=SCENARIOS)
    args=parser.parse_args(argv)
    records=run_smoke(args.policy,args.scenario_dir)
    print(json.dumps({"status":"offline_application_smoke","scenarios":records}, ensure_ascii=True, sort_keys=True))
    return 0 if all(item["replay_status"] == "passed" for item in records) else 1

if __name__ == "__main__":
    raise SystemExit(main())
