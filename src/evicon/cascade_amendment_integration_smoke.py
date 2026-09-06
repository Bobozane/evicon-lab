"""Offline H-D 48-run FakeProvider integration and amendment receipt smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from .cascade_agent_integration_smoke import _FakeProvider
from .cascade_intervention_application import CascadeScheduleStatus
from .cascade_protocol import CascadeScenarioLoader
from .cascade_real_agent_runner import CascadeRealAgentRunRecord, CascadeRealAgentRunner
from .provenance_cascade_amendment import load_hd_config, write_amendment_receipt
from .provenance_cascade_preregistration import CascadeCondition
from .request_ledger import RequestLedger

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd.v2.toml"
_DEFAULT_RECEIPT = _ROOT / "outputs/study-locks/provenance_cascade_pilot_hd_amendment_receipt.json"


def _stable_ledger_projection(path: Path) -> list[dict[str, object]]:
    return [
        {
            "fingerprint": item.fingerprint,
            "request_key": item.request_key,
            "condition": item.condition,
            "phase": item.phase,
            "agent_id": item.agent_id,
            "round_id": item.round_id,
            "status": item.status.value,
            "attempt_count": item.attempt_count,
            "reserved_tokens": item.reserved_tokens,
            "prompt_tokens": item.prompt_tokens,
            "completion_tokens": item.completion_tokens,
            "total_tokens": item.total_tokens,
            "error_code": item.error_code,
        }
        for item in RequestLedger(path).entries()
    ]


def _hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _context_snapshot(record: CascadeRealAgentRunRecord, agent_id: str, round_id: int) -> dict[str, object]:
    context = next(item for item in record.round_contexts if item.snapshot.agent_id == agent_id and item.snapshot.round_id == round_id)
    return context.snapshot.model_dump(mode="json")


def run_smoke(config_path: str | Path = _DEFAULT_CONFIG) -> dict[str, object]:
    config_file = Path(config_path).resolve()
    config, amendment = load_hd_config(config_file)
    base = config_file.parent
    materials = {item.scenario_id: item for item in config.scenario_materials}
    runner = CascadeRealAgentRunner()
    records: dict[tuple[str, int, CascadeCondition], CascadeRealAgentRunRecord] = {}
    counts: dict[tuple[str, CascadeCondition], dict[str, object]] = defaultdict(
        lambda: {
            "proposal_evaluation_count": 0,
            "scheduled_count": 0,
            "applied_count": 0,
            "rejected_count": 0,
            "actions": set(),
            "reason_codes": set(),
        }
    )
    provider_calls = 0
    stable_ledgers: list[dict[str, object]] = []
    with TemporaryDirectory(prefix="evicon-hd-amendment-") as temporary:
        root = Path(temporary)
        for run in config.runs:
            material = materials[run.scenario_id]
            scenario = CascadeScenarioLoader.load((base / material.config_path).resolve())
            provider = _FakeProvider()
            ledger_path = root / run.run_id / config.output.ledger_filename
            checkpoint_path = root / run.run_id / "agent_checkpoint.json"
            record = runner.run_scenario(
                scenario,
                run.seed,
                run.condition,
                provider=provider,
                run_id=run.run_id,
                ledger_path=ledger_path,
                checkpoint_path=checkpoint_path,
                model_name="fake-cascade-agent",
                agent_temperature=config.provider.agent_temperature,
                agent_max_tokens=config.provider.agent_max_tokens,
                request_cap=run.expected_provider_requests,
                completion_reservation_cap=run.completion_reservation,
            )
            if record.replay is None or record.replay.status.value != "passed":
                raise RuntimeError("amendment_smoke_replay_failed")
            if record.logical_request_count != run.expected_provider_requests or provider.calls != run.expected_provider_requests:
                raise RuntimeError("amendment_smoke_request_count_mismatch")
            raw_ledger = ledger_path.read_text(encoding="utf-8").lower()
            raw_checkpoint = checkpoint_path.read_text(encoding="utf-8").lower()
            forbidden = ("system_prompt", "user_prompt", "api_key", "provider_metadata", "ground_truth_label", "source_independence_label")
            if any(token in raw_ledger or token in raw_checkpoint for token in forbidden):
                raise RuntimeError("amendment_smoke_private_or_provider_data_leak")
            stable_ledgers.extend(_stable_ledger_projection(ledger_path))
            provider_calls += provider.calls
            records[(run.scenario_id, run.seed, run.condition)] = record
            bucket = counts[(run.scenario_id, run.condition)]
            bucket["proposal_evaluation_count"] = int(bucket["proposal_evaluation_count"]) + record.proposal_count
            bucket["scheduled_count"] = int(bucket["scheduled_count"]) + len(record.application_ledger.schedules)
            bucket["applied_count"] = int(bucket["applied_count"]) + sum(item.status is CascadeScheduleStatus.APPLIED for item in record.application_ledger.schedules)
            bucket["rejected_count"] = int(bucket["rejected_count"]) + sum(item.status is CascadeScheduleStatus.REJECTED for item in record.application_ledger.schedules)
            bucket["actions"].update(item.action.value for item in record.application_ledger.schedules)
            bucket["reason_codes"].update(code for item in record.application_ledger.schedules for code in item.reason_codes)

    false_id = amendment.trigger_observation.scenario_id
    for seed in config.seeds:
        baseline = records[(false_id, seed, CascadeCondition.NO_INTERVENTION)]
        aware = records[(false_id, seed, CascadeCondition.PROVENANCE_AWARE_CONTROLLER)]
        if _context_snapshot(baseline, "network-agent-01", 1) != _context_snapshot(aware, "network-agent-01", 1):
            raise RuntimeError("proposal_changed_creating_round_snapshot")
        schedules = aware.application_ledger.schedules
        if not schedules or not any(
            item.created_round_id == 1
            and item.effective_round_id == 2
            and item.status is CascadeScheduleStatus.APPLIED
            and "visible_unverified_same_root_repetition" in item.reason_codes
            and item.visible_source_root_ids == ("root-fm",)
            for item in schedules
        ):
            raise RuntimeError("formal_same_root_trigger_not_applied")
        round_two = next(item for item in aware.round_contexts if item.snapshot.agent_id == "network-agent-01" and item.snapshot.round_id == 2)
        if not round_two.directives or any(item.effective_round_id != 2 for item in round_two.directives):
            raise RuntimeError("directive_not_applied_next_round")

    for seed in config.seeds:
        for condition in config.conditions:
            consensus = records[("cascade-independent-true-consensus", seed, condition)]
            unresolved = records[("cascade-unresolved-disagreement", seed, condition)]
            if consensus.application_ledger.schedules or unresolved.application_ledger.schedules:
                raise RuntimeError("protected_scenario_intervention_detected")
            correction = records[("cascade-true-minority-correction", seed, condition)]
            if any(item.action.value in {"hide", "suppress", "downgrade", "remove"} for item in correction.application_ledger.schedules):
                raise RuntimeError("supported_correction_protection_failed")

    condition_summaries: list[dict[str, object]] = []
    for material in config.scenario_materials:
        for condition in config.conditions:
            bucket = counts[(material.scenario_id, condition)]
            condition_summaries.append({
                "scenario_id": material.scenario_id,
                "condition": condition.value,
                "proposal_evaluation_count": bucket["proposal_evaluation_count"],
                "scheduled_count": bucket["scheduled_count"],
                "applied_count": bucket["applied_count"],
                "rejected_count": bucket["rejected_count"],
                "actions": sorted(bucket["actions"]),
                "reason_codes": sorted(bucket["reason_codes"]),
                "replay_status": "passed",
            })
    scheduled = sum(int(item["scheduled_count"]) for item in condition_summaries)
    applied = sum(int(item["applied_count"]) for item in condition_summaries)
    rejected = sum(int(item["rejected_count"]) for item in condition_summaries)
    proposal_evaluations = sum(int(item["proposal_evaluation_count"]) for item in condition_summaries)
    return {
        "status": "completed_pending_human_approval",
        "amendment_id": amendment.amendment_id,
        "amendment_version": amendment.amendment_version,
        "run_count": len(records),
        "matched_group_count": len(config.matched_groups),
        "provider_call_count": provider_calls,
        "logical_request_count": len({item["fingerprint"] for item in stable_ledgers if item["status"] == "started"}),
        "transport_attempt_count": sum(item["status"] == "started" for item in stable_ledgers),
        "retry_count": 0,
        "recovery_count": 0,
        "proposal_evaluation_count": proposal_evaluations,
        "scheduled_count": scheduled,
        "applied_count": applied,
        "rejected_count": rejected,
        "condition_summaries": condition_summaries,
        "replay_status": "passed",
        "ledger_sha256": _hash(stable_ledgers),
        "request_cap": config.budget.request_cap,
        "completion_reservation_cap": config.budget.completion_reservation_cap,
        "network": "disabled",
        "private_truth_exposed": False,
        "calibration_fixture_excluded": True,
        "development_only": True,
        "not_paper_result": True,
        "no_causal_conclusion": True,
        "temporary_results_only": True,
        "ready_for_real_pilot": False,
        "human_approval_required": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline H-D FakeProvider amendment smoke.")
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG))
    parser.add_argument("--write-receipt", action="store_true")
    parser.add_argument("--receipt-output", default=str(_DEFAULT_RECEIPT))
    args = parser.parse_args(argv)
    summary = run_smoke(args.config)
    if args.write_receipt:
        receipt = write_amendment_receipt(args.config, summary, args.receipt_output)
        summary = {**summary, "receipt_sha256": hashlib.sha256(Path(args.receipt_output).read_bytes()).hexdigest(), "receipt_status": receipt.status}
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
