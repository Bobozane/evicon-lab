"""Offline lock for the planned protocol-blind v2 confirmatory manifest.

The lock is a design receipt only. It reads the manifest, the independently
reviewed synthetic test scenario, and the frozen ProbeSet metadata; it never
constructs a provider or reads environment configuration.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .frozen_wvs7_manifest import load_frozen_manifest, sha256_file
from .wvs7_confirmatory_study import (
    EXPECTED_CONDITIONS,
    EXPECTED_PRIMARY_METRICS,
    EXPECTED_V2_PROBE_SHA256,
    EXPECTED_V2_PROBE_SET_ID,
    EXPECTED_V2_TEMPLATE,
    ConfirmatoryStudyError,
    load_confirmatory_manifest,
    validate_confirmatory_manifest,
)
from .wvs7_test_scenario import TEST_SCENARIO_ID, TestScenarioAuthoringError, load_authoring, load_review, validate_test_scenario

_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SEEDS = (20260820, 20260821, 20260822)
EXPECTED_LOGICAL_REQUESTS_PER_GROUP = 384
EXPECTED_COMPLETION_RESERVATION_PER_GROUP = 51_200
EXPECTED_OBSERVED_TOTAL_TOKENS_PER_GROUP = 190_602


class ConfirmatoryPlanLockError(ValueError):
    """Stable, safe lock failure without source text or provider details."""


class ConfirmatoryPlanLockReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lock_id: str = Field(min_length=1)
    study_id: str = Field(min_length=1)
    status: Literal["locked"]
    plan_only: Literal[True]
    test_scenario_id: str
    probe_set_id: Literal["wvs7-english-core-23"]
    frozen_probe_set_sha256: str
    input_file_sha256: dict[str, str]
    seed_count: int = Field(ge=0)
    condition_count: int = Field(ge=0)
    run_count: int = Field(ge=0)
    matched_group_count: int = Field(ge=0)
    seeds: list[int]
    conditions: list[str]
    logical_requests_per_matched_group: int = Field(ge=0)
    total_logical_requests: int = Field(ge=0)
    completion_reservation_per_matched_group: int = Field(ge=0)
    total_completion_reservation: int = Field(ge=0)
    observed_total_token_reference_per_matched_group: int = Field(ge=0)
    observed_total_token_reference: int = Field(ge=0)
    primary_metrics: list[str]
    excluded_development_run_ids: list[str]
    locked_on: str


class ConfirmatoryPlanLockResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["locked", "blocked"]
    lock_receipt_path: str | None = None
    study_id: str
    blocking_reasons: list[str] = Field(default_factory=list)


def _rooted(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else _ROOT / value


def _safe_hash(path: Path, code: str) -> str:
    try:
        return sha256_file(path)
    except Exception as exc:
        raise ConfirmatoryPlanLockError(code) from exc


def _check_plan_contract(manifest_path: Path):
    try:
        manifest = load_confirmatory_manifest(manifest_path)
    except ConfirmatoryStudyError as exc:
        raise ConfirmatoryPlanLockError("confirmatory_manifest_invalid") from exc
    report = validate_confirmatory_manifest(manifest, verify_frozen_file=True)
    if report.status != "ready_for_confirmatory":
        raise ConfirmatoryPlanLockError("confirmatory_manifest_blocked:" + ",".join(report.blocking_reasons))
    if tuple(manifest.seeds) != EXPECTED_SEEDS:
        raise ConfirmatoryPlanLockError("confirmatory_seed_plan_mismatch")
    if tuple(manifest.conditions) != EXPECTED_CONDITIONS:
        raise ConfirmatoryPlanLockError("confirmatory_condition_plan_mismatch")
    if manifest.test_scenario_ids != [TEST_SCENARIO_ID]:
        raise ConfirmatoryPlanLockError("confirmatory_test_scenario_mismatch")
    if len(manifest.runs) != 12:
        raise ConfirmatoryPlanLockError("confirmatory_run_count_mismatch")
    run_ids = [run.run_id for run in manifest.runs]
    if len(run_ids) != len(set(run_ids)):
        raise ConfirmatoryPlanLockError("duplicate_run_id")
    groups = {run.matched_group_id for run in manifest.runs}
    if len(groups) != 3:
        raise ConfirmatoryPlanLockError("matched_group_count_mismatch")
    expected_total = len(EXPECTED_SEEDS) * EXPECTED_LOGICAL_REQUESTS_PER_GROUP
    expected_reservation = len(EXPECTED_SEEDS) * EXPECTED_COMPLETION_RESERVATION_PER_GROUP
    expected_observed = len(EXPECTED_SEEDS) * EXPECTED_OBSERVED_TOTAL_TOKENS_PER_GROUP
    if (
        manifest.logical_requests_per_matched_group != EXPECTED_LOGICAL_REQUESTS_PER_GROUP
        or manifest.completion_reservation_per_matched_group != EXPECTED_COMPLETION_RESERVATION_PER_GROUP
        or manifest.observational_total_token_reference_per_matched_group != EXPECTED_OBSERVED_TOTAL_TOKENS_PER_GROUP
        or manifest.total_logical_requests != expected_total
        or manifest.total_completion_reservation != expected_reservation
        or manifest.observational_total_token_reference != expected_observed
    ):
        raise ConfirmatoryPlanLockError("cost_plan_mismatch")
    if manifest.probe_template_version != EXPECTED_V2_TEMPLATE or manifest.probe_set_id != EXPECTED_V2_PROBE_SET_ID:
        raise ConfirmatoryPlanLockError("probe_contract_mismatch")
    if tuple(manifest.primary_metrics) != EXPECTED_PRIMARY_METRICS:
        raise ConfirmatoryPlanLockError("primary_metrics_mismatch")
    return manifest


def lock_confirmatory_plan(
    manifest_path: str | Path = "configs/studies/wvs7_english_protocol_blind_v2_confirmatory_plan.toml",
    *,
    output_path: str | Path = "outputs/study-locks/wvs7_protocol_blind_v2_confirmatory_plan_lock.json",
) -> ConfirmatoryPlanLockResult:
    """Validate and lock the design plan without changing any input artifact."""
    manifest_file = _rooted(manifest_path)
    manifest = _check_plan_contract(manifest_file)
    scenario_path = _rooted(manifest.test_scenario_path)
    authoring_path = _rooted(manifest.test_scenario_authoring_path)
    review_path = _rooted(manifest.test_scenario_review_path)
    try:
        scenario_report = validate_test_scenario(load_authoring(authoring_path), load_review(review_path), require_review=True)
    except (TestScenarioAuthoringError, OSError, ValueError) as exc:
        raise ConfirmatoryPlanLockError("test_scenario_review_invalid") from exc
    if scenario_report.status != "ready" or scenario_report.scenario_id != TEST_SCENARIO_ID:
        raise ConfirmatoryPlanLockError("test_scenario_review_not_accepted")
    if _safe_hash(scenario_path, "test_scenario_missing") != manifest.test_scenario_sha256:
        raise ConfirmatoryPlanLockError("test_scenario_hash_mismatch")

    probe_path = _rooted(manifest.frozen_probe_set_path)
    frozen_manifest_path = probe_path.with_name("english_core_wvs7_23_frozen_manifest.json")
    try:
        frozen = load_frozen_manifest(frozen_manifest_path)
    except Exception as exc:
        raise ConfirmatoryPlanLockError("frozen_probe_manifest_invalid") from exc
    if frozen.probe_set_id != EXPECTED_V2_PROBE_SET_ID or frozen.item_count != 23:
        raise ConfirmatoryPlanLockError("frozen_probe_set_contract_mismatch")
    if _safe_hash(probe_path, "frozen_probe_set_missing") != EXPECTED_V2_PROBE_SHA256 or frozen.probe_set_sha256 != EXPECTED_V2_PROBE_SHA256:
        raise ConfirmatoryPlanLockError("frozen_probe_set_hash_mismatch")

    output = _rooted(output_path)
    if output.exists():
        raise ConfirmatoryPlanLockError("lock_output_already_exists")
    input_paths = {
        "confirmatory_manifest": manifest_file,
        "test_scenario": scenario_path,
        "test_scenario_authoring": authoring_path,
        "test_scenario_review": review_path,
        "frozen_probe_set": probe_path,
        "frozen_probe_manifest": frozen_manifest_path,
    }
    receipt = ConfirmatoryPlanLockReceipt(
        lock_id="wvs7-protocol-blind-v2-confirmatory-plan-lock",
        study_id=manifest.study_id,
        status="locked",
        plan_only=True,
        test_scenario_id=TEST_SCENARIO_ID,
        probe_set_id=EXPECTED_V2_PROBE_SET_ID,
        frozen_probe_set_sha256=EXPECTED_V2_PROBE_SHA256,
        input_file_sha256={name: _safe_hash(path, "lock_input_missing") for name, path in input_paths.items()},
        seed_count=len(manifest.seeds),
        condition_count=len(manifest.conditions),
        run_count=len(manifest.runs),
        matched_group_count=len({run.matched_group_id for run in manifest.runs}),
        seeds=list(manifest.seeds),
        conditions=[condition.value for condition in manifest.conditions],
        logical_requests_per_matched_group=manifest.logical_requests_per_matched_group,
        total_logical_requests=manifest.total_logical_requests,
        completion_reservation_per_matched_group=manifest.completion_reservation_per_matched_group,
        total_completion_reservation=manifest.total_completion_reservation,
        observed_total_token_reference_per_matched_group=manifest.observational_total_token_reference_per_matched_group,
        observed_total_token_reference=manifest.observational_total_token_reference,
        primary_metrics=list(manifest.primary_metrics),
        excluded_development_run_ids=list(manifest.excluded_development_run_ids),
        locked_on=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
    return ConfirmatoryPlanLockResult(status="locked", lock_receipt_path=str(output), study_id=manifest.study_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lock a future protocol-blind v2 confirmatory design offline.")
    parser.add_argument("--manifest", "--config", dest="manifest", required=True)
    parser.add_argument("--output", default="outputs/study-locks/wvs7_protocol_blind_v2_confirmatory_plan_lock.json")
    args = parser.parse_args(argv)
    try:
        result = lock_confirmatory_plan(args.manifest, output_path=args.output)
    except ConfirmatoryPlanLockError as exc:
        print(json.dumps({"status": "blocked", "blocking_reasons": [str(exc)]}, sort_keys=True))
        return 1
    print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ConfirmatoryPlanLockError",
    "ConfirmatoryPlanLockReceipt",
    "ConfirmatoryPlanLockResult",
    "lock_confirmatory_plan",
]
