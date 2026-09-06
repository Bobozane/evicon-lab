"""Read-only technical amendment audit for the H-G outcome replay repair."""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from .cascade_agent_response import CascadeAgentResponse
from .cascade_application_replay import CascadeApplicationReplayValidator
from .cascade_hg_outcome_replay import HG_OUTCOME_REPLAY_VERSION, HGOutcomeReplayValidator
from .cascade_intervention_application import CascadeApplicationLedger, ControlledCascadeProtocolRunner
from .cascade_outcomes import CascadeOutcomeLedger, ClaimStance, OutcomeSource, PublicClaimOutcome
from .cascade_protocol import ActorAction, CascadeProtocolRunner
from .cascade_real_agent_runner import CascadeAgentCheckpoint, CascadeRealAgentRunRecord, _checkpoint_binding
from .provenance_cascade_exposure import ExposureLedger
from .provenance_cascade_identifiability import load_hg_config, sha256_file
from .request_ledger import RequestLedger, RequestLedgerStatus

_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH: Final = _ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg.v1.toml"
PROTOCOL_PATH: Final = _ROOT / "src/evicon/cascade_agent_protocol_hg.py"
OLD_VALIDATOR_PATH: Final = _ROOT / "src/evicon/cascade_outcome_replay.py"
NEW_VALIDATOR_PATH: Final = _ROOT / "src/evicon/cascade_hg_outcome_replay.py"
AMENDMENT_PATH: Final = _ROOT / "configs/provenance_cascade/pilot/amendments/provenance_cascade_pilot_hg_public_identifiability_amendment.v1.toml"
APPROVAL_PATH: Final = _ROOT / "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg_approval_template.toml"
COMPATIBILITY_PATH: Final = _ROOT / "outputs/study-locks/provenance_cascade_hg_compatibility_receipt.json"
RESULT_ROOT: Final = _ROOT / "results/provenance-cascade-pilot-hg-v1"
BATCH_PATH: Final = RESULT_ROOT / "pilot_batch_record.json"
DEFAULT_RECEIPT_PATH: Final = _ROOT / "outputs/study-locks/provenance_cascade_hg_outcome_replay_technical_amendment_receipt.json"
FAILED_RUN_ID: Final = "hg-cascade-hg-true-minority-correction-20260911-no_intervention"
AMENDMENT_VERSION: Final = "provenance_cascade_hg_outcome_replay_technical_amendment.v1"
EXPECTED_RECEIPT_SHA256: Final = "8e1a3651aedc4987001fc0596bbbb5d0a016d4e542b07deda977dc602dfbc2a9"


class HGReplayAmendmentError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HGReplayTechnicalAmendmentReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    amendment_id: str = "provenance-cascade-hg-outcome-replay-technical-amendment-v1"
    amendment_version: str = AMENDMENT_VERSION
    status: str = "offline_validated_ready_for_resume"
    registered_on: date
    batch_id: str = "evicon-provenance-cascade-pilot-hg"
    batch_record_sha256: str = Field(min_length=64, max_length=64)
    config_sha256: str = Field(min_length=64, max_length=64)
    protocol_sha256: str = Field(min_length=64, max_length=64)
    amendment_sha256: str = Field(min_length=64, max_length=64)
    approval_sha256: str = Field(min_length=64, max_length=64)
    compatibility_receipt_sha256: str = Field(min_length=64, max_length=64)
    scenario_material_bindings_sha256: str = Field(min_length=64, max_length=64)
    old_validator_sha256: str = Field(min_length=64, max_length=64)
    new_validator_sha256: str = Field(min_length=64, max_length=64)
    outcome_replay_contract_version: str = HG_OUTCOME_REPLAY_VERSION
    completed_run_count: int = Field(ge=0)
    failed_run_id: str
    failed_run_logical_request_count: int = Field(ge=0)
    failed_run_completed_fingerprint_count: int = Field(ge=0)
    failed_run_checkpoint_entry_count: int = Field(ge=0)
    failed_run_transport_attempt_count: int = Field(ge=0)
    failed_run_resume_provider_calls_required: int = Field(ge=0)
    completed_run_cascade_replay_passed_count: int = Field(ge=0)
    completed_run_application_replay_passed_count: int = Field(ge=0)
    completed_run_outcome_replay_passed_count: int = Field(ge=0)
    failed_run_cascade_replay_status: str
    failed_run_application_replay_status: str
    failed_run_outcome_replay_status: str
    completed_fingerprints_replayed: bool = False
    changes_replay_validation_only: bool = True
    allows_visible_cross_claim_correction_content: bool = True
    prompt_unchanged: bool = True
    scenario_unchanged: bool = True
    conditions_unchanged: bool = True
    seeds_unchanged: bool = True
    metrics_unchanged: bool = True
    budget_unchanged: bool = True
    request_cap_unchanged: bool = True
    model_unchanged: bool = True
    outcome_generation_unchanged: bool = True
    existing_results_rewritten: bool = False
    network: str = "disabled"
    provider_constructed: bool = False
    private_truth_exposed: bool = False
    development_only: bool = True
    pilot_only: bool = True
    not_paper_result: bool = True
    no_causal_conclusion: bool = True


class _FailedReplayAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    logical_request_count: int
    completed_fingerprint_count: int
    checkpoint_entry_count: int
    transport_attempt_count: int
    resume_provider_calls_required: int
    cascade_replay_status: str
    application_replay_status: str
    outcome_replay_status: str


def _sha_payload(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scenario_bindings(config: object) -> str:
    materials = getattr(config, "scenario_materials")
    return _sha_payload([
        {
            "scenario_id": item.scenario_id,
            "scenario_sha256": item.scenario_sha256,
            "graph_sha256": item.graph_sha256,
            "truth_sha256": item.truth_sha256,
        }
        for item in materials
    ])


def _load_batch() -> object:
    from .provenance_cascade_hg_pilot import HGPilotBatchRecord
    if not BATCH_PATH.is_file():
        raise HGReplayAmendmentError("hg_batch_record_missing")
    try:
        return HGPilotBatchRecord.model_validate_json(BATCH_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HGReplayAmendmentError("hg_batch_record_invalid") from exc


def _validate_design_bindings(batch: object, config: object) -> None:
    expected = {
        "config_sha256": sha256_file(CONFIG_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "amendment_sha256": sha256_file(AMENDMENT_PATH),
        "approval_sha256": sha256_file(APPROVAL_PATH),
        "compatibility_receipt_sha256": sha256_file(COMPATIBILITY_PATH),
    }
    for field_name, digest in expected.items():
        if getattr(batch, field_name) != digest:
            raise HGReplayAmendmentError(f"{field_name}_mismatch")
    if batch.study_id != config.study_id or len(batch.runs) != 48:
        raise HGReplayAmendmentError("hg_batch_plan_mismatch")
    if batch.status.value != "failed" or batch.failure_code != "cascade_outcome_replay_failed":
        raise HGReplayAmendmentError("hg_batch_failure_state_mismatch")


def _validate_completed_runs(batch: object, scenarios: dict[str, object]) -> tuple[int, int, int, int]:
    completed = [item for item in batch.runs if item.status.value == "completed"]
    if len(completed) != 12:
        raise HGReplayAmendmentError("completed_run_count_mismatch")
    cascade_passed = application_passed = outcome_passed = 0
    for state in completed:
        path = RESULT_ROOT / state.run_id / "run_record.json"
        if not path.is_file() or sha256_file(path) != state.run_record_sha256:
            raise HGReplayAmendmentError("completed_run_record_hash_mismatch")
        try:
            record = CascadeRealAgentRunRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HGReplayAmendmentError("completed_run_record_invalid") from exc
        if (
            record.exposure_ledger is None
            or record.application_ledger is None
            or record.outcome_ledger is None
            or record.scenario_id != state.scenario_id
            or record.seed != state.seed
            or record.condition.value != state.condition
        ):
            raise HGReplayAmendmentError("completed_run_sidecar_mismatch")
        try:
            report = HGOutcomeReplayValidator.validate(
                scenarios[state.scenario_id].graph,
                record.exposure_ledger,
                record.outcome_ledger,
                record.application_ledger,
                record.round_contexts,
            )
        except Exception as exc:
            raise HGReplayAmendmentError("completed_run_replay_failed") from exc
        cascade_passed += report.cascade_replay.status.value == "passed"
        application_passed += report.application_replay.status.value == "passed"
        outcome_passed += report.status.value == "passed"
    return len(completed), cascade_passed, application_passed, outcome_passed


def _audit_failed_run(config: object, scenarios: dict[str, object], batch: object) -> _FailedReplayAudit:
    state = next((item for item in batch.runs if item.run_id == FAILED_RUN_ID), None)
    if state is None or state.status.value != "failed" or state.error_code != "cascade_outcome_replay_failed":
        raise HGReplayAmendmentError("failed_run_state_mismatch")
    spec = next((item for item in config.runs if item.run_id == FAILED_RUN_ID), None)
    if spec is None or spec.condition.value != "no_intervention":
        raise HGReplayAmendmentError("failed_run_spec_mismatch")
    run_dir = RESULT_ROOT / FAILED_RUN_ID
    checkpoint_path = run_dir / "agent_checkpoint.json"
    ledger_path = run_dir / "request_ledger.jsonl"
    if not checkpoint_path.is_file() or not ledger_path.is_file():
        raise HGReplayAmendmentError("failed_run_recovery_input_missing")
    try:
        checkpoint = CascadeAgentCheckpoint.model_validate_json(checkpoint_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HGReplayAmendmentError("failed_run_checkpoint_invalid") from exc
    scenario = scenarios[spec.scenario_id]
    expected_binding = _checkpoint_binding(
        scenario=scenario,
        seed=spec.seed,
        condition=spec.condition,
        run_id=spec.run_id,
        model_name=batch.model_name,
        temperature=0.2,
        max_tokens=config.agent_max_tokens,
    )
    if checkpoint.binding_sha256 != expected_binding:
        raise HGReplayAmendmentError("failed_run_checkpoint_binding_mismatch")
    ledger = RequestLedger(ledger_path)
    entries = ledger.entries()
    summary = ledger.summary(
        request_cap=spec.expected_provider_requests,
        completion_reservation_cap=spec.completion_reservation,
    )
    completed = {
        entry.fingerprint
        for entry in entries
        if entry.status is RequestLedgerStatus.COMPLETED
    }
    started = [entry for entry in entries if entry.status is RequestLedgerStatus.STARTED]
    checkpoint_fingerprints = {entry.fingerprint for entry in checkpoint.entries}
    if (
        len(checkpoint.entries) != 18
        or len(checkpoint_fingerprints) != 18
        or completed != checkpoint_fingerprints
        or len(started) != 18
        or summary.unique_logical_request_count != 18
        or summary.completed_count != 18
        or summary.failed_count != 0
        or summary.transport_attempt_count != 18
    ):
        raise HGReplayAmendmentError("failed_run_completed_fingerprint_coverage_invalid")
    responses = {
        (item.agent_id, item.round_id): CascadeAgentResponse(
            stance=item.stance,
            content_ids_used=item.content_ids_used,
            evidence_ids_used=item.evidence_ids_used,
            share_content_id=item.share_content_id,
        )
        for item in checkpoint.entries
    }
    base = CascadeProtocolRunner()
    controlled = ControlledCascadeProtocolRunner(base)
    events: list[object] = []
    outcomes: list[PublicClaimOutcome] = []
    contexts: list[object] = []
    application = CascadeApplicationLedger()
    for round_id in range(scenario.max_rounds):
        round_snapshot = tuple(base.build_round_snapshot(scenario, tuple(events), round_id))
        contexts_for_round = controlled.build_next_round_context(
            scenario, tuple(events), round_id, application
        )
        contexts.extend(contexts_for_round)
        for snapshot in round_snapshot:
            response = responses.get((snapshot.agent_id, round_id))
            if response is None:
                raise HGReplayAmendmentError("failed_run_checkpoint_coordinate_missing")
            if snapshot.visible_claim_ids:
                outcomes.append(PublicClaimOutcome(
                    scenario_id=scenario.scenario_id,
                    agent_id=snapshot.agent_id,
                    claim_id=snapshot.visible_claim_ids[0],
                    round_id=round_id,
                    stance=response.stance,
                    content_ids=response.content_ids_used,
                    evidence_card_ids=response.evidence_ids_used,
                    source=OutcomeSource.ACTOR_OBSERVATION,
                ))
            else:
                for claim in scenario.graph.claims:
                    outcomes.append(PublicClaimOutcome(
                        scenario_id=scenario.scenario_id,
                        agent_id=snapshot.agent_id,
                        claim_id=claim.claim_id,
                        round_id=round_id,
                        stance=ClaimStance.NO_POSITION,
                        source=OutcomeSource.ACTOR_OBSERVATION,
                    ))
        base._append_initial_content(scenario, round_id, events)
        for index, schedule in enumerate(item for item in scenario.actor_schedule if item.round_id == round_id):
            if schedule.action is ActorAction.ABSTAIN:
                continue
            response = responses[(schedule.actor_id, round_id)]
            node = next(item for item in scenario.graph.nodes if item.node_id == schedule.provenance_node_id)
            required_share = schedule.parent_content_id or node.content_id
            if response.share_content_id != required_share:
                continue
            for target_index, target_agent_id in enumerate(schedule.target_agent_ids):
                events.append(base.append_public_event(
                    scenario=scenario,
                    events=tuple(events),
                    snapshots=tuple(item.snapshot for item in contexts_for_round),
                    entry=schedule,
                    target_agent_id=target_agent_id,
                    event_id=f"{FAILED_RUN_ID}-r{round_id}-e{index}-t{target_index}",
                    round_snapshot=round_snapshot,
                ))
    exposure = ExposureLedger(
        scenario_id=scenario.scenario_id,
        agent_ids=scenario.agent_ids,
        events=tuple(events),
        snapshots=tuple(item.snapshot for item in contexts),
    )
    outcome = CascadeOutcomeLedger(
        scenario_id=scenario.scenario_id,
        agent_ids=scenario.agent_ids,
        outcomes=tuple(outcomes),
    )
    try:
        report = HGOutcomeReplayValidator.validate(
            scenario.graph, exposure, outcome, application, tuple(contexts)
        )
        application_report = CascadeApplicationReplayValidator.validate(application, tuple(contexts))
    except Exception as exc:
        raise HGReplayAmendmentError("failed_run_offline_replay_failed") from exc
    return _FailedReplayAudit(
        logical_request_count=summary.unique_logical_request_count,
        completed_fingerprint_count=len(completed),
        checkpoint_entry_count=len(checkpoint.entries),
        transport_attempt_count=summary.transport_attempt_count,
        resume_provider_calls_required=0,
        cascade_replay_status=report.cascade_replay.status.value,
        application_replay_status=application_report.status.value,
        outcome_replay_status=report.status.value,
    )


def audit_technical_amendment(*, registered_on: date | None = None) -> HGReplayTechnicalAmendmentReceipt:
    config, scenarios, _ = load_hg_config(CONFIG_PATH)
    batch = _load_batch()
    _validate_design_bindings(batch, config)
    completed, cascade_passed, application_passed, outcome_passed = _validate_completed_runs(batch, scenarios)
    failed = _audit_failed_run(config, scenarios, batch)
    receipt = HGReplayTechnicalAmendmentReceipt(
        registered_on=registered_on or date.today(),
        batch_record_sha256=sha256_file(BATCH_PATH),
        config_sha256=sha256_file(CONFIG_PATH),
        protocol_sha256=sha256_file(PROTOCOL_PATH),
        amendment_sha256=sha256_file(AMENDMENT_PATH),
        approval_sha256=sha256_file(APPROVAL_PATH),
        compatibility_receipt_sha256=sha256_file(COMPATIBILITY_PATH),
        scenario_material_bindings_sha256=_scenario_bindings(config),
        old_validator_sha256=sha256_file(OLD_VALIDATOR_PATH),
        new_validator_sha256=sha256_file(NEW_VALIDATOR_PATH),
        completed_run_count=completed,
        failed_run_id=FAILED_RUN_ID,
        failed_run_logical_request_count=failed.logical_request_count,
        failed_run_completed_fingerprint_count=failed.completed_fingerprint_count,
        failed_run_checkpoint_entry_count=failed.checkpoint_entry_count,
        failed_run_transport_attempt_count=failed.transport_attempt_count,
        failed_run_resume_provider_calls_required=failed.resume_provider_calls_required,
        completed_run_cascade_replay_passed_count=cascade_passed,
        completed_run_application_replay_passed_count=application_passed,
        completed_run_outcome_replay_passed_count=outcome_passed,
        failed_run_cascade_replay_status=failed.cascade_replay_status,
        failed_run_application_replay_status=failed.application_replay_status,
        failed_run_outcome_replay_status=failed.outcome_replay_status,
    )
    serialized = receipt.model_dump_json().lower()
    forbidden = (
        "system_prompt", "user_prompt", "provider_metadata", "api_key",
        "ground_truth_label", "source_independence_label", "model_response",
    )
    if any(item in serialized for item in forbidden):
        raise HGReplayAmendmentError("technical_receipt_sensitive_field_detected")
    return receipt


def write_technical_amendment_receipt(
    path: str | Path = DEFAULT_RECEIPT_PATH,
) -> tuple[HGReplayTechnicalAmendmentReceipt, str]:
    output = Path(path).resolve()
    if output.exists():
        raise HGReplayAmendmentError("technical_amendment_receipt_exists")
    receipt = audit_technical_amendment()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(output)
    return receipt, sha256_file(output)


def validate_technical_amendment_receipt(
    path: str | Path = DEFAULT_RECEIPT_PATH,
) -> tuple[HGReplayTechnicalAmendmentReceipt, str]:
    """Validate the immutable pre-resume receipt against current design files.

    Once the authorized resume completes, the live batch record necessarily changes
    from failed to completed. The receipt remains a historical recovery credential;
    it must not be regenerated or rebound to the final batch.
    """
    receipt_path = Path(path).resolve()
    if not receipt_path.is_file():
        raise HGReplayAmendmentError("technical_amendment_receipt_missing")
    digest = sha256_file(receipt_path)
    if digest != EXPECTED_RECEIPT_SHA256:
        raise HGReplayAmendmentError("technical_amendment_receipt_hash_mismatch")
    try:
        stored = HGReplayTechnicalAmendmentReceipt.model_validate_json(
            receipt_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise HGReplayAmendmentError("technical_amendment_receipt_invalid") from exc
    config, _, _ = load_hg_config(CONFIG_PATH)
    batch = _load_batch()
    expected = {
        "config_sha256": sha256_file(CONFIG_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "amendment_sha256": sha256_file(AMENDMENT_PATH),
        "approval_sha256": sha256_file(APPROVAL_PATH),
        "compatibility_receipt_sha256": sha256_file(COMPATIBILITY_PATH),
        "old_validator_sha256": sha256_file(OLD_VALIDATOR_PATH),
        "new_validator_sha256": sha256_file(NEW_VALIDATOR_PATH),
    }
    for field_name, expected_digest in expected.items():
        if getattr(stored, field_name) != expected_digest:
            raise HGReplayAmendmentError("technical_amendment_receipt_binding_mismatch")
    if (
        stored.scenario_material_bindings_sha256 != _scenario_bindings(config)
        or stored.outcome_replay_contract_version != HG_OUTCOME_REPLAY_VERSION
        or stored.completed_run_count != 12
        or stored.failed_run_logical_request_count != 18
        or stored.failed_run_completed_fingerprint_count != 18
        or stored.failed_run_checkpoint_entry_count != 18
        or stored.failed_run_transport_attempt_count != 18
        or stored.failed_run_resume_provider_calls_required != 0
        or stored.failed_run_cascade_replay_status != "passed"
        or stored.failed_run_application_replay_status != "passed"
        or stored.failed_run_outcome_replay_status != "passed"
        or stored.completed_fingerprints_replayed
        or not stored.changes_replay_validation_only
        or stored.existing_results_rewritten
    ):
        raise HGReplayAmendmentError("technical_amendment_receipt_binding_mismatch")
    _validate_design_bindings_after_resume(batch, config)
    return stored, digest


def _validate_design_bindings_after_resume(batch: object, config: object) -> None:
    expected = {
        "config_sha256": sha256_file(CONFIG_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "amendment_sha256": sha256_file(AMENDMENT_PATH),
        "approval_sha256": sha256_file(APPROVAL_PATH),
        "compatibility_receipt_sha256": sha256_file(COMPATIBILITY_PATH),
    }
    for field_name, digest in expected.items():
        if getattr(batch, field_name) != digest:
            raise HGReplayAmendmentError(f"{field_name}_mismatch")
    if batch.study_id != config.study_id or len(batch.runs) != 48:
        raise HGReplayAmendmentError("hg_batch_plan_mismatch")
    status = batch.status.value
    if status == "failed":
        if batch.failure_code != "cascade_outcome_replay_failed":
            raise HGReplayAmendmentError("hg_batch_failure_state_mismatch")
        return
    if status != "completed" or batch.failure_code is not None:
        raise HGReplayAmendmentError("hg_batch_completion_state_mismatch")
    if any(
        item.status.value != "completed"
        or item.error_code is not None
        or item.cascade_replay_status != "passed"
        or item.application_replay_status != "passed"
        or item.outcome_replay_status != "passed"
        for item in batch.runs
    ):
        raise HGReplayAmendmentError("hg_batch_completion_state_mismatch")


def safe_summary(receipt: HGReplayTechnicalAmendmentReceipt, digest: str) -> dict[str, object]:
    return {
        "status": receipt.status,
        "amendment_version": receipt.amendment_version,
        "receipt_sha256": digest,
        "batch_record_sha256": receipt.batch_record_sha256,
        "old_validator_sha256": receipt.old_validator_sha256,
        "new_validator_sha256": receipt.new_validator_sha256,
        "outcome_replay_contract_version": receipt.outcome_replay_contract_version,
        "completed_run_count": receipt.completed_run_count,
        "failed_run_logical_request_count": receipt.failed_run_logical_request_count,
        "failed_run_completed_fingerprint_count": receipt.failed_run_completed_fingerprint_count,
        "failed_run_resume_provider_calls_required": receipt.failed_run_resume_provider_calls_required,
        "completed_run_replay_passed_count": receipt.completed_run_outcome_replay_passed_count,
        "failed_run_replay": {
            "cascade": receipt.failed_run_cascade_replay_status,
            "application": receipt.failed_run_application_replay_status,
            "outcome": receipt.failed_run_outcome_replay_status,
        },
        "network": "disabled",
        "provider_constructed": False,
        "existing_results_rewritten": False,
        "private_truth_exposed": False,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


__all__ = [
    "AMENDMENT_VERSION",
    "EXPECTED_RECEIPT_SHA256",
    "DEFAULT_RECEIPT_PATH",
    "HGReplayAmendmentError",
    "HGReplayTechnicalAmendmentReceipt",
    "audit_technical_amendment",
    "safe_summary",
    "validate_technical_amendment_receipt",
    "write_technical_amendment_receipt",
]
