"""Opt-in real-Agent adapter for provenance-cascade H-C.

This module is a sidecar to the deterministic outcome runner. It consumes only
one public round-start view and an already-applied structured directive, then
records parsed public outcomes. It never loads evaluator truth or writes
formal pilot results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cascade_agent_prompts import CascadeAgentPromptContext, CascadeAgentRuntimeConfig, render_cascade_agent_turn
from .cascade_agent_runtime import CascadeAgentRuntime, CascadeAgentRuntimeStatus
from .cascade_controller import CascadeControllerPolicyConfig, CascadeControllerPolicyLoader, propose, CascadeControllerValidationError
from .cascade_intervention_application import (
    CascadeApplicationLedger, CascadeApplicationStatus, CascadeScheduleStatus,
    ControlledCascadeProtocolRunner, ControlledRoundContext, snapshot_sha256,
)
from .cascade_outcome_replay import CascadeOutcomeReplayValidator, CascadeOutcomeReplayReport
from .cascade_outcomes import CascadeOutcomeLedger, ClaimStance, OutcomeSource, PublicClaimOutcome
from .cascade_protocol import ActorAction, CascadeProtocolRunner, CascadeScenarioSpec
from .llm_contract import LLMProvider
from .provenance_cascade_exposure import ControllerPublicView, ExposureLedger
from .provenance_cascade_preregistration import CascadeCondition
from .request_ledger import LedgeredProvider, RequestLedger, RequestLedgerError, request_fingerprint_facts
from .cascade_agent_response import CascadeAgentResponse

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_POLICY = _ROOT / "configs" / "provenance_cascade" / "cascade_controller_policy.v1.toml"


def _sha(value: object) -> str:
    data = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


class CascadeAgentCheckpointEntry(BaseModel):
    """Public parsed response checkpoint; never stores prompts or raw replies."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    fingerprint: str = Field(min_length=64, max_length=64)
    request_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    stance: ClaimStance
    content_ids_used: tuple[str, ...] = ()
    evidence_ids_used: tuple[str, ...] = ()
    share_content_id: str | None = None


class CascadeAgentCheckpoint(BaseModel):
    """Safe, resumable cache of successfully parsed public responses."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    binding_sha256: str = Field(min_length=64, max_length=64)
    entries: tuple[CascadeAgentCheckpointEntry, ...] = ()

    @model_validator(mode="after")
    def unique_fingerprints(self) -> "CascadeAgentCheckpoint":
        if len({entry.fingerprint for entry in self.entries}) != len(self.entries):
            raise ValueError("checkpoint fingerprints must be unique")
        return self


def _checkpoint_binding(*, scenario: CascadeScenarioSpec, seed: int, condition: CascadeCondition, run_id: str, model_name: str, temperature: float, max_tokens: int) -> str:
    return _sha({
        "scenario_id": scenario.scenario_id,
        "scenario_hash": _sha(scenario.model_dump(mode="json")),
        "seed": seed, "condition": condition.value, "run_id": run_id,
        "model_name": model_name, "temperature": temperature, "max_tokens": max_tokens,
    })


def _load_checkpoint(path: Path, binding: str) -> CascadeAgentCheckpoint:
    try:
        checkpoint = CascadeAgentCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CascadeRealAgentRunError("checkpoint_invalid") from exc
    if checkpoint.binding_sha256 != binding:
        raise CascadeRealAgentRunError("checkpoint_binding_mismatch")
    return checkpoint


def _write_checkpoint(path: Path, checkpoint: CascadeAgentCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(checkpoint.model_dump_json(), encoding="utf-8")
    temporary.replace(path)


class CascadeRealAgentRunError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class CascadeRealAgentRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    scenario_id: str
    seed: int = Field(gt=0)
    condition: CascadeCondition
    agent_ids: tuple[str, ...] = Field(min_length=6, max_length=6)
    round_count: int = Field(ge=1)
    logical_request_count: int = Field(ge=0)
    provider_call_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    content_event_count: int = Field(ge=0)
    exposure_event_count: int = Field(ge=0)
    snapshot_count: int = Field(ge=0)
    outcome_count: int = Field(ge=0)
    proposal_count: int = Field(ge=0)
    directive_applied_count: int = Field(ge=0)
    exposure_ledger_sha256: str = Field(min_length=64, max_length=64)
    application_ledger_sha256: str = Field(min_length=64, max_length=64)
    outcome_ledger_sha256: str = Field(min_length=64, max_length=64)
    replay: CascadeOutcomeReplayReport | None = None
    status: str = "completed"
    error_code: str | None = None
    request_ledger_path: str | None = None
    exposure_ledger: ExposureLedger | None = None
    application_ledger: CascadeApplicationLedger | None = None
    outcome_ledger: CascadeOutcomeLedger | None = None
    round_contexts: tuple[ControlledRoundContext, ...] = ()
    development_only: bool = True
    not_paper_result: bool = True
    no_causal_conclusion: bool = True

    @model_validator(mode="after")
    def safety(self) -> "CascadeRealAgentRunRecord":
        if not (self.development_only and self.not_paper_result and self.no_causal_conclusion):
            raise ValueError("H-C records require safety flags")
        return self


class CascadeRealAgentRunner:
    """Run one scenario using an injected AgentRuntime and ledgered provider."""

    def __init__(self, *, base_runner: CascadeProtocolRunner | None = None, policy_config: CascadeControllerPolicyConfig | None = None) -> None:
        self.base_runner = base_runner or CascadeProtocolRunner()
        self.controlled_runner = ControlledCascadeProtocolRunner(self.base_runner)
        self.policy_config = policy_config or CascadeControllerPolicyLoader.load(_DEFAULT_POLICY)
        self.runtime = CascadeAgentRuntime()

    def run_scenario(
        self,
        scenario: CascadeScenarioSpec,
        seed: int,
        condition: CascadeCondition | str,
        *,
        provider: LLMProvider,
        run_id: str,
        ledger_path: str | Path,
        model_name: str = "fake-cascade-agent",
        agent_temperature: float = 0.2,
        agent_max_tokens: int = 256,
        request_cap: int = 18,
        completion_reservation_cap: int = 4608,
        resume: bool = False,
        policy_config: CascadeControllerPolicyConfig | None = None,
        checkpoint_path: str | Path | None = None,
        recovery_fingerprints: set[str] | None = None,
        recovery_kind: str | None = None,
        max_recovery_attempts: int = 0,
        runtime: Any | None = None,
        context_builder: Any | None = None,
        outcome_replay_validator: Any | None = None,
        initial_events_builder: Any | None = None,
        claim_selector: Any | None = None,
        proposal_builder: Any | None = None,
        proposal_validator: Any | None = None,
        post_round_events_builder: Any | None = None,
        actor_schedule_provider: Any | None = None,
    ) -> CascadeRealAgentRunRecord:
        try:
            condition_id = condition if isinstance(condition, CascadeCondition) else CascadeCondition(condition)
        except (TypeError, ValueError) as exc:
            raise CascadeRealAgentRunError("condition_invalid") from exc
        if len(scenario.agent_ids) != 6 or scenario.max_rounds != 3:
            raise CascadeRealAgentRunError("scenario_shape_invalid")
        config = policy_config or self.policy_config
        ledger = RequestLedger(ledger_path)
        binding = _checkpoint_binding(scenario=scenario, seed=seed, condition=condition_id, run_id=run_id, model_name=model_name, temperature=agent_temperature, max_tokens=agent_max_tokens)
        checkpoint_file = Path(checkpoint_path) if checkpoint_path is not None else None
        checkpoint = _load_checkpoint(checkpoint_file, binding) if resume and checkpoint_file is not None and checkpoint_file.exists() else CascadeAgentCheckpoint(binding_sha256=binding)
        checkpoint_by_fingerprint = {entry.fingerprint: entry for entry in checkpoint.entries}
        ledgered = LedgeredProvider(
            provider, ledger=ledger, request_cap=request_cap,
            completion_reservation_cap=completion_reservation_cap,
            resume=resume, recovery_fingerprints=recovery_fingerprints,
            recovery_kind=recovery_kind,
            recovery_phases={"agent_turn"} if recovery_fingerprints else None,
            max_recovery_attempts=max_recovery_attempts,
        )
        selected_runtime = runtime or self.runtime
        selected_context_builder = context_builder or CascadeAgentPromptContext.from_public_view
        selected_outcome_replay_validator = outcome_replay_validator or CascadeOutcomeReplayValidator
        events: list[Any] = list(initial_events_builder(scenario) if initial_events_builder is not None else ())
        outcomes: list[PublicClaimOutcome] = []
        application = CascadeApplicationLedger()
        contexts: list[ControlledRoundContext] = []
        proposal_count = 0
        try:
            for round_id in range(scenario.max_rounds):
                round_snapshot = tuple(self.base_runner.build_round_snapshot(scenario, tuple(events), round_id))
                for scheduled in tuple(application.schedules):
                    if scheduled.status is not CascadeScheduleStatus.SCHEDULED or scheduled.effective_round_id != round_id:
                        continue
                    target = next((s for s in round_snapshot if s.agent_id == scheduled.target_agent_id), None)
                    if target is None:
                        raise CascadeRealAgentRunError("directive_target_unknown")
                    view = ControllerPublicView.from_snapshot(target, scenario.graph)
                    application, applied = application.apply_schedule(
                        scheduled.schedule_id, view=view, snapshot_hash=scheduled.input_snapshot_sha256, current_round_id=round_id,
                    )
                    if applied.status is CascadeApplicationStatus.REJECTED:
                        raise CascadeRealAgentRunError("scheduled_directive_rejected")
                contexts_for_round = self.controlled_runner.build_next_round_context(scenario, tuple(events), round_id, application)
                contexts.extend(contexts_for_round)
                directives = {c.snapshot.agent_id: (c.directives[0] if c.directives else None) for c in contexts_for_round}
                round_responses: dict[str, CascadeAgentResponse] = {}
                round_claim_ids: dict[str, str | None] = {}
                for snapshot in round_snapshot:
                    view = ControllerPublicView.from_snapshot(snapshot, scenario.graph)
                    runtime_config = CascadeAgentRuntimeConfig(model_name=model_name, temperature=agent_temperature, max_tokens=agent_max_tokens, seed=seed)
                    claim_id = (
                        claim_selector(scenario, snapshot, round_id)
                        if claim_selector is not None
                        else (snapshot.visible_claim_ids[0] if snapshot.visible_claim_ids else None)
                    )
                    if claim_id is not None and claim_id not in snapshot.visible_claim_ids:
                        raise CascadeRealAgentRunError("selected_claim_not_visible")
                    round_claim_ids[snapshot.agent_id] = claim_id
                    if claim_id is not None:
                        context = selected_context_builder(view, claim_id=claim_id, directive=directives.get(snapshot.agent_id), runtime_config=runtime_config)
                    else:
                        context = CascadeAgentPromptContext(scenario_id=scenario.scenario_id, agent_id=snapshot.agent_id, round_id=round_id, claim_id=None, runtime_config=runtime_config)
                    renderer = getattr(selected_runtime, "render_request", render_cascade_agent_turn)
                    rendered_request = renderer(context)
                    request = rendered_request.model_copy(update={"metadata": {**rendered_request.metadata, "protocol": condition_id.value, "condition": condition_id.value, "phase": "agent", "run_id": run_id, "seed": seed}})
                    fingerprint = str(request_fingerprint_facts(request)["fingerprint"])
                    cached = checkpoint_by_fingerprint.get(fingerprint) if resume else None
                    if cached is not None:
                        response = CascadeAgentResponse(
                            stance=cached.stance, content_ids_used=cached.content_ids_used,
                            evidence_ids_used=cached.evidence_ids_used, share_content_id=cached.share_content_id,
                        )
                    else:
                        result = selected_runtime.execute(
                            context, ledgered,
                            request_metadata={"protocol": condition_id.value, "condition": condition_id.value, "phase": "agent", "run_id": run_id, "seed": seed},
                        )
                        if result.status is not CascadeAgentRuntimeStatus.COMPLETED or result.response is None:
                            raise CascadeRealAgentRunError(result.audit.error_code or "agent_runtime_failed")
                        response = result.response
                        if checkpoint_file is not None:
                            entry = CascadeAgentCheckpointEntry(
                                fingerprint=fingerprint, request_id=result.audit.request_id or request.request_id,
                                agent_id=snapshot.agent_id, round_id=round_id, stance=response.stance,
                                content_ids_used=response.content_ids_used, evidence_ids_used=response.evidence_ids_used,
                                share_content_id=response.share_content_id,
                            )
                            checkpoint = CascadeAgentCheckpoint(binding_sha256=binding, entries=(*checkpoint.entries, entry))
                            checkpoint_by_fingerprint[fingerprint] = entry
                            _write_checkpoint(checkpoint_file, checkpoint)
                    claim_id = round_claim_ids[snapshot.agent_id]
                    if claim_id is not None:
                        outcomes.append(PublicClaimOutcome(
                            scenario_id=scenario.scenario_id, agent_id=snapshot.agent_id, claim_id=claim_id,
                            round_id=round_id, stance=response.stance,
                            content_ids=response.content_ids_used, evidence_card_ids=response.evidence_ids_used,
                            source=OutcomeSource.ACTOR_OBSERVATION,
                        ))
                    else:
                        # No public claim was exposed; this is a safe no-position coordinate.
                        for claim in scenario.graph.claims:
                            outcomes.append(PublicClaimOutcome(
                                scenario_id=scenario.scenario_id, agent_id=snapshot.agent_id, claim_id=claim.claim_id,
                                round_id=round_id, stance=ClaimStance.NO_POSITION, source=OutcomeSource.ACTOR_OBSERVATION,
                            ))
                    round_responses[snapshot.agent_id] = response
                self.base_runner._append_initial_content(scenario, round_id, events)
                if post_round_events_builder is not None:
                    events.extend(post_round_events_builder(scenario, round_id, tuple(events)))
                scheduled_entries = (
                    actor_schedule_provider(scenario, round_id)
                    if actor_schedule_provider is not None
                    else tuple(item for item in scenario.actor_schedule if item.round_id == round_id)
                )
                for index, entry in enumerate(scheduled_entries):
                    if entry.action is ActorAction.ABSTAIN:
                        continue
                    response = round_responses.get(entry.actor_id)
                    node = next(node for node in scenario.graph.nodes if node.node_id == entry.provenance_node_id)
                    required_share = entry.parent_content_id or node.content_id
                    # The model may only enable a pre-declared propagation slot
                    # by sharing its already visible parent/content. It cannot
                    # choose a new node, parent, source root, or evidence.
                    if response is None or response.share_content_id != required_share:
                        continue
                    for target_index, target_agent_id in enumerate(entry.target_agent_ids):
                        events.append(self.base_runner.append_public_event(
                            scenario=scenario, events=tuple(events), snapshots=tuple(c.snapshot for c in contexts_for_round),
                            entry=entry, target_agent_id=target_agent_id,
                            event_id=f"{run_id}-r{round_id}-e{index}-t{target_index}", round_snapshot=round_snapshot,
                        ))
                if round_id < scenario.max_rounds - 1 and condition_id is not CascadeCondition.NO_INTERVENTION:
                    for snapshot in round_snapshot:
                        if not snapshot.visible_claim_ids:
                            continue
                        view = ControllerPublicView.from_snapshot(snapshot, scenario.graph)
                        try:
                            proposal = (
                                proposal_builder(condition_id, view, config)
                                if proposal_builder is not None
                                else propose(condition_id, view, config)
                            )
                        except CascadeControllerValidationError as exc:
                            raise CascadeRealAgentRunError("controller_view_invalid") from exc
                        proposal_count += 1
                        application, scheduled_result = application.schedule_proposal(
                            proposal, view, snapshot_hash=snapshot_sha256(snapshot), config=config,
                            proposal_validator=proposal_validator,
                        )
                        if scheduled_result.status not in {CascadeApplicationStatus.NO_EFFECT, CascadeApplicationStatus.SCHEDULED}:
                            raise CascadeRealAgentRunError("proposal_application_rejected")
        except RequestLedgerError as exc:
            # Ledger messages are stable, content-free control codes. Keep the
            # public error semantic without forwarding an exception payload.
            allowed = {
                "completed_request_fingerprint_exists",
                "failed_request_requires_resume",
                "incomplete_request_requires_resume",
                "request_cap_reached",
                "completion_reservation_cap_reached",
                "parser_recovery_fingerprint_changed",
                "parser_recovery_attempt_limit_reached",
            }
            code = exc.message if exc.message in allowed else exc.code.value
            raise CascadeRealAgentRunError(code) from exc
        exposure = ExposureLedger(scenario_id=scenario.scenario_id, agent_ids=scenario.agent_ids, events=tuple(events), snapshots=tuple(c.snapshot for c in contexts))
        outcome = CascadeOutcomeLedger(scenario_id=scenario.scenario_id, agent_ids=scenario.agent_ids, outcomes=tuple(outcomes))
        try:
            replay = selected_outcome_replay_validator.validate(scenario.graph, exposure, outcome, application, tuple(contexts))
        except Exception as exc:
            raise CascadeRealAgentRunError("cascade_outcome_replay_failed") from exc
        summary = ledger.summary(request_cap=request_cap, completion_reservation_cap=completion_reservation_cap)
        return CascadeRealAgentRunRecord(
            run_id=run_id, scenario_id=scenario.scenario_id, seed=seed, condition=condition_id, agent_ids=scenario.agent_ids,
            round_count=scenario.max_rounds, logical_request_count=summary.unique_logical_request_count,
            provider_call_count=summary.provider_call_count, retry_count=max(0, summary.transport_attempt_count - summary.provider_call_count),
            content_event_count=len(events), exposure_event_count=len(events), snapshot_count=len(contexts), outcome_count=len(outcome.outcomes),
            proposal_count=proposal_count, directive_applied_count=sum(i.status is CascadeScheduleStatus.APPLIED for i in application.schedules),
            exposure_ledger_sha256=_sha(exposure.model_dump(mode="json")), application_ledger_sha256=_sha(application.model_dump(mode="json")),
            outcome_ledger_sha256=outcome.sha256(), replay=replay, request_ledger_path=str(ledger_path), exposure_ledger=exposure,
            application_ledger=application, outcome_ledger=outcome, round_contexts=tuple(contexts),
        )


__all__ = ["CascadeAgentCheckpoint", "CascadeAgentCheckpointEntry", "CascadeRealAgentRunError", "CascadeRealAgentRunRecord", "CascadeRealAgentRunner"]
