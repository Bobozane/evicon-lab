"""Auditable, next-round-only application of cascade controller proposals.

Stage 24E.1 intentionally does not alter ``CascadeProtocolRunner`` or its
public exposure ledger.  A proposal is bound to one current public view and a
caller-supplied snapshot hash.  The application layer returns immutable
schedules and structured next-round directives; a future runner may consume a
validated directive sidecar without treating it as evidence or a new source.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .cascade_controller import (
    CascadeAction,
    CascadeControllerPolicyConfig,
    CascadeControllerValidationError,
    CascadeInterventionProposal,
    CascadeProposalError,
    propose,
    validate_cascade_proposal,
)
from .cascade_protocol import CascadeScenarioSpec
from .provenance_cascade_exposure import CascadeExposureSnapshot, ControllerPublicView
from .provenance_cascade_preregistration import CascadeCondition
from .models._validation import identifier_list, normalized_text


class CascadeScheduleStatus(str, Enum):
    SCHEDULED = "scheduled"
    APPLIED = "applied"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class CascadeApplicationStatus(str, Enum):
    NO_EFFECT = "no_effect"
    SCHEDULED = "scheduled"
    APPLIED = "applied"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class CascadeDirectiveType(str, Enum):
    VERIFICATION_REQUEST = "verification_request"
    REASONING_REQUEST = "reasoning_request"
    PRIORITY_EVIDENCE = "priority_evidence"


class CascadeApplicationReasonCode(str, Enum):
    SCHEDULED_FOR_NEXT_ROUND = "scheduled_for_next_round"
    APPLIED_NEXT_ROUND = "applied_next_round"
    ABSTAIN_NO_EFFECT = "abstain_no_effect"
    DUPLICATE_PROPOSAL = "duplicate_proposal"
    SCHEDULE_CONFLICT = "schedule_conflict"
    SNAPSHOT_HASH_MISMATCH = "snapshot_hash_mismatch"
    VIEW_HASH_MISMATCH = "view_hash_mismatch"
    EFFECTIVE_ROUND_INVALID = "effective_round_invalid"
    PROPOSAL_INVALID = "proposal_invalid"
    CONTENT_NOT_VISIBLE = "content_not_visible"
    EVIDENCE_NOT_VISIBLE = "evidence_not_visible"
    PRIORITY_EVIDENCE_NOT_AUTHORIZED = "priority_evidence_not_authorized"
    CURRENT_ROUND_ALREADY_PASSED = "current_round_already_passed"
    DUPLICATE_APPLICATION = "duplicate_application"
    DIRECTIVE_PRIVATE_FIELD = "directive_private_field"
    APPLICATION_INPUT_INVALID = "application_input_invalid"


class CascadeApplicationError(ValueError):
    """Stable, non-sensitive application error."""

    def __init__(self, code: CascadeApplicationReasonCode | str) -> None:
        self.code = code.value if isinstance(code, CascadeApplicationReasonCode) else str(code)
        super().__init__(self.code)


class StructuredCascadeDirective(BaseModel):
    """A public, non-linguistic instruction sidecar for the next round."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    directive_id: str = Field(min_length=1)
    directive_type: CascadeDirectiveType
    scenario_id: str = Field(min_length=1)
    target_agent_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    effective_round_id: int = Field(ge=0)
    visible_content_ids: tuple[str, ...] = ()
    visible_evidence_ids: tuple[str, ...] = ()
    visible_same_root_repetition_count: int = Field(default=0, ge=0)
    visible_root_count: int = Field(default=0, ge=0)
    reason_codes: tuple[str, ...] = ()
    status: Literal["structured_only"] = "structured_only"

    @field_validator("directive_id", "scenario_id", "target_agent_id", "claim_id")
    @classmethod
    def clean_ids(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "id"))

    @field_validator("visible_content_ids", "visible_evidence_ids")
    @classmethod
    def unique_ids(cls, value: tuple[str, ...], info: object) -> tuple[str, ...]:
        return tuple(identifier_list(list(value), getattr(info, "field_name", "ids")))

    @field_validator("reason_codes")
    @classmethod
    def stable_reasons(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(identifier_list(list(value), "reason_codes"))


class ScheduledCascadeIntervention(BaseModel):
    """Immutable schedule bound to one verified current public view."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schedule_id: str = Field(min_length=1)
    proposal_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    target_agent_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    created_round_id: int = Field(ge=0)
    effective_round_id: int = Field(ge=0)
    action: CascadeAction
    visible_content_ids: tuple[str, ...] = ()
    visible_evidence_ids: tuple[str, ...] = ()
    visible_source_root_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    input_snapshot_sha256: str = Field(min_length=1)
    input_view_sha256: str = Field(min_length=64, max_length=64)
    status: CascadeScheduleStatus = CascadeScheduleStatus.SCHEDULED
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @field_validator("schedule_id", "proposal_id", "scenario_id", "target_agent_id", "claim_id")
    @classmethod
    def clean_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "id"))

    @field_validator("input_snapshot_sha256", "input_view_sha256")
    @classmethod
    def schedule_digest(cls, value: str, info: object) -> str:
        value = value.lower().strip()
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"{getattr(info, 'field_name', 'hash')} must be a SHA-256 digest")
        return value

    @field_validator("visible_content_ids", "visible_evidence_ids", "visible_source_root_ids")
    @classmethod
    def IDs(cls, value: tuple[str, ...], info: object) -> tuple[str, ...]:
        return tuple(identifier_list(list(value), getattr(info, "field_name", "ids")))

    @field_validator("reason_codes", "warnings", "errors")
    @classmethod
    def codes(cls, value: tuple[str, ...], info: object) -> tuple[str, ...]:
        return tuple(identifier_list(list(value), getattr(info, "field_name", "codes")))

    @model_validator(mode="after")
    def schedule_shape(self) -> "ScheduledCascadeIntervention":
        if self.action is CascadeAction.ABSTAIN:
            raise ValueError("abstain cannot be scheduled")
        if self.effective_round_id != self.created_round_id + 1:
            raise ValueError("effective round must be the next round")
        if self.status is CascadeScheduleStatus.SCHEDULED and self.errors:
            raise ValueError("scheduled item cannot contain errors")
        return self


class CascadeInterventionAuditEvent(BaseModel):
    """Safe state transition record; no prompt or model content is stored."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1)
    event_type: Literal["schedule_created", "schedule_applied", "schedule_rejected", "schedule_cancelled"]
    schedule_id: str = Field(min_length=1)
    proposal_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    target_agent_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    action: CascadeAction
    created_round_id: int = Field(ge=0)
    effective_round_id: int = Field(ge=0)
    reason_codes: tuple[str, ...] = ()
    input_snapshot_sha256: str = Field(min_length=1)
    input_view_sha256: str = Field(min_length=64, max_length=64)
    output_transformation_sha256: str | None = None
    status: CascadeScheduleStatus

    @field_validator("event_id", "schedule_id", "proposal_id", "scenario_id", "target_agent_id", "claim_id")
    @classmethod
    def safe_ids(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "id"))

    @field_validator("input_snapshot_sha256", "input_view_sha256")
    @classmethod
    def audit_digests(cls, value: str, info: object) -> str:
        value = value.lower().strip()
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"{getattr(info, 'field_name', 'hash')} must be a SHA-256 digest")
        return value

    @field_validator("output_transformation_sha256")
    @classmethod
    def output_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.lower().strip()
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("output_transformation_sha256 must be a SHA-256 digest")
        return value


class CascadeApplicationResult(BaseModel):
    """Result of one schedule or apply attempt; it carries no mutable ledger."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CascadeApplicationStatus
    schedule: ScheduledCascadeIntervention | None = None
    directive: StructuredCascadeDirective | None = None
    audit_events: tuple[CascadeInterventionAuditEvent, ...] = ()
    scheduled_count: int = Field(default=0, ge=0)
    applied_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    cancelled_count: int = Field(default=0, ge=0)
    error_code: str | None = None

    @property
    def exposure_transformation(self) -> StructuredCascadeDirective | None:
        """Name used by offline evaluators for the safe next-round sidecar."""
        return self.directive

    @model_validator(mode="after")
    def result_shape(self) -> "CascadeApplicationResult":
        if self.status is CascadeApplicationStatus.NO_EFFECT:
            if self.schedule is not None or self.directive is not None or self.audit_events:
                raise ValueError("no_effect must not create schedule, directive, or audit event")
        if self.status is CascadeApplicationStatus.APPLIED and self.directive is None:
            raise ValueError("applied result requires directive")
        return self


class CascadeApplicationLedger(BaseModel):
    """Append-only application state; every operation returns a new ledger."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schedules: tuple[ScheduledCascadeIntervention, ...] = ()
    audit_events: tuple[CascadeInterventionAuditEvent, ...] = ()

    @model_validator(mode="after")
    def unique_coordinates(self) -> "CascadeApplicationLedger":
        ids = [item.schedule_id for item in self.schedules]
        proposal_ids = [item.proposal_id for item in self.schedules]
        if len(ids) != len(set(ids)) or len(proposal_ids) != len(set(proposal_ids)):
            raise ValueError("application ledger has duplicate schedule or proposal")
        event_ids = [event.event_id for event in self.audit_events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("application ledger has duplicate audit event")
        return self

    def schedule_proposal(
        self,
        proposal: CascadeInterventionProposal,
        view: ControllerPublicView,
        *,
        snapshot_hash: str,
        config: CascadeControllerPolicyConfig,
        proposal_validator: object | None = None,
    ) -> tuple["CascadeApplicationLedger", CascadeApplicationResult]:
        """Bind a valid current-round proposal for exactly one next round."""
        try:
            validator = proposal_validator or validate_cascade_proposal
            validator(proposal, view, config)
            if proposal.valid is not True:
                raise CascadeApplicationError(CascadeApplicationReasonCode.PROPOSAL_INVALID)
            if proposal.action is CascadeAction.ABSTAIN:
                return self, CascadeApplicationResult(
                    status=CascadeApplicationStatus.NO_EFFECT,
                    error_code=CascadeApplicationReasonCode.ABSTAIN_NO_EFFECT.value,
                )
            if proposal.round_id != view.round_id:
                raise CascadeApplicationError(CascadeApplicationReasonCode.CURRENT_ROUND_ALREADY_PASSED)
            source_hash = public_view_sha256(view)
            snapshot_hash = _safe_snapshot_hash(snapshot_hash)
            schedule_id = _schedule_id(proposal, source_hash, snapshot_hash)
            existing = next((item for item in self.schedules if item.proposal_id == proposal.proposal_id), None)
            if existing is not None:
                raise CascadeApplicationError(CascadeApplicationReasonCode.DUPLICATE_PROPOSAL)
            conflict = next((item for item in self.schedules if item.scenario_id == proposal.scenario_id and item.target_agent_id == proposal.target_agent_id and item.claim_id == proposal.claim_id and item.effective_round_id == proposal.round_id + 1 and item.status in {CascadeScheduleStatus.SCHEDULED, CascadeScheduleStatus.APPLIED}), None)
            if conflict is not None:
                raise CascadeApplicationError(CascadeApplicationReasonCode.SCHEDULE_CONFLICT)
            schedule = ScheduledCascadeIntervention(
                schedule_id=schedule_id,
                proposal_id=proposal.proposal_id,
                scenario_id=proposal.scenario_id,
                target_agent_id=proposal.target_agent_id,
                claim_id=proposal.claim_id,
                created_round_id=proposal.round_id,
                effective_round_id=proposal.round_id + 1,
                action=proposal.action,
                visible_content_ids=proposal.used_content_ids,
                visible_evidence_ids=proposal.used_evidence_ids,
                visible_source_root_ids=proposal.used_source_root_ids,
                reason_codes=tuple(code.value for code in proposal.reason_codes),
                input_snapshot_sha256=snapshot_hash,
                input_view_sha256=source_hash,
            )
            event = _audit(schedule, "schedule_created", CascadeScheduleStatus.SCHEDULED)
            ledger = self.model_copy(update={"schedules": self.schedules + (schedule,), "audit_events": self.audit_events + (event,)})
            return ledger, CascadeApplicationResult(status=CascadeApplicationStatus.SCHEDULED, schedule=schedule, audit_events=(event,), scheduled_count=1)
        except CascadeControllerValidationError as exc:
            return self._append_rejection(proposal, view, snapshot_hash, exc.code)
        except CascadeApplicationError as exc:
            return self._append_rejection(proposal, view, snapshot_hash, exc.code)
        except (ValueError, TypeError):
            return self._append_rejection(proposal, view, snapshot_hash, CascadeApplicationReasonCode.APPLICATION_INPUT_INVALID.value)

    def _append_rejection(self, proposal: CascadeInterventionProposal, view: ControllerPublicView, snapshot_hash: str, code: str) -> tuple["CascadeApplicationLedger", CascadeApplicationResult]:
        result = self._rejected_result(proposal, view, snapshot_hash, code)
        if not result.audit_events:
            return self, result
        event = result.audit_events[0]
        schedules = self.schedules
        if result.schedule is not None and result.schedule.proposal_id not in {item.proposal_id for item in schedules}:
            schedules = schedules + (result.schedule,)
        if event.event_id in {item.event_id for item in self.audit_events}:
            return self.model_copy(update={"schedules": schedules}), result
        return self.model_copy(update={"schedules": schedules, "audit_events": self.audit_events + (event,)}), result

    def _rejected_result(self, proposal: CascadeInterventionProposal, view: ControllerPublicView, snapshot_hash: str, code: str) -> CascadeApplicationResult:
        try:
            safe_snapshot = _safe_snapshot_hash(snapshot_hash)
        except CascadeApplicationError:
            safe_snapshot = hashlib.sha256(b"invalid_snapshot_hash").hexdigest()
        source_hash = public_view_sha256(view)
        schedule_id = _schedule_id(proposal, source_hash, safe_snapshot)
        rejected_schedule = ScheduledCascadeIntervention(
            schedule_id=schedule_id,
            proposal_id=proposal.proposal_id,
            scenario_id=proposal.scenario_id,
            target_agent_id=proposal.target_agent_id,
            claim_id=proposal.claim_id,
            created_round_id=proposal.round_id,
            effective_round_id=proposal.round_id + 1,
            action=proposal.action,
            visible_content_ids=proposal.used_content_ids,
            visible_evidence_ids=proposal.used_evidence_ids,
            visible_source_root_ids=proposal.used_source_root_ids,
            reason_codes=tuple(code.value for code in proposal.reason_codes),
            input_snapshot_sha256=safe_snapshot,
            input_view_sha256=source_hash,
            status=CascadeScheduleStatus.REJECTED,
            errors=(code,),
        )
        event = _audit(rejected_schedule, "schedule_rejected", CascadeScheduleStatus.REJECTED, reason_codes=(code,))
        return CascadeApplicationResult(status=CascadeApplicationStatus.REJECTED, schedule=rejected_schedule, audit_events=(event,), rejected_count=1, error_code=code)

    def apply_schedule(
        self,
        schedule_id: str,
        *,
        view: ControllerPublicView,
        snapshot_hash: str,
        current_round_id: int,
    ) -> tuple["CascadeApplicationLedger", CascadeApplicationResult]:
        """Apply one schedule only at its declared next round."""
        schedule = next((item for item in self.schedules if item.schedule_id == schedule_id), None)
        if schedule is None:
            return self, CascadeApplicationResult(status=CascadeApplicationStatus.REJECTED, rejected_count=1, error_code="schedule_not_found")
        try:
            if schedule.status is not CascadeScheduleStatus.SCHEDULED:
                raise CascadeApplicationError(CascadeApplicationReasonCode.DUPLICATE_APPLICATION)
            if current_round_id != schedule.effective_round_id:
                raise CascadeApplicationError(CascadeApplicationReasonCode.EFFECTIVE_ROUND_INVALID)
            if view.scenario_id != schedule.scenario_id or view.agent_id != schedule.target_agent_id or view.round_id != current_round_id:
                raise CascadeApplicationError(CascadeApplicationReasonCode.APPLICATION_INPUT_INVALID)
            # The next round necessarily has a different public-view hash.
            # Keep the source view hash in the audit binding, and validate the
            # next-round public coordinates and referenced IDs below.
            if _safe_snapshot_hash(snapshot_hash) != schedule.input_snapshot_sha256:
                raise CascadeApplicationError(CascadeApplicationReasonCode.SNAPSHOT_HASH_MISMATCH)
            visible_content = {node.content_id for node in view.provenance_nodes if node.claim_id == schedule.claim_id}
            visible_evidence = {card.evidence_id for card in view.evidence_cards}
            if not set(schedule.visible_content_ids).issubset(visible_content):
                raise CascadeApplicationError(CascadeApplicationReasonCode.CONTENT_NOT_VISIBLE)
            if not set(schedule.visible_evidence_ids).issubset(visible_evidence):
                raise CascadeApplicationError(CascadeApplicationReasonCode.EVIDENCE_NOT_VISIBLE)
            directive = _directive(schedule)
            if directive.directive_type is CascadeDirectiveType.PRIORITY_EVIDENCE and not schedule.visible_evidence_ids:
                raise CascadeApplicationError(CascadeApplicationReasonCode.PRIORITY_EVIDENCE_NOT_AUTHORIZED)
            transformation_hash = _directive_sha256(directive)
            applied = schedule.model_copy(update={"status": CascadeScheduleStatus.APPLIED})
            event = _audit(applied, "schedule_applied", CascadeScheduleStatus.APPLIED, transformation_hash)
            schedules = tuple(applied if item.schedule_id == schedule_id else item for item in self.schedules)
            ledger = self.model_copy(update={"schedules": schedules, "audit_events": self.audit_events + (event,)})
            return ledger, CascadeApplicationResult(status=CascadeApplicationStatus.APPLIED, schedule=applied, directive=directive, audit_events=(event,), applied_count=1)
        except CascadeApplicationError as exc:
            event = _audit(schedule, "schedule_rejected", CascadeScheduleStatus.REJECTED, reason_codes=(exc.code,))
            ledger = self.model_copy(update={"audit_events": self.audit_events + (event,)})
            return ledger, CascadeApplicationResult(status=CascadeApplicationStatus.REJECTED, schedule=schedule, audit_events=(event,), rejected_count=1, error_code=exc.code)

    def cancel_schedule(self, schedule_id: str, *, reason_code: str = "schedule_cancelled") -> tuple["CascadeApplicationLedger", CascadeApplicationResult]:
        schedule = next((item for item in self.schedules if item.schedule_id == schedule_id), None)
        if schedule is None or schedule.status is not CascadeScheduleStatus.SCHEDULED:
            return self, CascadeApplicationResult(status=CascadeApplicationStatus.REJECTED, rejected_count=1, error_code="schedule_not_cancellable")
        cancelled = schedule.model_copy(update={"status": CascadeScheduleStatus.CANCELLED, "warnings": (reason_code,)})
        event = _audit(cancelled, "schedule_cancelled", CascadeScheduleStatus.CANCELLED, reason_codes=(reason_code,))
        return self.model_copy(update={"schedules": tuple(cancelled if item.schedule_id == schedule_id else item for item in self.schedules), "audit_events": self.audit_events + (event,)}), CascadeApplicationResult(status=CascadeApplicationStatus.CANCELLED, schedule=cancelled, audit_events=(event,), cancelled_count=1)


class ControlledRoundContext(BaseModel):
    """Base snapshot plus next-round sidecar directives; base snapshot is unchanged."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot: CascadeExposureSnapshot
    directives: tuple[StructuredCascadeDirective, ...] = ()
    transformation_sha256: str = Field(min_length=64, max_length=64)

    @field_validator("transformation_sha256")
    @classmethod
    def digest(cls, value: str) -> str:
        value = value.lower().strip()
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("transformation_sha256 must be a SHA-256 digest")
        return value


class ControlledCascadeProtocolRunner:
    """Opt-in adapter that never changes the historical cascade runner."""

    def __init__(self, base_runner: object | None = None) -> None:
        from .cascade_protocol import CascadeProtocolRunner
        self.base_runner = base_runner or CascadeProtocolRunner()

    def build_next_round_context(
        self,
        scenario: CascadeScenarioSpec,
        events: tuple[object, ...],
        round_id: int,
        ledger: CascadeApplicationLedger,
    ) -> tuple[ControlledRoundContext, ...]:
        """Create base snapshots, then attach only applied directives."""
        snapshots = self.base_runner.build_round_snapshot(scenario, events, round_id)
        applied = [item for item in ledger.schedules if item.status is CascadeScheduleStatus.APPLIED and item.effective_round_id == round_id]
        contexts: list[ControlledRoundContext] = []
        for snapshot in snapshots:
            directives = tuple(_directive(item) for item in applied if item.target_agent_id == snapshot.agent_id)
            digest = _sha256_json({"snapshot": snapshot.model_dump(mode="json"), "directives": [item.model_dump(mode="json") for item in directives]})
            contexts.append(ControlledRoundContext(snapshot=snapshot, directives=directives, transformation_sha256=digest))
        return tuple(contexts)


def public_view_sha256(view: ControllerPublicView) -> str:
    return _sha256_json(view.model_dump(mode="json"))


def snapshot_sha256(snapshot: CascadeExposureSnapshot) -> str:
    return _sha256_json(snapshot.model_dump(mode="json"))


def _safe_snapshot_hash(value: str) -> str:
    value = normalized_text(value, "snapshot_hash").lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise CascadeApplicationError(CascadeApplicationReasonCode.SNAPSHOT_HASH_MISMATCH)
    return value


def _schedule_id(proposal: CascadeInterventionProposal, view_hash: str, snapshot_hash: str) -> str:
    return "cascade-schedule-" + _sha256_json({"proposal_id": proposal.proposal_id, "view_hash": view_hash, "snapshot_hash": snapshot_hash, "effective_round": proposal.round_id + 1})[:24]


def _directive(schedule: ScheduledCascadeIntervention) -> StructuredCascadeDirective:
    if schedule.action is CascadeAction.REQUEST_INDEPENDENT_SOURCE:
        kind = CascadeDirectiveType.VERIFICATION_REQUEST
    elif schedule.action is CascadeAction.REQUEST_EVIDENCE_BASED_REASONING:
        kind = CascadeDirectiveType.REASONING_REQUEST
    elif schedule.action is CascadeAction.EVIDENCE_FIRST_EXPOSURE:
        kind = CascadeDirectiveType.PRIORITY_EVIDENCE
    else:
        raise CascadeApplicationError(CascadeApplicationReasonCode.ABSTAIN_NO_EFFECT)
    return StructuredCascadeDirective(
        directive_id=f"{schedule.schedule_id}-directive",
        directive_type=kind,
        scenario_id=schedule.scenario_id,
        target_agent_id=schedule.target_agent_id,
        claim_id=schedule.claim_id,
        effective_round_id=schedule.effective_round_id,
        visible_content_ids=schedule.visible_content_ids,
        visible_evidence_ids=schedule.visible_evidence_ids,
        visible_same_root_repetition_count=len(schedule.visible_content_ids) if kind is CascadeDirectiveType.VERIFICATION_REQUEST else 0,
        visible_root_count=len(schedule.visible_source_root_ids),
        reason_codes=schedule.reason_codes,
    )


def _directive_sha256(directive: StructuredCascadeDirective) -> str:
    return _sha256_json(directive.model_dump(mode="json"))


def _audit(schedule: ScheduledCascadeIntervention, event_type: str, status: CascadeScheduleStatus, transformation_hash: str | None = None, reason_codes: tuple[str, ...] = ()) -> CascadeInterventionAuditEvent:
    return CascadeInterventionAuditEvent(
        event_id=f"{schedule.schedule_id}-{event_type.removeprefix('schedule_')}",
        event_type=event_type,
        schedule_id=schedule.schedule_id,
        proposal_id=schedule.proposal_id,
        scenario_id=schedule.scenario_id,
        target_agent_id=schedule.target_agent_id,
        claim_id=schedule.claim_id,
        action=schedule.action,
        created_round_id=schedule.created_round_id,
        effective_round_id=schedule.effective_round_id,
        reason_codes=reason_codes or schedule.reason_codes,
        input_snapshot_sha256=schedule.input_snapshot_sha256,
        input_view_sha256=schedule.input_view_sha256,
        output_transformation_sha256=transformation_hash,
        status=status,
    )


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# Explicit aliases make the application boundary discoverable without changing
# the older controller/executor modules.
CascadeInterventionApplication = CascadeApplicationLedger

__all__ = [
    "CascadeApplicationError", "CascadeApplicationLedger", "CascadeApplicationReasonCode", "CascadeApplicationResult", "CascadeApplicationStatus",
    "CascadeDirectiveType", "CascadeInterventionApplication", "CascadeInterventionAuditEvent", "CascadeScheduleStatus", "ControlledCascadeProtocolRunner",
    "ControlledRoundContext", "ScheduledCascadeIntervention", "StructuredCascadeDirective", "public_view_sha256", "snapshot_sha256",
]
