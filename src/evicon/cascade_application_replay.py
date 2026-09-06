"""Sidecar replay validation for proposal schedules and next-round directives."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from .cascade_intervention_application import (
    CascadeApplicationLedger,
    CascadeDirectiveType,
    CascadeScheduleStatus,
    ControlledRoundContext,
)


class CascadeApplicationReplayStatus(str, Enum):
    PASSED = "passed"
    BLOCKED = "blocked"


class CascadeApplicationReplayReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CascadeApplicationReplayStatus
    error_code: str | None = None
    schedule_count: int = Field(ge=0)
    applied_count: int = Field(ge=0)
    audit_event_count: int = Field(ge=0)
    directive_count: int = Field(ge=0)


class CascadeApplicationReplayError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class CascadeApplicationReplayValidator:
    """Validate application transitions without loading private truth."""

    @staticmethod
    def validate(
        ledger: CascadeApplicationLedger,
        contexts: tuple[ControlledRoundContext, ...] = (),
    ) -> CascadeApplicationReplayReport:
        try:
            proposal_ids = {item.proposal_id for item in ledger.schedules}
            if any(event.schedule_id not in {item.schedule_id for item in ledger.schedules} for event in ledger.audit_events):
                raise CascadeApplicationReplayError("orphan_audit_event")
            for item in ledger.schedules:
                if item.action.value == "abstain":
                    raise CascadeApplicationReplayError("abstain_scheduled")
                if item.effective_round_id != item.created_round_id + 1:
                    raise CascadeApplicationReplayError("effective_round_invalid")
                events = [event for event in ledger.audit_events if event.schedule_id == item.schedule_id]
                if not events:
                    raise CascadeApplicationReplayError("schedule_audit_missing")
                created = [event for event in events if event.event_type == "schedule_created"]
                applied = [event for event in events if event.event_type == "schedule_applied"]
                rejected = [event for event in events if event.event_type == "schedule_rejected"]
                cancelled = [event for event in events if event.event_type == "schedule_cancelled"]
                if item.status is CascadeScheduleStatus.SCHEDULED and not created:
                    raise CascadeApplicationReplayError("schedule_created_missing")
                if len(created) != 1:
                    raise CascadeApplicationReplayError("duplicate_schedule_created")
                if item.status is CascadeScheduleStatus.APPLIED and (not created or len(applied) != 1):
                    raise CascadeApplicationReplayError("schedule_applied_pairing_invalid")
                if item.status is CascadeScheduleStatus.REJECTED and not rejected:
                    raise CascadeApplicationReplayError("schedule_rejected_pairing_invalid")
                if item.status is CascadeScheduleStatus.CANCELLED and not cancelled:
                    raise CascadeApplicationReplayError("schedule_cancelled_pairing_invalid")
                if len(applied) > 1:
                    raise CascadeApplicationReplayError("duplicate_application")
                if applied and events.index(created[0]) > events.index(applied[0]):
                    raise CascadeApplicationReplayError("application_before_schedule")
                for event in events:
                    if event.proposal_id not in proposal_ids or event.scenario_id != item.scenario_id:
                        raise CascadeApplicationReplayError("audit_coordinate_invalid")
                    if event.created_round_id + 1 != event.effective_round_id:
                        raise CascadeApplicationReplayError("audit_round_invalid")
                    if event.action is not item.action or event.target_agent_id != item.target_agent_id or event.claim_id != item.claim_id:
                        raise CascadeApplicationReplayError("audit_action_coordinate_invalid")
                    if event.input_snapshot_sha256 != item.input_snapshot_sha256 or event.input_view_sha256 != item.input_view_sha256:
                        raise CascadeApplicationReplayError("audit_input_hash_mismatch")
                if applied and any(event.output_transformation_sha256 is None for event in applied):
                    raise CascadeApplicationReplayError("applied_transformation_hash_missing")
            for context in contexts:
                for directive in context.directives:
                    matching = [item for item in ledger.schedules if item.schedule_id == directive.directive_id.removesuffix("-directive")]
                    if len(matching) != 1:
                        raise CascadeApplicationReplayError("directive_schedule_missing")
                    schedule = matching[0]
                    if schedule.status is not CascadeScheduleStatus.APPLIED:
                        raise CascadeApplicationReplayError("directive_from_unapplied_schedule")
                    if directive.effective_round_id != context.snapshot.round_id:
                        raise CascadeApplicationReplayError("directive_round_mismatch")
                    if directive.scenario_id != context.snapshot.scenario_id or directive.target_agent_id != context.snapshot.agent_id or directive.claim_id != schedule.claim_id:
                        raise CascadeApplicationReplayError("directive_target_mismatch")
                    if not set(directive.visible_content_ids).issubset(schedule.visible_content_ids):
                        raise CascadeApplicationReplayError("directive_content_expanded")
                    if not set(directive.visible_evidence_ids).issubset(schedule.visible_evidence_ids):
                        raise CascadeApplicationReplayError("directive_evidence_expanded")
                    if directive.directive_type is CascadeDirectiveType.PRIORITY_EVIDENCE and not directive.visible_evidence_ids:
                        raise CascadeApplicationReplayError("priority_evidence_not_authorized")
            return CascadeApplicationReplayReport(
                status=CascadeApplicationReplayStatus.PASSED,
                schedule_count=len(ledger.schedules),
                applied_count=sum(item.status is CascadeScheduleStatus.APPLIED for item in ledger.schedules),
                audit_event_count=len(ledger.audit_events),
                directive_count=sum(len(context.directives) for context in contexts),
            )
        except CascadeApplicationReplayError:
            raise
        except Exception as exc:
            raise CascadeApplicationReplayError("application_replay_invalid") from exc


__all__ = ["CascadeApplicationReplayError", "CascadeApplicationReplayReport", "CascadeApplicationReplayStatus", "CascadeApplicationReplayValidator"]
