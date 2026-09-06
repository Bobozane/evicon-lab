"""Replay checks for the opt-in provenance-cascade outcome runner.

This module is deliberately a sidecar.  It does not alter the historical
``CascadeReplayValidator`` or the application replay contract.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from .cascade_application_replay import CascadeApplicationReplayReport, CascadeApplicationReplayValidator
from .cascade_intervention_application import CascadeApplicationLedger, ControlledRoundContext
from .cascade_outcomes import CascadeOutcomeLedger
from .cascade_replay import CascadeReplayReport, CascadeReplayValidator
from .provenance_cascade import ProvenanceGraph
from .provenance_cascade_exposure import ExposureLedger


class CascadeOutcomeReplayStatus(str, Enum):
    PASSED = "passed"
    BLOCKED = "blocked"


class CascadeOutcomeReplayReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CascadeOutcomeReplayStatus
    cascade_replay: CascadeReplayReport
    application_replay: CascadeApplicationReplayReport
    outcome_count: int = Field(ge=0)
    error_code: str | None = None


class CascadeOutcomeReplayError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class CascadeOutcomeReplayValidator:
    """Validate public exposure, application sidecars, and outcomes together."""

    @staticmethod
    def validate(
        graph: ProvenanceGraph,
        exposure: ExposureLedger,
        outcomes: CascadeOutcomeLedger,
        application: CascadeApplicationLedger | None = None,
        contexts: tuple[ControlledRoundContext, ...] = (),
    ) -> CascadeOutcomeReplayReport:
        try:
            cascade_report = CascadeReplayValidator.validate(graph, exposure)
            outcomes.validate_against_public(graph, exposure, exact_snapshot=True)
            app_ledger = application or CascadeApplicationLedger()
            app_report = CascadeApplicationReplayValidator.validate(app_ledger, contexts)
            if outcomes.scenario_id != graph.scenario_id:
                raise CascadeOutcomeReplayError("outcome_scenario_mismatch")
            return CascadeOutcomeReplayReport(
                status=CascadeOutcomeReplayStatus.PASSED,
                cascade_replay=cascade_report,
                application_replay=app_report,
                outcome_count=len(outcomes.outcomes),
            )
        except CascadeOutcomeReplayError:
            raise
        except Exception as exc:
            # Do not surface parser/model text or public message content.
            raise CascadeOutcomeReplayError("cascade_outcome_replay_invalid") from exc


__all__ = [
    "CascadeOutcomeReplayError",
    "CascadeOutcomeReplayReport",
    "CascadeOutcomeReplayStatus",
    "CascadeOutcomeReplayValidator",
]
