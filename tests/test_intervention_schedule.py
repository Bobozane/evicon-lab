"""Strict schedule alignment and serialization coverage."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from controlled_cases import plan

from evicon.intervention_application import InterventionSchedule
from evicon.models import InterventionAction


def test_schedule_json_round_trip_and_round_lookup() -> None:
    scheduled_plan = plan(InterventionAction.NO_OP, round_id=0)
    schedule = InterventionSchedule(
        schedule_id="schedule-json",
        run_id="controlled-run",
        scenario_id="controlled-scenario",
        plans=[scheduled_plan],
        metadata={"fixture": "schedule"},
    )

    assert InterventionSchedule.model_validate_json(schedule.model_dump_json()) == schedule
    assert schedule.plan_for_round(0) == scheduled_plan
    assert schedule.plan_for_round(1) is None


def test_schedule_rejects_duplicate_round_misaligned_or_invalid_plans() -> None:
    first = plan(InterventionAction.NO_OP, round_id=0)
    second = plan(InterventionAction.REQUEST_EVIDENCE, round_id=0)
    with pytest.raises(ValidationError, match="unique round_id"):
        InterventionSchedule(
            schedule_id="duplicate",
            run_id="controlled-run",
            scenario_id="controlled-scenario",
            plans=[first, second],
            metadata={},
        )
    with pytest.raises(ValidationError, match="plan.run_id"):
        InterventionSchedule(
            schedule_id="wrong-run",
            run_id="different-run",
            scenario_id="controlled-scenario",
            plans=[first],
            metadata={},
        )
    with pytest.raises(ValidationError, match="must be valid"):
        InterventionSchedule(
            schedule_id="invalid",
            run_id="controlled-run",
            scenario_id="controlled-scenario",
            plans=[first.model_copy(update={"valid": False, "validation_errors": ["unknown_target_agent"]})],
            metadata={},
        )
