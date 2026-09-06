"""Pure action-template rendering coverage."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mediator_cases import plan_for, prompt_context_for

from evicon.mediator_prompts import TEMPLATE_VERSIONS, PromptContext, render_action_instruction
from evicon.models import InterventionAction


@pytest.mark.parametrize(
    "action",
    [
        InterventionAction.REQUEST_EVIDENCE,
        InterventionAction.BLIND_EVIDENCE_REFLECTION,
        InterventionAction.SOLICIT_DISSENT,
        InterventionAction.ADAPTIVE_EXPOSURE,
        InterventionAction.MINORITY_REPORT,
        InterventionAction.RESTRUCTURE,
    ],
)
def test_each_supported_action_has_a_deterministic_distinct_template(action: InterventionAction) -> None:
    plan = plan_for(action)
    context = prompt_context_for(plan)

    first = render_action_instruction(plan, context, seed=11)
    second = render_action_instruction(plan, context, seed=11)

    assert first == second
    assert first is not None
    assert first.metadata["template_version"] == TEMPLATE_VERSIONS[action]
    assert action.value in first.user_prompt
    assert first.metadata["action"] == action.value
    assert first.metadata["target_agent_ids"] == plan.target_agent_ids


def test_no_op_does_not_render_an_llm_request() -> None:
    plan = plan_for(InterventionAction.NO_OP)

    assert render_action_instruction(plan, prompt_context_for(plan)) is None


def test_private_probe_or_profile_content_is_rejected_before_rendering() -> None:
    plan = plan_for(InterventionAction.SOLICIT_DISSENT)
    values = prompt_context_for(plan).model_dump()
    values["metadata"] = {"hidden_probe": "private"}
    with pytest.raises(ValidationError, match="private"):
        PromptContext.model_validate(values)

    values = prompt_context_for(plan).model_dump()
    values["public_dialogue_summary"] = "hidden profile values"
    with pytest.raises(ValidationError, match="private"):
        PromptContext.model_validate(values)


def test_renderer_refuses_context_that_attempts_to_change_fixed_decision() -> None:
    plan = plan_for(InterventionAction.SOLICIT_DISSENT)
    context = prompt_context_for(plan)

    with pytest.raises(ValueError, match="action"):
        render_action_instruction(
            plan,
            context.model_copy(update={"action": InterventionAction.RESTRUCTURE}),
        )
    with pytest.raises(ValueError, match="target_agent_ids"):
        render_action_instruction(
            plan,
            context.model_copy(update={"target_agent_ids": ["agent-b"]}),
        )
