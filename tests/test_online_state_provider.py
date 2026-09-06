"""Public-only deterministic online-state provider coverage."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from pydantic import ValidationError

from adaptive_cases import config, scenario, stable_signal

from evicon.adaptive_control import FakeOnlineStateProvider
from evicon.models import DialogueState


def _state() -> DialogueState:
    value = scenario()
    return DialogueState(
        run_id="online-run",
        scenario_id=value.scenario_id,
        current_round=0,
        agents=value.agents,
        turns=[],
        evidence_cards=value.evidence_cards,
        value_profiles=[],
        intervention_budget=0.5,
        metadata={},
    )


def test_fake_online_state_provider_is_deterministic_and_does_not_modify_state(tmp_path: Path) -> None:
    state = _state()
    run_config = config(tmp_path, "online-run")
    provider = FakeOnlineStateProvider({0: stable_signal()})
    before = copy.deepcopy(state.model_dump())

    first = provider.observe(state, scenario(), run_config, 0)
    second = provider.observe(state, scenario(), run_config, 0)

    assert first == second
    assert first.remaining_budget == run_config.intervention_budget
    assert state.model_dump() == before


def test_hidden_probe_or_profile_fields_are_rejected() -> None:
    provider = FakeOnlineStateProvider({0: {**stable_signal(), "hidden_probe_response": "private"}})
    with pytest.raises(ValidationError, match="hidden_probe_response"):
        provider.observe(_state(), scenario(), config(Path("results"), "online-run"), 0)
