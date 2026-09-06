"""Strict scenario-file loading tests for deterministic local runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from evicon.scenario_loader import ScenarioLoadError, ScenarioLoader


def scenario_text(
    *,
    first_agent: str = "agent-1",
    second_agent: str = "agent-2",
    second_evidence_id: str = "evidence-2",
    restricted_audience: str = "agent-1",
    introduced_round: int = 1,
    include_title: bool = True,
    extra_field: bool = False,
) -> str:
    title = 'title = "Fixture scenario"\n' if include_title else ""
    extra = 'unexpected = true\n' if extra_field else ""
    return f'''scenario_id = "fixture-scenario"
{title}description = "A strict local scenario fixture."
max_rounds = 2
{extra}
[[agents]]
agent_id = "{first_agent}"
role = "participant"

[[agents]]
agent_id = "{second_agent}"
role = "participant"

[[evidence_cards]]
evidence_id = "evidence-1"
claim = "First fixture evidence."
source = "local"
supports = ["claim-1"]
contradicts = []
introduced_round = 0
visible_to = ["*"]
reliability = 0.9

[[evidence_cards]]
evidence_id = "{second_evidence_id}"
claim = "Second fixture evidence."
source = "local"
supports = []
contradicts = ["claim-1"]
introduced_round = {introduced_round}
visible_to = ["{restricted_audience}"]
reliability = 0.8
'''


def test_checked_in_smoke_scenario_loads_and_round_trips() -> None:
    project_root = Path(__file__).resolve().parents[1]
    scenario = ScenarioLoader.load(project_root / "configs/scenarios/smoke_scenario.toml")

    assert scenario.scenario_id == "local-smoke-scenario"
    assert len(scenario.agents) == 2
    assert len(scenario.evidence_cards) == 2
    assert type(scenario).model_validate_json(scenario.model_dump_json()) == scenario


@pytest.mark.parametrize(
    ("name", "contents", "field"),
    [
        ("duplicate_agent.toml", scenario_text(second_agent="agent-1"), "agent_id"),
        ("duplicate_evidence.toml", scenario_text(second_evidence_id="evidence-1"), "evidence_id"),
        ("unknown_audience.toml", scenario_text(restricted_audience="absent-agent"), "visible_to"),
        ("bad_round.toml", scenario_text(introduced_round=-1), "introduced_round"),
        ("missing_title.toml", scenario_text(include_title=False), "title"),
        ("extra_field.toml", scenario_text(extra_field=True), "unexpected"),
    ],
)
def test_invalid_scenarios_are_rejected_with_path_and_field(
    tmp_path: Path,
    name: str,
    contents: str,
    field: str,
) -> None:
    path = tmp_path / name
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ScenarioLoadError) as error:
        ScenarioLoader.load(path)

    assert str(path) in str(error.value)
    assert field in str(error.value)
