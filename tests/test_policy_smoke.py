"""Local-only behavior checks for the InterventionPolicy smoke command."""

from __future__ import annotations

import json

from evicon.models import InterventionAction
from evicon.policy_smoke import main, smoke_cases, smoke_config


def test_policy_smoke_cases_have_expected_decision_paths() -> None:
    labels = [label for label, _ in smoke_cases()]

    assert labels == [
        "stable",
        "social-collapse-without-evidence",
        "repeated-failure-restructure",
    ]
    assert smoke_config().version == "policy-smoke-v1"


def test_policy_smoke_prints_only_structured_decision_fields(capsys: object) -> None:
    assert main() == 0
    output = capsys.readouterr().out.strip().splitlines()  # type: ignore[attr-defined]
    rows = [json.loads(line) for line in output]

    assert [row["case"] for row in rows] == [
        "stable",
        "social-collapse-without-evidence",
        "repeated-failure-restructure",
    ]
    assert set(rows[0]) == {
        "case",
        "action",
        "target_agent_ids",
        "risk_score",
        "estimated_cost",
        "reason_codes",
        "cooldown_until",
    }
    assert rows[0]["action"] == InterventionAction.NO_OP.value
    assert rows[1]["action"] == InterventionAction.SOLICIT_DISSENT.value
    assert rows[2]["action"] == InterventionAction.RESTRUCTURE.value


def test_policy_smoke_never_creates_files_or_calls_network() -> None:
    # The smoke module only constructs frozen Pydantic inputs and prints JSON.
    assert all(policy_input.metadata["purpose"] == "local smoke" for _, policy_input in smoke_cases())
