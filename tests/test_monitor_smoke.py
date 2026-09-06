"""Contract checks for monitor serialization and the local smoke CLI."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from evicon.monitor import MonitorConfig
from evicon.monitor_smoke import main, smoke_config


def test_monitor_config_json_round_trips() -> None:
    configured = smoke_config()

    assert MonitorConfig.model_validate_json(configured.model_dump_json()) == configured


def test_monitor_config_rejects_unknown_fields() -> None:
    values = smoke_config().model_dump()
    values["unvalidated_threshold"] = 0.5

    with pytest.raises(ValidationError, match="unvalidated_threshold"):
        MonitorConfig.model_validate(values)


def test_monitor_smoke_prints_three_limited_structured_cases(capsys: pytest.CaptureFixture[str]) -> None:
    assert main() == 0
    payloads = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert [payload["case"] for payload in payloads] == [
        "stable",
        "social-collapse-without-evidence",
        "evidence-supported-update",
    ]
    assert all(
        set(payload) == {"case", "risk_score", "risk_level", "should_intervene", "reasons"}
        for payload in payloads
    )
