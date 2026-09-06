"""Focused tests for user-facing configuration-loader errors."""

from __future__ import annotations

from pathlib import Path

import pytest

from evicon.config import ConfigError, load_run_config


def test_toml_config_rejects_non_positive_max_rounds(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_max_rounds.toml"
    config_path.write_text(
        """run_id = "bad"
scenario_id = "scenario"
model_name = "fake-llm"
protocol = "independent"
agent_count = 2
max_rounds = 0
seed = 1
intervention_budget = 0.0
output_dir = "results"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="max_rounds"):
        load_run_config(config_path)
