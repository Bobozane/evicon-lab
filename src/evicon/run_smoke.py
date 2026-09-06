"""Command-line entry point for a no-network deterministic protocol run."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ConfigError, load_run_config
from .scenario_loader import ScenarioLoadError, ScenarioLoader
from .runner import ProtocolRunner




def main() -> int:
    parser = argparse.ArgumentParser(description="Run EviCon-Lab's deterministic local smoke protocol.")
    parser.add_argument("--config", required=True, help="Path to a TOML or JSON RunConfig file.")
    args = parser.parse_args()

    try:
        config = load_run_config(Path(args.config))
        if config.scenario_file is None:
            raise ConfigError("scenario_file is required for a smoke run")
        scenario = ScenarioLoader.load(Path(config.scenario_file))
        record = ProtocolRunner(config, scenario=scenario).run()
    except (ConfigError, ScenarioLoadError, FileExistsError, ValueError, OSError) as exc:
        parser.error(str(exc))

    output_directory = Path(config.output_dir) / config.run_id
    print(f"run_id={record.config.run_id}")
    print(f"scenario_id={record.scenario.scenario_id if record.scenario else config.scenario_id}")
    print(f"protocol={record.config.protocol.value}")
    print(f"rounds={record.config.max_rounds}")
    print(f"turns={len(record.turns)}")
    print(f"output_path={output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
