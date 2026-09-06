"""CLI for a deterministic, offline-only value-probe smoke run."""

from __future__ import annotations

import argparse
from pathlib import Path

from .models import ProbeRunConfig
from .probe_loader import ProbeLoadError, ProbeSetLoader
from .probe_results import ProbeResultStoreError, write_probe_results
from .probe_runner import ProbeResponseParseError, ProbeRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated deterministic EviCon-Lab value probes.")
    parser.add_argument("--probe-config", required=True, help="Path to a TOML or JSON ProbeSet file.")
    parser.add_argument("--run-id", required=True, help="New result directory identifier.")
    parser.add_argument("--round-id", type=int, default=0, help="Non-negative probe round identifier.")
    parser.add_argument("--seed", type=int, default=2026, help="Deterministic FakeLLM seed.")
    parser.add_argument("--holdout", action="store_true", help="Run holdout items instead of standard items.")
    parser.add_argument("--agent-id", action="append", dest="agent_ids", help="Agent ID; repeat as needed.")
    parser.add_argument("--output-dir", default="results", help="Parent directory for isolated result files.")
    args = parser.parse_args()

    try:
        probe_set = ProbeSetLoader.load(Path(args.probe_config))
        config = ProbeRunConfig(
            probe_set_id=probe_set.probe_set_id,
            round_id=args.round_id,
            agent_ids=args.agent_ids or ["agent-1", "agent-2"],
            model_name="fake-llm",
            seed=args.seed,
            is_holdout=args.holdout,
        )
        results = ProbeRunner(probe_set, config).run()
        result_path = write_probe_results(args.output_dir, args.run_id, results)
    except (
        ProbeLoadError,
        ProbeResponseParseError,
        ProbeResultStoreError,
        ValueError,
        OSError,
    ) as exc:
        parser.error(str(exc))

    print(f"probe_set_id={probe_set.probe_set_id}")
    print(f"agent_count={len(config.agent_ids)}")
    print(f"round_id={config.round_id}")
    print(f"is_holdout={config.is_holdout}")
    print(f"result_path={result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
