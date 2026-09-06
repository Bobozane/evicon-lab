"""Opt-in two-agent real Agent protocol smoke; never a formal experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .agent_runtime import AgentRequestSettings, AgentRuntime
from .models import AgentSpec, ProtocolCondition, RunConfig, ScenarioSpec
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ReasoningEffort
from .real_agent_runner import RealAgentProtocolRunner
from .replay import validate_replay


_SMOKE_MAX_TOKENS = 256


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one explicitly enabled two-agent real protocol smoke.")
    parser.add_argument("--allow-network", action="store_true", help="Allow at most four provider requests.")
    parser.add_argument("--run-id", help="Required unique output identifier when network access is enabled.")
    parser.add_argument("--output-dir", default="results", help="Parent directory for the one smoke run.")
    arguments = parser.parse_args(argv)
    if not arguments.allow_network:
        print("network_disabled")
        return 0
    if not arguments.run_id:
        parser.error("--run-id is required with --allow-network")
    if Path(arguments.run_id).name != arguments.run_id:
        parser.error("--run-id must be a simple directory name")

    config = ProviderConfig.from_env(allow_network=True).model_copy(
        update={
            "max_retries": 0,
            "max_tokens": _SMOKE_MAX_TOKENS,
            "temperature": 0.2,
            "reasoning_effort": ReasoningEffort.NONE,
        }
    )
    model_name = config.model_name or "unconfigured-model"
    scenario = _smoke_scenario()
    run_config = RunConfig(
        run_id=arguments.run_id,
        scenario_id=scenario.scenario_id,
        model_name=model_name,
        protocol=ProtocolCondition.SOCIAL_ONLY,
        agent_count=2,
        max_rounds=2,
        seed=config.seed,
        intervention_budget=0.0,
        output_dir=arguments.output_dir,
    )
    runtime = AgentRuntime(
        AgentRequestSettings(
            model_name=model_name,
            temperature=0.2,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
    )
    runner = RealAgentProtocolRunner(
        run_config,
        scenario=scenario,
        agent_runtime=runtime,
        provider=OpenAICompatibleProvider(config),
    )
    record = runner.run()
    output_directory = runner.output_directory
    replay = validate_replay(
        output_directory / "events.jsonl",
        output_directory / "run_record.json",
    )
    print(
        json.dumps(
            {
                "run_id": record.config.run_id,
                "protocol": record.config.protocol.value,
                "agent_count": record.config.agent_count,
                "rounds": record.config.max_rounds,
                "turn_count": len(record.turns),
                "model": record.config.model_name,
                "total_token_usage": runner.total_token_usage,
                "output_directory": str(output_directory),
                "replay_status": replay.validation,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


def _smoke_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="real-protocol-smoke-scenario",
        title="Two-agent public protocol smoke",
        description="A local fixture for validating a real Agent protocol call boundary.",
        initial_context="A public scenario asks two participants to state concise positions.",
        agents=[
            AgentSpec(
                agent_id="agent-a",
                role="public deliberation participant",
                initial_value_labels=["fairness"],
                metadata={},
            ),
            AgentSpec(
                agent_id="agent-b",
                role="public deliberation participant",
                initial_value_labels=["safety"],
                metadata={},
            ),
        ],
        evidence_cards=[],
        max_rounds=2,
        metadata={},
    )


if __name__ == "__main__":
    raise SystemExit(main())
