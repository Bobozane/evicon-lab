"""One opt-in measured protocol smoke; it is not a formal experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .agent_runtime import AgentRequestSettings, AgentRuntime
from .contextual_probe_runtime import ContextualProbeRequestSettings, ContextualProbeRuntime
from .measured_protocol_experiment import MeasuredProtocolExperimentRunner
from .models import AgentSpec, ProbeSet, ProtocolCondition, RunConfig, ScenarioSpec, ValueProbeItem
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ReasoningEffort
from .replay import validate_replay


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one explicitly enabled measured real-Agent smoke.")
    parser.add_argument("--allow-network", action="store_true", help="Allow at most eight provider requests.")
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
    output_directory = Path(arguments.output_dir) / arguments.run_id
    if output_directory.exists():
        parser.error("--run-id already has an output directory")

    base_config = ProviderConfig.from_env(allow_network=True)
    agent_config = base_config.model_copy(
        update={
            "max_retries": 0,
            "max_tokens": 256,
            "temperature": 0.2,
            "reasoning_effort": ReasoningEffort.NONE,
        }
    )
    probe_config = base_config.model_copy(
        update={
            "max_retries": 0,
            "max_tokens": 128,
            "temperature": 0.0,
            "reasoning_effort": ReasoningEffort.NONE,
        }
    )
    model_name = agent_config.model_name or "unconfigured-model"
    scenario = _smoke_scenario()
    config = RunConfig(
        run_id=arguments.run_id,
        scenario_id=scenario.scenario_id,
        model_name=model_name,
        protocol=ProtocolCondition.SOCIAL_ONLY,
        agent_count=2,
        max_rounds=2,
        seed=agent_config.seed,
        intervention_budget=0.0,
        output_dir=arguments.output_dir,
    )
    experiment = MeasuredProtocolExperimentRunner(
        config,
        scenario=scenario,
        probe_set=_smoke_probe_set(),
        agent_runtime=AgentRuntime(
            AgentRequestSettings(model_name=model_name, temperature=0.2, max_tokens=256, seed=agent_config.seed)
        ),
        contextual_probe_runtime=ContextualProbeRuntime(
            ContextualProbeRequestSettings(
                model_name=probe_config.model_name or model_name,
                temperature=0.0,
                max_tokens=128,
                seed=probe_config.seed,
            )
        ),
        agent_provider=OpenAICompatibleProvider(agent_config),
        probe_provider=OpenAICompatibleProvider(probe_config),
        probe_seed=probe_config.seed,
        pre_target_round=0,
        post_target_round=1,
        execute_holdout_probe=True,
    )
    result = experiment.run()
    replay_status: str | None = None
    if result.run_record is not None and result.run_record.status.value == "completed":
        replay_status = validate_replay(
            output_directory / "events.jsonl",
            output_directory / "run_record.json",
        ).validation
    record = result.measurement_record
    output = {
        "run_id": arguments.run_id,
        "protocol": config.protocol.value,
        "agent_count": config.agent_count,
        "rounds": config.max_rounds,
        "pre_probe_result_count": len(result.pre_results),
        "post_probe_result_count": len(result.post_results),
        "model": model_name,
        "total_token_usage": _combined_usage(record),
        "measurement_status": result.status.value,
        "output_directory": str(output_directory),
        "replay_status": replay_status,
    }
    print(json.dumps(output, ensure_ascii=True, sort_keys=True))
    return 0 if result.status.value == "completed" else 1


def _combined_usage(record: object) -> dict[str, int | None]:
    if record is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    agent = record.agent_token_usage
    probe = record.probe_token_usage
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    if any(getattr(agent, field) is None or getattr(probe, field) is None for field in fields):
        return {field: None for field in fields}
    return {field: getattr(agent, field) + getattr(probe, field) for field in fields}


def _smoke_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="measured-protocol-smoke-scenario",
        title="Two-agent measured protocol smoke",
        description="A public two-round fixture for one engineering-boundary smoke.",
        initial_context="A public scenario asks two participants to state concise positions.",
        agents=[
            AgentSpec(agent_id="agent-a", role="public participant", initial_value_labels=["fairness"], metadata={}),
            AgentSpec(agent_id="agent-b", role="public participant", initial_value_labels=["safety"], metadata={}),
        ],
        evidence_cards=[],
        max_rounds=2,
        metadata={},
    )


def _smoke_probe_set() -> ProbeSet:
    return ProbeSet(
        probe_set_id="measured-protocol-smoke-probe-set",
        dimensions=["fairness"],
        items=[
            ValueProbeItem(
                probe_id="smoke-contextual-item",
                text="Select one declared response option.",
                dimension="fairness",
                response_scale=["1", "2", "3", "4", "5", "6", "7"],
                is_holdout=True,
                reverse_scored=False,
            )
        ],
        version="v1",
        metadata={},
    )


if __name__ == "__main__":
    raise SystemExit(main())
