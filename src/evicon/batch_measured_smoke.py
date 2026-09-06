"""Safe plan and opt-in execution entry point for a four-condition batch."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .agent_runtime import AgentRequestSettings, AgentRuntime
from .batch_experiment import (
    BASELINE_PROTOCOLS,
    BatchMeasuredExperimentRunner,
    BatchRuntimeParameters,
    build_four_condition_manifest,
)
from .contextual_probe_runtime import ContextualProbeRequestSettings, ContextualProbeRuntime
from .models import AgentSpec, EvidenceCard, ProbeSet, ValueProbeItem, ScenarioSpec
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ReasoningEffort


_SMOKE_SEED = 0
_EXPECTED_REQUEST_UPPER_BOUND = 32


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan or explicitly run one measured four-condition batch smoke.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true", help="Build and print a safe plan without provider calls or writes.")
    mode.add_argument("--allow-network", action="store_true", help="Allow at most 32 provider requests for this fixed smoke batch.")
    parser.add_argument("--batch-id", help="Required identifier for plan-only or network-enabled operation.")
    parser.add_argument("--output-dir", default="results", help="Parent directory for per-condition run outputs.")
    parser.add_argument("--batch-output-dir", default="results/batches", help="Parent directory for batch audit files.")
    arguments = parser.parse_args(argv)
    if not arguments.plan_only and not arguments.allow_network:
        print("network_disabled")
        return 0
    if not arguments.batch_id:
        parser.error("--batch-id is required with --plan-only or --allow-network")
    if Path(arguments.batch_id).name != arguments.batch_id:
        parser.error("--batch-id must be a simple directory name")

    if arguments.plan_only:
        manifest = _smoke_manifest(
            batch_id=arguments.batch_id,
            model_name="batch-plan-only-model",
            probe_model_name="batch-plan-only-model",
            output_dir=arguments.output_dir,
            batch_output_dir=arguments.batch_output_dir,
        )
        print(json.dumps(_plan_summary(manifest), ensure_ascii=True, sort_keys=True))
        return 0

    if (Path(arguments.batch_output_dir) / arguments.batch_id).exists():
        parser.error("--batch-id already has an output directory")
    if any((Path(arguments.output_dir) / _run_id(arguments.batch_id, condition)).exists() for condition in BASELINE_PROTOCOLS):
        parser.error("a planned run ID already has an output directory")

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
    probe_model_name = probe_config.model_name or model_name
    manifest = _smoke_manifest(
        batch_id=arguments.batch_id,
        model_name=model_name,
        probe_model_name=probe_model_name,
        output_dir=arguments.output_dir,
        batch_output_dir=arguments.batch_output_dir,
    )
    print(json.dumps(_plan_summary(manifest), ensure_ascii=True, sort_keys=True))
    experiment = BatchMeasuredExperimentRunner(
        manifest,
        scenario=_smoke_scenario(),
        probe_set=_smoke_probe_set(),
        agent_runtime={
            _SMOKE_SEED: AgentRuntime(
                AgentRequestSettings(model_name=model_name, temperature=0.2, max_tokens=256, seed=_SMOKE_SEED)
            )
        },
        contextual_probe_runtime={
            _SMOKE_SEED: ContextualProbeRuntime(
                ContextualProbeRequestSettings(
                    model_name=probe_model_name,
                    temperature=0.0,
                    max_tokens=128,
                    seed=_SMOKE_SEED,
                )
            )
        },
        agent_provider=OpenAICompatibleProvider(agent_config),
        probe_provider=OpenAICompatibleProvider(probe_config),
    )
    record = experiment.run()
    print(
        json.dumps(
            {
                **_plan_summary(manifest),
                "planned_run_count": len(manifest.runs),
                "completed_run_count": record.completed_run_count,
                "failed_run_count": record.failed_run_count,
                "actual_request_count": _EXPECTED_REQUEST_UPPER_BOUND if record.status.value == "completed" else None,
                "batch_status": record.status.value,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if record.status.value == "completed" else 1


def _plan_summary(manifest: object) -> dict[str, object]:
    return {
        "batch_id": manifest.batch_id,
        "condition_count": len(manifest.conditions),
        "seed_count": len(manifest.seeds),
        "planned_run_count": len(manifest.runs),
        "completed_run_count": 0,
        "failed_run_count": 0,
        "expected_request_upper_bound": _EXPECTED_REQUEST_UPPER_BOUND,
        "batch_output_directory": str(Path(manifest.batch_output_dir) / manifest.batch_id),
    }


def _smoke_manifest(
    *,
    batch_id: str,
    model_name: str,
    probe_model_name: str,
    output_dir: str,
    batch_output_dir: str,
):
    return build_four_condition_manifest(
        batch_id=batch_id,
        scenario=_smoke_scenario(),
        probe_set=_smoke_probe_set(),
        model_name=model_name,
        probe_model_name=probe_model_name,
        seeds=[_SMOKE_SEED],
        output_dir=output_dir,
        batch_output_dir=batch_output_dir,
        runtime_parameters=BatchRuntimeParameters(
            agent_temperature=0.2,
            agent_max_tokens=256,
            probe_temperature=0.0,
            probe_max_tokens=128,
        ),
        is_holdout=True,
        pre_target_round=0,
        post_target_round=1,
        version="batch_smoke.v1",
        metadata={"purpose": "engineering smoke plan"},
    )


def _run_id(batch_id: str, condition: object) -> str:
    return f"{batch_id}-{condition.value}-seed-{_SMOKE_SEED}"


def _smoke_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="batch-measured-smoke-scenario",
        title="Two-agent evidence batch smoke",
        description="A public fixture for a four-condition batch engineering boundary.",
        initial_context="A public scenario asks two participants to assess a stated constraint.",
        agents=[
            AgentSpec(agent_id="agent-a", role="public participant", initial_value_labels=["fairness"], metadata={}),
            AgentSpec(agent_id="agent-b", role="public participant", initial_value_labels=["safety"], metadata={}),
        ],
        evidence_cards=[
            EvidenceCard(
                evidence_id="smoke-public-evidence",
                claim="A public source reports a shared factual constraint.",
                source="local-public-source",
                supports=["constraint"],
                contradicts=[],
                introduced_round=0,
                visible_to=["agent-a", "agent-b"],
                reliability=0.8,
            )
        ],
        max_rounds=2,
        metadata={},
    )


def _smoke_probe_set() -> ProbeSet:
    return ProbeSet(
        probe_set_id="batch-measured-smoke-probe-set",
        dimensions=["fairness"],
        items=[
            ValueProbeItem(
                probe_id="batch-smoke-probe-item",
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
