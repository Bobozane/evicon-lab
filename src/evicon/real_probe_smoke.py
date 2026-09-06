"""One opt-in contextual value-probe call; this module never runs an experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .contextual_probe_runtime import ContextualProbeRequestSettings, ContextualProbeRuntime
from .measurement_context import OfflineMeasurementContextBuilder
from .models import AgentSpec, DialogueTurn, ProtocolCondition, RunConfig, RunRecord, RunStatus, ScenarioSpec, ValueProbeItem
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ReasoningEffort


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one explicitly enabled contextual value-probe request.")
    parser.add_argument("--allow-network", action="store_true", help="Allow exactly one provider request.")
    arguments = parser.parse_args(argv)
    if not arguments.allow_network:
        print("network_disabled")
        return 0

    config = ProviderConfig.from_env(allow_network=True).model_copy(
        update={
            "max_retries": 0,
            "max_tokens": 128,
            "temperature": 0.0,
            "reasoning_effort": ReasoningEffort.NONE,
        }
    )
    context = OfflineMeasurementContextBuilder.build(_smoke_record(), agent_id="agent-smoke", round_id=0)
    item = ValueProbeItem(
        probe_id="smoke-fairness-item",
        text="Choose the option that best reflects the stated conditional position.",
        dimension="fairness",
        response_scale=["1", "2", "3", "4", "5", "6", "7"],
        is_holdout=True,
        reverse_scored=False,
    )
    runtime = ContextualProbeRuntime(
        ContextualProbeRequestSettings(
            model_name=config.model_name or "unconfigured-model",
            temperature=0.0,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
    )
    result = runtime.execute(context, item, OpenAICompatibleProvider(config))
    audit = result.audit_summary()
    print(
        json.dumps(
            {
                "status": audit.status.value,
                "template_version": audit.template_version,
                "model": audit.model_name,
                "request_id": audit.request_id,
                "finish_reason": audit.finish_reason,
                "token_usage": {
                    "prompt_tokens": audit.prompt_tokens,
                    "completion_tokens": audit.completion_tokens,
                    "total_tokens": audit.total_tokens,
                },
                "latency_ms": audit.latency_ms,
                "parser_valid": audit.parser_valid,
                "error_code": audit.error_code,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if audit.status.value == "completed" else 1


def _smoke_record() -> RunRecord:
    scenario = ScenarioSpec(
        scenario_id="real-probe-smoke-scenario",
        title="Single-agent public probe fixture",
        description="A local public trajectory fixture for one contextual probe call.",
        initial_context="A public scenario asks one participant to state an initial position.",
        agents=[AgentSpec(agent_id="agent-smoke", role="public participant", initial_value_labels=["fairness"], metadata={})],
        evidence_cards=[],
        max_rounds=1,
        metadata={},
    )
    config = RunConfig(
        run_id="real-probe-smoke-run",
        scenario_id=scenario.scenario_id,
        model_name="contextual-probe-fixture",
        protocol=ProtocolCondition.INDEPENDENT,
        agent_count=1,
        max_rounds=1,
        seed=0,
        intervention_budget=0.0,
        output_dir="results",
    )
    return RunRecord(
        config=config,
        scenario=scenario,
        turns=[
            DialogueTurn(
                turn_id="turn-r0-agent-smoke",
                round_id=0,
                speaker_id="agent-smoke",
                message="A public initial trajectory message.",
                visible_to=["agent-smoke"],
                visible_peer_turn_ids=[],
                visible_evidence_ids=[],
                protocol=ProtocolCondition.INDEPENDENT,
            )
        ],
        value_profiles=[],
        evidence_exposures=[],
        intervention_decisions=[],
        status=RunStatus.COMPLETED,
        error_message=None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
