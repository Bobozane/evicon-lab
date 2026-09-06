"""One opt-in Agent provider request; this module is not an experiment runner."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .agent_prompts import AGENT_TURN_TEMPLATE_VERSION, AgentPromptContext
from .agent_runtime import AgentRequestSettings, AgentRuntime
from .exposure import ExposureSnapshot
from .models import ProtocolCondition
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ReasoningEffort


_SMOKE_MAX_TOKENS = 256


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one explicitly enabled real Agent request.")
    parser.add_argument("--allow-network", action="store_true", help="Allow exactly one provider request.")
    arguments = parser.parse_args(argv)
    if not arguments.allow_network:
        print("network_disabled")
        return 0

    config = ProviderConfig.from_env(allow_network=True).model_copy(
        update={
            "max_retries": 0,
            "max_tokens": _SMOKE_MAX_TOKENS,
            "temperature": 0.2,
            "reasoning_effort": ReasoningEffort.NONE,
        }
    )
    context = _public_round_zero_context()
    runtime = AgentRuntime(
        AgentRequestSettings(
            model_name=config.model_name or "unconfigured-model",
            temperature=0.2,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
    )
    result = runtime.execute(context, OpenAICompatibleProvider(config))
    audit = result.audit_summary()
    output = {
        "status": audit.status.value,
        "template_version": audit.template_version or AGENT_TURN_TEMPLATE_VERSION,
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
        "evidence_ids_used": audit.evidence_ids_used,
    }
    print(json.dumps(output, ensure_ascii=True, sort_keys=True))
    return 0 if audit.status.value == "completed" else 1


def _public_round_zero_context() -> AgentPromptContext:
    """Build a public-only independent-protocol fixture with no peer or evidence inputs."""
    scenario_context = "A public scenario asks the participant to state a concise initial position."
    return AgentPromptContext(
        agent_id="agent-smoke",
        role="public deliberation participant",
        initial_value_labels=["fairness", "safety"],
        round_id=0,
        protocol=ProtocolCondition.INDEPENDENT,
        scenario_context=scenario_context,
        exposure_snapshot=ExposureSnapshot(
            agent_id="agent-smoke",
            round_id=0,
            protocol=ProtocolCondition.INDEPENDENT,
            scenario_context=scenario_context,
            visible_history=[],
            visible_peer_turn_ids=[],
            visible_evidence_ids=[],
        ),
        visible_evidence_cards=[],
    )


if __name__ == "__main__":
    raise SystemExit(main())
