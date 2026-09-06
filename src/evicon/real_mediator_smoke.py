"""One opt-in, real-provider mediator pilot; it is never an experiment runner."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .mediator_prompt_smoke import _context_for, _plan_for
from .mediator_prompts import TEMPLATE_VERSIONS
from .mediator_runtime import MediatorRequestSettings, MediatorRuntime
from .models import InterventionAction
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ReasoningEffort


_SMOKE_MAX_TOKENS = 256


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one explicitly enabled real mediator pilot request.")
    parser.add_argument("--allow-network", action="store_true", help="Allow exactly one provider request.")
    arguments = parser.parse_args(argv)
    if not arguments.allow_network:
        print("network_disabled")
        return 0

    config = ProviderConfig.from_env(allow_network=True).model_copy(
        update={
            "max_retries": 0,
            "max_tokens": _SMOKE_MAX_TOKENS,
            "reasoning_effort": ReasoningEffort.NONE,
        }
    )
    plan = _plan_for(InterventionAction.SOLICIT_DISSENT)
    context = _context_for(plan)
    runtime = MediatorRuntime(
        MediatorRequestSettings(
            model_name=config.model_name or "unconfigured-model",
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
    )
    result = runtime.execute(plan, context, OpenAICompatibleProvider(config))
    audit = result.audit_summary()
    output = {
        "status": audit.status.value,
        "action": plan.action.value,
        "template_version": audit.template_version or TEMPLATE_VERSIONS[plan.action],
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
    }
    print(json.dumps(output, ensure_ascii=True, sort_keys=True))
    return 0 if audit.status.value == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
