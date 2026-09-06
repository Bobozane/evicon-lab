"""Minimal, opt-in provider smoke command; it is never a full experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .llm_contract import LLMProviderError
from .mediator_prompt_smoke import _context_for, _plan_for
from .mediator_prompts import render_action_instruction
from .mediator_response import parse_mediator_response
from .models import InterventionAction
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ReasoningEffort


# This remains deliberately bounded: it is enough for the strict response schema
# while avoiding the broad generation budget used by a real experiment.
_SMOKE_MAX_TOKENS = 256


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one explicitly enabled OpenAI-compatible provider request.")
    parser.add_argument("--allow-network", action="store_true", help="Allow exactly one provider request.")
    parser.add_argument("--print-response", action="store_true", help="Print a bounded response preview after a request.")
    arguments = parser.parse_args(argv)
    if not arguments.allow_network:
        print("network_disabled")
        return 0

    config = ProviderConfig.from_env(allow_network=True).model_copy(
        update={
            "max_tokens": _SMOKE_MAX_TOKENS,
            "reasoning_effort": ReasoningEffort.NONE,
        }
    )
    plan = _plan_for(InterventionAction.REQUEST_EVIDENCE)
    context = _context_for(plan)
    request = render_action_instruction(
        plan,
        context,
        model_name=config.model_name or "unconfigured-model",
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        seed=config.seed,
    )
    if request is None:
        raise RuntimeError("provider smoke could not render its fixed request")
    provider = OpenAICompatibleProvider(config)
    try:
        response = provider.complete(request)
    except LLMProviderError as error:
        print(
            json.dumps(
                {
                    "provider": "openai_compatible",
                    "model": config.model_name,
                    "request_id": request.request_id,
                    "status": "provider_error",
                    "parser_valid": None,
                    "parser_error_code": error.code.value,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 2

    parsed = parse_mediator_response(response.content, plan, context)
    output: dict[str, object] = {
        "provider": "openai_compatible",
        "model": response.model_name,
        "request_id": response.request_id,
        "status": "completed" if parsed.valid else "invalid_response",
        "finish_reason": response.finish_reason,
        "token_usage": {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
        },
        "latency_ms": response.latency_ms,
        "parser_valid": parsed.valid,
        "parser_error_code": (
            parsed.validation_errors[0].value if parsed.validation_errors else None
        ),
    }
    if arguments.print_response:
        output["response_preview"] = response.content[:400]
    print(json.dumps(output, ensure_ascii=True, sort_keys=True))
    return 0 if parsed.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
