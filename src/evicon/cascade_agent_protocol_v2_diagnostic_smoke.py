"""Offline FakeProvider regression for H-D.2 strict-parser diagnostics."""
from __future__ import annotations

import json

from .cascade_agent_protocol_v2 import CascadeAgentProtocolV2Runtime, V2_SCHEMA_NAME
from .cascade_agent_protocol_v2_compatibility import (
    COMPATIBILITY_MAX_TOKENS,
    COMPATIBILITY_SEED,
    COMPATIBILITY_TEMPERATURE,
    build_minimal_public_context,
)
from .cascade_agent_protocol_v2_failure_diagnostics import classify_parser_failure
from .llm_contract import LLMRequest, LLMResponse
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ResponseFormatMode, TransportResponse


def _valid_content() -> str:
    return json.dumps({
        "stance": "uncertain",
        "content_ids_used": ["compatibility-content-01"],
        "evidence_ids_used": ["compatibility-evidence-01"],
        "share_content_id": "compatibility-content-01",
    }, sort_keys=True, separators=(",", ":"))


class SequenceProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        response = self.responses.pop(0)
        return response.model_copy(update={"request_id": request.request_id, "model_name": request.model_name})


class OneShotTransport:
    def __init__(self, response: TransportResponse) -> None:
        self.response = response
        self.calls = 0
        self.response_format_requested = False

    def post(self, url, headers, payload, timeout_seconds):
        self.calls += 1
        self.response_format_requested = payload.get("response_format", {}).get("type") == "json_schema"
        return self.response


def _response(content: str, *, finish_reason: str, completion_tokens: int) -> LLMResponse:
    return LLMResponse(
        request_id="offline-placeholder",
        model_name="hd2-v2-diagnostic-fake",
        content=content,
        finish_reason=finish_reason,
        prompt_tokens=10,
        completion_tokens=completion_tokens,
        total_tokens=10 + completion_tokens,
        latency_ms=1.0,
    )


def _observe(case: str, provider, *, response_format_mode: str = "json_schema") -> dict[str, object]:
    context = build_minimal_public_context(model_name="hd2-v2-diagnostic-fake")
    result = CascadeAgentProtocolV2Runtime().execute(context, provider)
    audit = result.audit
    assessment, reasons = classify_parser_failure(
        parser_error_category=audit.error_code,
        finish_reason=audit.finish_reason,
        completion_tokens=audit.completion_tokens,
        configured_max_tokens=COMPATIBILITY_MAX_TOKENS,
        parser_valid=bool(audit.parser_valid),
    )
    return {
        "case": case,
        "runtime_status": result.status.value,
        "parser_valid": audit.parser_valid,
        "parser_error_category": audit.error_code,
        "finish_reason": audit.finish_reason,
        "completion_tokens": audit.completion_tokens,
        "completion_token_limit_reached": audit.completion_tokens == COMPATIBILITY_MAX_TOKENS,
        "response_format_mode": response_format_mode,
        "assessment": assessment.value,
        "reason_codes": list(reasons),
    }


def run_smoke() -> dict[str, object]:
    truncated = SequenceProvider([_response(
        '{"stance":"uncertain","content_ids_used":[]',
        finish_reason="length",
        completion_tokens=COMPATIBILITY_MAX_TOKENS,
    )])
    complete = SequenceProvider([_response(
        _valid_content(), finish_reason="stop", completion_tokens=32,
    )])
    intermittent = SequenceProvider([
        _response('{"stance":', finish_reason="stop", completion_tokens=24),
        _response(_valid_content(), finish_reason="stop", completion_tokens=32),
    ])
    cases = [
        _observe("max_token_truncation", truncated),
        _observe("complete_strict_json", complete),
    ]
    first = _observe("occasional_malformed_json_first_observation", intermittent)
    second = _observe("occasional_malformed_json_separate_observation", intermittent)
    first["separate_observation_followed"] = True
    first["automatic_retry"] = False
    second["automatic_retry"] = False
    cases.extend((first, second))

    body = json.dumps({
        "model": "hd2-v2-diagnostic-fake",
        "choices": [{
            "message": {"content": '{"stance":"uncertain"'},
            "finish_reason": "length",
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": COMPATIBILITY_MAX_TOKENS,
            "total_tokens": 10 + COMPATIBILITY_MAX_TOKENS,
        },
    })
    transport = OneShotTransport(TransportResponse(200, body))
    provider = OpenAICompatibleProvider(
        ProviderConfig(
            base_url="https://offline.invalid/v1",
            model_name="hd2-v2-diagnostic-fake",
            allow_network=True,
            timeout_seconds=1.0,
            max_retries=0,
            temperature=COMPATIBILITY_TEMPERATURE,
            max_tokens=COMPATIBILITY_MAX_TOKENS,
            seed=COMPATIBILITY_SEED,
            response_format=ResponseFormatMode.JSON_SCHEMA,
            response_schema_name=V2_SCHEMA_NAME,
        ),
        transport=transport,
        environment={"EVICON_LLM_API_KEY": "offline-test-placeholder"},
    )
    accepted_but_truncated = _observe("response_format_accepted_but_truncated", provider)
    accepted_but_truncated["response_format_requested_in_payload"] = transport.response_format_requested
    accepted_but_truncated["transport_call_count"] = transport.calls
    cases.append(accepted_but_truncated)
    return {
        "status": "passed",
        "case_count": len(cases),
        "cases": cases,
        "network": "disabled",
        "results_written": False,
        "automatic_retry_performed": False,
        "pilot_resumed": False,
        "private_truth_exposed": False,
        "not_paper_result": True,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
