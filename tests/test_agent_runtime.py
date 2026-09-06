"""Offline tests for the isolated Agent runtime boundary."""

from __future__ import annotations

from agent_cases import LocalAgentProvider, agent_response_content, public_context

from evicon.agent_runtime import AgentRequestSettings, AgentRuntime, AgentRuntimeStatus
from evicon.llm_contract import LLMProviderError, ProviderErrorCode


def test_agent_runtime_uses_default_reproducible_request_settings_and_parses_response() -> None:
    provider = LocalAgentProvider(content=agent_response_content(evidence_ids_used=["evidence-0"]))
    runtime = AgentRuntime()

    result = runtime.execute(public_context(), provider)

    assert result.status is AgentRuntimeStatus.COMPLETED
    assert result.response is not None and result.response.valid is True
    assert result.audit.evidence_ids_used == ["evidence-0"]
    assert provider.requests[0].temperature == 0.2
    assert provider.requests[0].max_tokens == 256
    assert provider.requests[0].seed == 0


def test_malformed_json_and_unknown_evidence_return_parser_errors() -> None:
    malformed = AgentRuntime().execute(public_context(), LocalAgentProvider(content="{not-json"))
    unavailable = AgentRuntime().execute(
        public_context(),
        LocalAgentProvider(content=agent_response_content(evidence_ids_used=["not-visible"])),
    )

    assert malformed.status is AgentRuntimeStatus.PARSER_INVALID
    assert malformed.audit.error_code == "malformed_json"
    assert unavailable.status is AgentRuntimeStatus.PARSER_INVALID
    assert unavailable.audit.error_code == "unavailable_evidence"


def test_provider_error_is_redacted_and_audit_omits_prompt_message_and_hidden_content() -> None:
    provider = LocalAgentProvider(
        error=LLMProviderError(
            ProviderErrorCode.HTTP_SERVER_ERROR,
            "api_key=must-not-appear prompt=must-not-appear",
        )
    )
    failed = AgentRuntime().execute(public_context(), provider)
    audit_json = failed.audit_summary().model_dump_json()

    assert failed.status is AgentRuntimeStatus.PROVIDER_ERROR
    assert failed.audit.error_code == "http_server_error"
    assert "api_key" not in audit_json
    assert "must-not-appear" not in audit_json
    assert "hidden_probe" not in audit_json

    stored_message = "unique-agent-message-stays-in-memory"
    completed = AgentRuntime(
        AgentRequestSettings(model_name="audit-test-model", temperature=0.2, max_tokens=256, seed=1)
    ).execute(public_context(), LocalAgentProvider(content=agent_response_content(message=stored_message)))
    completed_audit = completed.audit_summary().model_dump_json()

    assert completed.response is not None and completed.response.message == stored_message
    assert stored_message not in completed_audit
    assert "scenario_context" not in completed_audit
