"""Tests for the no-write contextual private probe runtime."""

from __future__ import annotations

import pytest

from evicon.contextual_probe_runtime import (
    ContextualProbeRequestSettings,
    ContextualProbeRuntime,
    ContextualProbeStatus,
    parse_contextual_probe_response,
    render_contextual_probe,
)
from evicon.llm_contract import LLMProviderError, ProviderErrorCode
from evicon.measurement_context import OfflineMeasurementContextBuilder
from evicon.models import ProtocolCondition

from measurement_cases import LocalProbeProvider, completed_record, probe_item, probe_response_content


def _context():
    return OfflineMeasurementContextBuilder.build(completed_record(), agent_id="agent-a", round_id=1)


def test_contextual_probe_runtime_parses_a_declared_scale_choice_with_deterministic_settings() -> None:
    provider = LocalProbeProvider(content=probe_response_content("4"))
    result = ContextualProbeRuntime().execute(_context(), probe_item(), provider)

    assert result.status is ContextualProbeStatus.COMPLETED
    assert result.response is not None
    assert result.response.raw_response == "4"
    assert result.response.normalized_score == 0.5
    assert provider.requests[0].temperature == 0.0
    assert provider.requests[0].max_tokens == 128
    assert provider.requests[0].seed == 0
    assert "probe_item" in provider.requests[0].user_prompt


@pytest.mark.parametrize(
    ("content", "error_code"),
    [
        ("{not-json", "malformed_json"),
        (probe_response_content("8"), "invalid_choice"),
        (probe_response_content("4", action="override"), "forbidden_control_field"),
        (probe_response_content("hidden probe"), "private_content"),
    ],
)
def test_parser_rejects_non_json_illegal_scale_controls_and_private_content(content: str, error_code: str) -> None:
    result = ContextualProbeRuntime().execute(_context(), probe_item(), LocalProbeProvider(content=content))

    assert result.status is ContextualProbeStatus.PARSER_INVALID
    assert result.response is None
    assert result.audit.error_code == error_code


def test_provider_failure_is_redacted_and_audit_excludes_private_material() -> None:
    item = probe_item()
    result = ContextualProbeRuntime(
        ContextualProbeRequestSettings(model_name="audit-probe-model", temperature=0.0, max_tokens=128, seed=3)
    ).execute(
        _context(),
        item,
        LocalProbeProvider(error=LLMProviderError(ProviderErrorCode.HTTP_SERVER_ERROR, "api_key=never include secret")),
    )
    audit_json = result.audit_summary().model_dump_json()

    assert result.status is ContextualProbeStatus.PROVIDER_ERROR
    assert result.audit.error_code == "http_server_error"
    for forbidden in ("api_key", "secret", item.text, "hidden_probe", "scenario_context", "provider_metadata"):
        assert forbidden not in audit_json


def test_renderer_and_parser_are_strict_and_do_not_create_files(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    context = _context()
    request = render_contextual_probe(context, probe_item(), model_name="local", seed=7)
    parsed, error = parse_contextual_probe_response('{"choice":"1"}', context=context, probe_item=probe_item())

    assert request.metadata["template_version"] == "contextual_value_probe.v2"
    assert parsed is not None and parsed.normalized_score == 0.0
    assert error is None
    assert list(tmp_path.iterdir()) == []


def test_renderer_keeps_protocol_local_and_makes_initial_measurement_condition_blind() -> None:
    context = _context().model_copy(update={"round_id": 0, "own_history": [], "visible_peer_turns": [], "visible_evidence_cards": []})
    social_context = context.model_copy(update={"protocol": ProtocolCondition.SOCIAL_ONLY})
    independent_context = context.model_copy(update={"protocol": ProtocolCondition.INDEPENDENT})

    social = render_contextual_probe(social_context, probe_item(), model_name="local", seed=7)
    independent = render_contextual_probe(independent_context, probe_item(), model_name="local", seed=7)

    assert social.user_prompt == independent.user_prompt
    assert '"protocol"' not in social.user_prompt
    assert social.metadata["protocol"] == "social_only"
    assert independent.metadata["protocol"] == "independent"
