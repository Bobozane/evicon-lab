from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

import evicon.conformity_source_relation_comprehension_qualification_runner_v1 as qualification_runner
from evicon.conformity_source_relation_comprehension_compatibility_approval_v1 import (
    DEFAULT_COMPATIBILITY_APPROVAL,
)
from evicon.conformity_source_relation_comprehension_compatibility_v1 import (
    execute_compatibility_v1,
)
from evicon.conformity_source_relation_comprehension_calibration_v1 import (
    RESPONSE_SCHEMA,
    SCHEMA_NAME,
)
from evicon.conformity_source_relation_comprehension_qualification_runner_v1 import (
    ATTEMPT_RECEIPT_NAME,
    COMPLETION_RESERVATION_CAP,
    DEFAULT_APPROVAL,
    REQUEST_CAP,
    SAFE_AUDIT_NAME,
    QualificationCaseAudit,
    QualificationReceipt,
    SourceRelationQualificationError,
    _has_complete_coverage,
    _execute_qualification_v1,
    execute_qualification_v1,
    load_qualification_receipt,
    safe_preflight,
)
from evicon.openai_provider import TransportResponse


ENVIRONMENT = {
    "EVICON_LLM_BASE_URL": "https://provider.invalid/v1",
    "EVICON_LLM_MODEL": "gpt-5.6-luna",
    "EVICON_LLM_API_KEY": "test-only-secret",
}


class RecordingTransport:
    def __init__(
        self,
        responder: Callable[[int, dict[str, object]], TransportResponse | BaseException],
    ) -> None:
        self._responder = responder
        self.calls: list[dict[str, object]] = []

    def post(self, url, headers, payload, timeout_seconds):  # type: ignore[no-untyped-def]
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        outcome = self._responder(len(self.calls), payload)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _provider_body(
    content: str,
    *,
    finish_reason: str = "stop",
    status_model: str = "untrusted-provider-model-name",
) -> TransportResponse:
    return TransportResponse(
        200,
        json.dumps(
            {
                "model": status_model,
                "choices": [
                    {"message": {"content": content}, "finish_reason": finish_reason}
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 30,
                    "total_tokens": 130,
                },
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
    )


def _compatibility_content() -> str:
    return json.dumps(
        {
            "visible_social_message_count": 2,
            "source_relation": "distinct_shown_roots",
            "visible_social_root_count": 2,
            "social_root_assignments": [
                {"content_id": "message-a", "source_root_id": "lineage-k7"},
                {"content_id": "message-b", "source_root_id": "lineage-p4"},
            ],
            "direct_evidence_card_ids": ["evidence-card-01"],
            "support_kind": "direct_evidence_present",
            "source_structure_alone_decides_truth": False,
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def _qualification_content(payload: dict[str, object]) -> str:
    messages = payload["messages"]
    assert isinstance(messages, list)
    user_message = messages[1]
    assert isinstance(user_message, dict)
    public = json.loads(user_message["content"])
    assert isinstance(public, dict)
    material = public["visible_material"]
    assignments = public["public_root_relations"]
    assert isinstance(material, list)
    assert isinstance(assignments, list)
    roots = [item["source_root_id"] for item in assignments]
    if all(root is None for root in roots):
        relation = "not_shown"
    elif len(set(roots)) == 1:
        relation = "same_shown_root"
    else:
        relation = "distinct_shown_roots"
    evidence_ids = [
        item["content_id"]
        for item in material
        if item["material_type"] == "direct_evidence_card"
    ]
    return json.dumps(
        {
            "visible_social_message_count": 2,
            "source_relation": relation,
            "visible_social_root_count": len({root for root in roots if root is not None}),
            "social_root_assignments": assignments,
            "direct_evidence_card_ids": evidence_ids,
            "support_kind": (
                "direct_evidence_present" if evidence_ids else "social_reports_only"
            ),
            "source_structure_alone_decides_truth": False,
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def _accepted_copy(source: str, target: Path) -> Path:
    raw = (Path(__file__).resolve().parents[1] / source).read_text(encoding="utf-8")
    raw = raw.replace('acceptance_status = "pending"', 'acceptance_status = "accepted"')
    raw = raw.replace('accepted_by = ""', 'accepted_by = "test-reviewer"')
    raw = raw.replace('accepted_on = ""', 'accepted_on = "2026-08-29"')
    target.write_text(raw, encoding="utf-8")
    return target


def _accepted_qualification_approval(tmp_path: Path) -> Path:
    return _accepted_copy(DEFAULT_APPROVAL, tmp_path / "qualification-approval.toml")


def _pending_qualification_approval(tmp_path: Path) -> Path:
    raw = (Path(__file__).resolve().parents[1] / DEFAULT_APPROVAL).read_text(
        encoding="utf-8"
    )
    raw = raw.replace('acceptance_status = "accepted"', 'acceptance_status = "pending"')
    raw = raw.replace('accepted_by = "researcher_user"', 'accepted_by = ""')
    raw = raw.replace('accepted_on = "2026-08-29"', 'accepted_on = ""')
    target = tmp_path / "pending-qualification-approval.toml"
    target.write_text(raw, encoding="utf-8")
    return target


def _compatible_receipt(tmp_path: Path) -> tuple[Path, Path]:
    approval = _accepted_copy(
        DEFAULT_COMPATIBILITY_APPROVAL,
        tmp_path / "compatibility-approval.toml",
    )
    transport = RecordingTransport(
        lambda _index, _payload: _provider_body(_compatibility_content())
    )
    receipt = tmp_path / "compatibility-receipt.json"
    result = execute_compatibility_v1(
        allow_network=True,
        environment=ENVIRONMENT,
        transport=transport,
        approval_path=approval,
        receipt_path=receipt,
    )
    assert result.status == "completed"
    assert len(transport.calls) == 1
    return receipt, approval


def _run(
    tmp_path: Path,
    transport: RecordingTransport,
    *,
    qualification_approval: Path | None = None,
    environment: Mapping[str, str] | None = ENVIRONMENT,
) -> Any:
    compatibility_receipt, compatibility_approval = _compatible_receipt(tmp_path)
    return _execute_qualification_v1(
        allow_network=True,
        confirm_run=True,
        confirm_request_cap=REQUEST_CAP,
        confirm_completion_reservation_cap=COMPLETION_RESERVATION_CAP,
        environment=environment,
        transport=transport,
        approval_path=qualification_approval
        or _accepted_qualification_approval(tmp_path),
        compatibility_receipt_path=compatibility_receipt,
        compatibility_approval_path=compatibility_approval,
        output_root=tmp_path / "qualification-output",
        attempt_claim_path=tmp_path / "qualification.claim",
    )


def _claim_path(tmp_path: Path) -> Path:
    return tmp_path / "qualification.claim"


def test_default_execution_is_inert_and_does_not_read_environment_or_transport(
    tmp_path: Path,
) -> None:
    class ExplodingEnvironment(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            raise AssertionError("environment must not be read")

        def __iter__(self):
            raise AssertionError("environment must not be read")

        def __len__(self) -> int:
            raise AssertionError("environment must not be read")

        def get(self, key: str, default=None):
            raise AssertionError("environment must not be read")

    class FailTransport:
        def post(self, *args, **kwargs):
            raise AssertionError("transport must not be called")

    result = execute_qualification_v1(
        allow_network=False,
        environment=ExplodingEnvironment(),
        transport=FailTransport(),
    )
    assert result.status == "network_disabled"
    assert result.qualification_status == "not_run"
    assert result.logical_request_count == 0
    assert result.transport_attempt_count == 0
    assert result.safety.api_key_read is False
    assert result.safety.provider_constructed is False
    assert not list(tmp_path.iterdir())


def test_pending_approval_blocks_before_environment_or_transport_access(
    tmp_path: Path,
) -> None:
    class ExplodingEnvironment(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            raise AssertionError("environment must not be read")

        def __iter__(self):
            raise AssertionError("environment must not be read")

        def __len__(self) -> int:
            raise AssertionError("environment must not be read")

        def get(self, key: str, default=None):
            raise AssertionError("environment must not be read")

    class FailTransport:
        def post(self, *args, **kwargs):
            raise AssertionError("transport must not be called")

    result = _execute_qualification_v1(
        allow_network=True,
        confirm_run=True,
        confirm_request_cap=REQUEST_CAP,
        confirm_completion_reservation_cap=COMPLETION_RESERVATION_CAP,
        environment=ExplodingEnvironment(),
        transport=FailTransport(),
        approval_path=_pending_qualification_approval(tmp_path),
        output_root=tmp_path / "must-not-exist",
        attempt_claim_path=tmp_path / "must-not-claim",
    )
    assert result.status == "blocked"
    assert result.error_code == "source_relation_qualification_approval_not_accepted"
    assert result.safety.api_key_read is False
    assert result.safety.provider_constructed is False
    assert not (tmp_path / "must-not-exist").exists()
    assert not (tmp_path / "must-not-claim").exists()


def test_safe_preflight_is_read_only_and_reports_remaining_pending_gates() -> None:
    preflight = safe_preflight()
    assert preflight["status"] == "source_relation_qualification_preflight_ready"
    assert preflight["network"] == "disabled"
    assert preflight["api_key_read"] is False
    assert preflight["provider_constructed"] is False
    assert preflight["results_written"] is False
    assert preflight["approval_status"] == "accepted"
    assert "human_approval_required" not in preflight["blocking_reasons"]
    assert preflight["compatibility_approval_accepted"] is True
    assert "provider_compatibility_approval_required" not in preflight["blocking_reasons"]
    assert "provider_compatibility_receipt_required" in preflight["blocking_reasons"]


def test_24_case_success_uses_exact_contract_and_only_safe_artifacts(
    tmp_path: Path,
) -> None:
    transport = RecordingTransport(
        lambda _index, payload: _provider_body(_qualification_content(payload))
    )
    result = _run(tmp_path, transport)
    assert result.status == "completed"
    assert result.qualification_status == "source_structure_qualified"
    assert result.logical_request_count == REQUEST_CAP
    assert result.transport_attempt_count == REQUEST_CAP
    assert result.completed_case_count == REQUEST_CAP
    assert result.parser_invalid_count == 0
    assert len(transport.calls) == REQUEST_CAP
    assert result.safety.qualification_executed is True
    assert result.safety.safe_case_audits_written is True
    assert result.safety.attempt_receipt_written is True

    for call in transport.calls:
        payload = call["payload"]
        assert payload["max_tokens"] == 128
        assert payload["temperature"] == 0.0
        assert payload["seed"] == 20261401
        assert payload["reasoning_effort"] == "none"
        assert call["timeout_seconds"] == 5.0
        assert payload["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": SCHEMA_NAME,
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            },
        }

    output_root = tmp_path / "qualification-output"
    assert {item.name for item in output_root.iterdir()} == {
        SAFE_AUDIT_NAME,
        ATTEMPT_RECEIPT_NAME,
    }
    audit_text = (output_root / SAFE_AUDIT_NAME).read_text(encoding="utf-8")
    assert len(audit_text.splitlines()) == REQUEST_CAP
    receipt = load_qualification_receipt(
        output_root / ATTEMPT_RECEIPT_NAME,
        approval_path=tmp_path / "qualification-approval.toml",
        compatibility_receipt_path=tmp_path / "compatibility-receipt.json",
        compatibility_approval_path=tmp_path / "compatibility-approval.toml",
    )
    assert receipt.status == "source_structure_qualified"
    assert receipt.completed_case_count == REQUEST_CAP
    assert receipt.failure_audit is None

    serialized = "\n".join(
        [
            audit_text,
            (output_root / ATTEMPT_RECEIPT_NAME).read_text(encoding="utf-8"),
            result.model_dump_json(),
        ]
    ).lower()
    for forbidden in (
        "test-only-secret",
        "authorization",
        "bearer",
        "system_prompt",
        "user_prompt",
        '"provider_metadata":',
        "lineage-k7",
        "lineage-p4",
        "message-a",
        "message-b",
        "evidence-card-01",
        "harborlink",
    ):
        assert forbidden not in serialized
    assert not _claim_path(tmp_path).exists()


def _malformed_content(_: str) -> str:
    return "not-json"


def _unknown_field_content(content: str) -> str:
    payload = json.loads(content)
    assert isinstance(payload, dict)
    payload["unknown_field"] = "forbidden"
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _unexposed_content_id(content: str) -> str:
    payload = json.loads(content)
    assert isinstance(payload, dict)
    assignments = payload["social_root_assignments"]
    assert isinstance(assignments, list)
    assignments[1]["content_id"] = "message-x"
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


@pytest.mark.parametrize(
    ("failure_at", "mutate", "finish_reason", "expected"),
    [
        (3, _malformed_content, "stop", "malformed_json"),
        (2, _unknown_field_content, "stop", "invalid_schema"),
        # The strict response schema rejects an unknown ID before the
        # semantic visibility validator is reached.
        (2, _unexposed_content_id, "stop", "invalid_schema"),
        (1, None, "length", "invalid_provider_response"),
    ],
)
def test_invalid_response_stops_without_retry_and_keeps_a_redacted_attempt_receipt(
    tmp_path: Path,
    failure_at: int,
    mutate: Callable[[str], str] | None,
    finish_reason: str,
    expected: str,
) -> None:
    def responder(index: int, payload: dict[str, object]) -> TransportResponse:
        valid_content = _qualification_content(payload)
        return _provider_body(
            mutate(valid_content) if index == failure_at and mutate is not None else valid_content,
            finish_reason=finish_reason if index == failure_at else "stop",
        )

    transport = RecordingTransport(responder)
    result = _run(tmp_path, transport)
    assert result.status == "failed"
    assert result.error_code == expected
    assert result.logical_request_count == failure_at
    assert result.transport_attempt_count == failure_at
    assert result.completed_case_count == failure_at - 1
    assert result.parser_invalid_count == int(expected in {"malformed_json", "invalid_schema"})
    assert len(transport.calls) == failure_at
    assert _claim_path(tmp_path).exists()

    output_root = tmp_path / "qualification-output"
    receipt = QualificationReceipt.model_validate(
        json.loads((output_root / ATTEMPT_RECEIPT_NAME).read_text(encoding="utf-8"))
    )
    assert receipt.status == "execution_failed"
    assert receipt.failure_audit is not None
    assert receipt.failure_audit.error_category == expected
    assert receipt.failure_audit.cumulative_transport_attempt_count == failure_at
    if failure_at > 1:
        assert len((output_root / SAFE_AUDIT_NAME).read_text().splitlines()) == failure_at - 1
    else:
        assert not (output_root / SAFE_AUDIT_NAME).exists()


@pytest.mark.parametrize(
    ("outcome", "expected", "status_class"),
    [
        (TimeoutError("sensitive-timeout"), "timeout", None),
        (ConnectionError("sensitive-connection"), "connection_failure", None),
        (
            TransportResponse(401, '{"error":{"message":"sensitive-auth"}}'),
            "authentication_failed",
            "4xx",
        ),
        (
            TransportResponse(429, '{"error":{"message":"sensitive-rate"}}'),
            "rate_limited",
            "4xx",
        ),
        (
            TransportResponse(500, '{"error":{"message":"sensitive-server"}}'),
            "http_server_error",
            "5xx",
        ),
        (
            TransportResponse(
                400,
                '{"error":{"message":"response_format json_schema unsupported secret"}}',
            ),
            "response_format_unsupported",
            "4xx",
        ),
    ],
)
def test_transport_errors_are_terminal_and_do_not_leak_details(
    tmp_path: Path,
    outcome: TransportResponse | BaseException,
    expected: str,
    status_class: str | None,
) -> None:
    transport = RecordingTransport(lambda _index, _payload: outcome)
    result = _run(tmp_path, transport)
    assert result.status == "failed"
    assert result.error_code == expected
    assert result.logical_request_count == 1
    assert result.transport_attempt_count == 1
    assert result.completed_case_count == 0
    assert len(transport.calls) == 1
    assert _claim_path(tmp_path).exists()
    receipt_text = (tmp_path / "qualification-output" / ATTEMPT_RECEIPT_NAME).read_text(
        encoding="utf-8"
    )
    receipt = QualificationReceipt.model_validate(json.loads(receipt_text))
    assert receipt.failure_audit is not None
    assert receipt.failure_audit.http_status_class == status_class
    for forbidden in (
        "sensitive-",
        "unsupported secret",
        "test-only-secret",
        "authorization",
        '"provider_metadata":',
    ):
        assert forbidden not in receipt_text.lower()


def test_binding_change_between_cases_stops_before_a_second_request(tmp_path: Path) -> None:
    approval = _accepted_qualification_approval(tmp_path)

    def responder(index: int, payload: dict[str, object]) -> TransportResponse:
        if index == 1:
            approval.write_text(
                approval.read_text(encoding="utf-8") + "\n# changed during request\n",
                encoding="utf-8",
            )
        return _provider_body(_qualification_content(payload))

    transport = RecordingTransport(responder)
    result = _run(tmp_path, transport, qualification_approval=approval)
    assert result.status == "failed"
    assert result.error_code == "source_relation_qualification_binding_changed_before_request"
    assert result.logical_request_count == 1
    assert result.transport_attempt_count == 1
    assert result.completed_case_count == 1
    assert len(transport.calls) == 1
    assert _claim_path(tmp_path).exists()


def test_receipt_loader_rejects_tampered_safe_audit(tmp_path: Path) -> None:
    transport = RecordingTransport(
        lambda _index, payload: _provider_body(_qualification_content(payload))
    )
    result = _run(tmp_path, transport)
    assert result.status == "completed"
    audit_path = tmp_path / "qualification-output" / SAFE_AUDIT_NAME
    audit_path.write_text(audit_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(
        SourceRelationQualificationError,
        match="source_relation_qualification_receipt_audit_mismatch",
    ):
        load_qualification_receipt(
            tmp_path / "qualification-output" / ATTEMPT_RECEIPT_NAME,
            approval_path=tmp_path / "qualification-approval.toml",
            compatibility_receipt_path=tmp_path / "compatibility-receipt.json",
            compatibility_approval_path=tmp_path / "compatibility-approval.toml",
        )


def test_missing_base_url_is_redacted_and_releases_the_unused_claim(tmp_path: Path) -> None:
    transport = RecordingTransport(
        lambda _index, payload: _provider_body(_qualification_content(payload))
    )
    result = _run(
        tmp_path,
        transport,
        environment={"EVICON_LLM_MODEL": "gpt-5.6-luna"},
    )
    assert result.status == "failed"
    assert result.error_code == "missing_base_url"
    assert result.attempt_claim_released is True
    assert result.safety.network_enabled is True
    assert result.safety.compatibility_receipt_loaded is True
    assert result.safety.provider_constructed is False
    assert result.safety.api_key_read is False
    assert not transport.calls
    assert not (tmp_path / "qualification-output").exists()
    assert not _claim_path(tmp_path).exists()


def test_missing_api_key_is_redacted_and_releases_the_unused_claim(tmp_path: Path) -> None:
    transport = RecordingTransport(
        lambda _index, payload: _provider_body(_qualification_content(payload))
    )
    result = _run(
        tmp_path,
        transport,
        environment={
            "EVICON_LLM_BASE_URL": "https://provider.invalid/v1",
            "EVICON_LLM_MODEL": "gpt-5.6-luna",
        },
    )
    assert result.status == "failed"
    assert result.error_code == "missing_api_key"
    assert result.attempt_claim_released is True
    assert result.safety.network_enabled is True
    assert result.safety.compatibility_receipt_loaded is True
    assert result.safety.provider_constructed is True
    assert result.safety.api_key_read is True
    assert not transport.calls
    assert not (tmp_path / "qualification-output").exists()
    assert not _claim_path(tmp_path).exists()


def test_binding_change_before_first_request_is_blocked_with_exact_safety(
    tmp_path: Path,
) -> None:
    approval = _accepted_qualification_approval(tmp_path)

    class MutatingEnvironment(dict[str, str]):
        def get(self, key: str, default=None):  # type: ignore[no-untyped-def]
            if key == "EVICON_LLM_API_KEY":
                approval.write_text(
                    approval.read_text(encoding="utf-8") + "\n# changed before request\n",
                    encoding="utf-8",
                )
            return super().get(key, default)

    transport = RecordingTransport(
        lambda _index, payload: _provider_body(_qualification_content(payload))
    )
    result = _run(
        tmp_path,
        transport,
        qualification_approval=approval,
        environment=MutatingEnvironment(ENVIRONMENT),
    )
    assert result.status == "blocked"
    assert result.error_code == "source_relation_qualification_binding_changed_before_request"
    assert result.attempt_claim_released is True
    assert result.safety.network_enabled is True
    assert result.safety.compatibility_receipt_loaded is True
    assert result.safety.provider_constructed is True
    assert result.safety.api_key_read is True
    assert not transport.calls
    assert not (tmp_path / "qualification-output").exists()
    assert not _claim_path(tmp_path).exists()


def test_unexpected_parser_error_is_terminal_and_does_not_be_labeled_invalid_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_parser(*args: object, **kwargs: object) -> object:
        raise RuntimeError("sensitive parser detail")

    monkeypatch.setattr(qualification_runner, "parse_response", unexpected_parser)
    transport = RecordingTransport(
        lambda _index, payload: _provider_body(_qualification_content(payload))
    )
    result = _run(tmp_path, transport)
    assert result.status == "failed"
    assert result.error_code == "internal_execution_failure"
    assert result.parser_invalid_count == 0
    assert result.logical_request_count == 1
    assert len(transport.calls) == 1
    receipt_text = (tmp_path / "qualification-output" / ATTEMPT_RECEIPT_NAME).read_text(
        encoding="utf-8"
    )
    assert "sensitive parser detail" not in receipt_text
    receipt = QualificationReceipt.model_validate(json.loads(receipt_text))
    assert receipt.failure_audit is not None
    assert receipt.failure_audit.parser_valid is None


def test_receipt_loader_rejects_tampered_schema_binding(tmp_path: Path) -> None:
    transport = RecordingTransport(
        lambda _index, payload: _provider_body(_qualification_content(payload))
    )
    assert _run(tmp_path, transport).status == "completed"
    receipt_path = tmp_path / "qualification-output" / ATTEMPT_RECEIPT_NAME
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_payload["schema"]["schema_sha256"] = "0" * 64
    receipt_path.write_text(
        json.dumps(receipt_payload, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(
        SourceRelationQualificationError,
        match="source_relation_qualification_receipt_binding_mismatch",
    ):
        load_qualification_receipt(
            receipt_path,
            approval_path=tmp_path / "qualification-approval.toml",
            compatibility_receipt_path=tmp_path / "compatibility-receipt.json",
            compatibility_approval_path=tmp_path / "compatibility-approval.toml",
        )


def test_artifact_write_after_receipt_commit_reports_the_actual_safe_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_atomic_create = qualification_runner._atomic_create

    def commit_then_fail(target: Path, payload: str) -> None:
        original_atomic_create(target, payload)
        if target.name == ATTEMPT_RECEIPT_NAME:
            raise OSError("sensitive fsync failure")

    monkeypatch.setattr(qualification_runner, "_atomic_create", commit_then_fail)
    transport = RecordingTransport(
        lambda _index, payload: _provider_body(_qualification_content(payload))
    )
    result = _run(tmp_path, transport)
    output_root = tmp_path / "qualification-output"
    assert result.status == "failed"
    assert result.error_code == "source_relation_qualification_artifact_write_failed"
    assert result.safety.safe_case_audits_written is True
    assert result.safety.attempt_receipt_written is True
    assert result.safe_case_audit_path == str(output_root / SAFE_AUDIT_NAME)
    assert result.receipt_path == str(output_root / ATTEMPT_RECEIPT_NAME)
    assert (output_root / SAFE_AUDIT_NAME).exists()
    assert (output_root / ATTEMPT_RECEIPT_NAME).exists()
    assert _claim_path(tmp_path).exists()


def test_successful_receipt_remains_completed_if_claim_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualification_runner, "_release_claim", lambda _path: False)
    transport = RecordingTransport(
        lambda _index, payload: _provider_body(_qualification_content(payload))
    )
    result = _run(tmp_path, transport)
    assert result.status == "completed"
    assert result.qualification_status == "source_structure_qualified"
    assert result.error_code is None
    assert result.attempt_claim_released is False
    assert result.claim_cleanup_status == "release_failed"
    assert (tmp_path / "qualification-output" / ATTEMPT_RECEIPT_NAME).exists()
    assert _claim_path(tmp_path).exists()


def test_coverage_rejects_a_globally_balanced_but_projection_mismatched_matrix(
    tmp_path: Path,
) -> None:
    transport = RecordingTransport(
        lambda _index, payload: _provider_body(_qualification_content(payload))
    )
    assert _run(tmp_path, transport).status == "completed"
    audit_lines = (tmp_path / "qualification-output" / SAFE_AUDIT_NAME).read_text(
        encoding="utf-8"
    ).splitlines()
    audits = [QualificationCaseAudit.model_validate_json(line) for line in audit_lines]
    first_index = next(
        index for index, item in enumerate(audits) if item.source_relation.value == "not_shown"
    )
    second_index = next(
        index
        for index, item in enumerate(audits)
        if item.source_relation.value == "same_shown_root"
        and (
            item.scenario_id,
            item.evidence_state,
            item.presentation_counterbalance,
        )
        != (
            audits[first_index].scenario_id,
            audits[first_index].evidence_state,
            audits[first_index].presentation_counterbalance,
        )
    )
    first = audits[first_index]
    second = audits[second_index]
    audits[first_index] = first.model_copy(update={"source_relation": second.source_relation})
    audits[second_index] = second.model_copy(update={"source_relation": first.source_relation})
    assert not _has_complete_coverage(audits)


def test_failed_receipt_with_completed_cases_requires_a_safe_audit_hash(
    tmp_path: Path,
) -> None:
    def responder(index: int, payload: dict[str, object]) -> TransportResponse:
        return _provider_body(
            _qualification_content(payload),
            finish_reason="length" if index == 2 else "stop",
        )

    result = _run(tmp_path, RecordingTransport(responder))
    assert result.status == "failed"
    receipt_payload = json.loads(
        (tmp_path / "qualification-output" / ATTEMPT_RECEIPT_NAME).read_text(
            encoding="utf-8"
        )
    )
    receipt_payload["safe_case_audit_sha256"] = None
    with pytest.raises(ValueError, match="completed_audit_hash_missing"):
        QualificationReceipt.model_validate(receipt_payload)
