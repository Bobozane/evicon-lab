from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from evicon.conformity_source_behavior_execution_approval import (
    DEFAULT_EXECUTION_APPROVAL,
)
from evicon.conformity_source_behavior_runner import (
    RECEIPT_NAME,
    SAFE_AUDIT_NAME,
    QualificationRunReceipt,
    execute_behavior_qualification,
    load_qualification_receipt,
    run_behavior_qualification,
)
from evicon.llm_contract import LLMResponse


ROOT = Path(__file__).resolve().parents[1]


class FakeBehaviorProvider:
    def __init__(self, *, invalid_at: int | None = None, finish_reason: str = "stop") -> None:
        self.invalid_at = invalid_at
        self.finish_reason = finish_reason
        self.calls = 0

    def complete(self, request) -> LLMResponse:
        self.calls += 1
        if self.invalid_at == self.calls:
            content = "not-json"
            reason = self.finish_reason
        else:
            content = json.dumps({
                "adoption_decision": "withhold",
                "sharing_decision": "do_not_share",
                "content_ids_used": [],
                "share_content_id": None,
            }, ensure_ascii=True, sort_keys=True)
            reason = "stop"
        return LLMResponse(
            request_id=request.request_id,
            model_name="gpt-5.6-luna",
            content=content,
            finish_reason=reason,
            prompt_tokens=30,
            completion_tokens=10,
            total_tokens=40,
            latency_ms=2.0,
            provider_metadata={"status_code": 200},
        )


def _accepted_execution_approval(tmp_path: Path) -> Path:
    raw = (ROOT / DEFAULT_EXECUTION_APPROVAL).read_text(encoding="utf-8")
    raw = raw.replace('acceptance_status = "pending"', 'acceptance_status = "accepted"')
    raw = raw.replace('accepted_by = ""', 'accepted_by = "test-reviewer"')
    raw = raw.replace('accepted_on = ""', 'accepted_on = "2026-08-28"')
    raw = re.sub(r"(confirm_[a-z0-9_]+) = false", r"\1 = true", raw)
    path = tmp_path / "execution-approval.toml"
    path.write_text(raw, encoding="utf-8")
    return path


def test_fake_provider_completes_exact_12_cases_with_safe_outputs(tmp_path: Path) -> None:
    provider = FakeBehaviorProvider()
    output = tmp_path / "run"
    approval = _accepted_execution_approval(tmp_path)
    result = run_behavior_qualification(
        provider,
        output_root=output,
        execution_approval_path=approval,
    )
    assert result.status == "completed"
    assert provider.calls == 12
    assert result.logical_request_count == 12
    assert result.transport_attempt_count == 12
    assert result.completed_case_count == 12
    assert result.safe_case_audit_count == 12
    assert result.parser_invalid_count == 0
    assert set(path.name for path in output.iterdir()) == {SAFE_AUDIT_NAME, RECEIPT_NAME}

    lines = (output / SAFE_AUDIT_NAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 12
    audits = [json.loads(line) for line in lines]
    assert all(set(item) == {
        "case_id", "scenario_id", "projection", "parser_status",
        "adoption_decision", "sharing_decision", "used_content_count",
        "visible_root_count",
    } for item in audits)
    assert {item["projection"] for item in audits} == {
        "source_free", "same_root", "independent_roots",
    }

    receipt = load_qualification_receipt(
        output / RECEIPT_NAME,
        execution_approval_path=approval,
    )
    assert receipt.logical_request_count == receipt.transport_attempt_count == 12
    assert receipt.safe_case_audit_count == 12
    assert receipt.projection_case_counts == {
        "independent_roots": 4, "same_root": 4, "source_free": 4,
    }
    assert receipt.token_usage.model_dump() == {
        "prompt_tokens": 360,
        "completion_tokens": 120,
        "total_tokens": 480,
    }
    assert receipt.generation_parameters.reasoning_effort == "none"
    assert receipt.safety.request_ledger_written is False
    assert receipt.safety.behavior_effect_estimated is False

    serialized = "\n".join(
        path.read_text(encoding="utf-8") for path in output.iterdir()
    ).lower()
    for forbidden in (
        '"system_prompt":', '"user_prompt":', '"full_response":',
        '"raw_response":', '"provider_metadata":', '"authorization":',
        "bearer ", '"api_key":',
        '"ground_truth":', '"evaluator_private_truth":', '"public_summary":',
    ):
        assert forbidden not in serialized


def test_parser_invalid_stops_immediately_and_writes_no_completion_receipt(
    tmp_path: Path,
) -> None:
    provider = FakeBehaviorProvider(invalid_at=4)
    output = tmp_path / "failed"
    result = run_behavior_qualification(
        provider,
        output_root=output,
        execution_approval_path=_accepted_execution_approval(tmp_path),
    )
    assert result.status == "failed"
    assert result.error_code == "source_behavior_parser_invalid"
    assert provider.calls == 4
    assert result.logical_request_count == result.transport_attempt_count == 4
    assert result.completed_case_count == result.safe_case_audit_count == 3
    assert result.parser_invalid_count == 1
    assert len((output / SAFE_AUDIT_NAME).read_text(encoding="utf-8").splitlines()) == 3
    assert not (output / RECEIPT_NAME).exists()


def test_finish_reason_length_stops_without_parsing_or_more_calls(tmp_path: Path) -> None:
    provider = FakeBehaviorProvider(invalid_at=1, finish_reason="length")
    output = tmp_path / "length"
    result = run_behavior_qualification(
        provider,
        output_root=output,
        execution_approval_path=_accepted_execution_approval(tmp_path),
    )
    assert result.status == "failed"
    assert result.error_code == "source_behavior_finish_reason_invalid"
    assert provider.calls == 1
    assert result.parser_invalid_count == 1
    assert result.safe_case_audit_count == 0
    assert not (output / RECEIPT_NAME).exists()


def test_existing_output_blocks_before_provider_call(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")
    provider = FakeBehaviorProvider()
    result = run_behavior_qualification(
        provider,
        output_root=output,
        execution_approval_path=_accepted_execution_approval(tmp_path),
    )
    assert result.status == "blocked"
    assert result.error_code == "source_behavior_output_exists"
    assert provider.calls == 0
    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_receipt_loader_rejects_tampered_safe_audit(tmp_path: Path) -> None:
    output = tmp_path / "run"
    approval = _accepted_execution_approval(tmp_path)
    result = run_behavior_qualification(
        FakeBehaviorProvider(),
        output_root=output,
        execution_approval_path=approval,
    )
    assert result.status == "completed"
    with (output / SAFE_AUDIT_NAME).open("a", encoding="utf-8") as handle:
        handle.write((output / SAFE_AUDIT_NAME).read_text(encoding="utf-8").splitlines()[0])
        handle.write("\n")
    try:
        load_qualification_receipt(
            output / RECEIPT_NAME,
            execution_approval_path=approval,
        )
    except Exception as exc:
        assert "source_behavior_qualification_receipt_binding_mismatch" in str(exc)
    else:
        raise AssertionError("tampered audit must be rejected")


def test_default_cli_policy_does_not_read_environment_or_write_output(tmp_path: Path) -> None:
    class ExplodingEnvironment(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            raise AssertionError("environment must not be read")

        def __iter__(self):
            raise AssertionError("environment must not be read")

        def __len__(self) -> int:
            raise AssertionError("environment must not be read")

        def get(self, key: str, default=None):
            raise AssertionError("environment must not be read")

    result = execute_behavior_qualification(
        allow_network=False,
        environment=ExplodingEnvironment(),
        output_root=tmp_path / "must-not-exist",
    )
    assert isinstance(result, dict)
    assert result["status"] == "source_behavior_execution_gate_offline_ready"
    assert result["blocking_reasons"] == [
        "behavior_qualification_network_authorization_required"
    ]
    assert result["api_key_read"] is False
    assert result["outputs_written"] is False
    assert not list(tmp_path.iterdir())


def test_model_mismatch_blocks_before_provider_construction(tmp_path: Path) -> None:
    result = execute_behavior_qualification(
        allow_network=True,
        confirm_run=True,
        confirm_request_cap=12,
        confirm_completion_reservation_cap=1536,
        environment={
            "EVICON_LLM_BASE_URL": "https://provider.invalid/v1",
            "EVICON_LLM_MODEL": "wrong-model",
            "EVICON_LLM_API_KEY": "must-not-be-used",
        },
        output_root=tmp_path / "must-not-exist",
    )
    assert not isinstance(result, dict)
    assert result.status == "blocked"
    assert result.error_code == "source_behavior_provider_model_mismatch"
    assert result.provider_constructed is False
    assert result.api_key_read is False
    assert not list(tmp_path.iterdir())


def test_explicit_cap_confirmation_is_required_before_approval_or_environment() -> None:
    result = execute_behavior_qualification(allow_network=True)
    assert not isinstance(result, dict)
    assert result.status == "blocked"
    assert result.error_code == "behavior_qualification_explicit_cap_confirmation_required"
