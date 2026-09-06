from __future__ import annotations

import json
from pathlib import Path

import pytest

from evicon.cascade_agent_protocol_v2 import V2_TEMPLATE_VERSION, load_hd2_config
from evicon.cascade_agent_protocol_v2_diagnostic_smoke import run_smoke
from evicon.cascade_agent_protocol_v2_failure_diagnostics import (
    FINISH_REASON_NOT_PERSISTED,
    HD2ParserDiagnosticError,
    ParserFailureAssessment,
    classify_parser_failure,
    diagnose_persisted_failure,
    write_diagnostic_receipt,
)
from evicon.cascade_agent_protocol_v2_pilot import (
    HD2_CONFIG_RELATIVE,
    HD2_PROTOCOL_RELATIVE,
    HD2PilotBatchRecord,
    HD2PilotRunState,
    HD2PilotRunStatus,
    sha256_file,
)
from evicon.cascade_real_agent_runner import CascadeAgentCheckpoint
from evicon.request_ledger import RequestLedger, RequestLedgerEntry, RequestLedgerStatus

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / HD2_CONFIG_RELATIVE
PROTOCOL = ROOT / HD2_PROTOCOL_RELATIVE
FINGERPRINT = "c" * 64


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    config, _ = load_hd2_config(CONFIG)
    spec = config.runs[3]
    states = []
    for item in config.runs:
        states.append(HD2PilotRunState(
            run_id=item.run_id,
            scenario_id=item.scenario_id,
            seed=item.seed,
            condition=item.condition.value,
            status=HD2PilotRunStatus.FAILED if item.run_id == spec.run_id else HD2PilotRunStatus.PLANNED,
            error_code="malformed_json" if item.run_id == spec.run_id else None,
        ))
    root = tmp_path / "pilot"
    batch_path = root / "pilot_batch_record.json"
    batch_path.parent.mkdir(parents=True)
    batch = HD2PilotBatchRecord(
        study_id=config.study_id,
        status=HD2PilotRunStatus.FAILED,
        binding_sha256="a" * 64,
        config_sha256=sha256_file(CONFIG),
        protocol_sha256=sha256_file(PROTOCOL),
        template_sha256=sha256_file(PROTOCOL),
        model_name="synthetic-model",
        output_root=str(root),
        runs=tuple(states),
        failure_code="malformed_json",
    )
    batch_path.write_text(batch.model_dump_json(indent=2), encoding="utf-8")
    run_dir = root / spec.run_id
    ledger_path = run_dir / "request_ledger.jsonl"
    ledger = RequestLedger(ledger_path)
    common = dict(
        request_id="safe-request-id",
        fingerprint=FINGERPRINT,
        request_key="d" * 64,
        model_name="synthetic-model",
        template_version=V2_TEMPLATE_VERSION,
        public_content_sha256="e" * 64,
        condition=spec.condition.value,
        phase="agent_turn",
        agent_id="network-agent-01",
        round_id=1,
        attempt_count=1,
        reserved_tokens=256,
    )
    ledger.append(RequestLedgerEntry(**common, status=RequestLedgerStatus.STARTED))
    ledger.append(RequestLedgerEntry(
        **common,
        status=RequestLedgerStatus.COMPLETED,
        prompt_tokens=419,
        completion_tokens=256,
        total_tokens=675,
        latency_ms=2.0,
    ))
    checkpoint_path = run_dir / "agent_checkpoint.json"
    checkpoint_path.write_text(CascadeAgentCheckpoint(binding_sha256="b" * 64).model_dump_json(), encoding="utf-8")
    return root, batch_path, ledger_path, checkpoint_path


def test_historical_safe_facts_are_classified_as_truncation_consistent() -> None:
    assessment, reasons = classify_parser_failure(
        parser_error_category="malformed_json",
        finish_reason=FINISH_REASON_NOT_PERSISTED,
        completion_tokens=256,
        configured_max_tokens=256,
    )
    assert assessment is ParserFailureAssessment.TRUNCATION_MORE_CONSISTENT
    assert "completion_tokens_equal_configured_max" in reasons
    assert "finish_reason_unavailable" in reasons


def test_non_limit_malformed_stop_is_classified_as_incidental() -> None:
    assessment, reasons = classify_parser_failure(
        parser_error_category="malformed_json",
        finish_reason="stop",
        completion_tokens=24,
        configured_max_tokens=256,
    )
    assert assessment is ParserFailureAssessment.INCIDENTAL_MALFORMED_MORE_CONSISTENT
    assert "finish_reason_non_length" in reasons


def test_diagnostic_reads_only_safe_records_and_preserves_inputs(tmp_path) -> None:
    root, batch_path, ledger_path, checkpoint_path = _fixture(tmp_path)
    before = {path: sha256_file(path) for path in (batch_path, ledger_path, checkpoint_path)}
    diagnostic = diagnose_persisted_failure(result_root=root, fingerprint=FINGERPRINT, config_path=CONFIG)
    assert diagnostic.finish_reason == FINISH_REASON_NOT_PERSISTED
    assert diagnostic.finish_reason_available is False
    assert diagnostic.completion_token_limit_reached is True
    assert diagnostic.parser_error_category == "malformed_json"
    assert diagnostic.response_format_mode == "json_schema"
    assert diagnostic.assessment is ParserFailureAssessment.TRUNCATION_MORE_CONSISTENT
    assert diagnostic.failed_fingerprint_checkpointed is False
    assert diagnostic.network_used_for_diagnostic is False
    assert before == {path: sha256_file(path) for path in before}


def test_diagnostic_receipt_is_no_overwrite_and_content_free(tmp_path) -> None:
    root, _batch, _ledger, _checkpoint = _fixture(tmp_path)
    diagnostic = diagnose_persisted_failure(result_root=root, fingerprint=FINGERPRINT, config_path=CONFIG)
    output = root / "hd2_parser_failure_diagnostic.json"
    digest = write_diagnostic_receipt(diagnostic, output)
    assert digest == sha256_file(output)
    assert write_diagnostic_receipt(diagnostic, output) == digest
    raw = output.read_text(encoding="utf-8").lower()
    for forbidden in (
        "system_prompt", "user_prompt", "api_key", "authorization", "provider_metadata",
        "ground_truth_label", "source_independence_label", "raw_response",
    ):
        assert forbidden not in raw
    changed = diagnostic.model_copy(update={"fingerprint": "f" * 64})
    with pytest.raises(HD2ParserDiagnosticError) as error:
        write_diagnostic_receipt(changed, output)
    assert error.value.code == "diagnostic_receipt_mismatch"


def test_checkpointed_failed_fingerprint_is_rejected(tmp_path) -> None:
    root, _batch, _ledger, checkpoint_path = _fixture(tmp_path)
    payload = json.loads(checkpoint_path.read_text())
    payload["entries"] = [{
        "fingerprint": FINGERPRINT,
        "request_id": "safe-request-id",
        "agent_id": "network-agent-01",
        "round_id": 1,
        "stance": "uncertain",
        "content_ids_used": [],
        "evidence_ids_used": [],
        "share_content_id": None,
    }]
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(HD2ParserDiagnosticError) as error:
        diagnose_persisted_failure(result_root=root, fingerprint=FINGERPRINT, config_path=CONFIG)
    assert error.value.code == "failed_fingerprint_was_checkpointed"


def test_fakeprovider_diagnostic_smoke_covers_all_required_cases() -> None:
    summary = run_smoke()
    assert summary["status"] == "passed"
    assert summary["network"] == "disabled"
    assert summary["automatic_retry_performed"] is False
    cases = {case["case"]: case for case in summary["cases"]}
    assert cases["max_token_truncation"]["assessment"] == "truncation_more_consistent_but_not_confirmed"
    assert cases["complete_strict_json"]["parser_valid"] is True
    assert cases["occasional_malformed_json_first_observation"]["assessment"] == "incidental_malformed_json_more_consistent"
    assert cases["occasional_malformed_json_first_observation"]["automatic_retry"] is False
    accepted = cases["response_format_accepted_but_truncated"]
    assert accepted["response_format_requested_in_payload"] is True
    assert accepted["transport_call_count"] == 1
    assert accepted["parser_valid"] is False
    dumped = json.dumps(summary, sort_keys=True).lower()
    for forbidden in ("system_prompt", "user_prompt", "api_key", "provider_metadata", "ground_truth_label"):
        assert forbidden not in dumped
