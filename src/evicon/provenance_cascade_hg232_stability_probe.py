"""Independent real/FakeProvider protocol-stability probe for H-G.2.3.2."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cascade_agent_protocol_hg232 import HG232_RESPONSE_JSON_SCHEMA, HG232_SCHEMA_NAME
from .cascade_agent_protocol_hg232_audit import load_response_audit
from .cascade_real_agent_runner import CascadeRealAgentRunError
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ResponseFormatMode
from .provenance_cascade_hg2 import sha256_file
from .provenance_cascade_hg21_calibration import HG21RunSpec, _atomic
from .provenance_cascade_hg23 import HG23FakeProvider
from .provenance_cascade_hg232_calibration import HG232CalibrationRunner
from .provenance_cascade_hg232_compatibility_receipt import validate_receipt as validate_compatibility_receipt
from .provenance_cascade_hg232_stability_contract import (
    DEFAULT_PROBE_APPROVAL,
    DEFAULT_PROBE_CONFIG,
    DEFAULT_PROBE_RECEIPT,
    HG232ProbeApproval,
    HG232ProbeError,
    HG232ProbeReceipt,
    _path,
    load_probe_approval,
    load_probe_config,
)
from .provenance_cascade_preregistration import CascadeCondition
from .request_ledger import RequestLedger

PROBE_RUNNER_VERSION = "provenance_cascade_hg232_protocol_stability_probe_runner.v1"
_FORBIDDEN_KEYS = {
    "system_prompt", "user_prompt", "prompt", "raw_response", "response_content",
    "api_key", "authorization", "headers", "provider_metadata",
    "ground_truth_label", "source_independence_label", "evaluator_truth",
}


def _contains_forbidden(value: Any) -> bool:
    if isinstance(value, dict):
        return any(str(key).lower() in _FORBIDDEN_KEYS or _contains_forbidden(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return "bearer " in lowered or "ground_truth_label" in lowered or "source_independence_label" in lowered
    return False


def _safe_scan(root: Path) -> bool:
    for path in root.rglob("*"):
        if not path.is_file() or path.name == "protocol_stability_probe_receipt.json":
            continue
        try:
            text = path.read_text(encoding="utf-8")
            values = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else [json.loads(text)]
        except Exception:
            return False
        if any(_contains_forbidden(value) for value in values):
            return False
    return True


def _probe_spec(config) -> HG21RunSpec:
    return HG21RunSpec(
        run_id=config.run_id,
        matched_group_id="hg232-stability-true-minority-correction-20261031",
        scenario_id=config.scenario_id,
        seed=config.seed,
        condition=CascadeCondition.GENERIC_DISSENT,
        expected_provider_requests=config.logical_request_cap,
        completion_reservation=config.completion_reservation_cap,
    )


def _replay_statuses(record) -> tuple[str, str, str]:
    if record.replay is None:
        raise HG232ProbeError("protocol_stability_probe_replay_missing")
    outcome = record.replay.status.value
    cascade = record.replay.cascade_replay.status.value
    application = record.replay.application_replay.status.value
    if (cascade, application, outcome) != ("passed", "passed", "passed"):
        raise HG232ProbeError("protocol_stability_probe_replay_failed")
    return cascade, application, outcome


def run_probe(
    *,
    provider,
    root: str | Path,
    model_name: str,
    network: str,
    write_receipt: bool = True,
) -> HG232ProbeReceipt:
    config, config_path = load_probe_config()
    root_path = Path(root).resolve()
    if root_path.exists():
        raise HG232ProbeError("protocol_stability_probe_output_exists")
    root_path.mkdir(parents=True, exist_ok=False)
    parent = HG232CalibrationRunner()
    spec = _probe_spec(config)
    parent.runs = (spec,)
    try:
        record, decisions = parent.run_one(
            spec,
            provider=provider,
            root=root_path,
            model_name=model_name,
            resume=False,
        )
        if len(decisions) != 18:
            raise HG232ProbeError("protocol_stability_probe_decisions_incomplete")
        run_dir = root_path / spec.run_id
        binding = parent._run_binding(spec, model_name)
        audits = load_response_audit(run_dir / "response_audit.jsonl", binding)
        if len(audits) != 18:
            raise HG232ProbeError("protocol_stability_probe_response_audit_incomplete")
        if any(item.status != "completed" or item.parser_valid is not True for item in audits):
            raise HG232ProbeError("protocol_stability_probe_parser_invalid")
        if any(item.finish_reason == "length" for item in audits):
            raise HG232ProbeError("protocol_stability_probe_finish_reason_length")
        if any(item.completion_token_limit_reached is True for item in audits):
            raise HG232ProbeError("protocol_stability_probe_token_limit_reached")
        if any(item.finish_reason != "stop" for item in audits):
            raise HG232ProbeError("protocol_stability_probe_finish_reason_invalid")
        cascade_status, application_status, outcome_status = _replay_statuses(record)
        ledger_path = run_dir / "request_ledger.jsonl"
        ledger = RequestLedger(ledger_path).summary(
            request_cap=config.logical_request_cap,
            completion_reservation_cap=config.completion_reservation_cap,
        )
        if (
            ledger.unique_logical_request_count != 18
            or ledger.provider_call_count != 18
            or ledger.transport_attempt_count != 18
        ):
            raise HG232ProbeError("protocol_stability_probe_request_count_mismatch")
        run_record_path = run_dir / "run_record.json"
        _atomic(run_record_path, record.model_copy(update={"request_ledger_path": "request_ledger.jsonl"}))
        if not _safe_scan(root_path):
            raise HG232ProbeError("protocol_stability_probe_sensitive_scan_failed")
        _, approval_path = load_probe_approval()
        receipt = HG232ProbeReceipt(
            receipt_id="provenance-cascade-hg232-protocol-stability-probe-receipt-v1",
            status="passed",
            probe_config_sha256=sha256_file(config_path),
            parent_config_sha256=config.parent_config_sha256,
            protocol_sha256=config.protocol_sha256,
            compatibility_receipt_sha256=config.compatibility_receipt_sha256,
            calibration_runner_sha256=config.calibration_runner_sha256,
            probe_runner_sha256=sha256_file(__file__),
            approval_sha256=sha256_file(approval_path),
            run_id=config.run_id,
            scenario_id=config.scenario_id,
            condition=config.condition,
            seed=config.seed,
            logical_request_count=18,
            provider_call_count=18,
            transport_attempt_count=18,
            completion_reservation_cap=36864,
            max_tokens=2048,
            response_audit_count=18,
            finish_reason_stop_count=18,
            completion_limit_reached_count=0,
            cascade_replay_status=cascade_status,
            application_replay_status=application_status,
            outcome_replay_status=outcome_status,
            sensitive_scan_status="passed",
            request_ledger_sha256=sha256_file(ledger_path),
            run_record_sha256=sha256_file(run_record_path),
            response_audit_sha256=sha256_file(run_dir / "response_audit.jsonl"),
            network=network,
            development_only=True,
            probe_only=True,
            merged_with_calibration=False,
            private_truth_exposed=False,
            not_paper_result=True,
            no_causal_conclusion=True,
        )
        if write_receipt:
            receipt_path = root_path / "protocol_stability_probe_receipt.json"
            if receipt_path.exists():
                raise HG232ProbeError("protocol_stability_probe_receipt_exists")
            _atomic(receipt_path, receipt)
        return receipt
    except Exception:
        raise


def preflight() -> dict[str, object]:
    reasons: list[str] = []
    try:
        config, config_path = load_probe_config()
        approval, _ = load_probe_approval()
        expected = {
            "config_sha256": sha256_file(config_path),
            "protocol_sha256": config.protocol_sha256,
            "compatibility_receipt_sha256": config.compatibility_receipt_sha256,
            "calibration_runner_sha256": config.calibration_runner_sha256,
            "probe_runner_sha256": sha256_file(__file__),
        }
        if any(getattr(approval, key) != value for key, value in expected.items()):
            reasons.append("protocol_stability_probe_approval_binding_mismatch")
        if approval.acceptance_status != "accepted" or not approval.network_execution_authorized:
            reasons.append("protocol_stability_probe_network_authorization_required")
        validate_compatibility_receipt(config.compatibility_receipt_path, expected_hash=config.compatibility_receipt_sha256)
        if _path(config.output_root).exists():
            reasons.append("protocol_stability_probe_output_exists")
        return {
            "status": "ready_for_protocol_stability_probe" if not reasons else "blocked",
            "ready_for_protocol_stability_probe": not reasons,
            "blocking_reasons": sorted(set(reasons)),
            "probe_id": config.probe_id,
            "run_id": config.run_id,
            "scenario_id": config.scenario_id,
            "condition": config.condition,
            "run_count": 1,
            "logical_request_cap": 18,
            "completion_reservation_cap": 36864,
            "max_tokens": 2048,
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "private_truth_exposed": False,
            "not_paper_result": True,
            "no_causal_conclusion": True,
        }
    except Exception as exc:
        return {
            "status": "blocked",
            "ready_for_protocol_stability_probe": False,
            "blocking_reasons": [getattr(exc, "code", "protocol_stability_probe_preflight_failed")],
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
        }


def run_fake_smoke() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="evicon-hg232-stability-probe-") as temporary:
        receipt = run_probe(
            provider=HG23FakeProvider(),
            root=Path(temporary) / "probe",
            model_name="gpt-5.6-luna",
            network="disabled",
            write_receipt=True,
        )
    return {
        "status": "fake_probe_passed",
        "run_count": 1,
        "logical_request_count": receipt.logical_request_count,
        "provider_call_count": receipt.provider_call_count,
        "transport_attempt_count": receipt.transport_attempt_count,
        "completion_reservation_cap": receipt.completion_reservation_cap,
        "cascade_replay_status": receipt.cascade_replay_status,
        "application_replay_status": receipt.application_replay_status,
        "outcome_replay_status": receipt.outcome_replay_status,
        "sensitive_scan_status": receipt.sensitive_scan_status,
        "network": "disabled",
        "results_written": False,
        "private_truth_exposed": False,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def execute_real(
    *,
    allow_network: bool = False,
    confirm_run: bool = False,
    confirm_request_cap: int | None = None,
    confirm_completion_reservation_cap: int | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    checks = (
        (allow_network, "allow_network_required"),
        (confirm_run, "confirm_run_required"),
        (confirm_request_cap == 18, "confirm_request_cap_must_equal_18"),
        (confirm_completion_reservation_cap == 36864, "confirm_completion_reservation_cap_must_equal_36864"),
    )
    for valid, code in checks:
        if not valid:
            return {"status": "blocked", "error_code": code, "provider_constructed": False, "network": "disabled"}
    gate = preflight()
    if not gate["ready_for_protocol_stability_probe"]:
        return {"status": "blocked", "error_code": gate["blocking_reasons"][0], "provider_constructed": False, "network": "disabled"}
    source = os.environ if environment is None else environment
    if not all(source.get(key) for key in ("EVICON_LLM_BASE_URL", "EVICON_LLM_MODEL", "EVICON_LLM_API_KEY")):
        return {"status": "blocked", "error_code": "provider_environment_incomplete", "provider_constructed": False, "network": "disabled"}
    config, _ = load_probe_config()
    compatibility = validate_compatibility_receipt(config.compatibility_receipt_path, expected_hash=config.compatibility_receipt_sha256)
    if source["EVICON_LLM_MODEL"] != compatibility.model:
        return {"status": "blocked", "error_code": "provider_model_mismatch", "provider_constructed": False, "network": "disabled"}
    base = ProviderConfig.from_env(allow_network=True, environment=source)
    settings = base.model_copy(update={
        "timeout_seconds": config.timeout_seconds,
        "max_retries": config.max_retries,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "seed": config.seed,
        "reasoning_effort": None,
        "response_format": ResponseFormatMode.JSON_SCHEMA,
        "response_schema_name": HG232_SCHEMA_NAME,
        "response_schema": HG232_RESPONSE_JSON_SCHEMA,
    })
    provider = OpenAICompatibleProvider(settings, environment=source)
    try:
        receipt = run_probe(
            provider=provider,
            root=_path(config.output_root),
            model_name=compatibility.model,
            network="enabled",
            write_receipt=True,
        )
    except (HG232ProbeError, CascadeRealAgentRunError) as exc:
        return {
            "status": "failed",
            "error_code": getattr(exc, "code", "protocol_stability_probe_failed"),
            "provider_constructed": True,
            "network": "enabled",
            "receipt_written": False,
        }
    return {
        "status": "passed",
        "run_count": 1,
        "logical_request_count": receipt.logical_request_count,
        "transport_attempt_count": receipt.transport_attempt_count,
        "receipt_path": DEFAULT_PROBE_RECEIPT,
        "provider_constructed": True,
        "network": "enabled",
        "private_truth_exposed": False,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated H-G.2.3.2 protocol-stability probe")
    parser.add_argument("--mode", choices=("preflight", "fake-smoke", "real-probe"), default="preflight")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--confirm-request-cap", type=int)
    parser.add_argument("--confirm-completion-reservation-cap", type=int)
    args = parser.parse_args(argv)
    if args.mode == "preflight":
        payload = preflight()
    elif args.mode == "fake-smoke":
        payload = run_fake_smoke()
    else:
        payload = execute_real(
            allow_network=args.allow_network,
            confirm_run=args.confirm_run,
            confirm_request_cap=args.confirm_request_cap,
            confirm_completion_reservation_cap=args.confirm_completion_reservation_cap,
        )
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if payload.get("status") in {"ready_for_protocol_stability_probe", "fake_probe_passed", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
