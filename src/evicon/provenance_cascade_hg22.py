"""Offline governance for the H-G.2.2 explicit share-contract amendment."""
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, model_validator

from .cascade_agent_protocol_hg2 import HG2PromptContext
from .cascade_agent_protocol_hg22 import (
    HG22ParserError,
    HG22Runtime,
    HG22_PROTOCOL_VERSION,
    HG22_RESPONSE_JSON_SCHEMA,
    HG22_SCHEMA_NAME,
    HG22_TEMPLATE_VERSION,
    parse_hg22_response,
    render_hg22_turn,
)
from .llm_contract import LLMResponse
from .provenance_cascade_hg2 import sha256_file
from .provenance_cascade_hg2_compatibility import build_minimal_hg2_context

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg22_calibration.v1.toml"
DEFAULT_APPROVAL = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg22_approval_template.toml"
DEFAULT_AMENDMENT_RECEIPT = "outputs/study-locks/provenance_cascade_hg22_share_contract_amendment_receipt.json"
DEFAULT_COMPATIBILITY_RECEIPT = "outputs/study-locks/provenance_cascade_hg22_compatibility_receipt.json"


class HG22Error(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


class HG22Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    study_id: Literal["evicon-provenance-cascade-hg22-share-contract-calibration"]
    config_version: Literal["provenance_cascade_hg22_share_contract.v1"]
    status: Literal["pending_exact_hash_approval_and_compatibility"]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    parent_execution_config_path: str
    parent_execution_config_sha256: str
    parent_protocol_path: str
    parent_protocol_sha256: str
    protocol_path: str
    protocol_sha256: str
    amendment_path: str
    amendment_sha256: str
    failed_hg21v2_batch_record_path: str
    failed_hg21v2_batch_record_sha256: str
    failed_hg21v2_request_ledger_path: str
    failed_hg21v2_request_ledger_sha256: str
    failed_hg21v2_checkpoint_path: str
    failed_hg21v2_checkpoint_sha256: str
    failed_hg21v2_decisions_path: str
    failed_hg21v2_decisions_sha256: str
    failed_hg21v2_parser_category: Literal["share_field"]
    failed_hg21v2_must_not_resume: Literal[True]
    diagnostic_missing_field_ruled_out: Literal[True]
    diagnostic_behavior_field_type_ruled_out: Literal[True]
    diagnostic_share_type_vs_semantic_not_losslessly_distinguishable: Literal[True]
    diagnostic_most_consistent_with: Literal["share_behavior_inconsistent"]
    protocol_version: Literal["provenance_cascade_agent_protocol.hg2_2_share_contract.v1"]
    template_version: Literal["cascade_agent_turn.hg2_epistemic_behavior.share_contract.v2"]
    response_format: Literal["json_schema"]
    response_schema_name: Literal["cascade_agent_epistemic_behavior_share_contract_v2"]
    structural_schema_only: Literal[True]
    local_semantic_validation_required: Literal[True]
    parser_invalid_stops: Literal[True]
    parser_recovery_enabled: Literal[False]
    prompt_semantics_changed_only_for_share_contract: Literal[True]
    epistemic_stances_changed: Literal[False]
    behavioral_decisions_changed: Literal[False]
    scenario_design_changed: Literal[False]
    roles_changed: Literal[False]
    controller_rules_changed: Literal[False]
    metric_definitions_changed: Literal[False]
    required_model_name: Literal["gpt-5.6-luna"]
    seed: Literal[20261021]
    scenario_count: Literal[4]
    condition_count: Literal[4]
    agent_count: Literal[6]
    round_count: Literal[3]
    run_count: Literal[16]
    matched_group_count: Literal[4]
    logical_requests_per_run: Literal[18]
    request_cap: Literal[288]
    agent_max_tokens: Literal[1024]
    completion_reservation_cap: Literal[294912]
    temperature: Literal[0.2]
    compatibility_max_retries: Literal[0]
    compatibility_timeout_seconds: Literal[5]
    calibration_max_retries: Literal[0]
    calibration_timeout_seconds: Literal[15]
    run_id_namespace: Literal["hg22"]
    output_root: Literal["results/provenance-cascade-hg22-eligibility-calibration-v1"]
    compatibility_receipt_path: Literal["outputs/study-locks/provenance_cascade_hg22_compatibility_receipt.json"]
    full_pilot_authorized: Literal[False]

    @model_validator(mode="after")
    def fixed_scope(self) -> "HG22Config":
        if self.request_cap != self.run_count * self.logical_requests_per_run:
            raise ValueError("request cap mismatch")
        if self.completion_reservation_cap != self.request_cap * self.agent_max_tokens:
            raise ValueError("completion reservation mismatch")
        return self


class HG22Approval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    approval_id: Literal["provenance-cascade-hg22-share-contract-approval-v1"]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    config_sha256: str
    protocol_sha256: str
    template_version: Literal["cascade_agent_turn.hg2_epistemic_behavior.share_contract.v2"]
    amendment_sha256: str
    amendment_receipt_sha256: str
    compatibility_receipt_sha256: str
    response_format: Literal["json_schema"]
    response_schema_name: Literal["cascade_agent_epistemic_behavior_share_contract_v2"]
    run_count: Literal[16]
    matched_group_count: Literal[4]
    logical_request_cap: Literal[288]
    completion_reservation_cap: Literal[294912]
    confirm_share_contract: bool
    confirm_structural_semantic_layering: bool
    confirm_parser_invalid_stops: bool
    confirm_hg21_failures_immutable: bool
    confirm_no_effectiveness_claim: bool
    network_execution_authorized: Literal[False]

    @model_validator(mode="after")
    def accepted_shape(self) -> "HG22Approval":
        confirmations = (
            self.confirm_share_contract,
            self.confirm_structural_semantic_layering,
            self.confirm_parser_invalid_stops,
            self.confirm_hg21_failures_immutable,
            self.confirm_no_effectiveness_claim,
        )
        if self.acceptance_status == "accepted" and (not self.accepted_by.strip() or not all(confirmations)):
            raise ValueError("accepted approval incomplete")
        return self


class HG22AmendmentReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    receipt_id: Literal["provenance-cascade-hg22-share-contract-amendment-v1"]
    status: Literal["offline_validated_pending_exact_hash_approval"]
    config_sha256: str
    protocol_sha256: str
    amendment_sha256: str
    failed_batch_record_sha256: str
    failed_request_ledger_sha256: str
    failed_checkpoint_sha256: str
    failed_decisions_sha256: str
    missing_field_ruled_out: Literal[True]
    behavior_field_type_ruled_out: Literal[True]
    share_type_vs_semantic_not_losslessly_distinguishable: Literal[True]
    most_consistent_with: Literal["share_behavior_inconsistent"]
    structural_schema_only: Literal[True]
    local_semantic_validation_required: Literal[True]
    fake_case_count: Literal[7]
    fake_case_passed_count: Literal[7]
    run_count: Literal[16]
    matched_group_count: Literal[4]
    logical_request_cap: Literal[288]
    completion_reservation_cap: Literal[294912]
    network: Literal["disabled"]
    results_written: Literal[False]
    private_truth_exposed: Literal[False]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]


def load_config(path: str | Path = DEFAULT_CONFIG) -> tuple[HG22Config, Path]:
    config_path = _path(path)
    try:
        config = HG22Config.model_validate(tomllib.loads(config_path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise HG22Error("hg22_config_invalid") from exc
    bindings = (
        (config.parent_execution_config_path, config.parent_execution_config_sha256),
        (config.parent_protocol_path, config.parent_protocol_sha256),
        (config.protocol_path, config.protocol_sha256),
        (config.amendment_path, config.amendment_sha256),
        (config.failed_hg21v2_batch_record_path, config.failed_hg21v2_batch_record_sha256),
        (config.failed_hg21v2_request_ledger_path, config.failed_hg21v2_request_ledger_sha256),
        (config.failed_hg21v2_checkpoint_path, config.failed_hg21v2_checkpoint_sha256),
        (config.failed_hg21v2_decisions_path, config.failed_hg21v2_decisions_sha256),
    )
    if any(sha256_file(_path(file_path)) != digest for file_path, digest in bindings):
        raise HG22Error("hg22_binding_hash_mismatch")
    return config, config_path


class _FakeProvider:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0

    def complete(self, request) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content=json.dumps(self.payload),
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            latency_ms=1.0,
        )


def _payload(context: HG2PromptContext, **updates: object) -> dict[str, object]:
    content_id = context.public_context.contents[0].content_id
    payload: dict[str, object] = {
        "epistemic_stance": "uncertain",
        "behavioral_decision": "share_with_caveat",
        "content_ids_used": [content_id],
        "evidence_ids_used": [],
        "share_content_id": content_id,
    }
    payload.update(updates)
    return payload


def run_fake_smoke() -> dict[str, object]:
    context = build_minimal_hg2_context()
    content_id = context.public_context.contents[0].content_id
    cases = (
        ("legal_share", _payload(context), None),
        ("legal_do_not_share", _payload(context, behavioral_decision="do_not_share", share_content_id=None), None),
        ("legal_defer", _payload(context, behavioral_decision="defer_action", share_content_id=None), None),
        ("missing_share_field", {key: value for key, value in _payload(context).items() if key != "share_content_id"}, HG22ParserError.MISSING_FIELD),
        ("share_type", _payload(context, share_content_id=7), HG22ParserError.SHARE_FIELD_TYPE),
        ("sharing_without_id", _payload(context, share_content_id=None), HG22ParserError.SHARE_BEHAVIOR_INCONSISTENT),
        ("nonsharing_with_id", _payload(context, behavioral_decision="do_not_share", share_content_id=content_id), HG22ParserError.SHARE_BEHAVIOR_INCONSISTENT),
    )
    passed = 0
    for _, payload, expected in cases:
        parsed, error = parse_hg22_response(json.dumps(payload), context)
        if (expected is None and parsed is not None and error is None) or (expected is not None and parsed is None and error is expected):
            passed += 1
    runtime = HG22Runtime()
    result = runtime.execute(context, _FakeProvider(_payload(context)))
    request = render_hg22_turn(context)
    return {
        "status": "offline_fake_smoke_passed" if passed == len(cases) and result.response is not None else "failed",
        "case_count": len(cases),
        "case_passed_count": passed,
        "runtime_parser_valid": result.audit.parser_valid,
        "protocol_version": HG22_PROTOCOL_VERSION,
        "template_version": HG22_TEMPLATE_VERSION,
        "schema_name": HG22_SCHEMA_NAME,
        "schema_structural_only": "if" not in json.dumps(HG22_RESPONSE_JSON_SCHEMA),
        "local_semantic_validation": True,
        "request_id_versioned": request.request_id.startswith("hg22-"),
        "network": "disabled",
        "provider_constructed": False,
        "api_key_read": False,
        "results_written": False,
        "private_truth_exposed": False,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def preflight(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    try:
        config, config_path = load_config(path)
        receipt_path, approval_path = _path(DEFAULT_AMENDMENT_RECEIPT), _path(DEFAULT_APPROVAL)
        receipt = HG22AmendmentReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
        approval = HG22Approval.model_validate(tomllib.loads(approval_path.read_text(encoding="utf-8")))
        expected_receipt = {
            "config_sha256": sha256_file(config_path),
            "protocol_sha256": config.protocol_sha256,
            "amendment_sha256": config.amendment_sha256,
            "failed_batch_record_sha256": config.failed_hg21v2_batch_record_sha256,
            "failed_request_ledger_sha256": config.failed_hg21v2_request_ledger_sha256,
            "failed_checkpoint_sha256": config.failed_hg21v2_checkpoint_sha256,
            "failed_decisions_sha256": config.failed_hg21v2_decisions_sha256,
        }
        if any(getattr(receipt, key) != value for key, value in expected_receipt.items()):
            raise HG22Error("hg22_amendment_receipt_mismatch")
        expected_approval = {
            "config_sha256": sha256_file(config_path),
            "protocol_sha256": config.protocol_sha256,
            "amendment_sha256": config.amendment_sha256,
            "amendment_receipt_sha256": sha256_file(receipt_path),
        }
        if any(getattr(approval, key) != value for key, value in expected_approval.items()):
            raise HG22Error("hg22_approval_binding_mismatch")
        smoke = run_fake_smoke()
        if smoke["status"] != "offline_fake_smoke_passed":
            raise HG22Error("hg22_fake_smoke_failed")
        reasons: list[str] = []
        if approval.acceptance_status != "accepted":
            reasons.append("exact_hash_human_approval_required")
        compatibility_path = _path(config.compatibility_receipt_path)
        if not compatibility_path.is_file() or not approval.compatibility_receipt_sha256:
            reasons.append("provider_compatibility_check_required")
        elif sha256_file(compatibility_path) != approval.compatibility_receipt_sha256:
            reasons.append("provider_compatibility_receipt_hash_mismatch")
        reasons.append("real_eligibility_calibration_not_authorized")
        if _path(config.output_root).exists():
            reasons.append("output_root_exists")
        return {
            "status": "blocked",
            "blocking_reasons": reasons,
            "ready_for_real_calibration": False,
            "config_sha256": sha256_file(config_path),
            "protocol_sha256": config.protocol_sha256,
            "amendment_receipt_sha256": sha256_file(receipt_path),
            "approval_sha256": sha256_file(approval_path),
            "diagnostic": {
                "missing_field_ruled_out": True,
                "behavior_field_type_ruled_out": True,
                "share_type_vs_semantic_not_losslessly_distinguishable": True,
                "most_consistent_with": "share_behavior_inconsistent",
            },
            "run_count": 16,
            "matched_group_count": 4,
            "logical_request_cap": 288,
            "completion_reservation_cap": 294912,
            "run_id_namespace": config.run_id_namespace,
            "output_root": config.output_root,
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "private_truth_exposed": False,
            "development_only": True,
            "calibration_only": True,
            "not_paper_result": True,
            "no_causal_conclusion": True,
        }
    except Exception as exc:
        return {
            "status": "blocked",
            "blocking_reasons": [getattr(exc, "code", "hg22_preflight_failed")],
            "ready_for_real_calibration": False,
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "private_truth_exposed": False,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline H-G.2.2 share-contract preflight")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--fake-smoke", action="store_true")
    args = parser.parse_args(argv)
    payload = run_fake_smoke() if args.fake_smoke else preflight(args.config)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if payload.get("status") == "offline_fake_smoke_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
