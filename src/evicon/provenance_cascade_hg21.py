"""Offline governance and preflight for the H-G.2.1 provider-subset amendment."""
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, model_validator

from .cascade_agent_protocol_hg2 import parse_hg2_response, render_hg2_turn
from .cascade_agent_protocol_hg21 import (
    HG21_PROTOCOL_VERSION,
    HG21_RESPONSE_JSON_SCHEMA,
    HG21_SCHEMA_NAME,
    HG21_TEMPLATE_VERSION,
    render_hg21_turn,
)
from .provenance_cascade_hg2 import run_hg2_fake_calibration, sha256_file
from .provenance_cascade_hg2_compatibility import build_minimal_hg2_context

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg21_calibration.v1.toml"
DEFAULT_APPROVAL = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg21_approval_template.toml"
DEFAULT_AMENDMENT_RECEIPT = "outputs/study-locks/provenance_cascade_hg21_provider_subset_amendment_receipt.json"
DEFAULT_COMPATIBILITY_RECEIPT = "outputs/study-locks/provenance_cascade_hg21_compatibility_receipt.json"


class HG21Error(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


class HG21Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    study_id: Literal["evicon-provenance-cascade-hg21-eligibility-calibration"]
    config_version: Literal["provenance_cascade_hg21_provider_subset.v1"]
    status: Literal["offline_design_pending_exact_hash_approval"]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    parent_study_id: Literal["evicon-provenance-cascade-hg2-eligibility-calibration"]
    parent_config_path: str
    parent_config_sha256: str
    parent_protocol_path: str
    parent_protocol_sha256: str
    parent_approval_path: str
    parent_approval_sha256: str
    parent_amendment_receipt_path: str
    parent_amendment_receipt_sha256: str
    parent_compatibility_status: Literal["response_format_unsupported"]
    parent_compatibility_receipt_created: Literal[False]
    amendment_path: str
    amendment_sha256: str
    protocol_path: str
    protocol_sha256: str
    strict_parser_path: str
    strict_parser_sha256: str
    protocol_version: Literal["provenance_cascade_agent_protocol.hg2_1_provider_subset.v1"]
    template_version: Literal["cascade_agent_turn.hg2_epistemic_behavior.v1"]
    response_format: Literal["json_schema"]
    response_schema_name: Literal["cascade_agent_epistemic_behavior_response_v1_provider_subset"]
    removed_provider_keyword: Literal["uniqueItems"]
    local_duplicate_validation_required: Literal[True]
    prompt_semantics_changed: Literal[False]
    output_contract_changed: Literal[False]
    scenario_design_changed: Literal[False]
    controller_rules_changed: Literal[False]
    metric_definitions_changed: Literal[False]
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
    output_root: Literal["results/provenance-cascade-hg21-eligibility-calibration-v1"]
    real_eligibility_calibration_required: Literal[True]
    full_pilot_authorized: Literal[False]

    @model_validator(mode="after")
    def fixed_scope(self) -> "HG21Config":
        if self.request_cap != self.run_count * self.logical_requests_per_run:
            raise ValueError("request cap mismatch")
        if self.completion_reservation_cap != self.request_cap * self.agent_max_tokens:
            raise ValueError("completion reservation mismatch")
        return self


class HG21Approval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    approval_id: Literal["provenance-cascade-hg21-provider-subset-approval-v1"]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    config_sha256: str
    protocol_sha256: str
    amendment_sha256: str
    amendment_receipt_sha256: str
    compatibility_receipt_sha256: str
    response_format: Literal["json_schema"]
    response_schema_name: Literal["cascade_agent_epistemic_behavior_response_v1_provider_subset"]
    agent_max_tokens: Literal[1024]
    temperature: Literal[0.2]
    seed: Literal[20261021]
    run_count: Literal[16]
    matched_group_count: Literal[4]
    logical_request_cap: Literal[288]
    completion_reservation_cap: Literal[294912]
    confirm_prompt_semantics_unchanged: bool
    confirm_parser_remains_strict: bool
    confirm_only_unique_items_removed: bool
    confirm_parent_hg2_immutable: bool
    confirm_real_calibration_before_full_pilot: bool
    network_execution_authorized: Literal[False]

    @model_validator(mode="after")
    def accepted_shape(self) -> "HG21Approval":
        confirmations = (
            self.confirm_prompt_semantics_unchanged,
            self.confirm_parser_remains_strict,
            self.confirm_only_unique_items_removed,
            self.confirm_parent_hg2_immutable,
            self.confirm_real_calibration_before_full_pilot,
        )
        if self.acceptance_status == "accepted" and (not self.accepted_by.strip() or not all(confirmations)):
            raise ValueError("accepted approval incomplete")
        return self


class HG21AmendmentReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    receipt_id: Literal["provenance-cascade-hg21-provider-subset-amendment-lock-v1"]
    status: Literal["offline_validated_pending_exact_hash_approval"]
    config_sha256: str
    protocol_sha256: str
    amendment_sha256: str
    parent_config_sha256: str
    parent_protocol_sha256: str
    parent_approval_sha256: str
    parent_amendment_receipt_sha256: str
    removed_provider_keyword: Literal["uniqueItems"]
    strict_parser_duplicate_rejection_passed: Literal[True]
    prompt_bytes_unchanged: Literal[True]
    fake_replay_passed_count: Literal[16]
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
    full_pilot_authorized: Literal[False]


def load_config(path: str | Path = DEFAULT_CONFIG) -> tuple[HG21Config, Path]:
    config_path = _path(path)
    try:
        config = HG21Config.model_validate(tomllib.loads(config_path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise HG21Error("hg21_config_invalid") from exc
    bindings = (
        (config.parent_config_path, config.parent_config_sha256),
        (config.parent_protocol_path, config.parent_protocol_sha256),
        (config.parent_approval_path, config.parent_approval_sha256),
        (config.parent_amendment_receipt_path, config.parent_amendment_receipt_sha256),
        (config.amendment_path, config.amendment_sha256),
        (config.protocol_path, config.protocol_sha256),
        (config.strict_parser_path, config.strict_parser_sha256),
    )
    if any(sha256_file(_path(file_path)) != digest for file_path, digest in bindings):
        raise HG21Error("hg21_binding_hash_mismatch")
    if _path("outputs/study-locks/provenance_cascade_hg2_compatibility_receipt.json").exists():
        raise HG21Error("parent_failed_check_receipt_must_not_exist")
    return config, config_path


def run_fake_smoke(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    config, _ = load_config(path)
    context = build_minimal_hg2_context()
    old_request = render_hg2_turn(context)
    amended_request = render_hg21_turn(context)
    if old_request.system_prompt != amended_request.system_prompt or old_request.user_prompt != amended_request.user_prompt:
        raise HG21Error("prompt_semantics_changed")
    if "uniqueItems" in json.dumps(HG21_RESPONSE_JSON_SCHEMA):
        raise HG21Error("unsupported_keyword_retained")
    duplicate = json.dumps({
        "epistemic_stance": "uncertain",
        "behavioral_decision": "share_with_caveat",
        "content_ids_used": ["hg11-compatibility-content-01", "hg11-compatibility-content-01"],
        "evidence_ids_used": [],
        "share_content_id": "hg11-compatibility-content-01",
    })
    parsed, error = parse_hg2_response(duplicate, context)
    if parsed is not None or error is None:
        raise HG21Error("strict_duplicate_parser_regressed")
    inherited = run_hg2_fake_calibration()
    return {
        "status": "offline_fake_smoke_passed",
        "prompt_bytes_unchanged": True,
        "strict_parser_duplicate_rejection_passed": True,
        "provider_schema_unique_items_present": False,
        "schema_name": HG21_SCHEMA_NAME,
        "protocol_version": HG21_PROTOCOL_VERSION,
        "template_version": HG21_TEMPLATE_VERSION,
        "fake_replay_passed_count": inherited["replay_passed_count"],
        "run_count": config.run_count,
        "matched_group_count": config.matched_group_count,
        "logical_request_cap": config.request_cap,
        "completion_reservation_cap": config.completion_reservation_cap,
        "network": "disabled",
        "provider_constructed": False,
        "api_key_read": False,
        "results_written": False,
        "private_truth_exposed": False,
        "effectiveness_claimed": False,
    }


def preflight(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    try:
        config, config_path = load_config(path)
        smoke = run_fake_smoke(path)
        receipt_path = _path(DEFAULT_AMENDMENT_RECEIPT)
        approval_path = _path(DEFAULT_APPROVAL)
        receipt = HG21AmendmentReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
        approval = HG21Approval.model_validate(tomllib.loads(approval_path.read_text(encoding="utf-8")))
        expected = {
            "config_sha256": sha256_file(config_path),
            "protocol_sha256": config.protocol_sha256,
            "amendment_sha256": config.amendment_sha256,
            "parent_config_sha256": config.parent_config_sha256,
            "parent_protocol_sha256": config.parent_protocol_sha256,
            "parent_approval_sha256": config.parent_approval_sha256,
            "parent_amendment_receipt_sha256": config.parent_amendment_receipt_sha256,
        }
        if any(getattr(receipt, key) != value for key, value in expected.items()):
            raise HG21Error("hg21_amendment_receipt_binding_mismatch")
        if (
            approval.config_sha256 != receipt.config_sha256
            or approval.protocol_sha256 != receipt.protocol_sha256
            or approval.amendment_sha256 != receipt.amendment_sha256
            or approval.amendment_receipt_sha256 != sha256_file(receipt_path)
        ):
            raise HG21Error("hg21_approval_binding_mismatch")
        reasons: list[str] = []
        if approval.acceptance_status != "accepted":
            reasons.append("exact_hash_human_approval_required")
        if not approval.compatibility_receipt_sha256:
            reasons.append("provider_compatibility_check_required")
        else:
            try:
                from .provenance_cascade_hg21_compatibility_receipt import validate_receipt
                validate_receipt(DEFAULT_COMPATIBILITY_RECEIPT, expected_hash=approval.compatibility_receipt_sha256)
            except Exception as exc:
                reasons.append(getattr(exc, "code", "provider_compatibility_receipt_invalid"))
        reasons.append("one_seed_real_eligibility_calibration_required")
        if _path(config.output_root).exists():
            reasons.append("output_root_exists")
        return {
            "status": "blocked",
            "blocking_reasons": reasons,
            "ready_for_real_eligibility_calibration": False,
            "full_pilot_authorized": False,
            "config_sha256": sha256_file(config_path),
            "protocol_sha256": config.protocol_sha256,
            "amendment_receipt_sha256": sha256_file(receipt_path),
            "approval_sha256": sha256_file(approval_path),
            "run_count": config.run_count,
            "matched_group_count": config.matched_group_count,
            "logical_request_cap": config.request_cap,
            "completion_reservation_cap": config.completion_reservation_cap,
            "fake_replay_passed_count": smoke["fake_replay_passed_count"],
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
            "blocking_reasons": [getattr(exc, "code", "hg21_preflight_failed")],
            "ready_for_real_eligibility_calibration": False,
            "full_pilot_authorized": False,
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "private_truth_exposed": False,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline H-G.2.1 provider-subset preflight")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--fake-smoke", action="store_true")
    args = parser.parse_args(argv)
    payload = run_fake_smoke(args.config) if args.fake_smoke else preflight(args.config)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_AMENDMENT_RECEIPT", "DEFAULT_APPROVAL", "DEFAULT_COMPATIBILITY_RECEIPT",
    "DEFAULT_CONFIG", "HG21AmendmentReceipt", "HG21Approval", "HG21Config", "HG21Error",
    "load_config", "preflight", "run_fake_smoke",
]

if __name__ == "__main__":
    raise SystemExit(main())
