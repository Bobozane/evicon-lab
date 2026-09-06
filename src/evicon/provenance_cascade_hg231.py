"""Offline governance for the H-G.2.3.1 Provider-subset amendment."""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from .cascade_agent_protocol_hg23 import HG23_RESPONSE_JSON_SCHEMA, parse_hg23_response, render_hg23_turn
from .cascade_agent_protocol_hg231 import (
    HG231_PROTOCOL_VERSION, HG231_RESPONSE_JSON_SCHEMA, HG231_SCHEMA_NAME,
    HG231_TEMPLATE_VERSION, render_hg231_turn,
)
from .provenance_cascade_hg2 import sha256_file
from .provenance_cascade_hg23 import run_fake_smoke as run_hg23_fake_smoke
from .provenance_cascade_hg23_compatibility import build_minimal_hg23_context

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg231_calibration.v1.toml"
DEFAULT_APPROVAL = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg231_approval_template.toml"
DEFAULT_AMENDMENT_RECEIPT = "outputs/study-locks/provenance_cascade_hg231_provider_subset_amendment_receipt.json"
DEFAULT_COMPATIBILITY_RECEIPT = "outputs/study-locks/provenance_cascade_hg231_compatibility_receipt.json"


class HG231Error(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


class HG231Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    study_id: Literal["evicon-provenance-cascade-hg231-provider-subset-calibration"]
    config_version: Literal["provenance_cascade_hg231_provider_subset.v1"]
    status: Literal["offline_validated_pending_exact_hash_approval"]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    parent_study_id: Literal["evicon-provenance-cascade-hg23-adoption-identifiability-calibration"]
    parent_config_path: str; parent_config_sha256: str
    parent_protocol_path: str; parent_protocol_sha256: str
    parent_approval_path: str; parent_approval_sha256: str
    parent_amendment_receipt_path: str; parent_amendment_receipt_sha256: str
    parent_compatibility_status: Literal["response_format_unsupported"]
    parent_compatibility_attempt_count: Literal[1]
    parent_compatibility_receipt_created: Literal[False]
    amendment_path: str; amendment_sha256: str
    protocol_path: str; protocol_sha256: str
    strict_parser_path: str; strict_parser_sha256: str
    protocol_version: Literal["provenance_cascade_agent_protocol.hg2_3_1_provider_subset.v1"]
    template_version: Literal["cascade_agent_turn.hg2_3_observable_adoption.v1"]
    response_format: Literal["json_schema"]
    response_schema_name: Literal["cascade_agent_epistemic_adoption_sharing_v1_provider_subset"]
    removed_provider_keyword: Literal["uniqueItems"]
    local_duplicate_validation_required: Literal[True]
    prompt_semantics_changed: Literal[False]
    output_contract_changed: Literal[False]
    scenario_design_changed: Literal[False]
    exposure_schedule_changed: Literal[False]
    controller_rules_changed: Literal[False]
    metric_names_changed: Literal[False]
    metric_operationalization_changed: Literal[False]
    evaluator_truth_changed: Literal[False]
    seed: Literal[20261031]
    scenario_count: Literal[4]; condition_count: Literal[4]
    agent_count: Literal[6]; round_count: Literal[3]
    run_count: Literal[16]; matched_group_count: Literal[4]
    logical_requests_per_run: Literal[18]; request_cap: Literal[288]
    agent_max_tokens: Literal[1024]
    completion_reservation_cap: Literal[294912]
    temperature: Literal[0.2]
    compatibility_max_retries: Literal[0]
    compatibility_timeout_seconds: Literal[5]
    calibration_max_retries: Literal[0]
    calibration_timeout_seconds: Literal[15]
    run_id_namespace: Literal["hg231"]
    output_root: Literal["results/provenance-cascade-hg231-adoption-identifiability-v1"]
    compatibility_receipt_path: str
    full_pilot_authorized: Literal[False]

    @model_validator(mode="after")
    def fixed_scope(self) -> "HG231Config":
        if self.request_cap != self.run_count * self.logical_requests_per_run:
            raise ValueError("request cap mismatch")
        if self.completion_reservation_cap != self.request_cap * self.agent_max_tokens:
            raise ValueError("completion reservation mismatch")
        return self


class HG231Approval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    approval_id: Literal["provenance-cascade-hg231-provider-subset-approval-v1"]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str; accepted_on: str
    config_sha256: str; protocol_sha256: str; amendment_sha256: str
    amendment_receipt_sha256: str; compatibility_receipt_sha256: str
    response_format: Literal["json_schema"]
    response_schema_name: Literal["cascade_agent_epistemic_adoption_sharing_v1_provider_subset"]
    run_count: Literal[16]; matched_group_count: Literal[4]
    logical_request_cap: Literal[288]
    completion_reservation_cap: Literal[294912]
    confirm_prompt_bytes_unchanged: bool
    confirm_local_parser_remains_strict: bool
    confirm_only_unique_items_removed: bool
    confirm_hg23_immutable: bool
    confirm_development_only: bool
    network_execution_authorized: Literal[False]

    @model_validator(mode="after")
    def accepted_shape(self) -> "HG231Approval":
        checks = (
            self.confirm_prompt_bytes_unchanged, self.confirm_local_parser_remains_strict,
            self.confirm_only_unique_items_removed, self.confirm_hg23_immutable,
            self.confirm_development_only,
        )
        if self.acceptance_status == "accepted" and (not self.accepted_by or not self.accepted_on or not all(checks)):
            raise ValueError("accepted approval incomplete")
        return self


class HG231AmendmentReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    receipt_id: Literal["provenance-cascade-hg231-provider-subset-amendment-v1"]
    status: Literal["offline_validated_pending_exact_hash_approval"]
    config_sha256: str; protocol_sha256: str; amendment_sha256: str
    parent_config_sha256: str; parent_protocol_sha256: str
    parent_approval_sha256: str; parent_amendment_receipt_sha256: str
    removed_provider_keyword: Literal["uniqueItems"]
    prompt_bytes_unchanged: Literal[True]
    schema_difference_only_unique_items: Literal[True]
    strict_duplicate_rejection_passed: Literal[True]
    fake_replay_passed_count: Literal[16]
    false_initial_adoption_eligible_count: int
    correction_transition_eligible_count: int
    harmful_conformity_eligible_count: int
    run_count: Literal[16]; matched_group_count: Literal[4]
    logical_request_cap: Literal[288]
    completion_reservation_cap: Literal[294912]
    network: Literal["disabled"]
    provider_constructed: Literal[False]
    results_written: Literal[False]
    private_truth_exposed: Literal[False]
    effectiveness_claimed: Literal[False]
    development_only: Literal[True]
    calibration_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]


def load_config(path: str | Path = DEFAULT_CONFIG) -> tuple[HG231Config, Path]:
    config_path = _path(path)
    try:
        config = HG231Config.model_validate(tomllib.loads(config_path.read_text(encoding="utf-8")))
    except (OSError, ValidationError, tomllib.TOMLDecodeError, ValueError) as exc:
        raise HG231Error("hg231_config_invalid") from exc
    bindings = (
        (config.parent_config_path, config.parent_config_sha256),
        (config.parent_protocol_path, config.parent_protocol_sha256),
        (config.parent_approval_path, config.parent_approval_sha256),
        (config.parent_amendment_receipt_path, config.parent_amendment_receipt_sha256),
        (config.amendment_path, config.amendment_sha256),
        (config.protocol_path, config.protocol_sha256),
        (config.strict_parser_path, config.strict_parser_sha256),
    )
    if any(not _path(file_path).is_file() or sha256_file(_path(file_path)) != digest for file_path, digest in bindings):
        raise HG231Error("hg231_binding_hash_mismatch")
    if _path("outputs/study-locks/provenance_cascade_hg23_compatibility_receipt.json").exists():
        raise HG231Error("parent_failed_check_receipt_must_not_exist")
    return config, config_path


def _without_unique(value):
    if isinstance(value, dict):
        return {key: _without_unique(item) for key, item in value.items() if key != "uniqueItems"}
    if isinstance(value, list):
        return [_without_unique(item) for item in value]
    return value


def run_fake_smoke(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    config, _ = load_config(path)
    context = build_minimal_hg23_context()
    parent_request, amended_request = render_hg23_turn(context), render_hg231_turn(context)
    if parent_request.system_prompt != amended_request.system_prompt or parent_request.user_prompt != amended_request.user_prompt:
        raise HG231Error("prompt_semantics_changed")
    if "uniqueItems" in json.dumps(HG231_RESPONSE_JSON_SCHEMA):
        raise HG231Error("unsupported_keyword_retained")
    if _without_unique(HG23_RESPONSE_JSON_SCHEMA) != HG231_RESPONSE_JSON_SCHEMA:
        raise HG231Error("provider_schema_change_exceeds_unique_items")
    content_id = context.public_context.contents[0].content_id
    duplicate = json.dumps({
        "epistemic_stance": "uncertain", "adoption_decision": "adopt_claim",
        "sharing_decision": "share_with_caveat",
        "content_ids_used": [content_id, content_id], "evidence_ids_used": [],
        "share_content_id": content_id,
    })
    parsed, error = parse_hg23_response(duplicate, context)
    if parsed is not None or error is None:
        raise HG231Error("strict_duplicate_parser_regressed")
    inherited = run_hg23_fake_smoke()
    return {
        "status": "offline_fake_smoke_passed",
        "prompt_bytes_unchanged": True,
        "schema_difference_only_unique_items": True,
        "strict_duplicate_rejection_passed": True,
        "schema_name": HG231_SCHEMA_NAME,
        "protocol_version": HG231_PROTOCOL_VERSION,
        "template_version": HG231_TEMPLATE_VERSION,
        "fake_replay_passed_count": inherited["replay_passed_count"],
        "false_initial_adoption_eligible_count": inherited["false_initial_adoption_eligible_count"],
        "correction_transition_eligible_count": inherited["correction_transition_eligible_count"],
        "harmful_conformity_eligible_count": inherited["harmful_conformity_eligible_count"],
        "run_count": config.run_count, "matched_group_count": config.matched_group_count,
        "logical_request_cap": config.request_cap,
        "completion_reservation_cap": config.completion_reservation_cap,
        "network": "disabled", "provider_constructed": False, "api_key_read": False,
        "results_written": False, "private_truth_exposed": False,
        "effectiveness_claimed": False,
    }


def write_amendment_receipt(path: str | Path = DEFAULT_AMENDMENT_RECEIPT) -> tuple[HG231AmendmentReceipt, str]:
    config, config_path = load_config()
    smoke = run_fake_smoke()
    receipt = HG231AmendmentReceipt(
        receipt_id="provenance-cascade-hg231-provider-subset-amendment-v1",
        status="offline_validated_pending_exact_hash_approval",
        config_sha256=sha256_file(config_path), protocol_sha256=config.protocol_sha256,
        amendment_sha256=config.amendment_sha256,
        parent_config_sha256=config.parent_config_sha256,
        parent_protocol_sha256=config.parent_protocol_sha256,
        parent_approval_sha256=config.parent_approval_sha256,
        parent_amendment_receipt_sha256=config.parent_amendment_receipt_sha256,
        removed_provider_keyword="uniqueItems", prompt_bytes_unchanged=True,
        schema_difference_only_unique_items=True, strict_duplicate_rejection_passed=True,
        fake_replay_passed_count=16,
        false_initial_adoption_eligible_count=int(smoke["false_initial_adoption_eligible_count"]),
        correction_transition_eligible_count=int(smoke["correction_transition_eligible_count"]),
        harmful_conformity_eligible_count=int(smoke["harmful_conformity_eligible_count"]),
        run_count=16, matched_group_count=4, logical_request_cap=288,
        completion_reservation_cap=294912, network="disabled",
        provider_constructed=False, results_written=False, private_truth_exposed=False,
        effectiveness_claimed=False, development_only=True, calibration_only=True,
        not_paper_result=True, no_causal_conclusion=True,
    )
    target = _path(path)
    if target.exists():
        existing = HG231AmendmentReceipt.model_validate_json(target.read_text(encoding="utf-8"))
        if existing != receipt:
            raise HG231Error("amendment_receipt_exists_with_different_content")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(receipt.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return receipt, hashlib.sha256(target.read_bytes()).hexdigest()


def preflight(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    try:
        config, config_path = load_config(path)
        smoke = run_fake_smoke(path)
        receipt_path, approval_path = _path(DEFAULT_AMENDMENT_RECEIPT), _path(DEFAULT_APPROVAL)
        receipt = HG231AmendmentReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
        approval = HG231Approval.model_validate(tomllib.loads(approval_path.read_text(encoding="utf-8")))
        if (
            receipt.config_sha256 != sha256_file(config_path)
            or receipt.protocol_sha256 != config.protocol_sha256
            or receipt.amendment_sha256 != config.amendment_sha256
        ):
            raise HG231Error("amendment_receipt_binding_mismatch")
        if (
            approval.config_sha256 != receipt.config_sha256
            or approval.protocol_sha256 != receipt.protocol_sha256
            or approval.amendment_sha256 != receipt.amendment_sha256
            or approval.amendment_receipt_sha256 != sha256_file(receipt_path)
        ):
            raise HG231Error("approval_binding_mismatch")
        reasons = []
        if approval.acceptance_status != "accepted":
            reasons.append("exact_hash_human_approval_required")
        if not approval.compatibility_receipt_sha256:
            reasons.append("provider_compatibility_check_required")
        elif not _path(DEFAULT_COMPATIBILITY_RECEIPT).is_file():
            reasons.append("provider_compatibility_receipt_missing")
        if _path(config.output_root).exists():
            reasons.append("output_root_exists")
        return {
            "status": "ready_for_exact_hash_approval" if set(reasons) <= {"exact_hash_human_approval_required", "provider_compatibility_check_required"} else "blocked",
            "ready_for_real_calibration": False,
            "blocking_reasons": reasons,
            "config_sha256": sha256_file(config_path), "protocol_sha256": config.protocol_sha256,
            "amendment_sha256": config.amendment_sha256,
            "amendment_receipt_sha256": sha256_file(receipt_path),
            "approval_sha256": sha256_file(approval_path),
            "run_count": 16, "matched_group_count": 4,
            "logical_request_cap": 288, "completion_reservation_cap": 294912,
            "fake_replay_passed_count": smoke["fake_replay_passed_count"],
            "network": "disabled", "provider_constructed": False, "api_key_read": False,
            "results_written": False, "private_truth_exposed": False,
            "not_paper_result": True, "no_causal_conclusion": True,
        }
    except Exception as exc:
        return {
            "status": "blocked", "ready_for_real_calibration": False,
            "blocking_reasons": [getattr(exc, "code", "hg231_preflight_failed")],
            "network": "disabled", "provider_constructed": False, "api_key_read": False,
            "results_written": False, "private_truth_exposed": False,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline H-G.2.3.1 provider-subset gate")
    parser.add_argument("--mode", choices=("preflight", "fake-smoke", "write-amendment-receipt"), default="preflight")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    if args.mode == "fake-smoke":
        payload = run_fake_smoke(args.config)
    elif args.mode == "write-amendment-receipt":
        receipt, digest = write_amendment_receipt()
        payload = {"status": receipt.status, "receipt_sha256": digest, "network": "disabled", "results_written": False}
    else:
        payload = preflight(args.config)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if payload["status"] in {"offline_fake_smoke_passed", "offline_validated_pending_exact_hash_approval", "ready_for_exact_hash_approval"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

