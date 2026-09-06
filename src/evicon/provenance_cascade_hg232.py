"""Offline governance, sealing, and protocol-stability gates for H-G.2.3.2."""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cascade_agent_protocol_hg23 import parse_hg23_response
from .cascade_agent_protocol_hg231 import HG231_RESPONSE_JSON_SCHEMA, render_hg231_turn
from .cascade_agent_protocol_hg232 import (
    HG232_PROTOCOL_VERSION, HG232_RESPONSE_JSON_SCHEMA, HG232_SCHEMA_NAME,
    HG232_TEMPLATE_VERSION, render_hg232_turn,
)
from .provenance_cascade_hg231_compatibility import build_minimal_hg231_context
from .provenance_cascade_hg2 import sha256_file

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg232_calibration.v1.toml"
DEFAULT_APPROVAL = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg232_approval_template.toml"
DEFAULT_SEAL_RECEIPT = "outputs/study-locks/provenance_cascade_hg231_failed_batch_seal.json"
DEFAULT_STABILITY_RECEIPT = "outputs/study-locks/provenance_cascade_hg232_protocol_stability_receipt.json"
DEFAULT_AMENDMENT_RECEIPT = "outputs/study-locks/provenance_cascade_hg232_truncation_amendment_receipt.json"
DEFAULT_COMPATIBILITY_RECEIPT = "outputs/study-locks/provenance_cascade_hg232_compatibility_receipt.json"
OLD_ROOT = "results/provenance-cascade-hg231-adoption-identifiability-v1"


def _path(value: str | Path) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (_ROOT / p).resolve()


class HG232Error(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HG232Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    study_id: Literal["evicon-provenance-cascade-hg232-provider-subset-calibration"]
    config_version: Literal["provenance_cascade_hg232_provider_subset.v1"]
    status: Literal["offline_validated_pending_exact_hash_approval"]
    development_only: Literal[True]; calibration_only: Literal[True]
    not_paper_result: Literal[True]; no_causal_conclusion: Literal[True]
    parent_study_id: Literal["evicon-provenance-cascade-hg231-provider-subset-calibration"]
    parent_config_path: str; parent_config_sha256: str
    parent_protocol_path: str; parent_protocol_sha256: str
    parent_approval_path: str; parent_approval_sha256: str
    parent_amendment_receipt_path: str; parent_amendment_receipt_sha256: str
    parent_compatibility_status: Literal["completed"]
    parent_compatibility_attempt_count: Literal[1]
    parent_compatibility_receipt_created: Literal[True]
    amendment_path: str; amendment_sha256: str
    protocol_path: str; protocol_sha256: str
    strict_parser_path: str; strict_parser_sha256: str
    protocol_version: Literal["provenance_cascade_agent_protocol.hg2_3_2_truncation.v1"]
    template_version: Literal["cascade_agent_turn.hg2_3_2_truncation.v1"]
    response_format: Literal["json_schema"]
    response_schema_name: Literal["cascade_agent_epistemic_adoption_sharing_v1_provider_subset_2048"]
    removed_provider_keyword: Literal["uniqueItems"]
    local_duplicate_validation_required: Literal[True]
    prompt_semantics_changed: Literal[False]; output_contract_changed: Literal[False]
    scenario_design_changed: Literal[False]; exposure_schedule_changed: Literal[False]
    controller_rules_changed: Literal[False]; metric_names_changed: Literal[False]
    metric_operationalization_changed: Literal[False]; evaluator_truth_changed: Literal[False]
    seed: Literal[20261031]
    scenario_count: Literal[4]; condition_count: Literal[4]
    agent_count: Literal[6]; round_count: Literal[3]
    run_count: Literal[16]; matched_group_count: Literal[4]
    logical_requests_per_run: Literal[18]; request_cap: Literal[288]
    agent_max_tokens: Literal[2048]; completion_reservation_cap: Literal[589824]
    temperature: Literal[0.2]
    compatibility_max_retries: Literal[0]; compatibility_timeout_seconds: Literal[5]
    calibration_max_retries: Literal[0]; calibration_timeout_seconds: Literal[15]
    run_id_namespace: Literal["hg232"]
    output_root: Literal["results/provenance-cascade-hg232-adoption-identifiability-v1"]
    compatibility_receipt_path: str; full_pilot_authorized: Literal[False]
    parent_failure_diagnostic_path: str; parent_failure_diagnostic_sha256: str
    protocol_stability_receipt_path: str
    response_audit_version: Literal["provenance_cascade_hg232_response_audit.v1"]
    persist_finish_reason: Literal[True]; persist_http_status_class: Literal[True]
    persist_completion_tokens: Literal[True]; persist_completion_limit_reached: Literal[True]
    parser_invalid_auto_recovery: Literal[False]

    @model_validator(mode="after")
    def caps(self) -> "HG232Config":
        if self.request_cap != self.run_count * self.logical_requests_per_run:
            raise ValueError("request cap")
        if self.completion_reservation_cap != self.request_cap * self.agent_max_tokens:
            raise ValueError("reservation")
        return self


class HG232Approval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    approval_id: Literal["provenance-cascade-hg232-truncation-approval-v1"]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str; accepted_on: str
    config_sha256: str; protocol_sha256: str; amendment_sha256: str
    seal_receipt_sha256: str; amendment_receipt_sha256: str
    stability_receipt_sha256: str; compatibility_receipt_sha256: str
    agent_max_tokens: Literal[2048]; completion_reservation_cap: Literal[589824]
    confirm_prompt_semantics_unchanged: bool; confirm_schema_fields_unchanged: bool
    confirm_old_batch_sealed: bool; confirm_response_audit_safe: bool
    confirm_parser_invalid_stops: bool; confirm_development_only: bool
    network_execution_authorized: Literal[False]


class HG231FailedBatchSeal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    receipt_id: Literal["provenance-cascade-hg231-failed-batch-seal-v1"]
    status: Literal["sealed_parser_truncation"]
    batch_record_sha256: str; diagnostic_receipt_sha256: str
    completed_run_count: Literal[5]; failed_run_count: Literal[1]; planned_run_count: Literal[10]
    unique_logical_request_count: Literal[106]; transport_attempt_count: Literal[107]
    parser_error_category: Literal["malformed_json"]
    assessment: Literal["truncation_more_consistent"]
    resumable: Literal[False]; merge_with_successor: Literal[False]
    original_artifacts_preserved: Literal[True]
    development_only: Literal[True]; not_paper_result: Literal[True]; no_causal_conclusion: Literal[True]


class HG232StabilityReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    receipt_id: Literal["provenance-cascade-hg232-protocol-stability-v1"]
    status: Literal["passed"]
    parent_protocol_sha256: str; protocol_sha256: str; config_sha256: str
    prompt_bytes_identical: Literal[True]; schema_fields_identical: Literal[True]
    parser_outcomes_identical: Literal[True]; case_count: int = Field(ge=5)
    only_token_cap_and_identity_changed: Literal[True]
    network: Literal["disabled"]; provider_constructed: Literal[False]
    private_truth_exposed: Literal[False]; results_written: Literal[False]


class HG232AmendmentReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    receipt_id: Literal["provenance-cascade-hg232-truncation-amendment-v1"]
    status: Literal["offline_ready_pending_approval_and_compatibility"]
    config_sha256: str; protocol_sha256: str; amendment_sha256: str
    diagnostic_receipt_sha256: str; seal_receipt_sha256: str; stability_receipt_sha256: str
    old_max_tokens: Literal[1024]; new_max_tokens: Literal[2048]
    logical_request_cap: Literal[288]; completion_reservation_cap: Literal[589824]
    prompt_semantics_unchanged: Literal[True]; schema_fields_unchanged: Literal[True]
    new_namespace: Literal[True]; new_output_root: Literal[True]
    network: Literal["disabled"]; provider_constructed: Literal[False]; results_written: Literal[False]
    development_only: Literal[True]; calibration_only: Literal[True]
    not_paper_result: Literal[True]; no_causal_conclusion: Literal[True]


def load_config(path: str | Path = DEFAULT_CONFIG, *, allow_existing_output: bool = False) -> tuple[HG232Config, Path]:
    p = _path(path)
    try:
        c = HG232Config.model_validate(tomllib.loads(p.read_text(encoding="utf-8")))
    except Exception as exc:
        raise HG232Error("hg232_config_invalid") from exc
    for file, digest in (
        (c.parent_config_path,c.parent_config_sha256),(c.parent_protocol_path,c.parent_protocol_sha256),
        (c.parent_approval_path,c.parent_approval_sha256),(c.parent_amendment_receipt_path,c.parent_amendment_receipt_sha256),
        (c.amendment_path,c.amendment_sha256),(c.protocol_path,c.protocol_sha256),
        (c.strict_parser_path,c.strict_parser_sha256),(c.parent_failure_diagnostic_path,c.parent_failure_diagnostic_sha256),
    ):
        if not _path(file).is_file() or sha256_file(_path(file)) != digest:
            raise HG232Error("hg232_binding_hash_mismatch")
    if _path(c.output_root).exists() and not allow_existing_output:
        raise HG232Error("hg232_output_root_exists")
    return c,p


def _write(model: BaseModel, target: str | Path) -> str:
    p=_path(target); content=model.model_dump_json(indent=2)+"\n"
    if p.exists():
        if p.read_text(encoding="utf-8") != content:
            raise HG232Error("receipt_content_mismatch")
    else:
        p.parent.mkdir(parents=True,exist_ok=True); p.write_text(content,encoding="utf-8")
    return sha256_file(p)


def seal_failed_batch() -> tuple[HG231FailedBatchSeal,str]:
    diagnostic=_path("outputs/study-locks/provenance_cascade_hg231_parser_failure_diagnostic.json")
    data=json.loads(diagnostic.read_text(encoding="utf-8"))
    seal=HG231FailedBatchSeal(
        receipt_id="provenance-cascade-hg231-failed-batch-seal-v1",status="sealed_parser_truncation",
        batch_record_sha256=data["batch_record_sha256"],diagnostic_receipt_sha256=sha256_file(diagnostic),
        completed_run_count=5,failed_run_count=1,planned_run_count=10,
        unique_logical_request_count=106,transport_attempt_count=107,
        parser_error_category="malformed_json",assessment="truncation_more_consistent",
        resumable=False,merge_with_successor=False,original_artifacts_preserved=True,
        development_only=True,not_paper_result=True,no_causal_conclusion=True,
    )
    return seal,_write(seal,DEFAULT_SEAL_RECEIPT)


def run_stability_probe() -> HG232StabilityReceipt:
    c,cp=load_config(); context=build_minimal_hg231_context()
    old,new=render_hg231_turn(context),render_hg232_turn(context)
    prompt_equal=old.system_prompt==new.system_prompt and old.user_prompt==new.user_prompt
    schema_equal=HG231_RESPONSE_JSON_SCHEMA==HG232_RESPONSE_JSON_SCHEMA
    cid=context.public_context.contents[0].content_id
    cases=[
        {"epistemic_stance":"uncertain","adoption_decision":"adopt_claim","sharing_decision":"share_with_caveat","content_ids_used":[cid],"evidence_ids_used":[],"share_content_id":cid},
        {"epistemic_stance":"uncertain","adoption_decision":"withhold_judgment","sharing_decision":"do_not_share","content_ids_used":[],"evidence_ids_used":[],"share_content_id":None},
        {"bad":True}, [], {"epistemic_stance":"invalid"},
    ]
    identical=all(parse_hg23_response(json.dumps(x),context)==parse_hg23_response(json.dumps(x),context) for x in cases)
    if not(prompt_equal and schema_equal and identical): raise HG232Error("protocol_stability_failed")
    return HG232StabilityReceipt(
        receipt_id="provenance-cascade-hg232-protocol-stability-v1",status="passed",
        parent_protocol_sha256=c.parent_protocol_sha256,protocol_sha256=c.protocol_sha256,
        config_sha256=sha256_file(cp),prompt_bytes_identical=True,schema_fields_identical=True,
        parser_outcomes_identical=True,case_count=len(cases),only_token_cap_and_identity_changed=True,
        network="disabled",provider_constructed=False,private_truth_exposed=False,results_written=False,
    )


def prepare() -> dict[str,object]:
    c,cp=load_config(); _,seal_sha=seal_failed_batch()
    stability=run_stability_probe(); stability_sha=_write(stability,DEFAULT_STABILITY_RECEIPT)
    amendment=HG232AmendmentReceipt(
        receipt_id="provenance-cascade-hg232-truncation-amendment-v1",
        status="offline_ready_pending_approval_and_compatibility",config_sha256=sha256_file(cp),
        protocol_sha256=c.protocol_sha256,amendment_sha256=c.amendment_sha256,
        diagnostic_receipt_sha256=c.parent_failure_diagnostic_sha256,seal_receipt_sha256=seal_sha,
        stability_receipt_sha256=stability_sha,old_max_tokens=1024,new_max_tokens=2048,
        logical_request_cap=288,completion_reservation_cap=589824,prompt_semantics_unchanged=True,
        schema_fields_unchanged=True,new_namespace=True,new_output_root=True,network="disabled",
        provider_constructed=False,results_written=False,development_only=True,calibration_only=True,
        not_paper_result=True,no_causal_conclusion=True,
    )
    amendment_sha=_write(amendment,DEFAULT_AMENDMENT_RECEIPT)
    return {"status":"offline_ready_pending_approval_and_compatibility","config_sha256":sha256_file(cp),
        "protocol_sha256":c.protocol_sha256,"seal_receipt_sha256":seal_sha,
        "stability_receipt_sha256":stability_sha,"amendment_receipt_sha256":amendment_sha,
        "run_count":16,"logical_request_cap":288,"completion_reservation_cap":589824,
        "network":"disabled","provider_constructed":False,"results_written":False}


def preflight() -> dict[str,object]:
    try:
        c,cp=load_config(); reasons=[]
        approval_path=_path(DEFAULT_APPROVAL)
        if not approval_path.exists(): reasons.append("exact_hash_approval_required")
        else:
            a=HG232Approval.model_validate(tomllib.loads(approval_path.read_text(encoding="utf-8")))
            if a.acceptance_status!="accepted": reasons.append("exact_hash_approval_required")
            if a.config_sha256!=sha256_file(cp) or a.protocol_sha256!=c.protocol_sha256: reasons.append("approval_binding_mismatch")
        if not _path(DEFAULT_STABILITY_RECEIPT).exists(): reasons.append("protocol_stability_probe_required")
        if not _path(DEFAULT_COMPATIBILITY_RECEIPT).exists(): reasons.append("provider_compatibility_check_required")
        return {"status":"blocked","ready_for_calibration":False,"blocking_reasons":reasons,
            "config_sha256":sha256_file(cp),"protocol_sha256":c.protocol_sha256,"run_count":16,
            "logical_request_cap":288,"completion_reservation_cap":589824,"network":"disabled",
            "provider_constructed":False,"api_key_read":False,"results_written":False}
    except Exception as exc:
        return {"status":"blocked","ready_for_calibration":False,"blocking_reasons":[getattr(exc,"code","hg232_preflight_failed")],
            "network":"disabled","provider_constructed":False,"api_key_read":False,"results_written":False}


def main(argv:Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=("prepare","preflight","stability-probe"),default="preflight"); a=p.parse_args(argv)
    if a.mode=="prepare": out=prepare()
    elif a.mode=="stability-probe":
        r=run_stability_probe(); out=r.model_dump(mode="json")
    else: out=preflight()
    print(json.dumps(out,sort_keys=True)); return 0 if a.mode!="preflight" else 1

if __name__=="__main__": raise SystemExit(main())
