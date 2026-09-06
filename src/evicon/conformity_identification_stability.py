"""Independent 30-request protocol stability gate for Study v1.1."""
from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_identification import IdentificationCondition, IdentificationError, IdentificationStage, load_identification_config, sha256_file
from .conformity_identification_protocol import parse_identification_response, render_identification_turn
from .conformity_identification_protocol_v11 import BranchPromptContext, parse_branch_response, render_branch_turn
from .conformity_identification_smoke import QualificationFakeProvider, _context
from .conformity_identification_v11 import load_v11_config

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STABILITY_CONFIG = "configs/provenance_cascade/identification/conformity_identification_stability_probe.v1.toml"
DEFAULT_STABILITY_APPROVAL = "configs/provenance_cascade/identification/conformity_identification_stability_probe_approval.toml"


class StabilityProbeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    probe_id: Literal["evicon-conformity-identification-stability-v1"]
    probe_version: Literal["conformity_identification_stability_probe.v1"]
    status: Literal["offline_validated"]
    development_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    merge_with_behavior_study: Literal[False]
    study_config_path: str
    study_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    branch_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_id: Literal["cascade-identification-false-majority-v1"]
    condition: Literal[IdentificationCondition.SAME_ROOT_SOCIAL]
    seed: Literal[20261101]
    agent_count: Literal[6]
    shared_t0_request_count: Literal[6]
    continuation_stage_count: Literal[4]
    continuation_request_count: Literal[24]
    logical_request_cap: Literal[30]
    agent_max_tokens: Literal[512]
    completion_reservation_cap: Literal[15360]
    max_retries: Literal[0]
    timeout_seconds: Literal[5]
    parser_invalid_stops: Literal[True]
    output_root: Literal["outputs/conformity-identification-stability-v1"]
    approval_path: str

    @model_validator(mode="after")
    def fixed_probe(self) -> "StabilityProbeConfig":
        if self.logical_request_cap != self.shared_t0_request_count + self.continuation_request_count:
            raise ValueError("probe request accounting mismatch")
        if self.continuation_request_count != self.agent_count * self.continuation_stage_count:
            raise ValueError("probe continuation accounting mismatch")
        if self.completion_reservation_cap != self.logical_request_cap * self.agent_max_tokens:
            raise ValueError("probe reservation mismatch")
        return self


class StabilityProbeApproval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    approval_id: Literal["evicon-conformity-identification-stability-approval-v1"]
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str
    accepted_on: str
    config_sha256: str
    study_config_sha256: str
    protocol_sha256: str
    branch_contract_sha256: str
    logical_request_cap: Literal[30]
    completion_reservation_cap: Literal[15360]
    confirm_independent_output: bool
    confirm_no_result_merge: bool
    confirm_parser_invalid_stops: bool
    confirm_no_overwrite: bool
    network_execution_authorized: Literal[False]

    @model_validator(mode="after")
    def accepted_complete(self) -> "StabilityProbeApproval":
        if self.acceptance_status == "accepted" and (
            not self.accepted_by or not self.accepted_on
            or not all((
                self.confirm_independent_output, self.confirm_no_result_merge,
                self.confirm_parser_invalid_stops, self.confirm_no_overwrite,
            ))
        ):
            raise ValueError("accepted stability approval incomplete")
        return self


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def load_stability_config(path: str | Path = DEFAULT_STABILITY_CONFIG) -> tuple[StabilityProbeConfig, StabilityProbeApproval]:
    resolved = _resolve(path)
    try:
        config = StabilityProbeConfig.model_validate(tomllib.loads(resolved.read_text(encoding="utf-8")))
    except Exception as exc:
        raise IdentificationError("stability_config_invalid") from exc
    study, _ = load_v11_config(config.study_config_path)
    if sha256_file(config.study_config_path) != config.study_config_sha256:
        raise IdentificationError("stability_study_hash_mismatch")
    if (study.protocol_sha256, study.branch_contract_sha256) != (config.protocol_sha256, config.branch_contract_sha256):
        raise IdentificationError("stability_contract_binding_mismatch")
    approval_path = _resolve(config.approval_path)
    try:
        approval = StabilityProbeApproval.model_validate(tomllib.loads(approval_path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise IdentificationError("stability_approval_invalid") from exc
    if approval.config_sha256 != sha256_file(resolved):
        raise IdentificationError("stability_approval_config_hash_mismatch")
    if (
        approval.study_config_sha256, approval.protocol_sha256, approval.branch_contract_sha256
    ) != (
        config.study_config_sha256, config.protocol_sha256, config.branch_contract_sha256
    ):
        raise IdentificationError("stability_approval_binding_mismatch")
    return config, approval


def run_fake_smoke(path: str = DEFAULT_STABILITY_CONFIG) -> dict[str, object]:
    config, approval = load_stability_config(path)
    study, _ = load_v11_config(config.study_config_path)
    _, scenarios = load_identification_config(study.parent_config_path)
    scenario = next(item for item in scenarios if item.scenario_id == config.scenario_id)
    provider = QualificationFakeProvider()
    request_ids: set[str] = set()
    stage_counts: dict[str, int] = {IdentificationStage.INITIAL_PRIVATE.value: 0}
    for stage in study.continuation_stages:
        stage_counts[stage.value] = 0
    for agent_id in study.agent_ids:
        initial_context = _context(scenario, config.condition, config.seed, agent_id, IdentificationStage.INITIAL_PRIVATE)
        initial_request = render_identification_turn(initial_context)
        request_ids.add(initial_request.request_id)
        initial = parse_identification_response(provider.complete(initial_request).content, initial_context)
        stage_counts[IdentificationStage.INITIAL_PRIVATE.value] += 1
        initial_hash = hashlib.sha256(initial.model_dump_json().encode()).hexdigest()
        for stage in study.continuation_stages:
            public = _context(scenario, config.condition, config.seed, agent_id, stage)
            branch = BranchPromptContext(
                public_context=public,
                matched_group_id=f"stability-{scenario.scenario_id}-{config.seed}",
                branch_id=f"stability-{scenario.scenario_id}-{config.seed}-{config.condition.value}",
                shared_t0_observation_sha256=initial_hash,
            )
            request = render_branch_turn(branch)
            if request.request_id in request_ids:
                raise IdentificationError("stability_duplicate_fingerprint")
            request_ids.add(request.request_id)
            parse_branch_response(provider.complete(request).content, branch)
            stage_counts[stage.value] += 1
    return {
        "status": "stability_fake_smoke_passed",
        "approval_status": approval.acceptance_status,
        "logical_request_count": provider.calls,
        "unique_request_fingerprint_count": len(request_ids),
        "stage_counts": stage_counts,
        "parser_invalid_count": 0,
        "effect_estimated": False,
        "network": "disabled",
        "provider": "qualification_fake",
        "results_written": False,
        "private_truth_exposed": False,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def preflight(path: str = DEFAULT_STABILITY_CONFIG) -> dict[str, object]:
    try:
        config, approval = load_stability_config(path)
        smoke = run_fake_smoke(path)
        reasons = []
        if approval.acceptance_status != "accepted":
            reasons.append("exact_hash_human_approval_required")
        reasons.extend(("provider_compatibility_check_required", "network_authorization_required"))
        return {
            "status": "offline_probe_ready",
            "ready_for_network": False,
            "blocking_reasons": reasons,
            "probe_id": config.probe_id,
            "config_sha256": sha256_file(path),
            "logical_request_cap": config.logical_request_cap,
            "completion_reservation_cap": config.completion_reservation_cap,
            "fake_logical_request_count": smoke["logical_request_count"],
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
            "ready_for_network": False,
            "blocking_reasons": [getattr(exc, "code", "stability_preflight_failed")],
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "private_truth_exposed": False,
        }


def safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


__all__ = [
    "DEFAULT_STABILITY_CONFIG", "StabilityProbeApproval", "StabilityProbeConfig",
    "load_stability_config", "preflight", "run_fake_smoke",
]
