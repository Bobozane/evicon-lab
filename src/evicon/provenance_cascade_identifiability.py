"""H-G public-content identifiability amendment, offline preflight, and fake smoke."""
from __future__ import annotations

import hashlib
import json
import statistics
import tempfile
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .cascade_agent_protocol_hg import (
    HG_PROTOCOL_VERSION,
    HG_SCHEMA_NAME,
    HG_TEMPLATE_VERSION,
    CascadeAgentProtocolHGRuntime,
    build_identifiable_prompt_context,
)
from .cascade_controller import CascadeControllerPolicyLoader
from .cascade_outcomes import ClaimStance
from .cascade_protocol import CascadeScenarioLoader, CascadeScenarioSpec
from .cascade_real_agent_runner import CascadeRealAgentRunRecord, CascadeRealAgentRunner
from .llm_contract import LLMResponse
from .provenance_cascade import (
    EvaluatorTruthFixture,
    EvaluatorTruthLoader,
    GroundTruthLabel,
    SourceIndependenceLabel,
    validate_fixture_pair,
)
from .provenance_cascade_preregistration import CascadeCondition, CascadeScenario

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_RELATIVE = "configs/provenance_cascade/pilot/provenance_cascade_pilot_hg.v1.toml"
_PROTOCOL_RELATIVE = "src/evicon/cascade_agent_protocol_hg.py"
_AMENDMENT_RECEIPT_RELATIVE = "outputs/study-locks/provenance_cascade_hg_identifiability_amendment_receipt.json"
_HG_REPLAY_TECHNICAL_RECEIPT_RELATIVE = "outputs/study-locks/provenance_cascade_hg_outcome_replay_technical_amendment_receipt.json"
_NON_IDENTIFIABLE_RECEIPT_RELATIVE = "outputs/study-locks/provenance_cascade_hd21_non_identifiable_receipt.json"
_EXPECTED_CONDITIONS = tuple(CascadeCondition)
_EXPECTED_SEEDS = (20260911, 20260912, 20260913)
_EXPECTED_AGENTS = tuple(f"network-agent-{index:02d}" for index in range(1, 7))


class HGIdentifiabilityError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HGMaterialBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_id: str
    scenario_type: CascadeScenario
    scenario_path: str
    scenario_sha256: str
    graph_path: str
    graph_sha256: str
    truth_path: str
    truth_sha256: str


class HGPilotRunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    matched_group_id: str
    scenario_id: str
    condition: CascadeCondition
    seed: int
    expected_provider_requests: Literal[18] = 18
    completion_reservation: Literal[9216] = 9216


class HGPilotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    study_id: Literal["evicon-provenance-cascade-pilot-hg"]
    config_version: Literal["provenance_cascade_hg_identifiability.v1"]
    protocol_version: Literal["provenance_cascade_agent_protocol.hg_identifiable.v1"]
    template_version: Literal["cascade_agent_turn.hg_public_content.v1"]
    status: Literal["offline_preflight"]
    development_only: Literal[True]
    pilot_only: Literal[True]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    parent_pilot_status: Literal["non_identifiable_pilot"]
    parent_config_path: str
    parent_config_sha256: str
    parent_pilot_receipt_path: str
    parent_pilot_receipt_sha256: str
    parent_analysis_report_path: str
    parent_analysis_report_sha256: str
    protocol_path: str
    protocol_sha256: str
    policy_path: str
    policy_sha256: str
    amendment_id: Literal["provenance_cascade_hg_public_identifiability.v1"]
    conditions: tuple[CascadeCondition, ...]
    seeds: tuple[int, ...]
    agent_ids: tuple[str, ...]
    max_rounds: Literal[3]
    topology_id: Literal["ring_6_bidirectional"]
    run_count: Literal[48]
    matched_group_count: Literal[12]
    logical_requests_per_run: Literal[18]
    request_cap: Literal[864]
    agent_max_tokens: Literal[512]
    completion_reservation_per_run: Literal[9216]
    completion_reservation_cap: Literal[442368]
    response_format: Literal["json_schema"]
    response_schema_name: Literal["cascade_agent_response_v2_1"]
    max_retries: Literal[1]
    timeout_seconds: Literal[15]
    output_root: Literal["results/provenance-cascade-pilot-hg-v1"]
    approval_path: str
    compatibility_receipt_path: str
    scenario_materials: tuple[HGMaterialBinding, ...]

    @model_validator(mode="after")
    def fixed_scope(self) -> "HGPilotConfig":
        if self.protocol_version != HG_PROTOCOL_VERSION or self.template_version != HG_TEMPLATE_VERSION:
            raise ValueError("H-G protocol binding mismatch")
        if self.response_schema_name != HG_SCHEMA_NAME:
            raise ValueError("H-G response schema mismatch")
        if self.conditions != _EXPECTED_CONDITIONS or self.seeds != _EXPECTED_SEEDS:
            raise ValueError("H-G matched design mismatch")
        if self.agent_ids != _EXPECTED_AGENTS:
            raise ValueError("H-G agent order mismatch")
        if len(self.scenario_materials) != 4 or len({item.scenario_id for item in self.scenario_materials}) != 4:
            raise ValueError("H-G requires four unique scenario materials")
        if {item.scenario_type for item in self.scenario_materials} != set(CascadeScenario):
            raise ValueError("H-G scenario types are incomplete")
        if self.request_cap != self.run_count * self.logical_requests_per_run:
            raise ValueError("H-G request cap mismatch")
        if self.completion_reservation_cap != self.request_cap * self.agent_max_tokens:
            raise ValueError("H-G completion reservation mismatch")
        if self.completion_reservation_per_run != self.logical_requests_per_run * self.agent_max_tokens:
            raise ValueError("H-G per-run reservation mismatch")
        return self

    @property
    def runs(self) -> tuple[HGPilotRunSpec, ...]:
        return tuple(
            HGPilotRunSpec(
                run_id=f"hg-{material.scenario_id}-{seed}-{condition.value}",
                matched_group_id=f"hg-{material.scenario_id}-{seed}",
                scenario_id=material.scenario_id,
                condition=condition,
                seed=seed,
            )
            for material in self.scenario_materials
            for seed in self.seeds
            for condition in self.conditions
        )



class HGApproval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    approval_id: str
    acceptance_status: Literal["pending", "accepted"]
    accepted_by: str = ""
    accepted_on: str = ""
    acceptance_scope: str = ""
    config_sha256: str
    protocol_sha256: str
    amendment_sha256: str
    amendment_receipt_sha256: str
    compatibility_receipt_sha256: str = ""
    response_format: Literal["json_schema"]
    response_schema_name: Literal["cascade_agent_response_v2_1"]
    run_count: Literal[48]
    logical_request_cap: Literal[864]
    completion_reservation_cap: Literal[442368]
    agent_max_tokens: Literal[512]
    max_retries: Literal[1]
    timeout_seconds: Literal[15]
    confirm_public_summary_amendment: bool = False
    confirm_behavior_observation_amendment: bool = False
    confirm_old_pilot_non_identifiable: bool = False
    confirm_no_result_merge: bool = False
    confirm_scope_48_runs: bool = False
    confirm_request_cap_864: bool = False
    confirm_completion_reservation_442368: bool = False
    confirm_new_output_root: bool = False
    confirm_new_provider_compatibility_required: bool = False
    confirm_no_overwrite: bool = False
    confirm_append_only_ledger: bool = False
    confirm_resume_rules: bool = False
    network_execution_authorized: Literal[False] = False
    notes: str = ""

    
    @property
    def ready(self) -> bool:
        return self.acceptance_status == "accepted" and bool(self.accepted_by and self.accepted_on) and all((
            self.confirm_public_summary_amendment,
            self.confirm_behavior_observation_amendment,
            self.confirm_old_pilot_non_identifiable,
            self.confirm_no_result_merge,
            self.confirm_scope_48_runs,
            self.confirm_request_cap_864,
            self.confirm_completion_reservation_442368,
            self.confirm_new_output_root,
            self.confirm_new_provider_compatibility_required,
            self.confirm_no_overwrite,
            self.confirm_append_only_ledger,
            self.confirm_resume_rules,
        ))

class HGFakeProvider:
    """Content-sensitive provider used only by the offline H-G smoke."""

    def __init__(self) -> None:
        self.calls = 0
        self.public_prompt_seen = 0
        self.directive_seen = 0

    def complete(self, request) -> LLMResponse:
        self.calls += 1
        payload = json.loads(request.user_prompt)
        claims = payload.get("visible_claims", [])
        contents = payload.get("visible_contents", [])
        evidence = payload.get("visible_evidence", [])
        target_id = payload.get("target_claim_id")
        target = next((item for item in claims if item.get("claim_id") == target_id), None)
        target_summary = str((target or {}).get("public_summary", "")).lower()
        content_summaries = " ".join(str(item.get("public_summary", "")) for item in contents).lower()
        directive = payload.get("directive")
        if target_summary or content_summaries:
            self.public_prompt_seen += 1
        if directive is not None:
            self.directive_seen += 1
        visible_content_ids = [item["content_id"] for item in contents if isinstance(item, dict) and isinstance(item.get("content_id"), str)]
        target_content_ids = [item["content_id"] for item in contents if isinstance(item, dict) and item.get("claim_id") == target_id and isinstance(item.get("content_id"), str)]
        visible_evidence_ids = [item["evidence_id"] for item in evidence if isinstance(item, dict) and isinstance(item.get("evidence_id"), str)]
        contradicting = [
            item["evidence_id"] for item in evidence
            if isinstance(item, dict) and target_id in item.get("contradicts", [])
        ]
        supporting = [
            item["evidence_id"] for item in evidence
            if isinstance(item, dict) and target_id in item.get("supports", [])
        ]
        if target is None:
            stance = "no_position"
        elif "available public accounts conflict" in target_summary:
            stance = "uncertain"
        elif "published northgate schedule keeps" in target_summary:
            stance = "endorses"
        elif "remove all weekend hours" in target_summary:
            stance = "rejects" if contradicting or "still lists saturday" in content_summaries else "endorses"
        elif "harborlink service will suspend" in target_summary:
            stance = "uncertain" if directive is not None else "endorses"
        elif "cooling rooms will remain open" in target_summary:
            stance = "endorses"
        else:
            stance = "uncertain"
        used_evidence = contradicting or supporting
        used_content = target_content_ids[:2]
        share = target_content_ids[0] if target_content_ids else None
        response = json.dumps(
            {
                "stance": stance,
                "content_ids_used": used_content,
                "evidence_ids_used": used_evidence,
                "share_content_id": share,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content=response,
            finish_reason="stop",
            prompt_tokens=24,
            completion_tokens=20,
            total_tokens=44,
            latency_ms=1.0,
        )


def _rooted(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_hg_config(path: str | Path = _CONFIG_RELATIVE) -> tuple[HGPilotConfig, dict[str, CascadeScenarioSpec], dict[str, EvaluatorTruthFixture]]:
    config_path = Path(path).resolve()
    try:
        config = HGPilotConfig.model_validate(tomllib.loads(config_path.read_text(encoding="utf-8")))
    except (OSError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise HGIdentifiabilityError("hg_config_invalid") from exc
    bindings = (
        (config.parent_config_path, config.parent_config_sha256, "parent_config_hash_mismatch"),
        (config.parent_pilot_receipt_path, config.parent_pilot_receipt_sha256, "parent_receipt_hash_mismatch"),
        (config.parent_analysis_report_path, config.parent_analysis_report_sha256, "parent_analysis_hash_mismatch"),
        (config.protocol_path, config.protocol_sha256, "hg_protocol_hash_mismatch"),
        (config.policy_path, config.policy_sha256, "hg_policy_hash_mismatch"),
    )
    for value, expected, code in bindings:
        candidate = _rooted(config_path, value)
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise HGIdentifiabilityError(code)
    scenarios: dict[str, CascadeScenarioSpec] = {}
    truths: dict[str, EvaluatorTruthFixture] = {}
    for material in config.scenario_materials:
        scenario_path = _rooted(config_path, material.scenario_path)
        graph_path = _rooted(config_path, material.graph_path)
        truth_path = _rooted(config_path, material.truth_path)
        for candidate, expected, code in (
            (scenario_path, material.scenario_sha256, "scenario_hash_mismatch"),
            (graph_path, material.graph_sha256, "graph_hash_mismatch"),
            (truth_path, material.truth_sha256, "truth_hash_mismatch"),
        ):
            if not candidate.is_file() or sha256_file(candidate) != expected:
                raise HGIdentifiabilityError(code)
        try:
            scenario = CascadeScenarioLoader.load(scenario_path)
            truth = EvaluatorTruthLoader.load(truth_path)
            validate_fixture_pair(scenario.graph, truth)
        except Exception as exc:
            raise HGIdentifiabilityError("material_binding_invalid") from exc
        if scenario.scenario_id != material.scenario_id or scenario.scenario_type is not material.scenario_type:
            raise HGIdentifiabilityError("material_coordinate_mismatch")
        if any(node.public_statement is None for node in scenario.graph.nodes):
            raise HGIdentifiabilityError("public_statement_missing")
        public_payload = graph_path.read_text(encoding="utf-8").lower()
        if any(marker in public_payload for marker in ("ground_truth_label", "source_independence_label", "evaluator_truth")):
            raise HGIdentifiabilityError("private_truth_in_public_material")
        scenarios[scenario.scenario_id] = scenario
        truths[scenario.scenario_id] = truth
    if len(config.runs) != 48 or len({item.run_id for item in config.runs}) != 48:
        raise HGIdentifiabilityError("run_plan_incomplete")
    if len({item.matched_group_id for item in config.runs}) != 12:
        raise HGIdentifiabilityError("matched_group_incomplete")
    return config, scenarios, truths


def _substantive_history(record: CascadeRealAgentRunRecord, agent_id: str, claim_id: str):
    assert record.outcome_ledger is not None
    return [
        item for item in record.outcome_ledger.outcomes
        if item.agent_id == agent_id and item.claim_id == claim_id and item.stance is not ClaimStance.NO_POSITION
    ]


def _behavior_eligibility(record: CascadeRealAgentRunRecord, truth: EvaluatorTruthFixture) -> dict[str, int]:
    assert record.exposure_ledger is not None and record.outcome_ledger is not None
    false_claims = {
        item.claim_id for item in truth.records if item.ground_truth_label is GroundTruthLabel.FALSE
    }
    corrections = {
        item.claim_id for item in truth.records
        if item.ground_truth_label is GroundTruthLabel.TRUE
        and item.source_independence_label is SourceIndependenceLabel.INDEPENDENT
    }
    exposed = {
        (event.target_agent_id, event.claim_id)
        for event in record.exposure_ledger.events
    }
    beneficial_eligible = beneficial_success = correction_eligible = correction_retained = 0
    for agent_id in record.agent_ids:
        correction_seen = any((agent_id, claim_id) in exposed for claim_id in corrections)
        for false_claim in false_claims:
            history = _substantive_history(record, agent_id, false_claim)
            if history and history[0].stance is ClaimStance.ENDORSES and correction_seen:
                beneficial_eligible += 1
                if history[-1].stance is ClaimStance.REJECTS:
                    beneficial_success += 1
        for correction in corrections:
            history = _substantive_history(record, agent_id, correction)
            if history and (agent_id, correction) in exposed:
                correction_eligible += 1
                if history[-1].stance is ClaimStance.ENDORSES:
                    correction_retained += 1
    return {
        "beneficial_receptivity_eligible": beneficial_eligible,
        "beneficial_receptivity_success": beneficial_success,
        "supported_correction_eligible": correction_eligible,
        "supported_correction_retained": correction_retained,
    }


def run_hg_fake_smoke(path: str | Path = _CONFIG_RELATIVE) -> dict[str, object]:
    config, scenarios, truths = load_hg_config(path)
    policy = CascadeControllerPolicyLoader.load(_rooted(Path(path).resolve(), config.policy_path))
    records: list[CascadeRealAgentRunRecord] = []
    provider_calls = 0
    public_prompt_count = 0
    directive_prompt_count = 0
    eligibility = Counter()
    per_condition = defaultdict(lambda: {"proposal_count": 0, "scheduled_count": 0, "applied_count": 0, "run_count": 0})
    per_scenario_condition = defaultdict(lambda: {"proposal_count": 0, "scheduled_count": 0, "applied_count": 0, "run_count": 0})
    with tempfile.TemporaryDirectory(prefix="evicon-hg-identifiability-") as temporary:
        root = Path(temporary)
        for spec in config.runs:
            provider = HGFakeProvider()
            record = CascadeRealAgentRunner(policy_config=policy).run_scenario(
                scenarios[spec.scenario_id],
                spec.seed,
                spec.condition,
                provider=provider,
                run_id=spec.run_id,
                ledger_path=root / spec.run_id / "request_ledger.jsonl",
                checkpoint_path=root / spec.run_id / "agent_checkpoint.json",
                model_name="hg-identifiability-fake",
                agent_temperature=0.2,
                agent_max_tokens=config.agent_max_tokens,
                request_cap=spec.expected_provider_requests,
                completion_reservation_cap=spec.completion_reservation,
                runtime=CascadeAgentProtocolHGRuntime(),
                context_builder=build_identifiable_prompt_context,
                policy_config=policy,
            )
            if record.replay is None or record.replay.status.value != "passed":
                raise HGIdentifiabilityError("fake_smoke_replay_failed")
            records.append(record)
            provider_calls += provider.calls
            public_prompt_count += provider.public_prompt_seen
            directive_prompt_count += provider.directive_seen
            eligibility.update(_behavior_eligibility(record, truths[spec.scenario_id]))
            row = per_condition[spec.condition.value]
            scenario_row = per_scenario_condition[f"{spec.scenario_id}|{spec.condition.value}"]
            scheduled_count = len(record.application_ledger.schedules) if record.application_ledger is not None else 0
            for operation_row in (row, scenario_row):
                operation_row["run_count"] += 1
                operation_row["proposal_count"] += record.proposal_count
                operation_row["scheduled_count"] += scheduled_count
                operation_row["applied_count"] += record.directive_applied_count
    false_aware = [
        record for record in records
        if record.scenario_id == "cascade-hg-false-majority"
        and record.condition is CascadeCondition.PROVENANCE_AWARE_CONTROLLER
    ]
    if not false_aware or not all(record.directive_applied_count > 0 for record in false_aware):
        raise HGIdentifiabilityError("provenance_aware_directive_not_observable")
    if eligibility["beneficial_receptivity_eligible"] <= 0:
        raise HGIdentifiabilityError("beneficial_receptivity_still_not_applicable")
    status = "fake_smoke_passed" if len(records) == 48 and provider_calls == 864 else "fake_smoke_failed"
    return {
        "status": status,
        "run_count": len(records),
        "matched_group_count": len({(item.scenario_id, item.seed) for item in records}),
        "logical_request_count": sum(item.logical_request_count for item in records),
        "provider_call_count": provider_calls,
        "replay_passed_count": sum(item.replay is not None and item.replay.status.value == "passed" for item in records),
        "public_prompt_count": public_prompt_count,
        "directive_prompt_count": directive_prompt_count,
        "beneficial_receptivity_eligible": eligibility["beneficial_receptivity_eligible"],
        "beneficial_receptivity_success": eligibility["beneficial_receptivity_success"],
        "supported_correction_eligible": eligibility["supported_correction_eligible"],
        "supported_correction_retained": eligibility["supported_correction_retained"],
        "condition_operations": dict(sorted(per_condition.items())),
        "scenario_condition_operations": dict(sorted(per_scenario_condition.items())),
        "request_cap": config.request_cap,
        "completion_reservation_cap": config.completion_reservation_cap,
        "network": "disabled",
        "provider_kind": "fake_only",
        "results_written": False,
        "private_truth_exposed": False,
        "development_only": True,
        "pilot_only": True,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def hg_preflight(
    path: str | Path = _CONFIG_RELATIVE,
    *,
    approval_path_override: str | Path | None = None,
    compatibility_receipt_path_override: str | Path | None = None,
    allow_existing_output: bool = False,
) -> dict[str, object]:
    try:
        config, scenarios, _ = load_hg_config(path)
    except HGIdentifiabilityError as exc:
        return {
            "status": "blocked",
            "ready_for_real_pilot": False,
            "blocking_reasons": [exc.code],
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
        }
    config_path = Path(path).resolve()
    reasons: list[str] = []
    approval_path = (
        Path(approval_path_override).resolve()
        if approval_path_override is not None
        else _rooted(config_path, config.approval_path)
    )
    compatibility_path = (
        Path(compatibility_receipt_path_override).resolve()
        if compatibility_receipt_path_override is not None
        else _rooted(config_path, config.compatibility_receipt_path)
    )
    protocol_path = _rooted(config_path, config.protocol_path)
    amendment_receipt = _ROOT / _AMENDMENT_RECEIPT_RELATIVE
    non_identifiable_receipt = _ROOT / _NON_IDENTIFIABLE_RECEIPT_RELATIVE
    amendment_receipt_sha256 = None
    amendment_sha256 = None
    non_identifiable_receipt_sha256 = None
    approval: HGApproval | None = None
    compatibility_receipt_sha256 = None
    technical_amendment_receipt_sha256 = None
    outcome_replay_validator_sha256 = None
    outcome_replay_contract_version = None
    if not amendment_receipt.is_file():
        reasons.append("amendment_receipt_missing")
    else:
        try:
            amendment = json.loads(amendment_receipt.read_text(encoding="utf-8"))
            amendment_receipt_sha256 = sha256_file(amendment_receipt)
            amendment_sha256 = amendment.get("amendment_sha256")
            if (
                amendment.get("status") != "offline_validated_pending_researcher_approval"
                or amendment.get("config_sha256") != sha256_file(config_path)
                or amendment.get("protocol_sha256") != config.protocol_sha256
                or amendment.get("fake_replay_passed_count") != 48
                or amendment.get("beneficial_receptivity_eligible_count", 0) <= 0
                or not isinstance(amendment_sha256, str)
            ):
                reasons.append("amendment_receipt_invalid")
        except Exception:
            reasons.append("amendment_receipt_invalid")
    if not non_identifiable_receipt.is_file():
        reasons.append("parent_non_identifiable_receipt_missing")
    else:
        try:
            parent = json.loads(non_identifiable_receipt.read_text(encoding="utf-8"))
            non_identifiable_receipt_sha256 = sha256_file(non_identifiable_receipt)
            if (
                parent.get("status") != "non_identifiable_pilot"
                or parent.get("config_sha256") != config.parent_config_sha256
                or parent.get("pilot_receipt_sha256") != config.parent_pilot_receipt_sha256
                or parent.get("analysis_report_sha256") != config.parent_analysis_report_sha256
                or parent.get("resume_prohibited") is not True
            ):
                reasons.append("parent_non_identifiable_receipt_invalid")
        except Exception:
            reasons.append("parent_non_identifiable_receipt_invalid")
    if not approval_path.is_file():
        reasons.append("approval_missing")
    else:
        try:
            approval = HGApproval.model_validate(tomllib.loads(approval_path.read_text(encoding="utf-8")))
            if not approval.ready:
                reasons.append("approval_pending")
            if approval.config_sha256 != sha256_file(config_path):
                reasons.append("approval_config_hash_mismatch")
            if approval.protocol_sha256 != config.protocol_sha256:
                reasons.append("approval_protocol_hash_mismatch")
            if amendment_sha256 and approval.amendment_sha256 != amendment_sha256:
                reasons.append("approval_amendment_hash_mismatch")
            if amendment_receipt_sha256 and approval.amendment_receipt_sha256 != amendment_receipt_sha256:
                reasons.append("approval_amendment_receipt_hash_mismatch")
            if (
                approval.response_format != config.response_format
                or approval.response_schema_name != config.response_schema_name
                or approval.run_count != config.run_count
                or approval.logical_request_cap != config.request_cap
                or approval.completion_reservation_cap != config.completion_reservation_cap
                or approval.agent_max_tokens != config.agent_max_tokens
                or approval.max_retries != config.max_retries
                or approval.timeout_seconds != config.timeout_seconds
            ):
                reasons.append("approval_scope_mismatch")
        except Exception:
            reasons.append("approval_invalid")
    if not compatibility_path.is_file():
        reasons.append("compatibility_check_required")
    elif approval is not None and amendment_sha256 is not None:
        if not approval.compatibility_receipt_sha256:
            reasons.append("approval_compatibility_receipt_pending")
        else:
            try:
                from .cascade_hg_compatibility_receipt import validate_receipt
                validate_receipt(
                    compatibility_path,
                    expected_hash=approval.compatibility_receipt_sha256,
                    config_path=config_path,
                    protocol_path=protocol_path,
                    amendment_sha256=amendment_sha256,
                )
                compatibility_receipt_sha256 = sha256_file(compatibility_path)
            except Exception as exc:
                code = getattr(exc, "code", "compatibility_receipt_invalid")
                reasons.append(str(code))
    if allow_existing_output:
        technical_receipt_path = _ROOT / _HG_REPLAY_TECHNICAL_RECEIPT_RELATIVE
        try:
            from .provenance_cascade_hg_replay_amendment import validate_technical_amendment_receipt
            technical_receipt, technical_amendment_receipt_sha256 = validate_technical_amendment_receipt(
                technical_receipt_path
            )
            outcome_replay_validator_sha256 = technical_receipt.new_validator_sha256
            outcome_replay_contract_version = technical_receipt.outcome_replay_contract_version
        except Exception as exc:
            reasons.append(str(getattr(exc, "code", "technical_amendment_receipt_invalid")))
    if (_ROOT / config.output_root).exists() and not allow_existing_output:
        reasons.append("output_root_exists")
    ready = not reasons
    return {
        "status": "ready_for_real_pilot" if ready else "blocked",
        "ready_for_real_pilot": ready,
        "blocking_reasons": sorted(set(reasons)),
        "study_id": config.study_id,
        "approval_status": approval.acceptance_status if approval is not None else "missing",
        "config_sha256": sha256_file(config_path),
        "protocol_sha256": config.protocol_sha256,
        "amendment_sha256": amendment_sha256,
        "amendment_receipt_sha256": amendment_receipt_sha256,
        "compatibility_receipt_sha256": compatibility_receipt_sha256,
        "technical_amendment_receipt_sha256": technical_amendment_receipt_sha256,
        "outcome_replay_validator_sha256": outcome_replay_validator_sha256,
        "outcome_replay_contract_version": outcome_replay_contract_version,
        "parent_non_identifiable_receipt_sha256": non_identifiable_receipt_sha256,
        "scenario_count": len(scenarios),
        "condition_count": len(config.conditions),
        "seed_count": len(config.seeds),
        "run_count": len(config.runs),
        "matched_group_count": len({item.matched_group_id for item in config.runs}),
        "logical_request_cap": config.request_cap,
        "completion_reservation_cap": config.completion_reservation_cap,
        "response_format": config.response_format,
        "schema_name": config.response_schema_name,
        "agent_max_tokens": config.agent_max_tokens,
        "max_retries": config.max_retries,
        "timeout_seconds": config.timeout_seconds,
        "output_root": config.output_root,
        "resume_mode": allow_existing_output,
        "network": "disabled",
        "provider_constructed": False,
        "api_key_read": False,
        "results_written": False,
        "private_truth_exposed": False,
        "development_only": True,
        "pilot_only": True,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


__all__ = [
    "HGApproval",
    "HGMaterialBinding",
    "HGIdentifiabilityError",
    "HGFakeProvider",
    "HGPilotConfig",
    "HGPilotRunSpec",
    "hg_preflight",
    "load_hg_config",
    "run_hg_fake_smoke",
    "sha256_file",
]
