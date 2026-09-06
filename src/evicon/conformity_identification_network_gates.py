"""Explicit, bounded network gates for identification qualification.

The default entry points do not construct a Provider or create output paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .conformity_identification import IdentificationError, IdentificationStage, load_identification_config, safe_json, sha256_file
from .conformity_identification_compatibility import RESPONSE_SCHEMA
from .conformity_identification_protocol import parse_identification_response, render_identification_turn
from .conformity_identification_protocol_v11 import BranchPromptContext, parse_branch_response, render_branch_turn
from .conformity_identification_qualification import (
    DEFAULT_QUALIFICATION_APPROVAL,
    approval_sha256,
    load_qualification_approval,
)
from .conformity_identification_smoke import QualificationFakeProvider, _context
from .conformity_identification_stability import DEFAULT_STABILITY_CONFIG, load_stability_config
from .conformity_identification_v11 import load_v11_config
from .conformity_source_manipulation import (
    DEFAULT_MANIPULATION_CONFIG,
    parse_manipulation_response,
    render_manipulation_request,
)
from .conformity_source_manipulation_smoke import ManipulationFakeProvider, build_cases
from .llm_contract import LLMProvider, LLMProviderError
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ResponseFormatMode
from .request_ledger import LedgeredProvider, RequestLedger

_ROOT = Path(__file__).resolve().parents[2]
MANIPULATION_OUTPUT = "outputs/conformity-source-manipulation-v1"
STABILITY_OUTPUT = "outputs/conformity-identification-stability-v1"
MANIPULATION_SCHEMA_NAME = "conformity_source_structure_response_v1"
MANIPULATION_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "visible_message_count": {"type": "integer", "enum": [2]},
        "visible_source_root_count": {"type": "integer", "enum": [0, 1, 2]},
        "root_assignments": {
            "type": "array", "minItems": 2, "maxItems": 2,
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "content_id": {"type": "string"},
                    "source_root_id": {"type": ["string", "null"]},
                },
                "required": ["content_id", "source_root_id"],
            },
        },
    },
    "required": ["visible_message_count", "visible_source_root_count", "root_assignments"],
}


class GateRunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["completed", "failed", "network_disabled", "blocked"]
    gate_id: str
    logical_request_count: int = Field(ge=0)
    transport_attempt_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    error_code: str | None = None
    receipt_path: str | None = None
    network: Literal["disabled", "enabled"]
    provider_constructed: bool
    results_written: bool
    private_truth_exposed: Literal[False] = False
    behavior_effect_estimated: Literal[False] = False
    development_only: Literal[True] = True
    not_paper_result: Literal[True] = True
    no_causal_conclusion: Literal[True] = True


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def _ensure_output(path: str | Path, allowed: str) -> Path:
    target = _resolve(path)
    try:
        target.relative_to(_ROOT)
        inside_repository = True
    except ValueError:
        inside_repository = False
    if inside_repository and target != _resolve(allowed):
        # Injected FakeProvider tests may use an absolute directory outside the
        # repository; any repository-local execution remains fixed.
        raise IdentificationError("qualification_output_path_mismatch")
    if target.exists():
        raise IdentificationError("qualification_output_exists")
    return target


def _safe_failure(gate_id: str, exc: Exception, *, network: str, provider_constructed: bool, results_written: bool) -> GateRunSummary:
    code = getattr(exc, "code", None)
    if hasattr(code, "value"):
        code = code.value
    if not isinstance(code, str):
        code = str(exc) if isinstance(exc, IdentificationError) else "qualification_gate_failed"
    return GateRunSummary(
        status="failed" if provider_constructed else "blocked", gate_id=gate_id,
        logical_request_count=0, transport_attempt_count=0, completed_count=0,
        failed_count=1 if provider_constructed else 0, error_code=code,
        network=network, provider_constructed=provider_constructed,
        results_written=results_written,
    )


def _write_receipt(target: Path, name: str, payload: dict[str, object]) -> str:
    target.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2)
    receipt = target / name
    if receipt.exists():
        raise IdentificationError("qualification_receipt_exists")
    receipt.write_text(serialized, encoding="utf-8")
    return str(receipt.relative_to(_ROOT)) if receipt.is_relative_to(_ROOT) else str(receipt)


def run_manipulation_gate(
    provider: LLMProvider,
    *,
    output_root: str | Path = MANIPULATION_OUTPUT,
    config_path: str | Path = DEFAULT_MANIPULATION_CONFIG,
    approval_path: str | Path = DEFAULT_QUALIFICATION_APPROVAL,
) -> GateRunSummary:
    gate_id = "evicon-conformity-source-manipulation-v1"
    try:
        approval = load_qualification_approval(approval_path)
        if approval.acceptance_status != "accepted":
            raise IdentificationError("exact_hash_human_approval_required")
        target = _ensure_output(output_root, MANIPULATION_OUTPUT)
        ledger = RequestLedger(target / "request_ledger.jsonl")
        wrapped = LedgeredProvider(
            provider, ledger=ledger, request_cap=12,
            completion_reservation_cap=1536, resume=False,
        )
        roots: Counter[str] = Counter()
        for context in build_cases(str(config_path)):
            request = render_manipulation_request(context)
            request = request.model_copy(update={"metadata": {
                **request.metadata, "protocol": context.case_id,
                "agent_id": "source-structure-check", "round_id": 0,
            }})
            response = wrapped.complete(request)
            parsed = parse_manipulation_response(response.content, context)
            roots[context.projection.value] += parsed.visible_source_root_count
        summary = ledger.summary(request_cap=12, completion_reservation_cap=1536)
        if summary.completed_count != 12 or summary.failed_count:
            raise IdentificationError("manipulation_request_completeness_failed")
        payload = {
            "status": "completed", "gate_id": gate_id,
            "approval_sha256": approval_sha256(approval_path),
            "config_sha256": sha256_file(config_path),
            "logical_request_count": summary.unique_logical_request_count,
            "transport_attempt_count": summary.transport_attempt_count,
            "root_count_totals": dict(sorted(roots.items())),
            "behavior_effect_estimated": False, "joined_to_behavior_study": False,
            "network_used": True, "private_truth_exposed": False,
            "not_paper_result": True, "no_causal_conclusion": True,
        }
        receipt_path = _write_receipt(target, "source_manipulation_receipt.json", payload)
        return GateRunSummary(
            status="completed", gate_id=gate_id,
            logical_request_count=summary.unique_logical_request_count,
            transport_attempt_count=summary.transport_attempt_count,
            completed_count=summary.completed_count, failed_count=0,
            receipt_path=receipt_path, network="enabled", provider_constructed=True,
            results_written=True,
        )
    except Exception as exc:
        return _safe_failure(gate_id, exc, network="enabled", provider_constructed=True, results_written=_resolve(output_root).exists())


def run_stability_gate(
    provider: LLMProvider,
    *,
    output_root: str | Path = STABILITY_OUTPUT,
    config_path: str | Path = DEFAULT_STABILITY_CONFIG,
    approval_path: str | Path = DEFAULT_QUALIFICATION_APPROVAL,
) -> GateRunSummary:
    gate_id = "evicon-conformity-identification-stability-v1"
    try:
        approval = load_qualification_approval(approval_path)
        if approval.acceptance_status != "accepted":
            raise IdentificationError("exact_hash_human_approval_required")
        config, _ = load_stability_config(config_path)
        target = _ensure_output(output_root, STABILITY_OUTPUT)
        study, _ = load_v11_config(config.study_config_path)
        _, scenarios = load_identification_config(study.parent_config_path)
        scenario = next(item for item in scenarios if item.scenario_id == config.scenario_id)
        ledger = RequestLedger(target / "request_ledger.jsonl")
        wrapped = LedgeredProvider(
            provider, ledger=ledger, request_cap=config.logical_request_cap,
            completion_reservation_cap=config.completion_reservation_cap, resume=False,
        )
        stage_counts: Counter[str] = Counter()
        for agent_id in study.agent_ids:
            initial_context = _context(
                scenario, config.condition, config.seed, agent_id,
                IdentificationStage.INITIAL_PRIVATE,
            )
            initial_request = render_identification_turn(initial_context)
            initial_request = initial_request.model_copy(update={"metadata": {
                **initial_request.metadata, "protocol": f"stability:{config.condition.value}",
                "agent_id": agent_id, "round_id": 0,
            }})
            initial = parse_identification_response(wrapped.complete(initial_request).content, initial_context)
            stage_counts[IdentificationStage.INITIAL_PRIVATE.value] += 1
            initial_hash = hashlib.sha256(initial.model_dump_json().encode()).hexdigest()
            for round_id, stage in enumerate(study.continuation_stages, start=1):
                public = _context(scenario, config.condition, config.seed, agent_id, stage)
                branch = BranchPromptContext(
                    public_context=public,
                    matched_group_id=f"stability-{scenario.scenario_id}-{config.seed}",
                    branch_id=f"stability-{scenario.scenario_id}-{config.seed}-{config.condition.value}",
                    shared_t0_observation_sha256=initial_hash,
                )
                branch_request = render_branch_turn(branch)
                branch_request = branch_request.model_copy(update={"metadata": {
                    **branch_request.metadata, "protocol": f"stability:{config.condition.value}",
                    "agent_id": agent_id, "round_id": round_id,
                }})
                parse_branch_response(wrapped.complete(branch_request).content, branch)
                stage_counts[stage.value] += 1
        summary = ledger.summary(
            request_cap=config.logical_request_cap,
            completion_reservation_cap=config.completion_reservation_cap,
        )
        if summary.completed_count != 30 or summary.failed_count:
            raise IdentificationError("stability_request_completeness_failed")
        payload = {
            "status": "completed", "gate_id": gate_id,
            "approval_sha256": approval_sha256(approval_path),
            "config_sha256": sha256_file(config_path),
            "logical_request_count": summary.unique_logical_request_count,
            "transport_attempt_count": summary.transport_attempt_count,
            "stage_counts": dict(sorted(stage_counts.items())),
            "parser_invalid_count": 0, "effect_estimated": False,
            "network_used": True, "private_truth_exposed": False,
            "not_paper_result": True, "no_causal_conclusion": True,
        }
        receipt_path = _write_receipt(target, "protocol_stability_receipt.json", payload)
        return GateRunSummary(
            status="completed", gate_id=gate_id,
            logical_request_count=summary.unique_logical_request_count,
            transport_attempt_count=summary.transport_attempt_count,
            completed_count=summary.completed_count, failed_count=0,
            receipt_path=receipt_path, network="enabled", provider_constructed=True,
            results_written=True,
        )
    except Exception as exc:
        return _safe_failure(gate_id, exc, network="enabled", provider_constructed=True, results_written=_resolve(output_root).exists())


def _provider(
    *, environment: Mapping[str, str] | None, max_tokens: int, temperature: float,
    seed: int, schema_name: str, schema: dict[str, object],
) -> OpenAICompatibleProvider:
    base = ProviderConfig.from_env(allow_network=True, environment=environment)
    config = base.model_copy(update={
        "allow_network": True, "timeout_seconds": 5.0, "max_retries": 0,
        "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
        "response_format": ResponseFormatMode.JSON_SCHEMA,
        "response_schema_name": schema_name, "response_schema": schema,
    })
    return OpenAICompatibleProvider(config, environment=environment)


def _blocked(gate_id: str, error: str) -> GateRunSummary:
    return GateRunSummary(
        status="network_disabled" if error == "network_authorization_required" else "blocked",
        gate_id=gate_id, logical_request_count=0, transport_attempt_count=0,
        completed_count=0, failed_count=0, error_code=error, network="disabled",
        provider_constructed=False, results_written=False,
    )


def manipulation_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded source manipulation qualification gate")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--confirm-request-cap", type=int)
    parser.add_argument("--confirm-completion-reservation-cap", type=int)
    parser.add_argument("--fake-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.fake_smoke:
        from .conformity_source_manipulation_smoke import run_fake_smoke
        payload = run_fake_smoke()
    elif not args.allow_network:
        payload = _blocked("evicon-conformity-source-manipulation-v1", "network_authorization_required").model_dump(mode="json")
    elif not args.confirm_run or args.confirm_request_cap != 12 or args.confirm_completion_reservation_cap != 1536:
        payload = _blocked("evicon-conformity-source-manipulation-v1", "explicit_cap_confirmation_required").model_dump(mode="json")
    else:
        try:
            approval = load_qualification_approval()
            if approval.acceptance_status != "accepted":
                raise IdentificationError("exact_hash_human_approval_required")
            provider = _provider(
                environment=None, max_tokens=128, temperature=0.0, seed=20261100,
                schema_name=MANIPULATION_SCHEMA_NAME, schema=MANIPULATION_SCHEMA,
            )
            payload = run_manipulation_gate(provider).model_dump(mode="json")
        except Exception as exc:
            payload = _blocked("evicon-conformity-source-manipulation-v1", getattr(exc, "code", "gate_precondition_failed")).model_dump(mode="json")
    print(safe_json(payload))
    return 0 if payload.get("status") in {"completed", "network_disabled", "manipulation_fake_smoke_passed"} else 1


def stability_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded identification stability qualification gate")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--confirm-request-cap", type=int)
    parser.add_argument("--confirm-completion-reservation-cap", type=int)
    parser.add_argument("--fake-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.fake_smoke:
        from .conformity_identification_stability import run_fake_smoke
        payload = run_fake_smoke()
    elif not args.allow_network:
        payload = _blocked("evicon-conformity-identification-stability-v1", "network_authorization_required").model_dump(mode="json")
    elif not args.confirm_run or args.confirm_request_cap != 30 or args.confirm_completion_reservation_cap != 15360:
        payload = _blocked("evicon-conformity-identification-stability-v1", "explicit_cap_confirmation_required").model_dump(mode="json")
    else:
        try:
            approval = load_qualification_approval()
            if approval.acceptance_status != "accepted":
                raise IdentificationError("exact_hash_human_approval_required")
            provider = _provider(
                environment=None, max_tokens=512, temperature=0.2, seed=20261101,
                schema_name="conformity_identification_response_v1", schema=RESPONSE_SCHEMA,
            )
            payload = run_stability_gate(provider).model_dump(mode="json")
        except Exception as exc:
            payload = _blocked("evicon-conformity-identification-stability-v1", getattr(exc, "code", "gate_precondition_failed")).model_dump(mode="json")
    print(safe_json(payload))
    return 0 if payload.get("status") in {"completed", "network_disabled", "stability_fake_smoke_passed"} else 1


__all__ = [
    "GateRunSummary", "MANIPULATION_SCHEMA", "run_manipulation_gate",
    "run_stability_gate", "manipulation_main", "stability_main",
]
