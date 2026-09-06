"""H-D.2 versioned strict JSON protocol for provenance-cascade Agents.

This sidecar deliberately leaves the v1 Agent parser/runtime and all historical
pilot artifacts untouched.  It reports only stable diagnostic categories.
"""
from __future__ import annotations

import hashlib
import json
import re
import tomllib
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .cascade_agent_prompts import CascadeAgentPromptContext, render_cascade_agent_turn
from .cascade_agent_runtime import CascadeAgentRuntimeAudit, CascadeAgentRuntimeStatus
from .cascade_outcomes import ClaimStance
from .cascade_protocol import CascadeScenarioLoader, CascadeScenarioSpec
from .cascade_real_agent_runner import CascadeRealAgentRunError
from .llm_contract import LLMProvider, LLMProviderError, LLMRequest
from .provenance_cascade_amendment import load_hd_config
from .provenance_cascade_preregistration import CascadeCondition
from .request_ledger import RequestLedgerError

V2_TEMPLATE_VERSION = "cascade_agent_turn.v2.strict_json"
V2_PROTOCOL_VERSION = "provenance_cascade_agent_protocol.v2"
V2_SCHEMA_NAME = "cascade_agent_response_v2"
_AGENT_IDS = tuple(f"network-agent-{index:02d}" for index in range(1, 7))
_CONDITIONS = tuple(CascadeCondition)
_SCENARIO_IDS = (
    "cascade-false-majority",
    "cascade-true-minority-correction",
    "cascade-independent-true-consensus",
    "cascade-unresolved-disagreement",
)
_SCENARIO_TYPES = {
    "cascade-false-majority": "false_majority",
    "cascade-true-minority-correction": "true_minority_correction",
    "cascade-independent-true-consensus": "independent_true_consensus",
    "cascade-unresolved-disagreement": "unresolved_disagreement",
}
_SHA = re.compile(r"^[0-9a-f]{64}$")


class CascadeAgentV2Diagnostic(str, Enum):
    """Stable parser classes; no field values or raw response are retained."""

    MALFORMED_JSON = "malformed_json"
    TOP_LEVEL_TYPE = "top_level_type"
    MISSING_FIELD = "missing_field"
    EXTRA_FIELD = "extra_field"
    FIELD_TYPE = "field_type"
    INVALID_STANCE = "invalid_stance"
    UNAVAILABLE_CONTENT_ID = "unavailable_content_id"
    UNAVAILABLE_EVIDENCE_ID = "unavailable_evidence_id"
    SHARE_FIELD = "share_field"


class CascadeAgentV2Response(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stance: ClaimStance
    content_ids_used: tuple[str, ...]
    evidence_ids_used: tuple[str, ...]
    share_content_id: str | None
    valid: bool = True
    validation_errors: tuple[CascadeAgentV2Diagnostic, ...] = ()

    @model_validator(mode="after")
    def valid_shape(self) -> "CascadeAgentV2Response":
        if self.valid and self.validation_errors:
            raise ValueError("valid response cannot contain diagnostics")
        if not self.valid and not self.validation_errors:
            raise ValueError("invalid response must contain diagnostics")
        return self


def _invalid(code: CascadeAgentV2Diagnostic) -> CascadeAgentV2Response:
    return CascadeAgentV2Response(
        stance=ClaimStance.NO_POSITION,
        content_ids_used=(), evidence_ids_used=(), share_content_id=None,
        valid=False, validation_errors=(code,),
    )


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate")
        result[key] = value
    return result


def _private(value: object) -> bool:
    tokens = ("ground_truth_label", "source_independence_label", "hidden_probe", "hidden_profile", "api_key", "authorization", "private_truth")
    if isinstance(value, str):
        lowered = value.lower()
        return any(token in lowered for token in tokens)
    if isinstance(value, dict):
        return any(_private(k) or _private(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_private(v) for v in value)
    return False


def parse_cascade_agent_response_v2(content: str, context: CascadeAgentPromptContext) -> CascadeAgentV2Response:
    try:
        decoded = json.loads(content, object_pairs_hook=_pairs)
    except ValueError as exc:
        return _invalid(CascadeAgentV2Diagnostic.EXTRA_FIELD if str(exc) == "duplicate" else CascadeAgentV2Diagnostic.MALFORMED_JSON)
    if not isinstance(decoded, dict):
        return _invalid(CascadeAgentV2Diagnostic.TOP_LEVEL_TYPE)
    allowed = {"stance", "content_ids_used", "evidence_ids_used", "share_content_id"}
    if set(decoded) - allowed:
        return _invalid(CascadeAgentV2Diagnostic.EXTRA_FIELD)
    if _private(decoded):
        return _invalid(CascadeAgentV2Diagnostic.EXTRA_FIELD)
    if set(decoded) != allowed:
        return _invalid(CascadeAgentV2Diagnostic.MISSING_FIELD)
    if not isinstance(decoded["stance"], str):
        return _invalid(CascadeAgentV2Diagnostic.FIELD_TYPE)
    try:
        stance = ClaimStance(decoded["stance"])
    except (TypeError, ValueError):
        return _invalid(CascadeAgentV2Diagnostic.INVALID_STANCE)
    for field in ("content_ids_used", "evidence_ids_used"):
        value = decoded[field]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value) or len(set(value)) != len(value):
            return _invalid(CascadeAgentV2Diagnostic.FIELD_TYPE)
    share = decoded["share_content_id"]
    if share is not None and not isinstance(share, str):
        return _invalid(CascadeAgentV2Diagnostic.SHARE_FIELD)
    visible_content = {item.content_id for item in context.contents}
    visible_evidence = {item.evidence_id for item in context.evidence}
    content_ids = tuple(decoded["content_ids_used"])
    evidence_ids = tuple(decoded["evidence_ids_used"])
    if not set(content_ids).issubset(visible_content):
        return _invalid(CascadeAgentV2Diagnostic.UNAVAILABLE_CONTENT_ID)
    if not set(evidence_ids).issubset(visible_evidence):
        return _invalid(CascadeAgentV2Diagnostic.UNAVAILABLE_EVIDENCE_ID)
    if share is not None and (share not in visible_content or share not in content_ids):
        return _invalid(CascadeAgentV2Diagnostic.SHARE_FIELD)
    return CascadeAgentV2Response(
        stance=stance, content_ids_used=content_ids, evidence_ids_used=evidence_ids,
        share_content_id=share, valid=True, validation_errors=(),
    )


def render_cascade_agent_turn_v2(context: CascadeAgentPromptContext) -> LLMRequest:
    """Render the old public projection under a new, fingerprintable contract."""
    base = render_cascade_agent_turn(context)
    payload = json.loads(base.user_prompt)
    payload["template_version"] = V2_TEMPLATE_VERSION
    payload["protocol_version"] = V2_PROTOCOL_VERSION
    user_prompt = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    system_prompt = (
        "Return exactly one JSON object with exactly four fields: stance, content_ids_used, "
        "evidence_ids_used, share_content_id. JSON only; no Markdown, no extra fields, and no omitted fields. "
        "Use only currently visible public content and evidence. A directive is a process constraint, not a fact, evidence, source root, or truth label."
    )
    request_id = "v2-" + hashlib.sha256((context.scenario_id + "|" + context.agent_id + "|" + str(context.round_id) + "|" + user_prompt).encode()).hexdigest()[:24]
    metadata = {**base.metadata, "contract_version": V2_PROTOCOL_VERSION, "template_version": V2_TEMPLATE_VERSION}
    return base.model_copy(update={"request_id": request_id, "system_prompt": system_prompt, "user_prompt": user_prompt, "metadata": metadata})


class CascadeAgentV2RuntimeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    status: CascadeAgentRuntimeStatus
    response: CascadeAgentV2Response | None = None
    audit: CascadeAgentRuntimeAudit


class CascadeAgentProtocolV2Runtime:
    """One-call v2 runtime. It never writes state or retries outside Provider."""

    def render_request(self, context: CascadeAgentPromptContext) -> LLMRequest:
        return render_cascade_agent_turn_v2(context)

    def execute(self, context: CascadeAgentPromptContext, provider: LLMProvider, *, request_metadata: Mapping[str, Any] | None = None) -> CascadeAgentV2RuntimeResult:
        try:
            request = self.render_request(context)
            if request_metadata:
                allowed = {key: value for key, value in request_metadata.items() if key in {"protocol", "condition", "phase", "run_id", "seed", "matched_group_id"}}
                request = request.model_copy(update={"metadata": {**request.metadata, **allowed}})
        except Exception:
            return self._result(context, CascadeAgentRuntimeStatus.RENDER_ERROR, error_code="request_render_failed")
        try:
            response = provider.complete(request)
        except LLMProviderError as error:
            ledger_codes = {"completed_request_fingerprint_exists", "failed_request_requires_resume", "incomplete_request_requires_resume", "request_cap_reached", "completion_reservation_cap_reached", "parser_recovery_fingerprint_changed", "parser_recovery_attempt_limit_reached"}
            code = error.message if isinstance(error, RequestLedgerError) and error.message in ledger_codes else error.code.value
            return self._result(context, CascadeAgentRuntimeStatus.PROVIDER_ERROR, request_id=request.request_id, error_code=code)
        except Exception:
            return self._result(context, CascadeAgentRuntimeStatus.PROVIDER_ERROR, request_id=request.request_id, error_code="provider_failure")
        parsed = parse_cascade_agent_response_v2(response.content, context)
        status = CascadeAgentRuntimeStatus.COMPLETED if parsed.valid else CascadeAgentRuntimeStatus.PARSER_INVALID
        return self._result(context, status, response=parsed, request_id=request.request_id,
                            model_name=response.model_name, finish_reason=response.finish_reason,
                            prompt_tokens=response.prompt_tokens, completion_tokens=response.completion_tokens,
                            total_tokens=response.total_tokens, latency_ms=response.latency_ms,
                            parser_valid=parsed.valid,
                            content_ids_used_count=len(parsed.content_ids_used) if parsed.valid else 0,
                            evidence_ids_used_count=len(parsed.evidence_ids_used) if parsed.valid else 0,
                            share_requested=parsed.share_content_id is not None if parsed.valid else False,
                            error_code=parsed.validation_errors[0].value if parsed.validation_errors else None)

    @staticmethod
    def _result(context: CascadeAgentPromptContext, status: CascadeAgentRuntimeStatus, *, response: CascadeAgentV2Response | None = None, request_id: str | None = None, model_name: str | None = None, finish_reason: str | None = None, prompt_tokens: int | None = None, completion_tokens: int | None = None, total_tokens: int | None = None, latency_ms: float | None = None, parser_valid: bool | None = None, content_ids_used_count: int = 0, evidence_ids_used_count: int = 0, share_requested: bool = False, error_code: str | None = None) -> CascadeAgentV2RuntimeResult:
        return CascadeAgentV2RuntimeResult(status=status, response=response, audit=CascadeAgentRuntimeAudit(
            status=status, scenario_id=context.scenario_id, agent_id=context.agent_id, round_id=context.round_id,
            template_version=V2_TEMPLATE_VERSION, model_name=model_name or context.runtime_config.model_name,
            request_id=request_id, finish_reason=finish_reason, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, total_tokens=total_tokens, latency_ms=latency_ms,
            parser_valid=parser_valid, content_ids_used_count=content_ids_used_count,
            evidence_ids_used_count=evidence_ids_used_count, share_requested=share_requested, error_code=error_code))


class HD2ConfigError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class HD2RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    matched_group_id: str
    scenario_id: str
    scenario_type: str
    condition: CascadeCondition
    seed: int
    agent_ids: tuple[str, ...]
    round_count: Literal[3]
    topology_id: Literal["ring_6_bidirectional"]
    expected_provider_requests: Literal[18]
    completion_reservation: Literal[4608]


class HD2Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    study_id: Literal["evicon-provenance-cascade-pilot-hd2"]
    config_version: Literal["provenance_cascade_hd2.v1"]
    protocol_version: Literal["provenance_cascade_agent_protocol.v2"]
    template_version: Literal["cascade_agent_turn.v2.strict_json"]
    status: Literal["offline_only"]
    development_only: Literal[True]
    calibration_only: Literal[False]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    parent_config_path: str
    parent_config_sha256: str
    scenario_ids: tuple[str, ...]
    conditions: tuple[CascadeCondition, ...]
    seeds: tuple[int, ...]
    agent_ids: tuple[str, ...]
    max_rounds: Literal[3]
    topology_id: Literal["ring_6_bidirectional"]
    request_cap: Literal[864]
    completion_reservation_cap: Literal[221184]
    logical_requests_per_run: Literal[18]
    completion_reservation_per_run: Literal[4608]
    model_env_var: Literal["EVICON_LLM_MODEL"]
    response_format: Literal["json_schema"]
    response_schema_name: Literal["cascade_agent_response_v2"]
    output_root: str
    runs: tuple[HD2RunSpec, ...]

    @field_validator("parent_config_sha256")
    @classmethod
    def hash_shape(cls, value: str) -> str:
        if not _SHA.fullmatch(value.lower()):
            raise ValueError("parent_config_sha256 must be SHA-256")
        return value.lower()

    @model_validator(mode="after")
    def fixed_scope(self) -> "HD2Config":
        if self.scenario_ids != _SCENARIO_IDS or self.conditions != _CONDITIONS or self.agent_ids != _AGENT_IDS:
            raise ValueError("H-D.2 scope is not fixed")
        if self.seeds != (20260911, 20260912, 20260913):
            raise ValueError("H-D.2 seeds are not fixed")
        if len(self.runs) != 48 or len({r.run_id for r in self.runs}) != 48:
            raise ValueError("H-D.2 run plan is incomplete")
        expected = [(scenario_id, seed, condition)
                    for scenario_id in self.scenario_ids
                    for seed in self.seeds
                    for condition in self.conditions]
        actual = [(run.scenario_id, run.seed, run.condition) for run in self.runs]
        if actual != expected:
            raise ValueError("H-D.2 run order or coordinates are not explicit")
        for run in self.runs:
            if run.scenario_type != _SCENARIO_TYPES[run.scenario_id]:
                raise ValueError("H-D.2 scenario type mismatch")
            if run.matched_group_id != f"hd2-{run.scenario_id}-{run.seed}":
                raise ValueError("H-D.2 matched group mismatch")
            if run.agent_ids != self.agent_ids or run.round_count != self.max_rounds:
                raise ValueError("H-D.2 run agent or round scope mismatch")
            if run.topology_id != self.topology_id or run.expected_provider_requests != self.logical_requests_per_run:
                raise ValueError("H-D.2 run topology or request scope mismatch")
            if run.completion_reservation != self.completion_reservation_per_run:
                raise ValueError("H-D.2 run reservation mismatch")
        return self


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_hd2_config(path: str | Path) -> tuple[HD2Config, Any]:
    config_path = Path(path).resolve()
    try:
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        config = HD2Config.model_validate(payload)
        parent_path = (config_path.parent / config.parent_config_path).resolve()
        if _sha256(parent_path) != config.parent_config_sha256:
            raise HD2ConfigError("parent_config_hash_mismatch")
        parent, amendment = load_hd_config(parent_path)
        if parent.budget.request_cap != config.request_cap or parent.budget.completion_reservation_cap != config.completion_reservation_cap:
            raise HD2ConfigError("budget_changed")
        if parent.scenario_materials and tuple(item.scenario_id for item in parent.scenario_materials) != config.scenario_ids:
            raise HD2ConfigError("scenario_scope_changed")
        return config, amendment
    except HD2ConfigError:
        raise
    except (OSError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise HD2ConfigError("hd2_config_invalid") from exc


__all__ = [
    "CascadeAgentV2Diagnostic", "CascadeAgentV2Response", "CascadeAgentProtocolV2Runtime",
    "HD2Config", "HD2ConfigError", "HD2RunSpec", "V2_PROTOCOL_VERSION", "V2_SCHEMA_NAME",
    "V2_TEMPLATE_VERSION", "load_hd2_config", "parse_cascade_agent_response_v2", "render_cascade_agent_turn_v2",
]
