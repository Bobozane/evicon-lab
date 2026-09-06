"""Bounded real-model pilot runner for the factorized source-behavior protocol.

The offline factorization simulator is the design check.  This module is the
next, deliberately separate gate: it can run a small 32-request pilot only
after an explicit network flag and exact cap confirmations.  It persists safe
case audits and one receipt, never prompts, raw completions, headers, or
provider metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tomllib
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .conformity_identification import IdentificationError, safe_json, sha256_file
from .conformity_source_behavior_offline_simulation_v1 import (
    DEFAULT_CONFIG as OFFLINE_CONFIG,
    RESPONSE_SCHEMA,
    RESPONSE_SCHEMA_SHA256,
    SCHEMA_NAME,
    TEMPLATE_VERSION,
    FactorCase,
    FactorCondition,
    build_cases,
    parse_factor_response,
    render_factor_request,
    safe_factor_observation,
)
from .llm_contract import LLMProvider, LLMRequest, LLMResponse
from .openai_provider import (
    OpenAICompatibleProvider,
    OpenAICompatibleTransport,
    ProviderConfig,
    ReasoningEffort,
    ResponseFormatMode,
    TransportResponse,
    UrllibTransport,
)


_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG: Final[str] = (
    "configs/provenance_cascade/identification/"
    "conformity_source_behavior_factorization_pilot.v1.toml"
)
DEFAULT_OUTPUT_ROOT: Final[str] = (
    "outputs/conformity-source-behavior-factorization-pilot-v1"
)
DEFAULT_RUNNER: Final[str] = "src/evicon/conformity_source_behavior_factorization_runner_v1.py"
SAFE_AUDIT_NAME: Final[str] = "safe_case_audits.jsonl"
RECEIPT_NAME: Final[str] = "pilot_receipt.json"
RUNNER_VERSION: Final[str] = "conformity_source_behavior_factorization_runner.v1"
RECEIPT_VERSION: Final[str] = "conformity_source_behavior_factorization_receipt.v1"
SCHEMA_VERSION: Final[str] = "conformity_source_behavior_factorized_response.v1"
REQUEST_CAP: Final[int] = 32
COMPLETION_RESERVATION_CAP: Final[int] = 8192
MAX_TOKENS: Final[int] = 256
TEMPERATURE: Final[float] = 0.0
MAX_RETRIES: Final[int] = 0
TIMEOUT_SECONDS: Final[float] = 5.0
REASONING_EFFORT: Final[ReasoningEffort] = ReasoningEffort.NONE
REQUIRED_SCENARIOS: Final[tuple[str, str]] = (
    "source-factor-v1-s01",
    "source-factor-v1-s02",
)
REQUIRED_SEEDS: Final[tuple[int, int]] = (20261301, 20261302)
REQUIRED_CONDITIONS: Final[tuple[FactorCondition, ...]] = tuple(FactorCondition)


class FactorizationPilotError(IdentificationError):
    """Stable, non-sensitive error boundary for the pilot gate."""


class FactorizationPilotConfig(BaseModel):
    """Immutable scope and generation lock for the one-time pilot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pilot_id: Literal["evicon-conformity-source-behavior-factorization-pilot-v1"]
    pilot_version: Literal["conformity_source_behavior_factorization_pilot.v1"]
    runner_version: Literal["conformity_source_behavior_factorization_runner.v1"]
    status: Literal["development_only"]
    not_paper_result: Literal[True]
    no_causal_conclusion: Literal[True]
    network_enabled: Literal[True]
    real_provider_permitted: Literal[True]
    api_key_read_permitted: Literal[True]
    results_write_permitted: Literal[True]
    evaluator_private_truth_available: Literal[False]
    historical_results_excluded: Literal[True]
    wvs_material_excluded: Literal[True]
    minimal_public_synthetic_content_only: Literal[True]
    protocol_path: Literal["src/evicon/conformity_source_behavior_offline_simulation_v1.py"]
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_version: Literal["conformity_source_behavior_offline_simulation.v1"]
    template_version: Literal["conformity_source_behavior_factorized_turn.v1"]
    response_schema_name: Literal["conformity_source_behavior_factorized_response_v1"]
    response_schema_version: Literal["conformity_source_behavior_factorized_response.v1"]
    required_model: str = Field(min_length=1, max_length=160)
    scenario_ids: tuple[
        Literal["source-factor-v1-s01"],
        Literal["source-factor-v1-s02"],
    ]
    seeds: tuple[Literal[20261301], Literal[20261302]]
    conditions: tuple[FactorCondition, ...]
    scenario_count: Literal[2]
    seed_count: Literal[2]
    case_count: Literal[32]
    logical_request_cap: Literal[32]
    completion_reservation_cap: Literal[8192]
    max_tokens: Literal[256]
    temperature: float
    max_retries: Literal[0]
    timeout_seconds: float
    reasoning_effort: Literal["none"]
    output_root: Literal[
        "outputs/conformity-source-behavior-factorization-pilot-v1"
    ]

    @model_validator(mode="after")
    def fixed_scope(self) -> "FactorizationPilotConfig":
        if self.scenario_ids != REQUIRED_SCENARIOS:
            raise ValueError("factorization_pilot_scenario_order_mismatch")
        if self.seeds != REQUIRED_SEEDS:
            raise ValueError("factorization_pilot_seed_order_mismatch")
        if self.conditions != REQUIRED_CONDITIONS:
            raise ValueError("factorization_pilot_condition_order_mismatch")
        if self.temperature != TEMPERATURE or self.timeout_seconds != TIMEOUT_SECONDS:
            raise ValueError("factorization_pilot_generation_parameter_mismatch")
        if self.case_count != self.scenario_count * self.seed_count * len(self.conditions):
            raise ValueError("factorization_pilot_case_count_mismatch")
        if self.completion_reservation_cap != self.case_count * self.max_tokens:
            raise ValueError("factorization_pilot_reservation_cap_mismatch")
        return self


class PilotSchemaContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    response_format: Literal["json_schema"] = "json_schema"
    schema_name: Literal["conformity_source_behavior_factorized_response_v1"] = SCHEMA_NAME
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: Literal["conformity_source_behavior_factorized_response.v1"] = SCHEMA_VERSION
    template_version: Literal["conformity_source_behavior_factorized_turn.v1"] = TEMPLATE_VERSION


class PilotGenerationParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tokens: Literal[256] = MAX_TOKENS
    temperature: float = TEMPERATURE
    max_retries: Literal[0] = MAX_RETRIES
    timeout_seconds: float = TIMEOUT_SECONDS
    reasoning_effort: Literal["none"] = "none"
    seeds: tuple[Literal[20261301], Literal[20261302]] = REQUIRED_SEEDS
    logical_request_cap: Literal[32] = REQUEST_CAP
    completion_reservation_cap: Literal[8192] = COMPLETION_RESERVATION_CAP


class PilotTokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def total_is_consistent(self) -> "PilotTokenUsage":
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("factorization_pilot_token_usage_mismatch")
        return self


class PilotCaseAudit(BaseModel):
    """The only per-case output surface persisted by the pilot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^source-factor-v1-g\d{2}-b\d{2}$")
    group_id: str = Field(pattern=r"^source-factor-v1-g\d{2}$")
    scenario_id: str = Field(pattern=r"^source-factor-v1-s0[1-4]$")
    seed: Literal[20261301, 20261302]
    condition: FactorCondition
    parser_status: Literal["valid"] = "valid"
    ordinal_judgment: str
    public_action: str
    share_decision: str
    confidence: int = Field(ge=1, le=7)
    used_content_count: int = Field(ge=1, le=3)
    visible_focal_root_count: int = Field(ge=0, le=2)
    visible_evidence_count: int = Field(ge=0, le=1)
    finish_reason: Literal["stop", "length", "other"]
    http_status_class: Literal["2xx", "3xx", "4xx", "5xx", "unknown"]
    token_usage: PilotTokenUsage
    latency_ms: float = Field(ge=0.0)
    attempt_count: Literal[1] = 1
    shared_t0_observation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PilotSafety(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    api_key_read: bool = False
    api_key_saved: Literal[False] = False
    network_enabled: bool = False
    provider_constructed: bool = False
    prompt_saved: Literal[False] = False
    full_response_saved: Literal[False] = False
    headers_saved: Literal[False] = False
    provider_metadata_saved: Literal[False] = False
    request_ledger_written: Literal[False] = False
    results_written: bool = False
    evaluator_private_truth_exposed: Literal[False] = False
    historical_results_used: Literal[False] = False
    wvs_material_used: Literal[False] = False
    minimal_public_synthetic_content_only: Literal[True] = True
    behavior_effect_estimated: Literal[False] = False
    no_causal_conclusion: Literal[True] = True
    not_paper_result: Literal[True] = True
    one_shot_per_case: Literal[True] = True


class FactorizationPilotReceipt(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, serialize_by_alias=True
    )

    receipt_version: Literal["conformity_source_behavior_factorization_receipt.v1"]
    runner_version: Literal["conformity_source_behavior_factorization_runner.v1"]
    status: Literal["completed"]
    pilot_id: Literal["evicon-conformity-source-behavior-factorization-pilot-v1"]
    model: str = Field(min_length=1, max_length=160)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runner_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    safe_case_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_contract: PilotSchemaContract = Field(alias="schema")
    generation_parameters: PilotGenerationParameters
    logical_request_count: Literal[32]
    transport_attempt_count: Literal[32]
    completed_case_count: Literal[32]
    parser_invalid_count: Literal[0]
    condition_case_counts: dict[str, int]
    scenario_case_counts: dict[str, int]
    seed_case_counts: dict[str, int]
    token_usage: PilotTokenUsage
    latency_ms: float = Field(ge=0.0)
    safety: PilotSafety

    @model_validator(mode="after")
    def exact_coverage(self) -> "FactorizationPilotReceipt":
        if self.condition_case_counts != {condition.value: 4 for condition in REQUIRED_CONDITIONS}:
            raise ValueError("factorization_pilot_condition_coverage_mismatch")
        if self.scenario_case_counts != {scenario: 16 for scenario in REQUIRED_SCENARIOS}:
            raise ValueError("factorization_pilot_scenario_coverage_mismatch")
        if self.seed_case_counts != {str(seed): 16 for seed in REQUIRED_SEEDS}:
            raise ValueError("factorization_pilot_seed_coverage_mismatch")
        return self


class FactorizationPilotSummary(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, serialize_by_alias=True
    )

    status: Literal["factorization_pilot_preflight_ready", "network_disabled", "blocked", "completed", "failed"]
    error_code: str | None = None
    model: str | None = None
    finish_reason: Literal["stop", "length", "other"] | None = None
    parser_valid: bool | None = None
    parser_error_category: Literal[
        "malformed_json",
        "top_level_type",
        "invalid_schema",
        "duplicate_content_id",
        "unavailable_content_id",
        "judgment_action_mismatch",
        "share_content_id_unexpected",
        "share_content_id_invalid",
        "unknown",
    ] | None = None
    http_status_class: Literal["2xx", "3xx", "4xx", "5xx", "unknown"] | None = None
    transport_category: str | None = None
    token_usage: PilotTokenUsage | None = None
    latency_ms: float | None = Field(default=None, ge=0.0)
    attempt_count: int = Field(default=0, ge=0, le=REQUEST_CAP)
    logical_request_count: int = Field(default=0, ge=0, le=REQUEST_CAP)
    transport_attempt_count: int = Field(default=0, ge=0, le=REQUEST_CAP)
    completed_case_count: int = Field(default=0, ge=0, le=REQUEST_CAP)
    parser_invalid_count: int = Field(default=0, ge=0, le=REQUEST_CAP)
    safe_case_audit_count: int = Field(default=0, ge=0, le=REQUEST_CAP)
    output_root: str | None = None
    safe_case_audit_path: str | None = None
    receipt_path: str | None = None
    network: Literal["disabled", "enabled"] = "disabled"
    schema_contract: PilotSchemaContract = Field(
        alias="schema",
        default_factory=lambda: PilotSchemaContract(schema_sha256=RESPONSE_SCHEMA_SHA256)
    )
    generation_parameters: PilotGenerationParameters = Field(default_factory=PilotGenerationParameters)
    safety: PilotSafety = Field(default_factory=PilotSafety)


class _CountingTransport:
    def __init__(self, transport: OpenAICompatibleTransport, *, max_calls: int = REQUEST_CAP) -> None:
        self.transport = transport
        self.max_calls = max_calls
        self.calls = 0

    def post(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> TransportResponse:
        if self.calls >= self.max_calls:
            raise FactorizationPilotError("factorization_pilot_transport_cap_reached")
        self.calls += 1
        return self.transport.post(url, headers, payload, timeout_seconds)


class _CappedProvider:
    def __init__(self, provider: LLMProvider, *, request_cap: int, reservation_cap: int) -> None:
        self.provider = provider
        self.request_cap = request_cap
        self.reservation_cap = reservation_cap
        self.calls = 0
        self.reserved_tokens = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self.calls >= self.request_cap:
            raise FactorizationPilotError("factorization_pilot_request_cap_reached")
        if self.reserved_tokens + request.max_tokens > self.reservation_cap:
            raise FactorizationPilotError("factorization_pilot_completion_reservation_cap_reached")
        self.calls += 1
        self.reserved_tokens += request.max_tokens
        return self.provider.complete(request)


class _SeededProvider:
    """Reuse one transport while applying the case seed to each request."""

    def __init__(
        self,
        base_config: ProviderConfig,
        *,
        transport: OpenAICompatibleTransport,
        environment: Mapping[str, str],
    ) -> None:
        self.base_config = base_config
        self.transport = transport
        self.environment = environment
        self.providers: dict[int, OpenAICompatibleProvider] = {}
        self.constructed_count = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        provider = self.providers.get(request.seed)
        if provider is None:
            provider = OpenAICompatibleProvider(
                self.base_config.model_copy(update={"seed": request.seed}),
                transport=self.transport,
                environment=self.environment,
            )
            self.providers[request.seed] = provider
            self.constructed_count += 1
        return provider.complete(request)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_ROOT / candidate).resolve()


def _safe_path(path: Path) -> str:
    try:
        return str(path.relative_to(_ROOT))
    except ValueError:
        return str(path)


def _http_status_class(status_code: object) -> str | None:
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        if 200 <= status_code < 300:
            return "2xx"
        if 300 <= status_code < 400:
            return "3xx"
        if 400 <= status_code < 500:
            return "4xx"
        if 500 <= status_code < 600:
            return "5xx"
    return None


def _safe_finish_reason(value: object) -> Literal["stop", "length", "other"]:
    if value == "stop":
        return "stop"
    if value == "length":
        return "length"
    return "other"


def _safe_error_code(error: Exception) -> str:
    code = getattr(error, "code", None)
    if hasattr(code, "value"):
        code = code.value
    if isinstance(code, str):
        return code
    return "factorization_pilot_failed"


_PARSER_ERROR_CATEGORIES: Final[frozenset[str]] = frozenset({
    "malformed_json",
    "top_level_type",
    "invalid_schema",
    "duplicate_content_id",
    "unavailable_content_id",
    "judgment_action_mismatch",
    "share_content_id_unexpected",
    "share_content_id_invalid",
})


def _safe_parser_error_category(error: Exception) -> Literal[
    "malformed_json",
    "top_level_type",
    "invalid_schema",
    "duplicate_content_id",
    "unavailable_content_id",
    "judgment_action_mismatch",
    "share_content_id_unexpected",
    "share_content_id_invalid",
    "unknown",
] | None:
    """Expose only the parser's fixed categorical reason, never response text."""

    category = str(error)
    if category in _PARSER_ERROR_CATEGORIES:
        return category  # type: ignore[return-value]
    return "unknown" if category else None


def _ensure_new_output(path: str | Path) -> Path:
    target = _resolve(path)
    if target.exists():
        raise FactorizationPilotError("factorization_pilot_output_exists")
    if target == _ROOT or target.parent == target:
        raise FactorizationPilotError("factorization_pilot_output_path_invalid")
    return target


def _append_audit(path: Path, audit: PilotCaseAudit) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(audit.model_dump_json())
        handle.write("\n")


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_field")
        result[key] = value
    return result


def load_pilot_config(path: str | Path = DEFAULT_CONFIG) -> FactorizationPilotConfig:
    resolved = _resolve(path)
    try:
        config = FactorizationPilotConfig.model_validate(
            tomllib.loads(resolved.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise FactorizationPilotError("factorization_pilot_config_invalid") from exc
    if sha256_file(config.protocol_path) != config.protocol_sha256:
        raise FactorizationPilotError("factorization_pilot_protocol_hash_mismatch")
    if RESPONSE_SCHEMA_SHA256 != hashlib.sha256(
        json.dumps(RESPONSE_SCHEMA, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest():
        raise FactorizationPilotError("factorization_pilot_schema_hash_mismatch")
    return config


def build_pilot_cases(path: str | Path = DEFAULT_CONFIG) -> tuple[FactorCase, ...]:
    """Filter the frozen offline matrix to the two-scenario, two-seed pilot."""

    config = load_pilot_config(path)
    cases = tuple(
        case
        for case in build_cases(OFFLINE_CONFIG)
        if case.scenario_id in config.scenario_ids and case.seed in config.seeds
    )
    if len(cases) != config.case_count:
        raise FactorizationPilotError("factorization_pilot_case_count_mismatch")
    expected = {
        (scenario, seed, condition)
        for scenario in config.scenario_ids
        for seed in config.seeds
        for condition in config.conditions
    }
    observed = {(case.scenario_id, case.seed, case.condition) for case in cases}
    if observed != expected:
        raise FactorizationPilotError("factorization_pilot_matrix_incomplete")
    return cases


def _token_usage(response: LLMResponse) -> PilotTokenUsage:
    values = (response.prompt_tokens, response.completion_tokens, response.total_tokens)
    if any(value is None for value in values):
        raise FactorizationPilotError("factorization_pilot_token_usage_missing")
    assert response.prompt_tokens is not None
    assert response.completion_tokens is not None
    assert response.total_tokens is not None
    try:
        return PilotTokenUsage(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        )
    except Exception as exc:
        raise FactorizationPilotError("factorization_pilot_token_usage_invalid") from exc


def _observation_hash(response) -> str:
    serialized = json.dumps(
        response.model_dump(mode="json"), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _case_audit(
    case: FactorCase,
    parsed,
    response: LLMResponse,
    *,
    shared_t0_hash: str,
) -> PilotCaseAudit:
    observation = safe_factor_observation(case, parsed, shared_t0_hash)
    status_class = _http_status_class(response.provider_metadata.get("status_code")) or "2xx"
    payload = observation.model_dump(mode="json")
    payload.update({
        "finish_reason": _safe_finish_reason(response.finish_reason),
        "http_status_class": status_class,
        "token_usage": _token_usage(response).model_dump(mode="json"),
        "latency_ms": response.latency_ms,
        "attempt_count": 1,
    })
    return PilotCaseAudit(
        **payload,
    )


def _read_audits(path: Path) -> tuple[PilotCaseAudit, ...]:
    try:
        return tuple(
            PilotCaseAudit.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except Exception as exc:
        raise FactorizationPilotError("factorization_pilot_safe_audit_invalid") from exc


def _write_receipt(path: Path, receipt: FactorizationPilotReceipt) -> None:
    if path.exists():
        raise FactorizationPilotError("factorization_pilot_receipt_exists")
    serialized = json.dumps(
        receipt.model_dump(mode="json"), ensure_ascii=True, sort_keys=True, indent=2
    ) + "\n"
    path.write_text(serialized, encoding="utf-8")


def _safety(*, network_enabled: bool, provider_constructed: bool, api_key_read: bool, results_written: bool = False) -> PilotSafety:
    return PilotSafety(
        api_key_read=api_key_read,
        network_enabled=network_enabled,
        provider_constructed=provider_constructed,
        results_written=results_written,
    )


def pilot_preflight(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    """Validate scope only; this path never reads environment or builds a provider."""

    try:
        config = load_pilot_config(path)
        cases = build_pilot_cases(path)
        return {
            "status": "factorization_pilot_preflight_ready",
            "ready_for_network": False,
            "blocking_reasons": ["explicit_network_and_cap_confirmation_required"],
            "pilot_id": config.pilot_id,
            "scenario_count": config.scenario_count,
            "seed_count": config.seed_count,
            "condition_count": len(config.conditions),
            "case_count": len(cases),
            "logical_request_cap": config.logical_request_cap,
            "completion_reservation_cap": config.completion_reservation_cap,
            "network": "disabled",
            "api_key_read": False,
            "provider_constructed": False,
            "results_written": False,
            "schema": PilotSchemaContract(schema_sha256=RESPONSE_SCHEMA_SHA256).model_dump(mode="json"),
            "generation_parameters": PilotGenerationParameters().model_dump(mode="json"),
            "safety": _safety(network_enabled=False, provider_constructed=False, api_key_read=False).model_dump(mode="json"),
        }
    except Exception as exc:
        return {
            "status": "blocked",
            "blocking_reasons": [_safe_error_code(exc)],
            "network": "disabled",
            "api_key_read": False,
            "provider_constructed": False,
            "results_written": False,
        }


def _require_provider_environment(
    environment: Mapping[str, str] | None,
    *,
    required_model: str,
) -> Mapping[str, str]:
    source = os.environ if environment is None else environment
    if not str(source.get("EVICON_LLM_BASE_URL") or "").strip():
        raise FactorizationPilotError("missing_base_url")
    if not str(source.get("EVICON_LLM_MODEL") or "").strip():
        raise FactorizationPilotError("missing_model")
    if not str(source.get("EVICON_LLM_API_KEY") or "").strip():
        raise FactorizationPilotError("missing_api_key")
    if source.get("EVICON_LLM_MODEL") != required_model:
        raise FactorizationPilotError("factorization_pilot_provider_model_mismatch")
    return source


def _summary_from_failure(
    error: Exception,
    *,
    network: Literal["disabled", "enabled"],
    provider_constructed: bool,
    api_key_read: bool,
    capped: _CappedProvider | None = None,
    transport: _CountingTransport | None = None,
    target: Path | None = None,
    audit_path: Path | None = None,
    parser_invalid_count: int = 0,
    model: str | None = None,
    response: LLMResponse | None = None,
    parser_valid: bool | None = None,
    parser_error_category: Literal[
        "malformed_json",
        "top_level_type",
        "invalid_schema",
        "duplicate_content_id",
        "unavailable_content_id",
        "judgment_action_mismatch",
        "share_content_id_unexpected",
        "share_content_id_invalid",
        "unknown",
    ] | None = None,
) -> FactorizationPilotSummary:
    status_code = getattr(error, "status_code", None)
    if response is not None:
        status_code = response.provider_metadata.get("status_code")
    token_usage = None
    latency_ms = None
    finish_reason = None
    if response is not None:
        finish_reason = _safe_finish_reason(response.finish_reason)
        latency_ms = response.latency_ms
        try:
            token_usage = _token_usage(response)
        except Exception:
            token_usage = None
    error_code = getattr(error, "code", None)
    if hasattr(error_code, "value"):
        error_code = error_code.value
    if parser_valid is None and error_code in {"malformed_response", "invalid_response"}:
        parser_valid = False
    return FactorizationPilotSummary(
        status="failed" if provider_constructed else "blocked",
        error_code=_safe_error_code(error),
        model=model or (response.model_name if response is not None else None),
        finish_reason=finish_reason,
        parser_valid=parser_valid,
        parser_error_category=parser_error_category,
        http_status_class=_http_status_class(status_code),
        transport_category=_safe_error_code(error),
        token_usage=token_usage,
        latency_ms=latency_ms,
        attempt_count=(capped.calls if capped is not None else 0),
        logical_request_count=(capped.calls if capped is not None else 0),
        transport_attempt_count=(transport.calls if transport is not None else (capped.calls if capped is not None else 0)),
        completed_case_count=(len(_read_audits(audit_path)) if audit_path is not None and audit_path.exists() else 0),
        parser_invalid_count=parser_invalid_count,
        safe_case_audit_count=(len(_read_audits(audit_path)) if audit_path is not None and audit_path.exists() else 0),
        output_root=_safe_path(target) if target is not None else None,
        safe_case_audit_path=_safe_path(audit_path) if audit_path is not None and audit_path.exists() else None,
        network=network,
        safety=_safety(
            network_enabled=network == "enabled",
            provider_constructed=provider_constructed,
            api_key_read=api_key_read,
            results_written=target is not None and target.exists(),
        ),
    )


def run_factorization_pilot(
    provider: LLMProvider,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    output_root: str | Path | None = None,
    transport_counter: _CountingTransport | None = None,
    network: Literal["disabled", "enabled"] = "enabled",
) -> FactorizationPilotSummary:
    """Run exactly the configured 32 cases once and write safe artifacts."""

    config = load_pilot_config(config_path)
    cases = build_pilot_cases(config_path)
    target = _ensure_new_output(output_root or config.output_root)
    target.mkdir(parents=True, exist_ok=False)
    audit_path = target / SAFE_AUDIT_NAME
    capped = _CappedProvider(
        provider,
        request_cap=config.logical_request_cap,
        reservation_cap=config.completion_reservation_cap,
    )
    completed = 0
    parser_invalid = 0
    t0_hashes: dict[str, str] = {}
    prompt_tokens = completion_tokens = total_tokens = 0
    latency_ms = 0.0
    model_name: str | None = None
    last_response: LLMResponse | None = None
    last_parser_valid: bool | None = None
    parser_error_category: Literal[
        "malformed_json",
        "top_level_type",
        "invalid_schema",
        "duplicate_content_id",
        "unavailable_content_id",
        "judgment_action_mismatch",
        "share_content_id_unexpected",
        "share_content_id_invalid",
        "unknown",
    ] | None = None

    try:
        # build_cases is ordered by group and condition, so private_initial is first.
        for case in cases:
            rendered_case = case.model_copy(update={"model_name": config.required_model})
            request = render_factor_request(rendered_case)
            response = capped.complete(request)
            last_response = response
            if response.model_name != config.required_model:
                raise FactorizationPilotError("factorization_pilot_provider_model_mismatch")
            try:
                parsed = parse_factor_response(response.content, rendered_case)
            except ValueError as exc:
                parser_invalid += 1
                last_parser_valid = False
                parser_error_category = _safe_parser_error_category(exc)
                raise FactorizationPilotError("factorization_pilot_parser_invalid") from exc
            last_parser_valid = True
            if response.finish_reason != "stop":
                raise FactorizationPilotError("factorization_pilot_finish_reason_invalid")
            usage = _token_usage(response)
            if case.condition is FactorCondition.PRIVATE_INITIAL:
                t0_hashes[case.group_id] = _observation_hash(parsed)
            elif case.group_id not in t0_hashes:
                raise FactorizationPilotError("factorization_pilot_shared_t0_missing")
            audit = _case_audit(
                rendered_case,
                parsed,
                response,
                shared_t0_hash=t0_hashes[case.group_id],
            )
            _append_audit(audit_path, audit)
            completed += 1
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens
            total_tokens += usage.total_tokens
            latency_ms += response.latency_ms
            model_name = response.model_name

        audits = _read_audits(audit_path)
        expected_conditions = Counter(case.condition.value for case in cases)
        expected_scenarios = Counter(case.scenario_id for case in cases)
        expected_seeds = Counter(str(case.seed) for case in cases)
        observed_conditions = Counter(item.condition.value for item in audits)
        observed_scenarios = Counter(item.scenario_id for item in audits)
        observed_seeds = Counter(str(item.seed) for item in audits)
        if (
            capped.calls != config.logical_request_cap
            or (transport_counter is not None and transport_counter.calls != config.logical_request_cap)
            or capped.reserved_tokens != config.completion_reservation_cap
            or len(audits) != config.case_count
            or len({item.case_id for item in audits}) != config.case_count
            or observed_conditions != expected_conditions
            or observed_scenarios != expected_scenarios
            or observed_seeds != expected_seeds
        ):
            raise FactorizationPilotError("factorization_pilot_completeness_failed")
        assert model_name is not None
        receipt_path = target / RECEIPT_NAME
        receipt = FactorizationPilotReceipt(
            receipt_version=RECEIPT_VERSION,
            runner_version=RUNNER_VERSION,
            status="completed",
            pilot_id=config.pilot_id,
            model=model_name,
            config_sha256=sha256_file(config_path),
            protocol_sha256=config.protocol_sha256,
            runner_sha256=sha256_file(DEFAULT_RUNNER),
            safe_case_audit_sha256=sha256_file(audit_path),
            schema=PilotSchemaContract(schema_sha256=RESPONSE_SCHEMA_SHA256),
            generation_parameters=PilotGenerationParameters(),
            logical_request_count=config.logical_request_cap,
            transport_attempt_count=transport_counter.calls if transport_counter is not None else capped.calls,
            completed_case_count=completed,
            parser_invalid_count=0,
            condition_case_counts=dict(sorted(observed_conditions.items())),
            scenario_case_counts=dict(sorted(observed_scenarios.items())),
            seed_case_counts=dict(sorted(observed_seeds.items())),
            token_usage=PilotTokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            latency_ms=latency_ms,
            safety=_safety(
                network_enabled=network == "enabled",
                provider_constructed=True,
                api_key_read=network == "enabled",
                results_written=True,
            ),
        )
        _write_receipt(receipt_path, receipt)
        return FactorizationPilotSummary(
            status="completed",
            model=model_name,
            finish_reason="stop",
            parser_valid=True,
            http_status_class="2xx",
            token_usage=receipt.token_usage,
            latency_ms=latency_ms,
            attempt_count=config.logical_request_cap,
            logical_request_count=config.logical_request_cap,
            transport_attempt_count=receipt.transport_attempt_count,
            completed_case_count=completed,
            parser_invalid_count=0,
            safe_case_audit_count=len(audits),
            output_root=_safe_path(target),
            safe_case_audit_path=_safe_path(audit_path),
            receipt_path=_safe_path(receipt_path),
            network=network,
            safety=_safety(
                network_enabled=network == "enabled",
                provider_constructed=True,
                api_key_read=network == "enabled",
                results_written=True,
            ),
        )
    except Exception as exc:
        return _summary_from_failure(
            exc,
            network=network,
            provider_constructed=True,
            api_key_read=network == "enabled",
            capped=capped,
            transport=transport_counter,
            target=target,
            audit_path=audit_path,
            parser_invalid_count=parser_invalid,
            model=model_name,
            response=last_response,
            parser_valid=last_parser_valid,
            parser_error_category=parser_error_category,
        )


def execute_factorization_pilot(
    *,
    allow_network: bool = False,
    confirm_run: bool = False,
    confirm_request_cap: int | None = None,
    confirm_completion_reservation_cap: int | None = None,
    environment: Mapping[str, str] | None = None,
    transport: OpenAICompatibleTransport | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    output_root: str | Path | None = None,
) -> FactorizationPilotSummary | dict[str, object]:
    """Gate the pilot; without all confirmations no environment is touched."""

    if not allow_network:
        return pilot_preflight(config_path)
    if (
        not confirm_run
        or confirm_request_cap != REQUEST_CAP
        or confirm_completion_reservation_cap != COMPLETION_RESERVATION_CAP
    ):
        return FactorizationPilotSummary(
            status="blocked",
            error_code="factorization_pilot_explicit_cap_confirmation_required",
        )
    try:
        config = load_pilot_config(config_path)
        target = _ensure_new_output(output_root or config.output_root)
        source = _require_provider_environment(environment, required_model=config.required_model)
        base = ProviderConfig.from_env(allow_network=True, environment=source)
        provider_config = base.model_copy(update={
            "allow_network": True,
            "timeout_seconds": config.timeout_seconds,
            "max_retries": config.max_retries,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "reasoning_effort": REASONING_EFFORT,
            "response_format": ResponseFormatMode.JSON_SCHEMA,
            "response_schema_name": SCHEMA_NAME,
            "response_schema": RESPONSE_SCHEMA,
        })
        counted_transport = _CountingTransport(transport or UrllibTransport())
        seeded_provider = _SeededProvider(
            provider_config,
            transport=counted_transport,
            environment=source,
        )
        result = run_factorization_pilot(
            seeded_provider,
            config_path=config_path,
            output_root=target,
            transport_counter=counted_transport,
            network="enabled",
        )
        return result.model_copy(update={
            "safety": result.safety.model_copy(update={
                "provider_constructed": seeded_provider.constructed_count > 0,
                "api_key_read": True,
            })
        })
    except Exception as exc:
        return _summary_from_failure(
            exc,
            network="enabled",
            provider_constructed=False,
            api_key_read=False,
        )


def load_pilot_receipt(
    path: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
) -> FactorizationPilotReceipt:
    """Validate the receipt and its safe audit binding."""

    receipt_path = _resolve(path)
    try:
        payload = json.loads(
            receipt_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_pairs,
        )
        receipt = FactorizationPilotReceipt.model_validate(payload)
        audit_path = receipt_path.parent / SAFE_AUDIT_NAME
        audits = _read_audits(audit_path)
        config = load_pilot_config(config_path)
        if (
            receipt.config_sha256 != sha256_file(config_path)
            or receipt.protocol_sha256 != config.protocol_sha256
            or receipt.runner_sha256 != sha256_file(DEFAULT_RUNNER)
            or receipt.safe_case_audit_sha256 != sha256_file(audit_path)
            or receipt.schema_contract.schema_sha256 != RESPONSE_SCHEMA_SHA256
            or len(audits) != config.case_count
        ):
            raise ValueError("factorization_pilot_receipt_binding_mismatch")
        return receipt
    except Exception as exc:
        if isinstance(exc, FactorizationPilotError):
            raise
        raise FactorizationPilotError("factorization_pilot_receipt_invalid") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the explicitly authorized source-behavior factorization pilot"
    )
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--confirm-request-cap", type=int)
    parser.add_argument("--confirm-completion-reservation-cap", type=int)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    result = execute_factorization_pilot(
        allow_network=args.allow_network,
        confirm_run=args.confirm_run,
        confirm_request_cap=args.confirm_request_cap,
        confirm_completion_reservation_cap=args.confirm_completion_reservation_cap,
        config_path=args.config,
        output_root=args.output_root,
    )
    payload = result.model_dump(mode="json") if isinstance(result, BaseModel) else result
    print(safe_json(payload))
    return 0 if payload.get("status") in {"factorization_pilot_preflight_ready", "network_disabled", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_RUNNER",
    "FactorizationPilotConfig",
    "FactorizationPilotError",
    "FactorizationPilotReceipt",
    "FactorizationPilotSummary",
    "PilotCaseAudit",
    "PilotGenerationParameters",
    "PilotSafety",
    "PilotSchemaContract",
    "PilotTokenUsage",
    "RECEIPT_NAME",
    "RECEIPT_VERSION",
    "REQUEST_CAP",
    "RUNNER_VERSION",
    "SAFE_AUDIT_NAME",
    "SCHEMA_VERSION",
    "build_pilot_cases",
    "execute_factorization_pilot",
    "load_pilot_config",
    "load_pilot_receipt",
    "main",
    "pilot_preflight",
    "run_factorization_pilot",
]
