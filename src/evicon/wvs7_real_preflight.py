"""Default-offline validation for one explicitly confirmed WVS baseline pilot."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .batch_experiment import BatchExperimentManifest, BatchRunSpec, BatchRuntimeParameters, expected_artifact_paths
from .freeze_wvs7_probe_set import EXPECTED_VARIABLE_IDS
from .llm_contract import LLMProvider, LLMProviderError, LLMRequest, ProviderErrorCode
from .openai_provider import OpenAICompatibleProvider, ProviderConfig, ReasoningEffort
from .wvs7_authoring import (
    WVS7AuthoringAcceptanceLoader,
    WVS7LanguageArm,
    WVS7ManualTranscriptionLoader,
    build_wvs7_authoring_plan,
    validate_manual_transcriptions,
)
from .wvs7_baseline_pilot import PreparedWVS7BaselinePilot, WVS7BaselinePilotError, prepare_wvs7_baseline_pilot
from .wvs7_candidates import WVS7CandidateLoader
from .wvs7_review import WVS7ReviewLoader


REAL_REQUEST_CAP = 384
REAL_COMPLETION_RESERVATION_CAP = 51_200
REAL_DEFAULT_RUN_ID = "wvs7-real-baseline-pilot-seed-001"
_CONNECTION_CHECK_MAX_TOKENS = 256
_ROOT = Path(__file__).resolve().parents[2]
_CANDIDATE = _ROOT / "configs/studies/wvs7_candidate_24.toml"
_REVIEW = _ROOT / "configs/studies/wvs7_candidate_24_codex_preliminary_review.toml"
_ACCEPTANCE = _ROOT / "configs/studies/wvs7_authoring_acceptance_researcher_2026-08-18.toml"
_TRANSCRIPTIONS = _ROOT / "private_wvs_transcriptions/english_core_wvs7_23.toml"


class WVS7RealPreflightError(ValueError):
    """Stable preflight error with no source text, prompts, or secret data."""


class WVS7RealPreflightReport(BaseModel):
    """Safe preflight envelope. Boolean configuration flags reveal no secret values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    run_id: str
    frozen_probe_set_sha256: str = Field(min_length=64, max_length=64)
    condition_count: int = Field(ge=0)
    seed_count: int = Field(ge=0)
    agent_count: int = Field(ge=0)
    rounds: int = Field(ge=0)
    pre_probe_count: int = Field(ge=0)
    post_probe_count: int = Field(ge=0)
    request_cap: int = Field(gt=0)
    completion_reservation_cap: int = Field(gt=0)
    matched_groups_complete: bool
    acceptance_valid: bool
    transcription_valid: bool
    output_paths_available: bool
    model_configured: bool
    base_url_configured: bool
    blocking_reasons: list[str] = Field(default_factory=list)


def condition_run_id(batch_run_id: str, condition: str) -> str:
    """Explicit stable per-condition ID derived from the confirmed batch run ID."""
    normalized = batch_run_id.strip()
    if not normalized or Path(normalized).name != normalized:
        raise WVS7RealPreflightError("run_id must be a simple identifier")
    return f"{normalized}-{condition}"


def validate_source_gates() -> None:
    """Revalidate the existing accepted English authoring and transcription gates."""
    try:
        candidate = WVS7CandidateLoader.load(_CANDIDATE)
        review = WVS7ReviewLoader.load(_REVIEW)
        acceptance = WVS7AuthoringAcceptanceLoader.load(_ACCEPTANCE)
        plan = build_wvs7_authoring_plan(candidate, review, acceptance, WVS7LanguageArm.ENGLISH_CORE)
        transcriptions = WVS7ManualTranscriptionLoader.load(_TRANSCRIPTIONS)
        count = validate_manual_transcriptions(plan, transcriptions)
    except Exception as exc:
        raise WVS7RealPreflightError("accepted_authoring_or_transcription_is_invalid") from exc
    if count != 23 or tuple(item.variable_id for item in plan.items) != EXPECTED_VARIABLE_IDS:
        raise WVS7RealPreflightError("accepted_authoring_or_transcription_is_invalid")


def build_real_batch_manifest(
    prepared: PreparedWVS7BaselinePilot,
    *,
    batch_run_id: str,
    model_name: str,
) -> BatchExperimentManifest:
    """Keep Stage 22B-5 matching coordinates while assigning fresh real run IDs."""
    if not model_name.strip():
        raise WVS7RealPreflightError("environment_model_required")
    specs: list[BatchRunSpec] = []
    for spec in prepared.batch_manifest.runs:
        run_id = condition_run_id(batch_run_id, spec.condition.value)
        specs.append(spec.model_copy(update={
            "run_id": run_id,
            "model_name": model_name,
            "probe_model_name": model_name,
            "output_dir": prepared.config.run_output_dir,
            "artifacts": expected_artifact_paths(prepared.config.run_output_dir, run_id),
            "runtime_parameters": BatchRuntimeParameters(
                agent_temperature=prepared.config.agent_temperature,
                agent_max_tokens=prepared.config.agent_max_tokens,
                probe_temperature=prepared.config.probe_temperature,
                probe_max_tokens=prepared.config.probe_max_tokens,
            ),
        }))
    candidate = prepared.batch_manifest.model_copy(update={
        "batch_id": batch_run_id,
        "model_name": model_name,
        "probe_model_name": model_name,
        "runs": specs,
        "run_order": [condition_run_id(batch_run_id, spec.condition.value) for spec in prepared.batch_manifest.runs],
        "metadata": {
            "pilot_only": True,
            "single_seed": True,
            "not_paper_result": True,
            "no_causal_conclusion": True,
            "no_mediator": True,
            "frozen_probe_set_sha256": prepared.config.frozen_probe_set_sha256,
        },
    })
    return BatchExperimentManifest.model_validate(candidate.model_dump(mode="python"))


def preflight_wvs7_real_pilot(
    config_path: str | Path,
    *,
    run_id: str = REAL_DEFAULT_RUN_ID,
    environment: dict[str, str] | None = None,
    resume: bool = False,
) -> tuple[PreparedWVS7BaselinePilot, WVS7RealPreflightReport]:
    """Perform all local checks. This function never constructs a provider or calls HTTP."""
    prepared = prepare_wvs7_baseline_pilot(config_path)
    _validate_fixed_real_scope(prepared)
    validate_source_gates()
    source = os.environ if environment is None else environment
    model_name = source.get("EVICON_LLM_MODEL") or "preflight-unconfigured-model"
    manifest = build_real_batch_manifest(prepared, batch_run_id=run_id, model_name=model_name)
    if prepared.config.max_retries < 0:  # Defensive: Pydantic already enforces this.
        raise WVS7RealPreflightError("retry_configuration_invalid")
    if not _outputs_are_under_results(prepared, manifest):
        raise WVS7RealPreflightError("real_pilot_outputs_must_be_under_results")
    output_paths = [Path(manifest.batch_output_dir) / run_id, *(Path(spec.artifacts.run_directory) for spec in manifest.runs)]
    available = not any(path.exists() for path in output_paths)
    resume_directory = Path(manifest.batch_output_dir) / run_id
    resumable = resume_directory.is_dir()
    ready = resumable if resume else available
    report = WVS7RealPreflightReport(
        status="ready" if ready else "blocked",
        run_id=run_id,
        frozen_probe_set_sha256=prepared.config.frozen_probe_set_sha256,
        condition_count=4,
        seed_count=1,
        agent_count=2,
        rounds=2,
        pre_probe_count=23,
        post_probe_count=23,
        request_cap=REAL_REQUEST_CAP,
        completion_reservation_cap=REAL_COMPLETION_RESERVATION_CAP,
        matched_groups_complete=len(manifest.runs) == 4 and len(manifest.seeds) == 1,
        acceptance_valid=True,
        transcription_valid=True,
        output_paths_available=available,
        model_configured=bool(source.get("EVICON_LLM_MODEL")),
        base_url_configured=bool(source.get("EVICON_LLM_BASE_URL")),
        blocking_reasons=(
            []
            if ready
            else (["resume_batch_not_found"] if resume else ["run_id_or_batch_output_already_exists"])
        ),
    )
    return prepared, report


def _validate_fixed_real_scope(prepared: PreparedWVS7BaselinePilot) -> None:
    """Reject configuration drift before any real-provider construction."""
    config = prepared.config
    expected_request_count = (
        4 * 2 * 2  # conditions * Agents * rounds
        + 4 * 2 * 23 * 2  # conditions * Agents * items * pre/post phases
    )
    expected_token_count = 4 * 2 * 2 * 256 + 4 * 2 * 23 * 2 * 128
    if (
        len(config.seeds) != 1
        or len(config.agent_ids) != 2
        or config.max_rounds != 2
        or len(prepared.batch_manifest.runs) != 4
        or len(prepared.probe_set.items_for_holdout(False)) != 23
        or config.agent_temperature != 0.2
        or config.agent_max_tokens != 256
        or config.probe_temperature != 0.0
        or config.probe_max_tokens != 128
        or expected_request_count != REAL_REQUEST_CAP
        or expected_token_count != REAL_COMPLETION_RESERVATION_CAP
    ):
        raise WVS7RealPreflightError("real_pilot_scope_or_budget_mismatch")


def connection_check(provider: LLMProvider, *, model_name: str) -> dict[str, object]:
    """Execute exactly one bounded provider request; callers own the network opt-in."""
    response = provider.complete(LLMRequest(
        request_id="wvs7-real-preflight-connection-check",
        model_name=model_name,
        system_prompt="Return only a JSON object.",
        user_prompt="{}",
        temperature=0.0,
        max_tokens=_CONNECTION_CHECK_MAX_TOKENS,
        seed=0,
        metadata={"template_version": "preflight_connection_check.v1"},
    ))
    return {
        "status": "completed",
        "model": response.model_name,
        "request_id": response.request_id,
        "finish_reason": response.finish_reason,
        "token_usage": {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
        },
        "latency_ms": response.latency_ms,
    }


def _outputs_are_under_results(prepared: PreparedWVS7BaselinePilot, manifest: BatchExperimentManifest) -> bool:
    return (
        Path(prepared.config.run_output_dir).parts[:1] == ("results",)
        and Path(manifest.batch_output_dir).parts[:1] == ("results",)
        and all(Path(spec.artifacts.run_directory).parts[:1] == ("results",) for spec in manifest.runs)
    )


def _safe_error_code(error: BaseException) -> str:
    message = str(error)
    if "already exists" in message:
        return "run_id_or_batch_output_already_exists"
    if "accepted_authoring" in message:
        return "accepted_authoring_or_transcription_is_invalid"
    if "environment_model" in message:
        return "environment_model_required"
    return "preflight_validation_failed"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate one real WVS baseline pilot without default network access.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=REAL_DEFAULT_RUN_ID)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--connection-check-only", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        prepared, report = preflight_wvs7_real_pilot(arguments.config, run_id=arguments.run_id)
        if arguments.connection_check_only:
            if not arguments.allow_network:
                print(json.dumps({"status": "blocked", "blocking_reasons": ["allow_network_required"]}, sort_keys=True))
                return 1
            config = ProviderConfig.from_env(allow_network=True).model_copy(update={
                "max_retries": 0,
                "max_tokens": _CONNECTION_CHECK_MAX_TOKENS,
                "reasoning_effort": ReasoningEffort.NONE,
            })
            if not config.model_name:
                print(json.dumps({"status": "blocked", "blocking_reasons": ["environment_model_required"]}, sort_keys=True))
                return 1
            print(json.dumps(connection_check(OpenAICompatibleProvider(config), model_name=config.model_name), ensure_ascii=True, sort_keys=True))
            return 0
        print(json.dumps(report.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))
        return 0 if report.status == "ready" else 1
    except LLMProviderError as exc:
        print(json.dumps({"status": "provider_error", "error_code": f"provider_{exc.code.value}"}, sort_keys=True))
        return 2
    except (WVS7BaselinePilotError, WVS7RealPreflightError, OSError, ValidationError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "blocking_reasons": [_safe_error_code(exc)]}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REAL_DEFAULT_RUN_ID",
    "REAL_REQUEST_CAP",
    "REAL_COMPLETION_RESERVATION_CAP",
    "WVS7RealPreflightError",
    "WVS7RealPreflightReport",
    "build_real_batch_manifest",
    "condition_run_id",
    "connection_check",
    "preflight_wvs7_real_pilot",
    "_validate_fixed_real_scope",
    "validate_source_gates",
]
