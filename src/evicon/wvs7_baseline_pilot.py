"""Matched, local-first WVS English baseline pilot orchestration.

The module reads a frozen ProbeSet, executes only explicitly declared baseline
conditions, and keeps the existing protocol, replay, and metric definitions
unchanged.  ``real-pilot`` is deliberately not enabled here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .agent_runtime import AgentRequestSettings, AgentRuntime
from .batch_experiment import (
    BASELINE_PROTOCOLS,
    BatchArtifactPaths,
    BatchExperimentManifest,
    BatchExperimentRecord,
    BatchExperimentStatus,
    BatchMatchedGroupRecord,
    BatchRunRecord,
    BatchRunSpec,
    BatchRunStatus,
    BatchRuntimeParameters,
    expected_artifact_paths,
)
from .contextual_probe_runtime import ContextualProbeRequestSettings, ContextualProbeRuntime
from .evaluation import OfflineEvaluator, write_evaluation_report
from .evaluation.models import (
    EvaluationCondition,
    EvaluationManifest,
    EvaluationMetricOptions,
    EvaluationRunEntry,
    MetricSuite,
)
from .fake_llm import FakeLLM, FakeLLMRequest
from .frozen_wvs7_manifest import FrozenWVS7Manifest, load_frozen_manifest, sha256_file
from .llm_contract import LLMProviderError, LLMRequest, LLMResponse, ProviderErrorCode
from .measured_protocol_experiment import MeasuredProtocolExperimentRunner, MeasurementExperimentStatus
from .models import ProbeSet, ProtocolCondition, ScenarioSpec
from .models._validation import Metadata, identifier_list, normalized_text
from .probe_loader import ProbeSetLoader
from .replay import validate_replay
from .scenario_loader import ScenarioLoader
from .freeze_wvs7_probe_set import DEMOCRACY_IDS, EXPECTED_VARIABLE_IDS, SCIENCE_IDS


class WVS7BaselinePilotError(ValueError):
    """Stable pilot-planning error without source item text or provider data."""


class WVS7BaselineMode(str, Enum):
    PLAN_ONLY = "plan-only"
    FAKE_SMOKE = "fake-smoke"
    REAL_PILOT = "real-pilot"


class WVS7BaselineRunDeclaration(BaseModel):
    """One explicit condition/seed/run ID declaration from the study plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    matched_group_id: str = Field(min_length=1)
    condition: ProtocolCondition
    seed: int

    @field_validator("run_id", "matched_group_id")
    @classmethod
    def identifiers(cls, value: str, info: object) -> str:
        normalized = normalized_text(value, getattr(info, "field_name", "identifier"))
        if Path(normalized).name != normalized:
            raise ValueError("run_id and matched_group_id must be simple identifiers")
        return normalized

    @field_validator("seed", mode="before")
    @classmethod
    def seed_value(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("seed must be an integer")
        return value


class WVS7BaselinePilotConfig(BaseModel):
    """Strict manifest for a four-condition English development pilot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    scenario_path: str = Field(min_length=1)
    frozen_probe_set_path: str = Field(min_length=1)
    frozen_manifest_path: str = Field(min_length=1)
    frozen_probe_set_sha256: str = Field(min_length=64, max_length=64)
    probe_set_id: Literal["wvs7-english-core-23"]
    agent_ids: list[str] = Field(min_length=2)
    max_rounds: int = Field(ge=2, le=3)
    seeds: list[int] = Field(min_length=1)
    model_name: str = Field(min_length=1)
    probe_model_name: str = Field(min_length=1)
    agent_temperature: float = Field(ge=0.0, le=2.0)
    agent_max_tokens: int = Field(gt=0)
    probe_temperature: float = Field(ge=0.0, le=2.0)
    probe_max_tokens: int = Field(gt=0)
    max_retries: int = Field(ge=0, le=5)
    coverage_threshold: float = Field(ge=0.0, le=1.0)
    run_output_dir: str = Field(min_length=1)
    batch_output_dir: str = Field(min_length=1)
    metadata: Metadata = Field(default_factory=dict)
    runs: list[WVS7BaselineRunDeclaration] = Field(min_length=4)
    run_order: list[str] = Field(min_length=4)

    @field_validator(
        "study_id", "version", "scenario_path", "frozen_probe_set_path", "frozen_manifest_path",
        "model_name", "probe_model_name", "run_output_dir", "batch_output_dir",
    )
    @classmethod
    def text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("agent_ids", "run_order")
    @classmethod
    def agents(cls, value: list[str]) -> list[str]:
        return identifier_list(value, "agent_ids")

    @field_validator("seeds")
    @classmethod
    def unique_seeds(cls, value: list[int]) -> list[int]:
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in value):
            raise ValueError("seeds must contain integers")
        if len(value) != len(set(value)):
            raise ValueError("seeds must not repeat")
        return value

    @field_validator("frozen_probe_set_sha256")
    @classmethod
    def sha256(cls, value: str) -> str:
        normalized = value.lower().strip()
        if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
            raise ValueError("frozen_probe_set_sha256 must be a SHA-256 digest")
        return normalized

    @model_validator(mode="after")
    def matched_groups(self) -> "WVS7BaselinePilotConfig":
        run_ids = [run.run_id for run in self.runs]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("runs must not repeat run_id values")
        if set(self.run_order) != set(run_ids) or len(self.run_order) != len(run_ids):
            raise ValueError("run_order must explicitly contain every run_id exactly once")
        expected_conditions = set(BASELINE_PROTOCOLS)
        grouped: dict[int, list[WVS7BaselineRunDeclaration]] = defaultdict(list)
        for run in self.runs:
            if run.seed not in self.seeds:
                raise ValueError("run seed must be listed in seeds")
            if run.condition not in expected_conditions:
                raise ValueError("run condition must be a baseline protocol")
            grouped[run.seed].append(run)
        for seed in self.seeds:
            group = grouped[seed]
            if len(group) != 4 or {run.condition for run in group} != expected_conditions:
                raise ValueError("every seed must declare exactly four baseline conditions")
            if len({run.matched_group_id for run in group}) != 1:
                raise ValueError("every seed must use one explicit matched_group_id")
        return self


class WVS7BaselineDryRun(BaseModel):
    """Safe, no-network request budget and matching summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: str
    condition_count: int = Field(ge=0)
    seed_count: int = Field(ge=0)
    agent_turn_count: int = Field(ge=0)
    pre_probe_request_count: int = Field(ge=0)
    post_probe_request_count: int = Field(ge=0)
    provider_request_upper_bound: int = Field(ge=0)
    token_upper_bound: int = Field(ge=0)
    matched_groups: list[dict[str, object]]
    frozen_probe_set_sha256: str


@dataclass(frozen=True)
class PreparedWVS7BaselinePilot:
    config: WVS7BaselinePilotConfig
    scenario: ScenarioSpec
    probe_set: ProbeSet
    frozen_manifest: FrozenWVS7Manifest
    batch_manifest: BatchExperimentManifest


class WVS7BaselineFakeLLMProvider:
    """A deterministic local LLMProvider adapter for baseline engineering smoke tests."""

    def __init__(self) -> None:
        self.request_count = 0
        self._fake_llm = FakeLLM()

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.request_count += 1
        template = request.metadata.get("template_version")
        if template == "agent_turn.v1":
            content = self._agent_content(request)
        elif template == "contextual_value_probe.v2":
            content = self._probe_content(request)
        else:
            raise LLMProviderError(ProviderErrorCode.INVALID_REQUEST, "unsupported local baseline request template")
        prompt_tokens = max(1, (len(request.system_prompt) + len(request.user_prompt) + 7) // 8)
        completion_tokens = max(1, (len(content) + 7) // 8)
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content=content,
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=0.0,
            provider_metadata={"provider_name": "wvs7_baseline_fake_llm.v1"},
        )

    def _agent_content(self, request: LLMRequest) -> str:
        agent_id = _required_metadata_string(request, "agent_id")
        round_id = _required_metadata_int(request, "round_id")
        protocol = _required_metadata_string(request, "protocol")
        peers = request.metadata.get("visible_peer_turn_ids", [])
        evidence = request.metadata.get("visible_evidence_ids", [])
        if not isinstance(peers, list) or not isinstance(evidence, list):
            raise LLMProviderError(ProviderErrorCode.INVALID_REQUEST, "local baseline visibility metadata is invalid")
        fake_response = self._fake_llm.complete(FakeLLMRequest(
            agent_id=agent_id,
            round_id=round_id,
            protocol=ProtocolCondition(protocol),
            # AgentRuntime has already rendered the strictly visible context.
            # FakeLLM only needs these bounded identifiers for deterministic
            # engineering smoke behavior.
            visible_peer_turn_ids=peers,
            visible_evidence_ids=evidence,
            seed=request.seed,
        ))
        used = [evidence[0]] if evidence else []
        return json.dumps(
            {
                "message": fake_response.message,
                "evidence_ids_used": used,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _probe_content(request: LLMRequest) -> str:
        try:
            payload = json.loads(request.user_prompt)
            context = payload["measurement_context"]
            item = payload["probe_item"]
            agent_id = context["agent_id"]
            protocol = _required_metadata_string(request, "protocol")
            round_id = context["round_id"]
            choices = item["response_scale"]
            probe_id = item["probe_id"]
        except (TypeError, KeyError, json.JSONDecodeError) as exc:
            raise LLMProviderError(ProviderErrorCode.INVALID_REQUEST, "local baseline probe request is invalid") from exc
        if not isinstance(choices, list) or not choices or not all(isinstance(choice, str) for choice in choices):
            raise LLMProviderError(ProviderErrorCode.INVALID_REQUEST, "local baseline probe scale is invalid")
        visible_peers = context.get("visible_peer_turns", [])
        visible_evidence = context.get("visible_evidence_cards", [])
        canonical = json.dumps(
            {
                "seed": request.seed,
                "agent_id": agent_id,
                "protocol": protocol,
                "round_id": round_id,
                "probe_id": probe_id,
                "peer_count": len(visible_peers),
                "evidence_count": len(visible_evidence),
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        index = int(hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8], 16) % len(choices)
        return json.dumps({"choice": choices[index]}, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


class WVS7BaselinePilotRunner:
    """Serial, resumable FakeLLM runner for explicitly declared matched groups."""

    def __init__(self, prepared: PreparedWVS7BaselinePilot, *, batch_id: str) -> None:
        if Path(batch_id).name != batch_id:
            raise WVS7BaselinePilotError("batch_id must be a simple identifier")
        self.prepared = prepared
        self.batch_id = batch_id

    @property
    def batch_directory(self) -> Path:
        return Path(self.prepared.config.batch_output_dir) / self.batch_id

    def run_fake_smoke(self, *, resume: bool = False) -> tuple[BatchExperimentRecord, dict[str, object] | None]:
        records = self._initial_or_resumed_records(resume=resume)
        provider = WVS7BaselineFakeLLMProvider()
        for seed in self.prepared.config.seeds:
            group_failed = False
            for spec in self._ordered_specs_for_seed(seed):
                # ``records`` is immutable Pydantic data replaced after every
                # transition, so look up the current state rather than a
                # stale snapshot taken before the loop.
                previous = next(record for record in records if record.run_id == spec.run_id)
                if previous.status is BatchRunStatus.COMPLETED:
                    continue
                if previous.status is BatchRunStatus.FAILED:
                    group_failed = True
                    continue
                if group_failed:
                    continue
                records = self._replace_record(records, spec.run_id, BatchRunStatus.RUNNING)
                self._write_record(self._record(records, BatchExperimentStatus.RUNNING, error_code=None))
                try:
                    result = self._run_condition(spec, provider)
                    if result.status is not MeasurementExperimentStatus.COMPLETED or result.run_record is None:
                        raise WVS7BaselinePilotError(result.status.value)
                    replay = validate_replay(spec.artifacts.events_path, spec.artifacts.run_record_path)
                    if replay.validation not in {"passed", "passed_with_failure"}:
                        raise WVS7BaselinePilotError("replay_validation_failed")
                except Exception:
                    records = self._replace_record(records, spec.run_id, BatchRunStatus.FAILED, error_code="condition_failed")
                    self._write_record(self._record(records, BatchExperimentStatus.FAILED, error_code="condition_failed"))
                    group_failed = True
                    continue
                records = self._replace_record(records, spec.run_id, BatchRunStatus.COMPLETED, measurement_status="completed")
                self._write_record(self._record(records, BatchExperimentStatus.RUNNING, error_code=None))

        final_status = BatchExperimentStatus.COMPLETED if all(record.status is BatchRunStatus.COMPLETED for record in records) else BatchExperimentStatus.FAILED
        final_record = self._record(records, final_status, error_code=None if final_status is BatchExperimentStatus.COMPLETED else "incomplete_matched_group")
        self._write_record(final_record)
        report_summary = self._evaluate_completed_batch() if final_status is BatchExperimentStatus.COMPLETED else None
        return final_record, report_summary

    def _initial_or_resumed_records(self, *, resume: bool) -> list[BatchRunRecord]:
        manifest_path = self.batch_directory / "batch_manifest.json"
        record_path = self.batch_directory / "batch_record.json"
        if self.batch_directory.exists():
            if not resume:
                raise FileExistsError("refusing to overwrite existing baseline batch directory")
            try:
                stored = BatchExperimentManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
                existing = BatchExperimentRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
            except (OSError, ValidationError, ValueError) as exc:
                raise WVS7BaselinePilotError("baseline batch recovery artifacts are invalid") from exc
            if stored.model_dump(mode="json") != self.prepared.batch_manifest.model_dump(mode="json"):
                raise WVS7BaselinePilotError("baseline batch manifest does not match the requested plan")
            return list(existing.runs)
        self.batch_directory.mkdir(parents=True, exist_ok=False)
        manifest_path.write_text(self.prepared.batch_manifest.model_dump_json(indent=2), encoding="utf-8")
        records = [
            BatchRunRecord(
                run_id=spec.run_id,
                matched_group_id=spec.matched_group_id,
                condition=spec.condition,
                status=BatchRunStatus.PLANNED,
                artifacts=spec.artifacts,
            )
            for spec in self.prepared.batch_manifest.runs
        ]
        self._write_record(self._record(records, BatchExperimentStatus.PLANNED, error_code=None))
        return records

    def _run_condition(self, spec: BatchRunSpec, provider: WVS7BaselineFakeLLMProvider) -> object:
        settings = self.prepared.config
        agent_runtime = AgentRuntime(
            AgentRequestSettings(
                model_name=settings.model_name,
                temperature=settings.agent_temperature,
                max_tokens=settings.agent_max_tokens,
                seed=spec.seed,
            )
        )
        probe_runtime = ContextualProbeRuntime(
            ContextualProbeRequestSettings(
                model_name=settings.probe_model_name,
                temperature=settings.probe_temperature,
                max_tokens=settings.probe_max_tokens,
                seed=spec.seed,
            )
        )
        runner = MeasuredProtocolExperimentRunner(
            config=_run_config_from_spec(spec),
            scenario=self.prepared.scenario,
            probe_set=self.prepared.probe_set,
            agent_runtime=agent_runtime,
            contextual_probe_runtime=probe_runtime,
            agent_provider=provider,
            probe_provider=provider,
            probe_seed=spec.probe_seed,
            pre_target_round=spec.pre_target_round,
            post_target_round=spec.post_target_round,
            execute_holdout_probe=False,
        )
        return runner.run()

    def _ordered_specs_for_seed(self, seed: int) -> list[BatchRunSpec]:
        by_id = {spec.run_id: spec for spec in self.prepared.batch_manifest.runs}
        return [by_id[run_id] for run_id in self.prepared.batch_manifest.run_order if by_id[run_id].seed == seed]

    def _replace_record(
        self,
        records: list[BatchRunRecord],
        run_id: str,
        status: BatchRunStatus,
        *,
        measurement_status: str | None = None,
        error_code: str | None = None,
    ) -> list[BatchRunRecord]:
        return [
            record.model_copy(update={"status": status, "measurement_status": measurement_status, "error_code": error_code})
            if record.run_id == run_id else record
            for record in records
        ]

    def _record(self, records: list[BatchRunRecord], status: BatchExperimentStatus, *, error_code: str | None) -> BatchExperimentRecord:
        groups: list[BatchMatchedGroupRecord] = []
        record_by_id = {record.run_id: record for record in records}
        for seed in self.prepared.config.seeds:
            specs = self._ordered_specs_for_seed(seed)
            group_records = [record_by_id[spec.run_id] for spec in specs]
            groups.append(BatchMatchedGroupRecord(
                matched_group_id=specs[0].matched_group_id,
                seed=seed,
                run_ids=[record.run_id for record in group_records],
                run_statuses=[record.status for record in group_records],
                is_complete=all(record.status is BatchRunStatus.COMPLETED for record in group_records),
            ))
        canonical = json.dumps(self.prepared.batch_manifest.model_dump(mode="json"), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return BatchExperimentRecord(
            batch_id=self.batch_id,
            manifest_sha256=hashlib.sha256(canonical).hexdigest(),
            status=status,
            runs=records,
            matched_groups=groups,
            completed_run_count=sum(record.status is BatchRunStatus.COMPLETED for record in records),
            failed_run_count=sum(record.status is BatchRunStatus.FAILED for record in records),
            error_code=error_code,
        )

    def _write_record(self, record: BatchExperimentRecord) -> None:
        (self.batch_directory / "batch_record.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")

    def _evaluate_completed_batch(self) -> dict[str, object]:
        evaluation_path = self.batch_directory / "offline_evaluation_manifest.json"
        report_path = self.batch_directory / "offline_evaluation_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            return _safe_evaluation_summary(report)
        entries = [
            EvaluationRunEntry(
                run_id=spec.run_id,
                condition=EvaluationCondition(spec.condition.value),
                protocol=spec.condition,
                run_record_path=str(Path(spec.artifacts.run_record_path).resolve()),
                events_path=str(Path(spec.artifacts.events_path).resolve()),
                probe_result_path=str(Path(spec.artifacts.post_probe_results_path).resolve()),
                seed=spec.seed,
                model_name=spec.model_name,
                role="final",
                counterfactual_group_id=spec.matched_group_id,
            )
            for spec in self.prepared.batch_manifest.runs
        ]
        manifest = EvaluationManifest(
            evaluation_id=f"{self.batch_id}-offline",
            scenario_id=self.prepared.scenario.scenario_id,
            metric_suite=[MetricSuite.BASELINE_DIVERSITY, MetricSuite.REPRESENTATION, MetricSuite.COUNTERFACTUAL, MetricSuite.AUDIT],
            runs=entries,
            probe_sets=[self.prepared.probe_set.probe_set_id],
            metric_options=EvaluationMetricOptions(coverage_threshold=self.prepared.config.coverage_threshold, strict=True),
            metadata={"source": "wvs7_baseline_fake_smoke", "not_paper_result": True},
        )
        evaluation_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        report = OfflineEvaluator(manifest, manifest_directory=self.batch_directory).evaluate()
        write_evaluation_report(report, report_path)
        return _safe_evaluation_summary(report.model_dump(mode="json"))


def load_wvs7_baseline_pilot_config(path: str | Path) -> WVS7BaselinePilotConfig:
    config_path = Path(path)
    try:
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        return WVS7BaselinePilotConfig.model_validate(payload)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise WVS7BaselinePilotError("baseline pilot configuration is invalid") from exc


def prepare_wvs7_baseline_pilot(config_path: str | Path) -> PreparedWVS7BaselinePilot:
    config = load_wvs7_baseline_pilot_config(config_path)
    root = Path(__file__).resolve().parents[2]
    scenario = ScenarioLoader.load(_resolve_project_path(config.scenario_path, root))
    probe_set_path = _resolve_project_path(config.frozen_probe_set_path, root)
    frozen_manifest = load_frozen_manifest(_resolve_project_path(config.frozen_manifest_path, root))
    _validate_frozen_probe_set(config, probe_set_path, frozen_manifest)
    probe_set = ProbeSetLoader.load(probe_set_path)
    _validate_scenario(config, scenario)
    batch_manifest = _build_batch_manifest(config, scenario, probe_set)
    return PreparedWVS7BaselinePilot(config, scenario, probe_set, frozen_manifest, batch_manifest)


def dry_run_wvs7_baseline_pilot(prepared: PreparedWVS7BaselinePilot) -> WVS7BaselineDryRun:
    config, scenario, probe_set = prepared.config, prepared.scenario, prepared.probe_set
    condition_count = len(BASELINE_PROTOCOLS)
    seed_count = len(config.seeds)
    turns = condition_count * seed_count * len(scenario.agents) * scenario.max_rounds
    probes_per_phase = condition_count * seed_count * len(scenario.agents) * len(probe_set.items_for_holdout(False))
    provider = turns + probes_per_phase * 2
    tokens = turns * config.agent_max_tokens + probes_per_phase * 2 * config.probe_max_tokens
    groups = [
        {
            "matched_group_id": _matched_group_for_seed(prepared.batch_manifest, seed),
            "seed": seed,
            "condition_count": len([run for run in prepared.batch_manifest.runs if run.seed == seed]),
            "complete": True,
        }
        for seed in config.seeds
    ]
    return WVS7BaselineDryRun(
        study_id=config.study_id,
        condition_count=condition_count,
        seed_count=seed_count,
        agent_turn_count=turns,
        pre_probe_request_count=probes_per_phase,
        post_probe_request_count=probes_per_phase,
        provider_request_upper_bound=provider,
        token_upper_bound=tokens,
        matched_groups=groups,
        frozen_probe_set_sha256=config.frozen_probe_set_sha256,
    )


def _validate_frozen_probe_set(config: WVS7BaselinePilotConfig, probe_path: Path, manifest: FrozenWVS7Manifest) -> None:
    actual_hash = sha256_file(probe_path)
    if actual_hash != config.frozen_probe_set_sha256 or actual_hash != manifest.probe_set_sha256:
        raise WVS7BaselinePilotError("frozen ProbeSet SHA-256 mismatch")
    if manifest.probe_set_id != "wvs7-english-core-23" or manifest.item_count != 23 or "Q48" in manifest.variable_ids:
        raise WVS7BaselinePilotError("frozen manifest has an invalid item contract")
    probe_set = ProbeSetLoader.load(probe_path)
    if probe_set.probe_set_id != config.probe_set_id or len(probe_set.items) != 23:
        raise WVS7BaselinePilotError("frozen ProbeSet ID or item count is invalid")
    variables = tuple(item.probe_id.removeprefix("wvs7-english-core-") for item in probe_set.items)
    if variables != EXPECTED_VARIABLE_IDS or "Q48" in variables:
        raise WVS7BaselinePilotError("frozen ProbeSet variables are invalid")
    metadata = probe_set.metadata
    variable_metadata = metadata.get("variable_metadata")
    if not isinstance(variable_metadata, dict):
        raise WVS7BaselinePilotError("frozen ProbeSet lacks variable metadata")
    if metadata.get("material_status") != "frozen" or metadata.get("language_arm") != "english_core":
        raise WVS7BaselinePilotError("frozen ProbeSet material metadata is invalid")
    if metadata.get("wvs_official_factor_structure") is not False or metadata.get("q158_q163_secondary_analysis") is not True:
        raise WVS7BaselinePilotError("frozen ProbeSet analysis metadata is invalid")
    for variable_id in EXPECTED_VARIABLE_IDS:
        row = variable_metadata.get(variable_id)
        if not isinstance(row, dict) or row.get("language_arm") != "english_core":
            raise WVS7BaselinePilotError("frozen ProbeSet contains a non-English or incomplete variable row")
    q111 = variable_metadata["Q111"]
    if not _has_code(q111, "3") or not str(q111["special_response_codes"]["3"]).startswith("volunteered_other"):
        raise WVS7BaselinePilotError("frozen ProbeSet Q111 special code is missing")
    for variable_id in ("Q149", "Q150"):
        row = variable_metadata[variable_id]
        if row.get("scoring_mode") != "binary_categorical_not_continuous":
            raise WVS7BaselinePilotError("frozen ProbeSet binary category metadata is missing")
    for variable_id in DEMOCRACY_IDS:
        row = variable_metadata[variable_id]
        if not _has_code(row, "0") or not str(row["special_response_codes"]["0"]).startswith("volunteered_against_democracy"):
            raise WVS7BaselinePilotError("frozen ProbeSet democracy special code is missing")
    for variable_id in SCIENCE_IDS:
        notes = variable_metadata[variable_id].get("authoring_notes")
        if not isinstance(notes, list) or "secondary_science_technology_attitude_analysis_not_pure_value_factor" not in notes:
            raise WVS7BaselinePilotError("frozen ProbeSet secondary-analysis metadata is missing")
    q246_notes = variable_metadata["Q246"].get("authoring_notes")
    if not isinstance(q246_notes, list) or "english_chinese_semantic_scope_requires_manual_cross_language_review" not in q246_notes:
        raise WVS7BaselinePilotError("frozen ProbeSet Q246 semantic-scope warning is missing")


def _validate_scenario(config: WVS7BaselinePilotConfig, scenario: ScenarioSpec) -> None:
    if scenario.max_rounds != config.max_rounds or [agent.agent_id for agent in scenario.agents] != config.agent_ids:
        raise WVS7BaselinePilotError("scenario agents or rounds do not match the pilot plan")
    if scenario.metadata.get("language") != "en" or scenario.metadata.get("source_kind") != "author_original":
        raise WVS7BaselinePilotError("baseline scenario must be original English development material")
    if len(scenario.evidence_cards) not in {2, 3}:
        raise WVS7BaselinePilotError("baseline scenario must declare two or three public evidence cards")


def _build_batch_manifest(config: WVS7BaselinePilotConfig, scenario: ScenarioSpec, probe_set: ProbeSet) -> BatchExperimentManifest:
    parameters = BatchRuntimeParameters(
        agent_temperature=config.agent_temperature,
        agent_max_tokens=config.agent_max_tokens,
        probe_temperature=config.probe_temperature,
        probe_max_tokens=config.probe_max_tokens,
    )
    selected = probe_set.items_for_holdout(False)
    declarations = {run.run_id: run for run in config.runs}
    specs = [
        BatchRunSpec(
            run_id=declaration.run_id,
            matched_group_id=declaration.matched_group_id,
            condition=declaration.condition,
            scenario_id=scenario.scenario_id,
            model_name=config.model_name,
            probe_model_name=config.probe_model_name,
            agent_ids=config.agent_ids,
            seed=declaration.seed,
            probe_seed=declaration.seed,
            max_rounds=config.max_rounds,
            probe_set_id=probe_set.probe_set_id,
            probe_item_ids=[item.probe_id for item in selected],
            is_holdout=False,
            pre_target_round=0,
            post_target_round=config.max_rounds - 1,
            runtime_parameters=parameters,
            output_dir=config.run_output_dir,
            artifacts=expected_artifact_paths(config.run_output_dir, declaration.run_id),
        )
        for declaration in config.runs
    ]
    return BatchExperimentManifest(
        batch_id=config.study_id,
        scenario_id=scenario.scenario_id,
        model_name=config.model_name,
        probe_model_name=config.probe_model_name,
        agent_ids=config.agent_ids,
        max_rounds=config.max_rounds,
        probe_set_id=probe_set.probe_set_id,
        probe_item_ids=[item.probe_id for item in selected],
        is_holdout=False,
        seeds=config.seeds,
        conditions=list(BASELINE_PROTOCOLS),
        runs=specs,
        run_order=config.run_order,
        batch_output_dir=config.batch_output_dir,
        version=config.version,
        metadata={
            "study_id": config.study_id,
            "frozen_probe_set_sha256": config.frozen_probe_set_sha256,
            "material_status": "pilot",
            "not_paper_result": True,
            "no_mediator": True,
        },
    )


def _run_config_from_spec(spec: BatchRunSpec):
    from .models import RunConfig

    return RunConfig(
        run_id=spec.run_id,
        scenario_id=spec.scenario_id,
        model_name=spec.model_name,
        protocol=spec.condition,
        agent_count=len(spec.agent_ids),
        max_rounds=spec.max_rounds,
        seed=spec.seed,
        intervention_budget=0.0,
        output_dir=spec.output_dir,
    )


def _matched_group_for_seed(manifest: BatchExperimentManifest, seed: int) -> str:
    return next(run.matched_group_id for run in manifest.runs if run.seed == seed)


def _has_code(row: object, code: str) -> bool:
    return isinstance(row, dict) and isinstance(row.get("special_response_codes"), dict) and code in row["special_response_codes"]


def _required_metadata_string(request: LLMRequest, field: str) -> str:
    value = request.metadata.get(field)
    if not isinstance(value, str) or not value:
        raise LLMProviderError(ProviderErrorCode.INVALID_REQUEST, "local baseline request metadata is invalid")
    return value


def _required_metadata_int(request: LLMRequest, field: str) -> int:
    value = request.metadata.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise LLMProviderError(ProviderErrorCode.INVALID_REQUEST, "local baseline request metadata is invalid")
    return value


def _resolve_project_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _safe_evaluation_summary(report: dict[str, Any]) -> dict[str, object]:
    metric_names = [metric.get("metric_name") for metric in report.get("metrics", []) if isinstance(metric, dict)]
    return {
        "status": report.get("status"),
        "metric_names": metric_names,
        "warning_count": len(report.get("warnings", [])) if isinstance(report.get("warnings"), list) else 0,
    }


def safe_dry_run_summary(summary: WVS7BaselineDryRun) -> dict[str, object]:
    return summary.model_dump(mode="json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an explicit local-first WVS English baseline pilot.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=[mode.value for mode in WVS7BaselineMode])
    parser.add_argument("--run-id", help="Required batch identifier for fake-smoke and real-pilot modes.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        prepared = prepare_wvs7_baseline_pilot(arguments.config)
        mode = WVS7BaselineMode(arguments.mode)
        if mode is WVS7BaselineMode.PLAN_ONLY:
            print(json.dumps(safe_dry_run_summary(dry_run_wvs7_baseline_pilot(prepared)), ensure_ascii=True, sort_keys=True))
            return 0
        if not arguments.run_id:
            raise WVS7BaselinePilotError("run_id is required for execution modes")
        if mode is WVS7BaselineMode.REAL_PILOT:
            if not arguments.allow_network:
                raise WVS7BaselinePilotError("allow_network_required")
            raise WVS7BaselinePilotError("real_pilot_not_enabled")
        if arguments.allow_network:
            raise WVS7BaselinePilotError("fake_smoke_does_not_allow_network")
        record, evaluation = WVS7BaselinePilotRunner(prepared, batch_id=arguments.run_id).run_fake_smoke(resume=arguments.resume)
        print(json.dumps({
            "batch_id": arguments.run_id,
            "status": record.status.value,
            "completed_run_count": record.completed_run_count,
            "failed_run_count": record.failed_run_count,
            "matched_groups_complete": [group.is_complete for group in record.matched_groups],
            "offline_evaluation": evaluation,
        }, ensure_ascii=True, sort_keys=True))
        return 0 if record.status is BatchExperimentStatus.COMPLETED else 1
    except (WVS7BaselinePilotError, OSError, ValidationError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "blocking_reasons": [_error_code(exc)]}, sort_keys=True))
        return 1


def _error_code(error: BaseException) -> str:
    message = str(error)
    known = (
        "frozen ProbeSet SHA-256 mismatch", "frozen ProbeSet", "frozen manifest", "scenario agents or rounds",
        "baseline pilot configuration", "real_pilot_not_enabled", "allow_network_required", "run_id is required",
        "fake_smoke_does_not_allow_network", "baseline batch", "condition_failed", "replay_validation_failed",
    )
    for code in known:
        if code in message:
            return code.replace(" ", "_")
    return "baseline_pilot_validation_failed"


__all__ = [
    "PreparedWVS7BaselinePilot",
    "WVS7BaselineDryRun",
    "WVS7BaselineFakeLLMProvider",
    "WVS7BaselineMode",
    "WVS7BaselinePilotConfig",
    "WVS7BaselinePilotError",
    "WVS7BaselinePilotRunner",
    "WVS7BaselineRunDeclaration",
    "dry_run_wvs7_baseline_pilot",
    "load_wvs7_baseline_pilot_config",
    "prepare_wvs7_baseline_pilot",
    "safe_dry_run_summary",
]


if __name__ == "__main__":
    raise SystemExit(main())
