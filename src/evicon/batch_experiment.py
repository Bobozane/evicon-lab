"""Auditable, fail-fast orchestration for matched four-condition experiments."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from .agent_runtime import AgentRuntime
from .contextual_probe_runtime import ContextualProbeRuntime
from .llm_contract import LLMProvider
from .measured_protocol_experiment import (
    MeasuredProtocolExperimentRunner,
    MeasurementExperimentStatus,
)
from .models import ProbeSet, ProtocolCondition, RunConfig, ScenarioSpec
from .models._validation import Metadata, identifier_list, normalized_text


BASELINE_PROTOCOLS: tuple[ProtocolCondition, ...] = (
    ProtocolCondition.INDEPENDENT,
    ProtocolCondition.SOCIAL_ONLY,
    ProtocolCondition.EVIDENCE_ONLY,
    ProtocolCondition.EVIDENCE_SOCIAL,
)

_UNSAFE_METADATA_TOKENS = (
    "prompt",
    "api_key",
    "apikey",
    "authorization",
    "secret",
    "hidden",
    "probe_answer",
    "raw_response",
    "provider_metadata",
)


class BatchRunStatus(str, Enum):
    """Safe lifecycle state for one explicitly planned run."""

    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchExperimentStatus(str, Enum):
    """Safe lifecycle state for a serial matched batch."""

    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchRuntimeParameters(BaseModel):
    """Non-secret parameters that must match within every matched group."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_temperature: float = Field(ge=0.0, le=2.0)
    agent_max_tokens: int = Field(gt=0)
    probe_temperature: float = Field(ge=0.0, le=2.0)
    probe_max_tokens: int = Field(gt=0)


class BatchArtifactPaths(BaseModel):
    """Expected, content-free paths for the existing single-condition artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_directory: str = Field(min_length=1)
    events_path: str = Field(min_length=1)
    run_record_path: str = Field(min_length=1)
    pre_probe_results_path: str = Field(min_length=1)
    post_probe_results_path: str = Field(min_length=1)
    measurement_record_path: str = Field(min_length=1)

    @field_validator(
        "run_directory",
        "events_path",
        "run_record_path",
        "pre_probe_results_path",
        "post_probe_results_path",
        "measurement_record_path",
    )
    @classmethod
    def validate_path(cls, value: str) -> str:
        return normalized_text(value, "artifact path")


class BatchRunSpec(BaseModel):
    """One planned condition/seed run with explicit matching coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    matched_group_id: str = Field(min_length=1)
    condition: ProtocolCondition
    status: BatchRunStatus = BatchRunStatus.PLANNED
    scenario_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    probe_model_name: str = Field(min_length=1)
    agent_ids: list[str] = Field(min_length=1)
    seed: int
    probe_seed: int
    max_rounds: int = Field(gt=0)
    probe_set_id: str = Field(min_length=1)
    probe_item_ids: list[str] = Field(min_length=1)
    is_holdout: bool
    pre_target_round: int = Field(ge=0)
    post_target_round: int = Field(ge=0)
    runtime_parameters: BatchRuntimeParameters
    output_dir: str = Field(min_length=1)
    artifacts: BatchArtifactPaths

    @field_validator(
        "run_id",
        "matched_group_id",
        "scenario_id",
        "model_name",
        "probe_model_name",
        "probe_set_id",
        "output_dir",
    )
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "text")
        normalized = normalized_text(value, field_name)
        if field_name in {"run_id", "matched_group_id"} and Path(normalized).name != normalized:
            raise ValueError(f"{field_name} must be a simple identifier")
        return normalized

    @field_validator("agent_ids", "probe_item_ids")
    @classmethod
    def validate_ids(cls, value: list[str], info: object) -> list[str]:
        return identifier_list(value, getattr(info, "field_name", "ids"))

    @field_validator("seed", "probe_seed", mode="before")
    @classmethod
    def validate_seed(cls, value: object, info: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{getattr(info, 'field_name', 'seed')} must be an integer")
        return value

    @model_validator(mode="after")
    def validate_coordinates(self) -> "BatchRunSpec":
        if self.condition not in BASELINE_PROTOCOLS:
            raise ValueError("condition must be one of the four baseline protocols")
        if self.status is not BatchRunStatus.PLANNED:
            raise ValueError("batch manifest run specs must start in planned status")
        if self.seed != self.probe_seed:
            raise ValueError("seed and probe_seed must match within an explicitly paired run")
        if self.pre_target_round != 0:
            raise ValueError("pre_target_round must be 0 for initial-context probes")
        if self.post_target_round >= self.max_rounds:
            raise ValueError("post_target_round must be before max_rounds")
        expected = expected_artifact_paths(self.output_dir, self.run_id)
        if self.artifacts.model_dump(mode="json") != expected.model_dump(mode="json"):
            raise ValueError("artifacts must exactly match the explicit output_dir and run_id")
        return self


class BatchExperimentManifest(BaseModel):
    """A fully explicit four-condition pairing plan; it contains no result data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    probe_model_name: str = Field(min_length=1)
    agent_ids: list[str] = Field(min_length=1)
    max_rounds: int = Field(gt=0)
    probe_set_id: str = Field(min_length=1)
    probe_item_ids: list[str] = Field(min_length=1)
    is_holdout: bool
    seeds: list[int] = Field(min_length=1)
    conditions: list[ProtocolCondition] = Field(min_length=4)
    runs: list[BatchRunSpec] = Field(min_length=4)
    run_order: list[str] = Field(min_length=4)
    batch_output_dir: str = Field(min_length=1)
    version: str = Field(min_length=1)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator(
        "batch_id",
        "scenario_id",
        "model_name",
        "probe_model_name",
        "probe_set_id",
        "batch_output_dir",
        "version",
    )
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "text")
        normalized = normalized_text(value, field_name)
        if field_name == "batch_id" and Path(normalized).name != normalized:
            raise ValueError("batch_id must be a simple identifier")
        return normalized

    @field_validator("agent_ids", "probe_item_ids", "run_order")
    @classmethod
    def validate_ids(cls, value: list[str], info: object) -> list[str]:
        return identifier_list(value, getattr(info, "field_name", "ids"))

    @field_validator("seeds", mode="before")
    @classmethod
    def validate_seeds(cls, value: object) -> list[int]:
        if not isinstance(value, list) or not value:
            raise ValueError("seeds must be a non-empty list of integers")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in value):
            raise ValueError("seeds must be a non-empty list of integers")
        if len(set(value)) != len(value):
            raise ValueError("seeds must not contain duplicates")
        return value

    @field_validator("metadata")
    @classmethod
    def validate_safe_metadata(cls, value: Metadata) -> Metadata:
        _reject_unsafe_metadata(value)
        return value

    @model_validator(mode="after")
    def validate_matched_plan(self) -> "BatchExperimentManifest":
        if tuple(self.conditions) != BASELINE_PROTOCOLS:
            raise ValueError("conditions must explicitly list the four baseline protocols in canonical order")
        if len(self.runs) != len(self.seeds) * len(BASELINE_PROTOCOLS):
            raise ValueError("runs must contain exactly one entry for every condition and seed")
        run_by_id = {run.run_id: run for run in self.runs}
        if len(run_by_id) != len(self.runs):
            raise ValueError("runs must not contain duplicate run_id values")
        if set(self.run_order) != set(run_by_id) or len(self.run_order) != len(run_by_id):
            raise ValueError("run_order must explicitly contain every run_id exactly once")

        seen_pairs: set[tuple[int, ProtocolCondition]] = set()
        groups_by_seed: dict[int, list[BatchRunSpec]] = {seed: [] for seed in self.seeds}
        for run in self.runs:
            if run.scenario_id != self.scenario_id or run.model_name != self.model_name:
                raise ValueError("every run must match the manifest scenario_id and model_name")
            if run.probe_model_name != self.probe_model_name or run.agent_ids != self.agent_ids:
                raise ValueError("every run must match the manifest probe model and ordered agent IDs")
            if run.max_rounds != self.max_rounds or run.probe_set_id != self.probe_set_id:
                raise ValueError("every run must match the manifest rounds and probe_set_id")
            if run.probe_item_ids != self.probe_item_ids or run.is_holdout != self.is_holdout:
                raise ValueError("every run must match the explicit probe item selection")
            if run.seed not in groups_by_seed:
                raise ValueError("run seed must be explicitly listed in manifest seeds")
            pair = (run.seed, run.condition)
            if pair in seen_pairs:
                raise ValueError("manifest must not contain duplicate condition/seed run entries")
            seen_pairs.add(pair)
            groups_by_seed[run.seed].append(run)

        for seed, group in groups_by_seed.items():
            if len(group) != len(BASELINE_PROTOCOLS) or {run.condition for run in group} != set(BASELINE_PROTOCOLS):
                raise ValueError("every seed must have one run for every baseline condition")
            group_ids = {run.matched_group_id for run in group}
            if len(group_ids) != 1:
                raise ValueError("all four condition runs for one seed must share a matched_group_id")
            shared = {
                "scenario_id": group[0].scenario_id,
                "model_name": group[0].model_name,
                "probe_model_name": group[0].probe_model_name,
                "agent_ids": group[0].agent_ids,
                "seed": group[0].seed,
                "probe_seed": group[0].probe_seed,
                "max_rounds": group[0].max_rounds,
                "probe_set_id": group[0].probe_set_id,
                "probe_item_ids": group[0].probe_item_ids,
                "is_holdout": group[0].is_holdout,
                "pre_target_round": group[0].pre_target_round,
                "post_target_round": group[0].post_target_round,
                "runtime_parameters": group[0].runtime_parameters.model_dump(mode="json"),
                "output_dir": group[0].output_dir,
            }
            for run in group[1:]:
                candidate = {
                    "scenario_id": run.scenario_id,
                    "model_name": run.model_name,
                    "probe_model_name": run.probe_model_name,
                    "agent_ids": run.agent_ids,
                    "seed": run.seed,
                    "probe_seed": run.probe_seed,
                    "max_rounds": run.max_rounds,
                    "probe_set_id": run.probe_set_id,
                    "probe_item_ids": run.probe_item_ids,
                    "is_holdout": run.is_holdout,
                    "pre_target_round": run.pre_target_round,
                    "post_target_round": run.post_target_round,
                    "runtime_parameters": run.runtime_parameters.model_dump(mode="json"),
                    "output_dir": run.output_dir,
                }
                if candidate != shared:
                    raise ValueError(f"matched_group_id for seed {seed} has inconsistent non-condition parameters")
        return self


class BatchRunRecord(BaseModel):
    """Safe per-run batch audit state without copying any single-run output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    matched_group_id: str = Field(min_length=1)
    condition: ProtocolCondition
    status: BatchRunStatus
    artifacts: BatchArtifactPaths
    measurement_status: str | None = None
    error_code: str | None = None


class BatchMatchedGroupRecord(BaseModel):
    """Explicit completeness state for one counterfactual seed group."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    matched_group_id: str = Field(min_length=1)
    seed: int
    run_ids: list[str] = Field(min_length=4)
    run_statuses: list[BatchRunStatus] = Field(min_length=4)
    is_complete: bool


class BatchExperimentRecord(BaseModel):
    """Safe, updatable batch state. It contains no prompts or result contents."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_id: str = Field(min_length=1)
    manifest_sha256: str = Field(min_length=64, max_length=64)
    status: BatchExperimentStatus
    runs: list[BatchRunRecord] = Field(min_length=4)
    matched_groups: list[BatchMatchedGroupRecord] = Field(min_length=1)
    completed_run_count: int = Field(ge=0)
    failed_run_count: int = Field(ge=0)
    error_code: str | None = None


class BatchMeasuredExperimentRunner:
    """Serially execute a manifest in explicit order, stopping at the first failure."""

    def __init__(
        self,
        manifest: BatchExperimentManifest,
        *,
        scenario: ScenarioSpec,
        probe_set: ProbeSet,
        agent_runtime: AgentRuntime | Mapping[int, AgentRuntime],
        contextual_probe_runtime: ContextualProbeRuntime | Mapping[int, ContextualProbeRuntime],
        agent_provider: LLMProvider,
        probe_provider: LLMProvider,
    ) -> None:
        self.manifest = manifest
        self.scenario = scenario
        self.probe_set = probe_set
        self.agent_runtime = agent_runtime
        self.contextual_probe_runtime = contextual_probe_runtime
        self.agent_provider = agent_provider
        self.probe_provider = probe_provider
        self._validate_static_contract()

    @property
    def output_directory(self) -> Path:
        return Path(self.manifest.batch_output_dir) / self.manifest.batch_id

    def run(self) -> BatchExperimentRecord:
        """Write an audit plan then execute only the manifest's declared sequence."""
        self._prepare_output_directories()
        manifest_path = self.output_directory / "batch_manifest.json"
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(self.manifest.model_dump(mode="json"), handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")

        records = [self._planned_record(run) for run in self._ordered_specs()]
        batch_record = self._build_record(BatchExperimentStatus.RUNNING, records, error_code=None)
        self._write_batch_record(batch_record)

        for index, spec in enumerate(self._ordered_specs()):
            records[index] = records[index].model_copy(update={"status": BatchRunStatus.RUNNING})
            batch_record = self._build_record(BatchExperimentStatus.RUNNING, records, error_code=None)
            self._write_batch_record(batch_record)
            try:
                result = self._single_runner(spec).run()
            except Exception:
                records[index] = records[index].model_copy(
                    update={"status": BatchRunStatus.FAILED, "error_code": "run_orchestration_failed"}
                )
                batch_record = self._build_record(BatchExperimentStatus.FAILED, records, error_code="run_orchestration_failed")
                self._write_batch_record(batch_record)
                return batch_record

            if result.status is not MeasurementExperimentStatus.COMPLETED:
                records[index] = records[index].model_copy(
                    update={
                        "status": BatchRunStatus.FAILED,
                        "measurement_status": result.status.value,
                        "error_code": result.status.value,
                    }
                )
                batch_record = self._build_record(BatchExperimentStatus.FAILED, records, error_code=result.status.value)
                self._write_batch_record(batch_record)
                return batch_record
            if not _artifacts_exist(spec.artifacts):
                records[index] = records[index].model_copy(
                    update={"status": BatchRunStatus.FAILED, "error_code": "expected_artifacts_missing"}
                )
                batch_record = self._build_record(BatchExperimentStatus.FAILED, records, error_code="expected_artifacts_missing")
                self._write_batch_record(batch_record)
                return batch_record
            records[index] = records[index].model_copy(
                update={"status": BatchRunStatus.COMPLETED, "measurement_status": result.status.value}
            )
            batch_record = self._build_record(BatchExperimentStatus.RUNNING, records, error_code=None)
            self._write_batch_record(batch_record)

        batch_record = self._build_record(BatchExperimentStatus.COMPLETED, records, error_code=None)
        self._write_batch_record(batch_record)
        return batch_record

    def _validate_static_contract(self) -> None:
        if self.scenario.scenario_id != self.manifest.scenario_id:
            raise ValueError("scenario.scenario_id must match manifest.scenario_id")
        if [agent.agent_id for agent in self.scenario.agents] != self.manifest.agent_ids:
            raise ValueError("scenario agent IDs must match manifest agent_ids in order")
        if self.scenario.max_rounds != self.manifest.max_rounds:
            raise ValueError("scenario.max_rounds must match manifest.max_rounds")
        if self.probe_set.probe_set_id != self.manifest.probe_set_id:
            raise ValueError("probe_set.probe_set_id must match manifest.probe_set_id")
        selected = self.probe_set.items_for_holdout(self.manifest.is_holdout)
        if [item.probe_id for item in selected] != self.manifest.probe_item_ids:
            raise ValueError("probe set selection must match manifest probe_item_ids")
        for spec in self.manifest.runs:
            agent_runtime = self._agent_runtime_for_seed(spec.seed)
            probe_runtime = self._probe_runtime_for_seed(spec.seed)
            agent_settings = agent_runtime._request_settings
            probe_settings = probe_runtime._request_settings
            if (
                agent_settings.model_name != spec.model_name
                or agent_settings.seed != spec.seed
                or agent_settings.temperature != spec.runtime_parameters.agent_temperature
                or agent_settings.max_tokens != spec.runtime_parameters.agent_max_tokens
            ):
                raise ValueError("injected AgentRuntime must match every explicit BatchRunSpec parameter")
            if (
                probe_settings.model_name != spec.probe_model_name
                or probe_settings.seed != spec.probe_seed
                or probe_settings.temperature != spec.runtime_parameters.probe_temperature
                or probe_settings.max_tokens != spec.runtime_parameters.probe_max_tokens
            ):
                raise ValueError("injected ContextualProbeRuntime must match every explicit BatchRunSpec parameter")

    def _prepare_output_directories(self) -> None:
        if self.output_directory.exists():
            raise FileExistsError(f"refusing to overwrite existing batch directory: {self.output_directory}")
        for spec in self.manifest.runs:
            if Path(spec.artifacts.run_directory).exists():
                raise FileExistsError(f"refusing to overwrite existing run directory: {spec.artifacts.run_directory}")
        self.output_directory.mkdir(parents=True, exist_ok=False)

    def _single_runner(self, spec: BatchRunSpec) -> MeasuredProtocolExperimentRunner:
        return MeasuredProtocolExperimentRunner(
            RunConfig(
                run_id=spec.run_id,
                scenario_id=spec.scenario_id,
                model_name=spec.model_name,
                protocol=spec.condition,
                agent_count=len(spec.agent_ids),
                max_rounds=spec.max_rounds,
                seed=spec.seed,
                intervention_budget=0.0,
                output_dir=spec.output_dir,
            ),
            scenario=self.scenario,
            probe_set=self.probe_set,
            agent_runtime=self._agent_runtime_for_seed(spec.seed),
            contextual_probe_runtime=self._probe_runtime_for_seed(spec.seed),
            agent_provider=self.agent_provider,
            probe_provider=self.probe_provider,
            probe_seed=spec.probe_seed,
            pre_target_round=spec.pre_target_round,
            post_target_round=spec.post_target_round,
            execute_holdout_probe=spec.is_holdout,
        )

    def _agent_runtime_for_seed(self, seed: int) -> AgentRuntime:
        if isinstance(self.agent_runtime, Mapping):
            if seed not in self.agent_runtime:
                raise ValueError(f"no injected AgentRuntime for seed {seed}")
            return self.agent_runtime[seed]
        return self.agent_runtime

    def _probe_runtime_for_seed(self, seed: int) -> ContextualProbeRuntime:
        if isinstance(self.contextual_probe_runtime, Mapping):
            if seed not in self.contextual_probe_runtime:
                raise ValueError(f"no injected ContextualProbeRuntime for seed {seed}")
            return self.contextual_probe_runtime[seed]
        return self.contextual_probe_runtime

    def _ordered_specs(self) -> list[BatchRunSpec]:
        by_id = {spec.run_id: spec for spec in self.manifest.runs}
        return [by_id[run_id] for run_id in self.manifest.run_order]

    @staticmethod
    def _planned_record(spec: BatchRunSpec) -> BatchRunRecord:
        return BatchRunRecord(
            run_id=spec.run_id,
            matched_group_id=spec.matched_group_id,
            condition=spec.condition,
            status=BatchRunStatus.PLANNED,
            artifacts=spec.artifacts,
        )

    def _build_record(
        self,
        status: BatchExperimentStatus,
        runs: list[BatchRunRecord],
        *,
        error_code: str | None,
    ) -> BatchExperimentRecord:
        groups: list[BatchMatchedGroupRecord] = []
        records_by_id = {record.run_id: record for record in runs}
        for seed in self.manifest.seeds:
            specs = [spec for spec in self._ordered_specs() if spec.seed == seed]
            group_records = [records_by_id[spec.run_id] for spec in specs]
            groups.append(
                BatchMatchedGroupRecord(
                    matched_group_id=specs[0].matched_group_id,
                    seed=seed,
                    run_ids=[record.run_id for record in group_records],
                    run_statuses=[record.status for record in group_records],
                    is_complete=all(record.status is BatchRunStatus.COMPLETED for record in group_records),
                )
            )
        manifest_bytes = json.dumps(self.manifest.model_dump(mode="json"), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return BatchExperimentRecord(
            batch_id=self.manifest.batch_id,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            status=status,
            runs=runs,
            matched_groups=groups,
            completed_run_count=sum(record.status is BatchRunStatus.COMPLETED for record in runs),
            failed_run_count=sum(record.status is BatchRunStatus.FAILED for record in runs),
            error_code=error_code,
        )

    def _write_batch_record(self, record: BatchExperimentRecord) -> None:
        path = self.output_directory / "batch_record.json"
        mode = "w" if path.exists() else "x"
        with path.open(mode, encoding="utf-8") as handle:
            json.dump(record.model_dump(mode="json"), handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")


def expected_artifact_paths(output_dir: str, run_id: str) -> BatchArtifactPaths:
    """Return the fixed public/private artifact path names without reading them."""
    run_directory = Path(output_dir) / run_id
    probes = run_directory / "probes"
    return BatchArtifactPaths(
        run_directory=str(run_directory),
        events_path=str(run_directory / "events.jsonl"),
        run_record_path=str(run_directory / "run_record.json"),
        pre_probe_results_path=str(probes / "pre_probe_results.jsonl"),
        post_probe_results_path=str(probes / "post_probe_results.jsonl"),
        measurement_record_path=str(run_directory / "measurement_record.json"),
    )


def build_four_condition_manifest(
    *,
    batch_id: str,
    scenario: ScenarioSpec,
    probe_set: ProbeSet,
    model_name: str,
    probe_model_name: str,
    seeds: list[int],
    output_dir: str,
    batch_output_dir: str,
    runtime_parameters: BatchRuntimeParameters,
    is_holdout: bool = True,
    pre_target_round: int = 0,
    post_target_round: int | None = None,
    version: str = "batch_manifest.v1",
    metadata: Metadata | None = None,
) -> BatchExperimentManifest:
    """Build all explicit condition/seed entries without inspecting any results."""
    selected = probe_set.items_for_holdout(is_holdout)
    if not selected:
        raise ValueError("probe set has no items for the requested is_holdout selection")
    target_round = scenario.max_rounds - 1 if post_target_round is None else post_target_round
    runs: list[BatchRunSpec] = []
    run_order: list[str] = []
    for seed in seeds:
        matched_group_id = f"{batch_id}-seed-{seed}"
        for condition in BASELINE_PROTOCOLS:
            run_id = f"{batch_id}-{condition.value}-seed-{seed}"
            runs.append(
                BatchRunSpec(
                    run_id=run_id,
                    matched_group_id=matched_group_id,
                    condition=condition,
                    scenario_id=scenario.scenario_id,
                    model_name=model_name,
                    probe_model_name=probe_model_name,
                    agent_ids=[agent.agent_id for agent in scenario.agents],
                    seed=seed,
                    probe_seed=seed,
                    max_rounds=scenario.max_rounds,
                    probe_set_id=probe_set.probe_set_id,
                    probe_item_ids=[item.probe_id for item in selected],
                    is_holdout=is_holdout,
                    pre_target_round=pre_target_round,
                    post_target_round=target_round,
                    runtime_parameters=runtime_parameters,
                    output_dir=output_dir,
                    artifacts=expected_artifact_paths(output_dir, run_id),
                )
            )
            run_order.append(run_id)
    return BatchExperimentManifest(
        batch_id=batch_id,
        scenario_id=scenario.scenario_id,
        model_name=model_name,
        probe_model_name=probe_model_name,
        agent_ids=[agent.agent_id for agent in scenario.agents],
        max_rounds=scenario.max_rounds,
        probe_set_id=probe_set.probe_set_id,
        probe_item_ids=[item.probe_id for item in selected],
        is_holdout=is_holdout,
        seeds=seeds,
        conditions=list(BASELINE_PROTOCOLS),
        runs=runs,
        run_order=run_order,
        batch_output_dir=batch_output_dir,
        version=version,
        metadata={} if metadata is None else metadata,
    )


def _artifacts_exist(paths: BatchArtifactPaths) -> bool:
    return all(
        Path(path).is_file()
        for path in (
            paths.events_path,
            paths.run_record_path,
            paths.pre_probe_results_path,
            paths.post_probe_results_path,
            paths.measurement_record_path,
        )
    )


def _reject_unsafe_metadata(value: Metadata) -> None:
    def visit(item: JsonValue, path: str) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                if any(token in key.lower() for token in _UNSAFE_METADATA_TOKENS):
                    raise ValueError(f"metadata contains unsafe key at {path}.{key}")
                visit(nested, f"{path}.{key}")
        elif isinstance(item, list):
            for index, nested in enumerate(item):
                visit(nested, f"{path}[{index}]")
        elif isinstance(item, str) and any(token in item.lower() for token in _UNSAFE_METADATA_TOKENS):
            raise ValueError(f"metadata contains unsafe content at {path}")

    visit(value, "metadata")


__all__ = [
    "BASELINE_PROTOCOLS",
    "BatchArtifactPaths",
    "BatchExperimentManifest",
    "BatchExperimentRecord",
    "BatchExperimentStatus",
    "BatchMatchedGroupRecord",
    "BatchMeasuredExperimentRunner",
    "BatchRunRecord",
    "BatchRunSpec",
    "BatchRunStatus",
    "BatchRuntimeParameters",
    "build_four_condition_manifest",
    "expected_artifact_paths",
]
