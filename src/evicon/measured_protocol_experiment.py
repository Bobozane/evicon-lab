"""One-condition, offline-measured real-Agent protocol orchestration.

This module deliberately composes existing isolated components. It does not
change public protocol semantics or feed measurement results to any controller.
"""

from __future__ import annotations

import hashlib
import json
import os
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .agent_runtime import AgentRuntime
from .contextual_probe_runtime import (
    ContextualProbeRequestSettings,
    ContextualProbeRuntime,
    ContextualProbeStatus,
)
from .llm_contract import LLMProvider
from .measurement_context import OfflineMeasurementContext, OfflineMeasurementContextBuilder
from .models import ProbeResult, ProbeSet, ProtocolCondition, RunConfig, RunRecord, RunStatus, ScenarioSpec, ValueProbeResponse
from .models._validation import identifier_list, normalized_text
from .private_probe_recovery import (
    PrivateProbeRecoveryCheckpoint,
    PrivateProbeRecoveryError,
    PrivateProbeRecoveryStore,
    ProbeRecoveryPhase,
    context_sha256,
)
from .probe_profiles import ProbeProfileBuilder
from .real_agent_runner import RealAgentProtocolRunner


class MeasurementExperimentStatus(str, Enum):
    """Terminal orchestration states; none indicates a research outcome."""

    COMPLETED = "completed"
    PRE_PROBE_FAILED = "pre_probe_failed"
    AGENT_RUN_FAILED = "agent_run_failed"
    POST_PROBE_FAILED = "post_probe_failed"


class MeasurementPhaseStatus(str, Enum):
    """Safe phase lifecycle values persisted in the measurement record."""

    NOT_STARTED = "not_started"
    COMPLETED = "completed"
    FAILED = "failed"


class TokenUsageSummary(BaseModel):
    """Safe aggregate usage; unknown values remain explicit nulls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class MeasurementRecord(BaseModel):
    """Safe experiment summary excluding prompts, turns, items, and answers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    protocol: ProtocolCondition
    model_name: str = Field(min_length=1)
    probe_model_name: str = Field(min_length=1)
    probe_set_id: str = Field(min_length=1)
    agent_ids: list[str] = Field(min_length=1)
    probe_item_ids: list[str] = Field(min_length=1)
    probe_seed: int
    is_holdout: bool
    pre_target_round: int = Field(ge=0)
    post_target_round: int = Field(ge=0)
    agent_run_status: RunStatus | None = None
    pre_probe_status: MeasurementPhaseStatus
    post_probe_status: MeasurementPhaseStatus
    pre_probe_result_count: int = Field(ge=0)
    post_probe_result_count: int = Field(ge=0)
    pre_probe_result_path: str | None = None
    post_probe_result_path: str | None = None
    pre_probe_sha256: str | None = None
    post_probe_sha256: str | None = None
    agent_token_usage: TokenUsageSummary
    probe_token_usage: TokenUsageSummary
    error_code: str | None = None

    @field_validator("run_id", "scenario_id", "model_name", "probe_model_name", "probe_set_id")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("agent_ids", "probe_item_ids")
    @classmethod
    def validate_ids(cls, value: list[str], info: object) -> list[str]:
        return identifier_list(value, getattr(info, "field_name", "ids"))

    @field_validator("probe_seed", mode="before")
    @classmethod
    def validate_seed(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("probe_seed must be an integer")
        return value


class MeasuredProtocolExperimentResult(BaseModel):
    """In-memory outcome. Private probe responses are never copied to audits."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    status: MeasurementExperimentStatus
    pre_results: list[ProbeResult] = Field(default_factory=list)
    post_results: list[ProbeResult] = Field(default_factory=list)
    run_record: RunRecord | None = None
    measurement_record: MeasurementRecord | None = None
    error_code: str | None = None


class MeasuredProtocolExperimentRunner:
    """Execute private pre probes, one public trajectory, then private post probes."""

    def __init__(
        self,
        config: RunConfig,
        *,
        scenario: ScenarioSpec,
        probe_set: ProbeSet,
        agent_runtime: AgentRuntime,
        contextual_probe_runtime: ContextualProbeRuntime,
        agent_provider: LLMProvider,
        probe_provider: LLMProvider,
        probe_seed: int,
        pre_target_round: int = 0,
        post_target_round: int | None = None,
        execute_holdout_probe: bool = True,
        recovery_checkpoint_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.scenario = scenario
        self.probe_set = probe_set
        self.agent_runtime = agent_runtime
        self.contextual_probe_runtime = contextual_probe_runtime
        self.agent_provider = agent_provider
        self.probe_provider = probe_provider
        self.probe_seed = probe_seed
        self.pre_target_round = pre_target_round
        self.post_target_round = scenario.max_rounds - 1 if post_target_round is None else post_target_round
        self.execute_holdout_probe = execute_holdout_probe
        self._recovery_store = (
            None if recovery_checkpoint_dir is None else PrivateProbeRecoveryStore(recovery_checkpoint_dir)
        )
        self._probe_audits: list[object] = []
        self._validate_static_contract()

    @property
    def output_directory(self) -> Path:
        return Path(self.config.output_dir) / self.config.run_id

    @property
    def selected_items(self) -> list[object]:
        return self.probe_set.items_for_holdout(self.execute_holdout_probe)

    def run(self, *, resume: bool = False) -> MeasuredProtocolExperimentResult:
        """Run the fixed sequence, optionally resuming a private probe checkpoint."""
        output_exists = self.output_directory.exists()
        if output_exists and not resume:
            raise FileExistsError(f"refusing to overwrite existing run directory: {self.output_directory}")

        pre_contexts = self._initial_contexts()
        try:
            pre_checkpoint = self._open_recovery_checkpoint(
                ProbeRecoveryPhase.PRE,
                pre_contexts,
                resume=resume,
            )
        except PrivateProbeRecoveryError as error:
            return MeasuredProtocolExperimentResult(
                status=MeasurementExperimentStatus.PRE_PROBE_FAILED,
                error_code=str(error),
            )
        pre_results, pre_error = self._run_probe_phase(pre_contexts, checkpoint=pre_checkpoint)
        if pre_error is not None:
            return MeasuredProtocolExperimentResult(
                status=MeasurementExperimentStatus.PRE_PROBE_FAILED,
                pre_results=pre_results,
                error_code=pre_error,
            )

        if output_exists:
            run_record = self._read_record_if_present()
            if run_record is None or run_record.status is not RunStatus.COMPLETED or run_record.config != self.config:
                return MeasuredProtocolExperimentResult(
                    status=MeasurementExperimentStatus.AGENT_RUN_FAILED,
                    pre_results=pre_results,
                    run_record=run_record,
                    error_code="agent_recovery_not_supported",
                )
            agent_usage = TokenUsageSummary()
        else:
            agent_runner = RealAgentProtocolRunner(
                self.config,
                scenario=self.scenario,
                agent_runtime=self.agent_runtime,
                provider=self.agent_provider,
            )
            try:
                run_record = agent_runner.run()
            except Exception:
                failed_record = self._read_record_if_present()
                record = self._measurement_record(
                    agent_run_status=RunStatus.FAILED,
                    pre_status=MeasurementPhaseStatus.COMPLETED,
                    post_status=MeasurementPhaseStatus.NOT_STARTED,
                    pre_count=len(pre_results),
                    post_count=0,
                    agent_usage=_usage_from_mapping(agent_runner.total_token_usage),
                    error_code="agent_run_failed",
                )
                self._write_measurement_record(record)
                return MeasuredProtocolExperimentResult(
                    status=MeasurementExperimentStatus.AGENT_RUN_FAILED,
                    pre_results=pre_results,
                    run_record=failed_record,
                    measurement_record=record,
                    error_code="agent_run_failed",
                )
            agent_usage = _usage_from_mapping(agent_runner.total_token_usage)

        post_contexts, context_error = self._post_contexts(run_record)
        if context_error is not None:
            record = self._measurement_record(
                agent_run_status=run_record.status,
                pre_status=MeasurementPhaseStatus.COMPLETED,
                post_status=MeasurementPhaseStatus.FAILED,
                pre_count=len(pre_results),
                post_count=0,
                agent_usage=agent_usage,
                error_code="post_context_failed",
            )
            self._write_measurement_record(record, replace=resume)
            return MeasuredProtocolExperimentResult(
                status=MeasurementExperimentStatus.POST_PROBE_FAILED,
                pre_results=pre_results,
                run_record=run_record,
                measurement_record=record,
                error_code="post_context_failed",
            )

        try:
            post_checkpoint = self._open_recovery_checkpoint(
                ProbeRecoveryPhase.POST,
                post_contexts,
                resume=resume,
            )
        except PrivateProbeRecoveryError as error:
            record = self._measurement_record(
                agent_run_status=run_record.status,
                pre_status=MeasurementPhaseStatus.COMPLETED,
                post_status=MeasurementPhaseStatus.FAILED,
                pre_count=len(pre_results),
                post_count=0,
                agent_usage=agent_usage,
                error_code=str(error),
            )
            self._write_measurement_record(record, replace=resume)
            return MeasuredProtocolExperimentResult(
                status=MeasurementExperimentStatus.POST_PROBE_FAILED,
                pre_results=pre_results,
                run_record=run_record,
                measurement_record=record,
                error_code=str(error),
            )
        post_results, post_error = self._run_probe_phase(post_contexts, checkpoint=post_checkpoint)
        if post_error is not None:
            record = self._measurement_record(
                agent_run_status=run_record.status,
                pre_status=MeasurementPhaseStatus.COMPLETED,
                post_status=MeasurementPhaseStatus.FAILED,
                pre_count=len(pre_results),
                post_count=len(post_results),
                agent_usage=agent_usage,
                error_code="post_probe_failed",
            )
            self._write_measurement_record(record, replace=resume)
            return MeasuredProtocolExperimentResult(
                status=MeasurementExperimentStatus.POST_PROBE_FAILED,
                pre_results=pre_results,
                post_results=post_results,
                run_record=run_record,
                measurement_record=record,
                error_code=post_error,
            )

        pre_path, post_path = self._write_phase_results(pre_results, post_results)
        record = self._measurement_record(
            agent_run_status=run_record.status,
            pre_status=MeasurementPhaseStatus.COMPLETED,
            post_status=MeasurementPhaseStatus.COMPLETED,
            pre_count=len(pre_results),
            post_count=len(post_results),
            pre_path=pre_path,
            post_path=post_path,
            agent_usage=agent_usage,
            error_code=None,
        )
        self._write_measurement_record(record, replace=resume)
        if self._recovery_store is not None:
            self._recovery_store.clear()
        return MeasuredProtocolExperimentResult(
            status=MeasurementExperimentStatus.COMPLETED,
            pre_results=pre_results,
            post_results=post_results,
            run_record=run_record,
            measurement_record=record,
        )

    def _validate_static_contract(self) -> None:
        if self.config.scenario_id != self.scenario.scenario_id:
            raise ValueError("config.scenario_id must match scenario.scenario_id")
        if self.config.agent_count != len(self.scenario.agents):
            raise ValueError("config.agent_count must match scenario agent count")
        if self.config.max_rounds != self.scenario.max_rounds:
            raise ValueError("config.max_rounds must match scenario.max_rounds")
        if type(self.probe_seed) is not int:
            raise ValueError("probe_seed must be an integer")
        if self.pre_target_round != 0:
            raise ValueError("pre_target_round must be 0 because pre probes use the initial public context")
        if type(self.post_target_round) is not int or not 0 <= self.post_target_round < self.config.max_rounds:
            raise ValueError("post_target_round must be within the configured run")
        if self.agent_runtime._request_settings.model_name != self.config.model_name:
            raise ValueError("config.model_name must match AgentRuntime request model_name")
        probe_settings: ContextualProbeRequestSettings = self.contextual_probe_runtime._request_settings
        if probe_settings.seed != self.probe_seed:
            raise ValueError("probe_seed must match ContextualProbeRuntime request seed")
        if not self.selected_items:
            raise ValueError("probe set has no items for the requested is_holdout selection")

    def _initial_contexts(self) -> list[OfflineMeasurementContext]:
        return [
            OfflineMeasurementContext(
                agent_id=agent.agent_id,
                role=agent.role,
                initial_value_labels=list(agent.initial_value_labels),
                round_id=self.pre_target_round,
                protocol=self.config.protocol,
                scenario_context=self.scenario.initial_context,
                own_history=[],
                visible_peer_turns=[],
                visible_evidence_cards=[],
            )
            for agent in self.scenario.agents
        ]

    def _post_contexts(self, run_record: RunRecord) -> tuple[list[OfflineMeasurementContext], str | None]:
        if run_record.status is not RunStatus.COMPLETED:
            return [], "run_not_completed"
        try:
            return [
                OfflineMeasurementContextBuilder.build(
                    run_record,
                    agent_id=agent.agent_id,
                    round_id=self.post_target_round,
                )
                for agent in self.scenario.agents
            ], None
        except (TypeError, ValueError):
            return [], "post_context_failed"

    def _open_recovery_checkpoint(
        self,
        phase: ProbeRecoveryPhase,
        contexts: list[OfflineMeasurementContext],
        *,
        resume: bool,
    ) -> PrivateProbeRecoveryCheckpoint | None:
        if self._recovery_store is None:
            return None
        settings: ContextualProbeRequestSettings = self.contextual_probe_runtime._request_settings
        identity = PrivateProbeRecoveryCheckpoint(
            run_id=self.config.run_id,
            scenario_id=self.scenario.scenario_id,
            protocol=self.config.protocol,
            probe_set_id=self.probe_set.probe_set_id,
            probe_model_name=settings.model_name,
            probe_seed=self.probe_seed,
            phase=phase,
            target_round=contexts[0].round_id,
            is_holdout=self.execute_holdout_probe,
            agent_ids=[context.agent_id for context in contexts],
            probe_item_ids=[item.probe_id for item in self.selected_items],
            context_sha256=[context_sha256(context) for context in contexts],
        )
        return self._recovery_store.open(
            identity,
            resume=resume,
            require_existing=resume and phase is ProbeRecoveryPhase.PRE,
        )

    def _run_probe_phase(
        self,
        contexts: list[OfflineMeasurementContext],
        *,
        checkpoint: PrivateProbeRecoveryCheckpoint | None,
    ) -> tuple[list[ProbeResult], str | None]:
        phase_results: list[ProbeResult] = []
        for context in contexts:
            responses: list[ValueProbeResponse] = []
            for item in self.selected_items:
                restored = (
                    checkpoint.response_for(agent_id=context.agent_id, probe_id=item.probe_id)
                    if checkpoint is not None
                    else None
                )
                if restored is not None:
                    responses.append(restored)
                    continue
                outcome = self.contextual_probe_runtime.execute(context, item, self.probe_provider)
                self._probe_audits.append(outcome.audit_summary())
                if outcome.status is not ContextualProbeStatus.COMPLETED or outcome.response is None:
                    return phase_results, outcome.audit.error_code or outcome.status.value
                responses.append(outcome.response)
                if checkpoint is not None and self._recovery_store is not None:
                    try:
                        checkpoint = self._recovery_store.append(checkpoint, outcome.response)
                    except PrivateProbeRecoveryError as error:
                        return phase_results, str(error)
            try:
                profile = ProbeProfileBuilder.build(
                    self.probe_set,
                    responses,
                    agent_id=context.agent_id,
                    round_id=context.round_id,
                    is_holdout=self.execute_holdout_probe,
                )
            except (TypeError, ValueError):
                return phase_results, "profile_aggregation_failed"
            phase_results.append(
                ProbeResult(
                    probe_set_id=self.probe_set.probe_set_id,
                    agent_id=context.agent_id,
                    round_id=context.round_id,
                    responses=responses,
                    value_profile=profile,
                    completed=True,
                    is_holdout=self.execute_holdout_probe,
                )
            )
        return phase_results, None

    def _write_phase_results(self, pre_results: list[ProbeResult], post_results: list[ProbeResult]) -> tuple[Path, Path]:
        probes_directory = self.output_directory / "probes"
        probes_directory.mkdir(exist_ok=False)
        pre_path = probes_directory / "pre_probe_results.jsonl"
        post_path = probes_directory / "post_probe_results.jsonl"
        self._write_jsonl(pre_path, pre_results)
        self._write_jsonl(post_path, post_results)
        return pre_path, post_path

    @staticmethod
    def _write_jsonl(path: Path, results: list[ProbeResult]) -> None:
        with path.open("x", encoding="utf-8") as handle:
            for result in results:
                handle.write(result.model_dump_json() + "\n")

    def _measurement_record(
        self,
        *,
        agent_run_status: RunStatus,
        pre_status: MeasurementPhaseStatus,
        post_status: MeasurementPhaseStatus,
        pre_count: int,
        post_count: int,
        agent_usage: TokenUsageSummary,
        error_code: str | None,
        pre_path: Path | None = None,
        post_path: Path | None = None,
    ) -> MeasurementRecord:
        probe_settings: ContextualProbeRequestSettings = self.contextual_probe_runtime._request_settings
        return MeasurementRecord(
            run_id=self.config.run_id,
            scenario_id=self.scenario.scenario_id,
            protocol=self.config.protocol,
            model_name=self.config.model_name,
            probe_model_name=probe_settings.model_name,
            probe_set_id=self.probe_set.probe_set_id,
            agent_ids=[agent.agent_id for agent in self.scenario.agents],
            probe_item_ids=[item.probe_id for item in self.selected_items],
            probe_seed=self.probe_seed,
            is_holdout=self.execute_holdout_probe,
            pre_target_round=self.pre_target_round,
            post_target_round=self.post_target_round,
            agent_run_status=agent_run_status,
            pre_probe_status=pre_status,
            post_probe_status=post_status,
            pre_probe_result_count=pre_count,
            post_probe_result_count=post_count,
            pre_probe_result_path=None if pre_path is None else str(pre_path.relative_to(self.output_directory)),
            post_probe_result_path=None if post_path is None else str(post_path.relative_to(self.output_directory)),
            pre_probe_sha256=None if pre_path is None else _sha256(pre_path),
            post_probe_sha256=None if post_path is None else _sha256(post_path),
            agent_token_usage=agent_usage,
            probe_token_usage=_usage_from_audits(self._probe_audits),
            error_code=error_code,
        )

    def _write_measurement_record(self, record: MeasurementRecord, *, replace: bool = False) -> None:
        output_path = self.output_directory / "measurement_record.json"
        payload = json.dumps(record.model_dump(mode="json"), ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        if replace:
            temporary = output_path.with_suffix(".tmp")
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, output_path)
            return
        with output_path.open("x", encoding="utf-8") as handle:
            handle.write(payload)

    def _read_record_if_present(self) -> RunRecord | None:
        path = self.output_directory / "run_record.json"
        if not path.is_file():
            return None
        try:
            return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None


def _usage_from_mapping(values: dict[str, int | None]) -> TokenUsageSummary:
    return TokenUsageSummary(**values)


def _usage_from_audits(audits: list[object]) -> TokenUsageSummary:
    values = [
        (audit.prompt_tokens, audit.completion_tokens, audit.total_tokens)
        for audit in audits
    ]
    if not values or any(None in item for item in values):
        return TokenUsageSummary()
    return TokenUsageSummary(
        prompt_tokens=sum(item[0] for item in values if item[0] is not None),
        completion_tokens=sum(item[1] for item in values if item[1] is not None),
        total_tokens=sum(item[2] for item in values if item[2] is not None),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "MeasuredProtocolExperimentResult",
    "MeasuredProtocolExperimentRunner",
    "MeasurementExperimentStatus",
    "MeasurementPhaseStatus",
    "MeasurementRecord",
    "TokenUsageSummary",
]
