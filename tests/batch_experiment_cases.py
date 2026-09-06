"""Local fixtures for deterministic Stage 20 batch orchestration tests."""

from __future__ import annotations

import json
from pathlib import Path

from evicon.agent_runtime import AgentRequestSettings, AgentRuntime
from evicon.batch_experiment import BatchMeasuredExperimentRunner, BatchRuntimeParameters, build_four_condition_manifest
from evicon.contextual_probe_runtime import ContextualProbeRequestSettings, ContextualProbeRuntime
from evicon.llm_contract import LLMProviderError, LLMRequest, LLMResponse
from evicon.models import ProbeSet, ProtocolCondition

from measured_experiment_cases import AGENT_MESSAGE, RAW_ANSWER, probe_set, scenario


class BatchLocalProvider:
    """A no-network provider double used for either batch request stream."""

    def __init__(
        self,
        *,
        kind: str,
        error_at: int | None = None,
        error: LLMProviderError | None = None,
    ) -> None:
        self.kind = kind
        self.error_at = error_at
        self.error = error
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.error_at == len(self.requests):
            assert self.error is not None
            raise self.error
        content = (
            json.dumps({"message": AGENT_MESSAGE, "evidence_ids_used": []})
            if self.kind == "agent"
            else json.dumps({"choice": RAW_ANSWER})
        )
        return LLMResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            content=content,
            finish_reason="stop",
            prompt_tokens=7,
            completion_tokens=5,
            total_tokens=12,
            latency_ms=1.0,
            provider_metadata={"provider_name": f"local_batch_{self.kind}"},
        )


def batch_setup(tmp_path: Path, *, batch_id: str = "batch-test", seed: int = 0):
    active_scenario = scenario(ProtocolCondition.EVIDENCE_SOCIAL)
    active_probe_set: ProbeSet = probe_set()
    manifest = build_four_condition_manifest(
        batch_id=batch_id,
        scenario=active_scenario,
        probe_set=active_probe_set,
        model_name="batch-agent-model",
        probe_model_name="batch-probe-model",
        seeds=[seed],
        output_dir=str(tmp_path / "results"),
        batch_output_dir=str(tmp_path / "results" / "batches"),
        runtime_parameters=BatchRuntimeParameters(
            agent_temperature=0.2,
            agent_max_tokens=256,
            probe_temperature=0.0,
            probe_max_tokens=128,
        ),
        is_holdout=True,
        pre_target_round=0,
        post_target_round=1,
        metadata={"purpose": "batch test"},
    )
    agent_runtime = {
        seed: AgentRuntime(
            AgentRequestSettings(model_name="batch-agent-model", temperature=0.2, max_tokens=256, seed=seed)
        )
    }
    probe_runtime = {
        seed: ContextualProbeRuntime(
            ContextualProbeRequestSettings(model_name="batch-probe-model", temperature=0.0, max_tokens=128, seed=seed)
        )
    }
    return manifest, active_scenario, active_probe_set, agent_runtime, probe_runtime


def batch_runner(
    tmp_path: Path,
    *,
    batch_id: str = "batch-test",
    agent_provider: BatchLocalProvider | None = None,
    probe_provider: BatchLocalProvider | None = None,
) -> tuple[BatchMeasuredExperimentRunner, BatchLocalProvider, BatchLocalProvider]:
    manifest, active_scenario, active_probe_set, agent_runtime, probe_runtime = batch_setup(tmp_path, batch_id=batch_id)
    active_agent_provider = agent_provider or BatchLocalProvider(kind="agent")
    active_probe_provider = probe_provider or BatchLocalProvider(kind="probe")
    return (
        BatchMeasuredExperimentRunner(
            manifest,
            scenario=active_scenario,
            probe_set=active_probe_set,
            agent_runtime=agent_runtime,
            contextual_probe_runtime=probe_runtime,
            agent_provider=active_agent_provider,
            probe_provider=active_probe_provider,
        ),
        active_agent_provider,
        active_probe_provider,
    )
