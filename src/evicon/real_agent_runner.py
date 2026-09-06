"""Opt-in protocol runner for parsed real Agent turns with no mediator integration."""

from __future__ import annotations

import json
from pathlib import Path

from .agent_prompts import AGENT_TURN_TEMPLATE_VERSION, AgentPromptContext
from .agent_runtime import AgentRuntime, AgentRuntimeStatus
from .events import EventType, JsonlEventLog
from .exposure import ExposurePlan, ExposureSnapshot
from .llm_contract import LLMProvider
from .models import (
    DialogueState,
    DialogueTurn,
    EvidenceCard,
    EvidenceExposure,
    ProtocolCondition,
    RunConfig,
    RunRecord,
    RunStatus,
    ScenarioSpec,
)


class RealAgentRunError(RuntimeError):
    """Stable terminal error for a failed Agent runtime invocation."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(f"real Agent run failed: {error_code}")


class RealAgentProtocolRunner:
    """Run the four base protocols through an injected AgentRuntime and provider."""

    def __init__(
        self,
        config: RunConfig,
        *,
        scenario: ScenarioSpec,
        agent_runtime: AgentRuntime,
        provider: LLMProvider,
    ) -> None:
        self.config = config
        self.scenario = scenario
        self.agent_runtime = agent_runtime
        self.provider = provider
        self._validate_config_against_scenario()
        self._validate_runtime_model()
        self.last_state: DialogueState | None = None
        self._usage_values: list[tuple[int | None, int | None, int | None]] = []

    def _validate_config_against_scenario(self) -> None:
        if self.config.scenario_id != self.scenario.scenario_id:
            raise ValueError("config.scenario_id must match scenario.scenario_id")
        if self.config.agent_count != len(self.scenario.agents):
            raise ValueError("config.agent_count must match the scenario agent count")
        if self.config.max_rounds != self.scenario.max_rounds:
            raise ValueError("config.max_rounds must match scenario.max_rounds")

    def _validate_runtime_model(self) -> None:
        runtime_model_name = self.agent_runtime._request_settings.model_name
        if self.config.model_name != runtime_model_name:
            raise ValueError("config.model_name must match AgentRuntime request model_name")

    @property
    def output_directory(self) -> Path:
        return Path(self.config.output_dir) / self.config.run_id

    @property
    def total_token_usage(self) -> dict[str, int | None]:
        """Return known aggregate usage, or explicit unknowns if any response omitted usage."""
        if not self._usage_values or any(None in values for values in self._usage_values):
            return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        prompt_tokens = sum(values[0] for values in self._usage_values if values[0] is not None)
        completion_tokens = sum(values[1] for values in self._usage_values if values[1] is not None)
        total_tokens = sum(values[2] for values in self._usage_values if values[2] is not None)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def run(self) -> RunRecord:
        """Execute public snapshots; errors write a failed record and terminate the run."""
        output_directory = self._prepare_output_directory()
        logger = JsonlEventLog(output_directory / "events.jsonl", self.config.run_id)
        turns: list[DialogueTurn] = []
        evidence_exposures: list[EvidenceExposure] = []
        state: DialogueState | None = None

        logger.append(
            EventType.RUN_STARTED,
            round_id=0,
            payload={
                "scenario_id": self.config.scenario_id,
                "scenario_title": self.scenario.title,
                "protocol": self.config.protocol.value,
                "model_name": self.config.model_name,
                "agent_count": self.config.agent_count,
                "max_rounds": self.config.max_rounds,
                "seed": self.config.seed,
                "agent_runtime": "real_agent_runtime.v1",
            },
        )
        try:
            for round_id in range(self.scenario.max_rounds):
                state = self._build_state(round_id, turns)
                logger.append(EventType.ROUND_STARTED, round_id=round_id, payload={})

                # All snapshots are created from the same pre-round state.
                snapshots = ExposurePlan(
                    self.config.protocol,
                    scenario_context=self.scenario.initial_context,
                ).create(state, round_id=round_id)
                for agent in state.agents:
                    snapshot = snapshots[agent.agent_id]
                    evidence_exposures.extend(self._record_exposures(snapshot))
                    logger.append(
                        EventType.EXPOSURE_CREATED,
                        round_id=round_id,
                        payload=self._exposure_payload(snapshot),
                    )
                    public_context = self._prompt_context(agent_id=agent.agent_id, snapshot=snapshot, state=state)
                    result = self.agent_runtime.execute(public_context, self.provider)
                    self._append_runtime_events(logger, snapshot, result)
                    if result.status is not AgentRuntimeStatus.COMPLETED or result.response is None:
                        raise RealAgentRunError(result.audit.error_code or result.status.value)

                    turn = self._build_turn(snapshot, result.response.message)
                    turns.append(turn)
                    logger.append(
                        EventType.TURN_COMPLETED,
                        round_id=round_id,
                        payload={
                            "turn_id": turn.turn_id,
                            "speaker_id": turn.speaker_id,
                            "visible_peer_turn_ids": turn.visible_peer_turn_ids,
                            "visible_evidence_ids": turn.visible_evidence_ids,
                        },
                    )

                state = self._build_state(round_id, turns)
                logger.append(
                    EventType.ROUND_COMPLETED,
                    round_id=round_id,
                    payload={"turn_count": len(turns)},
                )

            self.last_state = state
            record = self._record(turns, evidence_exposures, RunStatus.COMPLETED, error_message=None)
            logger.append(
                EventType.RUN_COMPLETED,
                round_id=self.scenario.max_rounds - 1,
                payload={"turn_count": len(turns)},
            )
            self._write_record(output_directory, record)
            return record
        except Exception as error:
            failed_round = state.current_round if state is not None else 0
            error_code = error.error_code if isinstance(error, RealAgentRunError) else "runner_failure"
            logger.append(
                EventType.RUN_FAILED,
                round_id=failed_round,
                payload={"error_code": error_code},
            )
            failed_record = self._record(
                turns,
                evidence_exposures,
                RunStatus.FAILED,
                error_message=f"run failed: {error_code}",
            )
            self._write_record(output_directory, failed_record)
            raise

    def _build_state(self, round_id: int, turns: list[DialogueTurn]) -> DialogueState:
        available_cards = [
            card for card in self.scenario.evidence_cards if card.is_available_at(round_id)
        ]
        return DialogueState(
            run_id=self.config.run_id,
            scenario_id=self.scenario.scenario_id,
            current_round=round_id,
            agents=self.scenario.agents,
            turns=turns,
            evidence_cards=available_cards,
            value_profiles=[],
            intervention_budget=self.config.intervention_budget,
            metadata={
                "protocol": self.config.protocol.value,
                "scenario_title": self.scenario.title,
                "seed": self.config.seed,
            },
        )

    def _prompt_context(
        self,
        *,
        agent_id: str,
        snapshot: ExposureSnapshot,
        state: DialogueState,
    ) -> AgentPromptContext:
        agent = next(item for item in state.agents if item.agent_id == agent_id)
        cards_by_id = {card.evidence_id: card for card in state.evidence_cards}
        cards = [cards_by_id[evidence_id] for evidence_id in snapshot.visible_evidence_ids]
        return AgentPromptContext(
            agent_id=agent.agent_id,
            role=agent.role,
            initial_value_labels=list(agent.initial_value_labels),
            round_id=snapshot.round_id,
            protocol=snapshot.protocol,
            scenario_context=snapshot.scenario_context,
            exposure_snapshot=snapshot,
            visible_evidence_cards=cards,
        )

    def _append_runtime_events(
        self,
        logger: JsonlEventLog,
        snapshot: ExposureSnapshot,
        result: object,
    ) -> None:
        audit = result.audit_summary()
        visibility = self._exposure_payload(snapshot)
        logger.append(
            EventType.LLM_REQUEST,
            round_id=snapshot.round_id,
            payload={
                **visibility,
                "request_id": audit.request_id,
                "template_version": audit.template_version or AGENT_TURN_TEMPLATE_VERSION,
                "model_name": audit.model_name,
                "seed": self.config.seed,
            },
        )
        response_payload: dict[str, object] = {
            "agent_id": snapshot.agent_id,
            "status": audit.status.value,
            "request_id": audit.request_id,
            "template_version": audit.template_version or AGENT_TURN_TEMPLATE_VERSION,
            "model_name": audit.model_name,
            "finish_reason": audit.finish_reason,
            "prompt_tokens": audit.prompt_tokens,
            "completion_tokens": audit.completion_tokens,
            "total_tokens": audit.total_tokens,
            "latency_ms": audit.latency_ms,
            "parser_valid": audit.parser_valid,
            "error_code": audit.error_code,
            "evidence_ids_used": audit.evidence_ids_used,
            "visible_peer_turn_ids": snapshot.visible_peer_turn_ids,
            "visible_evidence_ids": snapshot.visible_evidence_ids,
        }
        if result.status is AgentRuntimeStatus.COMPLETED and result.response is not None:
            response_payload["message"] = result.response.message
            self._usage_values.append(
                (audit.prompt_tokens, audit.completion_tokens, audit.total_tokens)
            )
        logger.append(EventType.LLM_RESPONSE, round_id=snapshot.round_id, payload=response_payload)

    def _record_exposures(self, snapshot: ExposureSnapshot) -> list[EvidenceExposure]:
        return [
            EvidenceExposure(
                evidence_id=evidence_id,
                round_id=snapshot.round_id,
                exposed_to=[snapshot.agent_id],
                exposure_reason=f"{snapshot.protocol.value} visibility",
            )
            for evidence_id in snapshot.visible_evidence_ids
        ]

    @staticmethod
    def _exposure_payload(snapshot: ExposureSnapshot) -> dict[str, object]:
        return {
            "agent_id": snapshot.agent_id,
            "protocol": snapshot.protocol.value,
            "scenario_context": snapshot.scenario_context,
            "visible_history_ids": [turn.turn_id for turn in snapshot.visible_history],
            "visible_peer_turn_ids": snapshot.visible_peer_turn_ids,
            "visible_evidence_ids": snapshot.visible_evidence_ids,
        }

    def _build_turn(self, snapshot: ExposureSnapshot, message: str) -> DialogueTurn:
        if self.config.protocol in {
            ProtocolCondition.SOCIAL_ONLY,
            ProtocolCondition.EVIDENCE_SOCIAL,
        }:
            visible_to = ["*"]
        else:
            visible_to = [snapshot.agent_id]
        return DialogueTurn(
            turn_id=f"turn-r{snapshot.round_id}-{snapshot.agent_id}",
            round_id=snapshot.round_id,
            speaker_id=snapshot.agent_id,
            message=message,
            visible_to=visible_to,
            visible_peer_turn_ids=snapshot.visible_peer_turn_ids,
            visible_evidence_ids=snapshot.visible_evidence_ids,
            protocol=self.config.protocol,
        )

    def _record(
        self,
        turns: list[DialogueTurn],
        evidence_exposures: list[EvidenceExposure],
        status: RunStatus,
        *,
        error_message: str | None,
    ) -> RunRecord:
        return RunRecord(
            config=self.config,
            scenario=self.scenario,
            turns=turns,
            value_profiles=[],
            evidence_exposures=evidence_exposures,
            intervention_decisions=[],
            status=status,
            error_message=error_message,
        )

    def _prepare_output_directory(self) -> Path:
        output_directory = self.output_directory
        if output_directory.exists():
            raise FileExistsError(f"refusing to overwrite existing run directory: {output_directory}")
        output_directory.mkdir(parents=True, exist_ok=False)
        return output_directory

    @staticmethod
    def _write_record(output_directory: Path, record: RunRecord) -> None:
        output_path = output_directory / "run_record.json"
        with output_path.open("x", encoding="utf-8") as handle:
            json.dump(record.model_dump(mode="json"), handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")


__all__ = ["RealAgentProtocolRunner", "RealAgentRunError"]
