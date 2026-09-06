"""Public adaptive runner that audits online control after every completed round."""

from __future__ import annotations

import json
from pathlib import Path

from . import adaptive_runner_policy as _policy
from .adaptive_control import OnlineStateProvider, TargetCandidateProvider
from .conformity_monitor import MonitorConfig
from .events import EventType, JsonlEventLog
from .fake_llm import FakeLLMRequest
from .models import DialogueState, DialogueTurn, EvidenceExposure, RunConfig, RunRecord, RunStatus, ScenarioSpec
from .policy import PolicyConfig
from .fake_llm import LocalProvider


class AdaptiveProtocolRunner(_policy.AdaptiveProtocolRunner):
    """FakeLLM controller which observes every finished round, including the last."""

    def run(self) -> RunRecord:
        output_directory = self._prepare_output_directory()
        logger = JsonlEventLog(output_directory / "events.jsonl", self.config.run_id)
        turns: list[DialogueTurn] = []
        evidence_exposures: list[EvidenceExposure] = []
        state: DialogueState | None = None
        applied_plan_ids: set[str] = set()

        logger.append(EventType.RUN_STARTED, round_id=0, payload=self._run_payload())
        try:
            for round_id in range(self.scenario.max_rounds):
                state = self._build_state(round_id, turns)
                logger.append(EventType.ROUND_STARTED, round_id=round_id, payload={})
                active_effect = self._apply_pending_plan(state, round_id, logger, applied_plan_ids)
                if self._pending_plan is None:
                    self._sync_controller_state(None)
                snapshots = self._snapshots_for_round(state, round_id, active_effect)
                turn_order = self._turn_order(state, active_effect)
                responses: list[tuple[object, str]] = []
                for agent_id in turn_order:
                    snapshot = snapshots[agent_id]
                    evidence_exposures.extend(self._record_exposures(snapshot))
                    logger.append(
                        EventType.EXPOSURE_CREATED,
                        round_id=round_id,
                        payload=self._exposure_payload(snapshot),
                    )
                    metadata = self._intervention_metadata_for(agent_id, active_effect)
                    request = FakeLLMRequest(
                        agent_id=agent_id,
                        round_id=round_id,
                        protocol=snapshot.protocol,
                        scenario_context=snapshot.scenario_context,
                        visible_history=snapshot.visible_history,
                        visible_peer_turn_ids=snapshot.visible_peer_turn_ids,
                        visible_evidence_ids=snapshot.visible_evidence_ids,
                        seed=self.config.seed,
                        intervention_metadata=metadata,
                    )
                    logger.append(
                        EventType.LLM_REQUEST,
                        round_id=round_id,
                        payload={
                            **self._exposure_payload(snapshot),
                            "seed": request.seed,
                            "intervention_metadata": (
                                metadata.model_dump(mode="json") if metadata is not None else None
                            ),
                        },
                    )
                    response = self.provider.complete(request)
                    logger.append(
                        EventType.LLM_RESPONSE,
                        round_id=round_id,
                        payload={
                            "agent_id": response.agent_id,
                            "message": response.message,
                            "request_fingerprint": response.request_fingerprint,
                            "visible_peer_turn_ids": response.visible_peer_turn_ids,
                            "visible_evidence_ids": response.visible_evidence_ids,
                        },
                    )
                    responses.append((snapshot, response.message))
                new_turns = [self._build_turn(snapshot, message) for snapshot, message in responses]
                turns.extend(new_turns)
                for turn in new_turns:
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
                logger.append(EventType.ROUND_COMPLETED, round_id=round_id, payload={"turn_count": len(turns)})
                self._control_after_round(state, round_id, logger)

            self.last_state = state
            record = self._record(turns, evidence_exposures, RunStatus.COMPLETED)
            logger.append(
                EventType.RUN_COMPLETED,
                round_id=self.scenario.max_rounds - 1,
                payload={"turn_count": len(turns)},
            )
            self._write_record(output_directory, record)
            self._write_controller_state(output_directory)
            return record
        except Exception as exc:
            if self._pending_plan is not None:
                self._release_pending(state.current_round if state is not None else 0, logger, "run_failed")
            failed_round = state.current_round if state is not None else 0
            logger.append(EventType.RUN_FAILED, round_id=failed_round, payload={"error_type": type(exc).__name__})
            failed_record = self._record(
                turns,
                evidence_exposures,
                RunStatus.FAILED,
                error_message=f"run failed: {type(exc).__name__}",
            )
            self._write_record(output_directory, failed_record)
            self._write_controller_state(output_directory)
            raise


def run_adaptive(
    run_config: RunConfig,
    scenario: ScenarioSpec,
    online_state_provider: OnlineStateProvider,
    target_candidate_provider: TargetCandidateProvider,
    monitor_config: MonitorConfig,
    policy_config: PolicyConfig,
    provider: LocalProvider | None = None,
) -> RunRecord:
    return AdaptiveProtocolRunner(
        run_config,
        scenario=scenario,
        online_state_provider=online_state_provider,
        target_candidate_provider=target_candidate_provider,
        monitor_config=monitor_config,
        policy_config=policy_config,
        provider=provider,
    ).run()


__all__ = ["AdaptiveProtocolRunner", "run_adaptive"]
