"""Public read-only input boundary for offline evaluation."""

from __future__ import annotations

from pathlib import Path

from . import loaders_core as _core
from .models import EvaluationManifest, EvaluationRunEntry

EvaluationInputError = _core.EvaluationInputError
EvaluationManifestLoader = _core.EvaluationManifestLoader
LoadedEvaluationRun = _core.LoadedEvaluationRun
LoadedProbeResults = _core.LoadedProbeResults
LoadedRunRecord = _core.LoadedRunRecord
ProbeResultLoader = _core.ProbeResultLoader
RunRecordLoader = _core.RunRecordLoader


def load_declared_run(
    entry: EvaluationRunEntry,
    manifest: EvaluationManifest,
    *,
    manifest_directory: str | Path,
) -> LoadedEvaluationRun:
    """Load one declared run and reject event metadata inconsistent with its record."""
    loaded = _core.load_declared_run(
        entry,
        manifest,
        manifest_directory=manifest_directory,
    )
    scenario = loaded.record.scenario
    assert scenario is not None
    started_payload = loaded.events[0]["payload"]
    if started_payload.get("scenario_id") != loaded.record.config.scenario_id:
        raise EvaluationInputError(
            f"events {loaded.input_audit[1].path}: run_started scenario_id does not match run record"
        )

    agent_ids = {agent.agent_id for agent in scenario.agents}
    for index, event in enumerate(loaded.events, start=1):
        round_id = event["round_id"]
        if round_id >= loaded.record.config.max_rounds:
            raise EvaluationInputError(
                f"events {loaded.input_audit[1].path}:{index}: round_id exceeds configured max_rounds"
            )
        payload = event["payload"]
        candidate = payload.get("agent_id", payload.get("speaker_id"))
        if candidate is not None and (not isinstance(candidate, str) or candidate not in agent_ids):
            raise EvaluationInputError(
                f"events {loaded.input_audit[1].path}:{index}: event names an unknown agent_id"
            )
    return loaded


__all__ = [
    "EvaluationInputError",
    "EvaluationManifestLoader",
    "LoadedEvaluationRun",
    "LoadedProbeResults",
    "LoadedRunRecord",
    "ProbeResultLoader",
    "RunRecordLoader",
    "load_declared_run",
]
