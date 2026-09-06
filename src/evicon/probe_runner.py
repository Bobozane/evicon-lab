"""Deterministic, isolated execution of value probes with FakeLLM only."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from .fake_llm import FakeLLM, FakeProbeRequest, FakeProbeResponse, ProbeProvider
from .models import ProbeResult, ProbeRunConfig, ProbeSet, ValueProbeItem, ValueProbeResponse
from .probe_profiles import ProbeProfileBuilder


class ProbeResponseParseError(ValueError):
    """Raised when a probe provider response does not satisfy the item contract."""


class ProbeRunner:
    """Execute probes without accepting DialogueState, evidence, or peer history."""

    def __init__(
        self,
        probe_set: ProbeSet,
        config: ProbeRunConfig,
        *,
        provider: ProbeProvider | None = None,
    ) -> None:
        if config.probe_set_id != probe_set.probe_set_id:
            raise ValueError("config.probe_set_id must match probe_set.probe_set_id")
        self.probe_set = probe_set
        self.config = config
        self.provider = provider or FakeLLM()

    def run(self) -> list[ProbeResult]:
        """Run one private item set per agent and return offline-only results."""
        selected_items = self.probe_set.items_for_holdout(self.config.is_holdout)
        if not selected_items:
            raise ValueError("probe set has no items for the requested is_holdout selection")
        results: list[ProbeResult] = []
        for agent_id in self.config.agent_ids:
            responses = [self._complete_item(agent_id, item) for item in selected_items]
            profile = ProbeProfileBuilder.build(
                self.probe_set,
                responses,
                agent_id=agent_id,
                round_id=self.config.round_id,
                is_holdout=self.config.is_holdout,
            )
            results.append(
                ProbeResult(
                    probe_set_id=self.probe_set.probe_set_id,
                    agent_id=agent_id,
                    round_id=self.config.round_id,
                    responses=responses,
                    value_profile=profile,
                    completed=True,
                    is_holdout=self.config.is_holdout,
                )
            )
        return results

    def _complete_item(self, agent_id: str, item: ValueProbeItem) -> ValueProbeResponse:
        request = FakeProbeRequest(
            agent_id=agent_id,
            round_id=self.config.round_id,
            probe=item,
            seed=self.config.seed,
        )
        raw_response = self.provider.complete_probe(request)
        return parse_probe_response(raw_response, request)


def parse_probe_response(
    raw_response: FakeProbeResponse | Mapping[str, Any],
    request: FakeProbeRequest,
) -> ValueProbeResponse:
    """Validate a structured provider response against one exact probe request."""
    try:
        payload = (
            raw_response.model_dump(mode="json")
            if isinstance(raw_response, FakeProbeResponse)
            else dict(raw_response)
        )
        parsed = FakeProbeResponse.model_validate(payload)
    except (TypeError, ValidationError) as exc:
        raise ProbeResponseParseError("probe response must be a valid structured object") from exc

    if (
        parsed.agent_id != request.agent_id
        or parsed.probe_id != request.probe.probe_id
        or parsed.round_id != request.round_id
    ):
        raise ProbeResponseParseError("probe response identifiers do not match its request")
    try:
        expected_score = request.probe.normalized_score_for(parsed.raw_response)
    except ValueError as exc:
        raise ProbeResponseParseError(str(exc)) from exc
    if not math.isclose(parsed.normalized_score, expected_score, abs_tol=1e-12):
        raise ProbeResponseParseError("probe response normalized_score does not match its response_scale")
    return ValueProbeResponse(
        agent_id=parsed.agent_id,
        probe_id=parsed.probe_id,
        round_id=parsed.round_id,
        raw_response=parsed.raw_response,
        normalized_score=parsed.normalized_score,
    )
