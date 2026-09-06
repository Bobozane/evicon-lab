"""Shared deterministic fixtures for contract and future protocol tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from evicon.models import (
    AgentSpec,
    DialogueState,
    DialogueTurn,
    EvidenceCard,
    ProtocolCondition,
    RunConfig,
    ValueProbeItem,
    ValueProfile,
)


_HG232_LEGACY_TEST_MODULES = {
    "test_provenance_cascade_hg232",
    "test_provenance_cascade_hg232_calibration",
    "test_provenance_cascade_hg232_compatibility",
    "test_provenance_cascade_hg232_stability_probe",
}
_HG232_FORMAL_OUTPUT = "results/provenance-cascade-hg232-adoption-identifiability-v1"


@pytest.fixture(autouse=True)
def isolate_legacy_hg232_output_root(request, monkeypatch, tmp_path: Path) -> None:
    """Keep pre-existing formal HG232 results outside legacy unit-test writes."""
    if request.module.__name__ not in _HG232_LEGACY_TEST_MODULES:
        return

    from evicon import provenance_cascade_hg232 as contract
    from evicon import provenance_cascade_hg232_calibration as calibration

    isolated_root = tmp_path / "formal-output-root"

    def isolated(resolver):
        def resolve(value):
            if str(value) == _HG232_FORMAL_OUTPUT:
                return isolated_root
            return resolver(value)

        return resolve

    monkeypatch.setattr(contract, "_path", isolated(contract._path))
    monkeypatch.setattr(calibration, "_path", isolated(calibration._path))


@pytest.fixture
def minimal_agents() -> list[AgentSpec]:
    return [
        AgentSpec(agent_id="agent-a", role="participant", initial_value_labels=["fairness"]),
        AgentSpec(agent_id="agent-b", role="participant", initial_value_labels=["autonomy"]),
    ]


@pytest.fixture
def minimal_evidence_card() -> EvidenceCard:
    return EvidenceCard(
        evidence_id="evidence-1",
        claim="The source reports a verified factual update.",
        source="local-fixture",
        supports=["claim-1"],
        contradicts=[],
        introduced_round=0,
        visible_to=["agent-a"],
        reliability=0.9,
    )


@pytest.fixture
def minimal_dialogue_state(
    minimal_agents: list[AgentSpec], minimal_evidence_card: EvidenceCard
) -> DialogueState:
    return DialogueState(
        run_id="run-fixture",
        scenario_id="scenario-fixture",
        current_round=0,
        agents=minimal_agents,
        turns=[
            DialogueTurn(
                turn_id="turn-1",
                round_id=0,
                speaker_id="agent-a",
                message="I will consider the available evidence.",
                visible_to=["agent-a"],
                visible_peer_turn_ids=[],
                visible_evidence_ids=["evidence-1"],
                protocol=ProtocolCondition.EVIDENCE_ONLY,
            )
        ],
        evidence_cards=[minimal_evidence_card],
        value_profiles=[
            ValueProfile(
                agent_id="agent-a",
                round_id=0,
                dimensions=["fairness"],
                scores=[0.75],
                source="fixture",
                probe_id="probe-1",
            )
        ],
        intervention_budget=1.0,
        metadata={"fixture": True},
    )


@pytest.fixture
def minimal_protocol_fixture(minimal_dialogue_state: DialogueState) -> dict[str, object]:
    """A no-network fixture for future FakeLLM and protocol tests."""
    return {
        "run_config": RunConfig(
            run_id="run-fixture",
            scenario_id="scenario-fixture",
            model_name="fake-model",
            protocol=ProtocolCondition.EVIDENCE_ONLY,
            agent_count=2,
            max_rounds=2,
            seed=7,
            intervention_budget=1.0,
            output_dir="results/fixture",
        ),
        "state": minimal_dialogue_state,
        "probe": ValueProbeItem(
            probe_id="probe-1",
            text="How important is fairness in this decision?",
            dimension="fairness",
            response_scale=["low", "medium", "high"],
            is_holdout=True,
        ),
    }
