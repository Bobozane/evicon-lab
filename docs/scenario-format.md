# Scenario Format

Stage 5 protocol runs load all participants and evidence from a strict `ScenarioSpec` file. TOML and JSON are supported. Unknown fields are rejected rather than ignored.

The checked-in example is `configs/scenarios/smoke_scenario.toml`.

```toml
scenario_id = "local-smoke-scenario"
title = "Scenario-backed local smoke"
description = "A deterministic local scenario."
initial_context = "Synthetic public context."
max_rounds = 2

[[agents]]
agent_id = "agent-1"
role = "participant"
initial_value_labels = ["fairness"]

[[evidence_cards]]
evidence_id = "evidence-public"
claim = "A local fixture claim."
source = "evicon-local-fixture"
supports = ["claim-1"]
contradicts = []
introduced_round = 0
visible_to = ["*"]
reliability = 1.0
```

`agents` contains `AgentSpec` objects. Agent IDs must be unique. `evidence_cards` contains `EvidenceCard` objects. Evidence IDs must be unique, `supports` and `contradicts` are lists of stable claim identifiers, and every non-public `visible_to` agent ID must occur in `agents`.

`introduced_round` is zero-based. A card can be exposed only at its introduction round or later, and only to an agent named by `visible_to` or to all agents through `"*"`. The scenario does not include a fixed value taxonomy or value-probe answers.

The configured `scenario_id`, `agent_count`, and `max_rounds` must match the loaded scenario exactly. The runner rejects mismatches to avoid two competing sources of truth.

Run the local example without an API key:

```bash
uv run python -m evicon.run_smoke --config configs/smoke.toml
```

This stage does not execute `ValueProbe`, compute metrics, implement `ConformityMonitor`, or implement any mediator policy.
