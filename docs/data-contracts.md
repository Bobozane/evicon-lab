# EviCon-Lab Data Contracts

## Scope

This document defines the stable, serializable contracts implemented in
`src/evicon/models`. They use Pydantic v2, reject undeclared fields, perform
no network access, and contain no metric, monitoring, mediator-policy, or
experiment-runner behavior.

All identifiers are non-blank strings. Ordered lists remain ordered in JSON.
The audience value `"*"` is reserved for public visibility and must be the
only member of an audience list. Normalized scores, reliability, and risk
scores are finite floats in the closed interval `[0, 1]`.

## Agent and value contracts

| Model | Fields | Purpose and invariants |
| --- | --- | --- |
| `AgentSpec` | `agent_id`, `role`, `initial_value_labels`, `metadata` | Static identity and initial labels for one run. Labels are unique, ordered strings; metadata is JSON-compatible. |
| `ValueProfile` | `agent_id`, `round_id`, `dimensions`, `scores`, `source`, `probe_id` | A value snapshot. Dimension and score order is stable, dimensions are unique, lengths must match, and scores are normalized. `dimension_scores` is a derived mapping only. |
| `ScenarioSpec` | `scenario_id`, `title`, `description`, `initial_context`, `agents`, `evidence_cards`, `max_rounds`, `metadata` | Strict scenario input for deterministic protocol runs. Agent and evidence IDs are unique; evidence audiences reference declared agents or `"*"`. |

No fixed taxonomy, including Schwartz dimensions, is encoded. `source` and
optional `probe_id` identify how a profile was obtained without deciding
whether it is socially or evidence driven.

## Probe contracts

| Model | Fields | Purpose and invariants |
| --- | --- | --- |
| `ValueProbeItem` | `probe_id`, `text`, `dimension`, `response_scale`, `is_holdout` | A probe definition. The ordered scale has at least two unique labels. |
| `ValueProbeResponse` | `agent_id`, `probe_id`, `round_id`, `raw_response`, `normalized_score` | A private answer record. It carries raw text and a separately normalized score. |
| `ProbeSet` | `probe_set_id`, `dimensions`, `items`, `version`, `metadata` | A strict, versioned offline probe definition. IDs are unique and dimensions are fully covered. |
| `ProbeRunConfig` | `probe_set_id`, `round_id`, `agent_ids`, `model_name`, `seed`, `is_holdout` | One isolated standard or holdout probe pass. |
| `ProbeResult` | `probe_set_id`, `agent_id`, `round_id`, `responses`, `value_profile`, `completed`, `is_holdout` | Offline-only result record stored outside dialogue and ordinary event logs. |

`DialogueState` intentionally has no `ValueProbeResponse` field. A held-out
answer is never automatically made public just because a dialogue state is
serialized or restored.

## Evidence contracts

| Model | Fields | Purpose and invariants |
| --- | --- | --- |
| `EvidenceCard` | `evidence_id`, `claim`, `source`, `supports`, `contradicts`, `introduced_round`, `visible_to`, `reliability` | An evidence claim with source, explicit claim links, introduction round, audience, and reliability. Support and contradiction links cannot overlap. |
| `EvidenceExposure` | `evidence_id`, `round_id`, `exposed_to`, `exposure_reason` | An event that records that a card was exposed to an audience and why. |

`EvidenceCard.is_available_at(round_id)` checks whether the card has appeared,
and `EvidenceCard.is_visible_to(agent_id)` checks card authorization. These are
provenance and access checks only; they do not judge natural-language truth.

## Dialogue contracts

| Model | Fields | Purpose and invariants |
| --- | --- | --- |
| `DialogueTurn` | `turn_id`, `round_id`, `speaker_id`, `message`, `visible_to`, `visible_peer_turn_ids`, `visible_evidence_ids`, `protocol` | One rendered utterance plus the peer-turn and evidence IDs seen by its speaker. |
| `DialogueState` | `run_id`, `scenario_id`, `current_round`, `agents`, `turns`, `evidence_cards`, `value_profiles`, `intervention_budget`, `metadata` | The complete restorable state for a run at one current round. |

`DialogueState` validates that agent, evidence, and turn IDs are unique; turn
speakers and audiences are known; profiles and records are not from a future
round; peer-turn references exist; and every cited evidence ID exists, was
introduced by that turn's round, and is visible to the speaker. It provides
`visible_turns_for(agent_id)` and `visible_evidence_for(agent_id)`, each with
an optional `through_round` bounded by `current_round`.

## Protocol and intervention contracts

| Model | Values or fields | Purpose and invariants |
| --- | --- | --- |
| `ProtocolCondition` | `independent`, `social_only`, `evidence_only`, `evidence_social` | The four controlled visibility conditions. |
| `InterventionAction` | `no_op`, `request_evidence`, `blind_evidence_reflection`, `solicit_dissent`, `adaptive_exposure`, `minority_report`, `restructure` | The closed action vocabulary for future policy code. |
| `InterventionDecision` | `action`, `target_agent_ids`, `reason`, `risk_score`, `estimated_cost`, `round_id` | An auditable decision record. `no_op` has no targets; every other action has at least one. |

These models record a future policy output only. They do not decide when to
intervene, select an agent, call an LLM, or generate a mediator message.

## Experiment contracts

| Model | Fields | Purpose and invariants |
| --- | --- | --- |
| `RunConfig` | `run_id`, `scenario_id`, `scenario_file`, `model_name`, `protocol`, `agent_count`, `max_rounds`, `seed`, `intervention_budget`, `output_dir` | Serializable runner input. Counts and rounds must be positive; the budget is non-negative. It performs no file-system action. |
| `RunRecord` | `config`, `scenario`, `turns`, `value_profiles`, `evidence_exposures`, `intervention_decisions`, `status`, `error_message` | Serializable output envelope that retains the ScenarioSpec snapshot required for replay validation. |
| `RunStatus` | `pending`, `running`, `completed`, `failed` | Closed lifecycle vocabulary for `RunRecord`. |

## Serialization and test fixture

Every contract supports Pydantic `model_dump_json()` and
`model_validate_json()`. The shared pytest fixture `minimal_protocol_fixture`
contains a two-agent state, one agent-scoped evidence card, a hold-out probe,
and a `fake-model` run configuration. It is deterministic, does not contain
an API key, and is the intended starting point for later FakeLLM and protocol
tests.

## Deliberately not implemented

- LLM clients, prompt rendering, caching, or API calls;
- diversity, representation, social-influence, quality, safety, or cost metrics;
- a `ConformityMonitor`;
- mediator policy, target selection, or action rendering;
- experiment scheduling, CSV writing, result generation, or analysis.
