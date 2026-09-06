# EviCon-Lab Implementation Plan

## Phase 1: Project boundary and reproducibility contract (complete)

- Create an isolated Python package with `src/evicon`, `tests`, `configs`,
  `scripts`, `results`, and `docs`.
- Configure Python packaging and pytest.
- Document research scope, upstream boundary, future implementation sequence,
  and output provenance requirements.
- Add a smoke test that exercises no model or agent behavior.

Acceptance: `pytest` succeeds; no agent, model provider, API client, metric,
or experiment workflow exists.

## Phase 2: Minimal typed experiment vocabulary (not implemented)

Implement Pydantic models with validation and serialization tests only. The
minimum data structures are:

| Structure | Minimum responsibility |
| --- | --- |
| `ValueProfile` | Agent identifier, named value dimensions, and profile provenance. |
| `ValueProbe` | Probe identifier, prompt template, value dimensions measured, and held-out flag. |
| `ExposureItem` | Ordered social or evidence exposure with source and visibility metadata. |
| `EvidenceCard` | Claim, source, reliability metadata, timestamp, and evidence identifier. |
| `EvidenceTracker` | Immutable association from an exposure or response to visible evidence identifiers. |
| `AgentResponse` | Agent identifier, stage, rendered prompt hash, structured answer, and referenced evidence IDs. |
| `InteractionEvent` | Run, round, order, sender, recipients, exposure IDs, and timestamps. |
| `RunConfig` | Condition, model settings, seed, prompt-set version, and cache policy. |
| `RunMetadata` | Immutable execution provenance for JSONL records. |

These models deliberately do not decide intervention policy or call an LLM.
Their field names, schemas, and persistence layout remain subject to review
before implementation.

Acceptance: model validation tests, JSON serialization round trips, no network
access, and a `FakeLLM` protocol contract without a real provider.

## Phase 3: Controlled protocol and observability (not implemented)

Add configuration loading, prompt rendering, a deterministic `FakeLLM`, cache
interfaces, JSONL event logging, and condition assignment. Keep social and
evidence channels separable, make exposure order explicit, and use hidden
probes only after exposure.

Acceptance: deterministic fake runs produce auditable JSONL with complete
metadata and no API credentials.

## Phase 4: Measurement layer (not implemented)

Implement pairwise diversity, structural diversity, representative coverage,
and evidence-attributed update measurement. Define the social-influence excess
loss estimator only after its counterfactual comparison and uncertainty method
are reviewed.

Acceptance: hand-constructed fixtures validate metrics and edge cases; metric
CSV is derived from JSONL rather than hard-coded.

## Phase 5: Monitor and selective mediation (not implemented)

Implement an interpretable `ConformityMonitor`, target selection, and a
policy-controlled `EvidenceAwareMediator`. Supported actions will be `no-op`,
`request-evidence`, `blind-evidence-reflection`, `solicit-dissent`,
`adaptive-exposure`, `minority-report`, and `restructure`.

Acceptance: unit tests prove policy selection is deterministic from recorded
signals and that the LLM cannot choose the target or action category.

## Phase 6: Runner, reporting, and statistical analysis (not implemented)

Implement a unified runner for all declared conditions, result CSV generation,
plots, confidence intervals, and predeclared comparisons of quality, safety,
and cost trade-offs.

Acceptance: seeded end-to-end fake experiments generate structured outputs and
analysis uses only recorded results.

## Phase 7: Real-model study execution (not implemented)

After review of prompts, budget, credentials, safety handling, licenses, and
sample-size plan, add real providers behind the tested interface and execute
the study. No performance claim is made before results are generated and
reviewed.

