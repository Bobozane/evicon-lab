# Upstream Implementation Mapping

## Audit scope and status

This document is a read-only audit of local sibling directories under
`/home/ubuntu`. It records the code observed on 2026-08-16 and does not import,
vendor, execute, or modify any upstream source. It supersedes no license text;
the license file shipped with each upstream directory is the authority.

| Work | Local source root reviewed | Local license observation | EviCon-Lab status |
| --- | --- | --- | --- |
| MultiAgent-Diversity | `/home/ubuntu/MultiAgent-Diversity-main` | No `LICENSE` or equivalent was found within three directory levels. | Methods only; do not copy code, prompts, data, outputs, or derived code. |
| ProMediate | `/home/ubuntu/promediate-main` | `LICENSE` is MIT, copyright 2026 Humalike. | Refer to architecture and independently reimplement EviCon interfaces. Preserve notices only if any source is later reused after review. |
| Social-Conformity-in-Large-Language-Models | `/home/ubuntu/Social-Conformity-in-Large-Language-Models-main` | `LICENSE` is MIT, copyright 2026 xuexucheng. | Refer to protocol and independently reimplement EviCon interfaces. Preserve notices only if any source is later reused after review. |
| ASVO | `/home/ubuntu/ASVO-master` | `LICENSE` is MIT, copyright 2024 FangweiZhong. | Refer to state-modeling concepts and independently reimplement EviCon interfaces. Preserve notices only if any source is later reused after review. |

The MIT observation applies to the repository software license, not
automatically to third-party materials embedded in a repository. In particular,
ProMediate's README says its prompt directory contains paper-appendix prompts
and its negotiation scenarios are constrained by Harvard Program on Negotiation
materials. Dataset, prompt, model-output, and asset provenance must therefore
be checked separately before any reuse.

## 1. MultiAgent-Diversity

### Observed data flow

The executable scripts at the project root use World Values Survey artifacts:

- `data/wvs.json` supplies ordered questions with `Q_id`, question text, and
  response options.
- `data/proportions_group_by_country.json` supplies country-level option
  distributions used as the human reference.
- `sec5_infer_api.py` or `sec5_infer_torch.py` produces one JSONL response
  file per culture. `evaluate.py` expects fields including `idx`, `q_id`, and
  `response`, then sorts records by `idx` and parses a boxed numeric answer.

The data schema is tightly coupled to country-conditioned WVS multiple-choice
responses. It does not represent evidence provenance, hidden probes, agent
state, exposure order, model parameters, random seed, or cache provenance.

### Observed interaction flow

The baseline response is generated independently per culture. For round one,
`sec6_social_exposure_api.py` loads the independent JSONL response of every
other culture for the same item. It builds a prompt that retains the target
culture identity and appends up to four other-culture answers under culture
labels before requesting a boxed answer. For Llama-named models it exposes only
the parsed numeric choices; otherwise it exposes each full response text.

`sec6_multi_turn_api.py` repeats the same construction for rounds two and
above, replacing the independent files with the previous-round files. Thus the
transition is:

```text
independent responses (round 0)
  -> round-1 exposure to four other responses
  -> round-r exposure to the other agents' round-(r-1) responses
```

The script iterates cultures in a fixed dictionary order, excludes the target,
and stops after four exposures. It appends JSONL results with the rendered
system and instruction strings. It has no condition randomization, explicit
evidence channel, hidden-probe separation, or run-level seed/model-parameter
record.

### Observed metric flow

`evaluate.py` provides three relevant measurements:

- Value alignment: compares an agent's answer per question with the modal
  human answer for its declared culture. It uses a normalized Euclidean
  distance over ordered option indices and reports `100 * (1 - distance)`.
- Pairwise diversity: constructs an answer vector for each culture-agent,
  computes normalized Euclidean distance across common questions for every
  unordered pair, and averages the pair distances.
- Structural diversity: builds an undirected pairwise-distance matrix, takes a
  minimum spanning tree, and divides total MST length by `N - 1`.

Both diversity scores are answer-space dispersion measures. Neither is a
representative-coverage metric, an evidence-attribution measure, nor a causal
estimate of social influence.

### EviCon-Lab interpretation

EviCon-Lab can reproduce the *experimental idea* of independent baseline,
social exposure, and repeated exposure, while independently defining its value
probes and log schema. The normalized pairwise-distance and MST formulas are
useful candidates for independently reimplemented metrics after a review of
question encoding and missing-answer behavior. Because no license was found in
the local project, no implementation, prompt, WVS data, response file, or
derived code may be copied into EviCon-Lab.

Relevant observed files: `evaluate.py`, `sec6_social_exposure_api.py`, and
`sec6_multi_turn_api.py`.

## 2. ProMediate

### Observed simulator structure

`promediate/models.py` defines a Pydantic `Thought` and `Turn`, then a
`Conversation` containing scenario, mode, difficulty, model metadata, and
ordered turns. `promediate/simulator/conversation.py` creates a participant
per scenario party and, for each turn, executes this priority order:

1. Ask the configured mediator whether it should engage. A mediator utterance
   consumes the turn when it engages.
2. Otherwise ask every participant to produce candidate thoughts, select the
   participant with the highest top-thought motivation score, and record that
   participant's articulated utterance.

`SimulationConfig` carries scenario, human and mediator clients, mediator
kind, difficulty/mode, thought count, maximum turns, and intervention
threshold. `runner.py` persists a JSON object containing both the full
conversation and its metrics bundle.

### Observed mediator structure

`GenericMediator` uses LLM prompts for two decisions: `decide_when`, then
`speak`. Its `WhenDecision` contains `should_engage`, a reason, and a rating.

`SocialMediator` owns the same decision authority but follows four LLM-driven
steps on every candidate mediator turn:

```text
when: analyze history and produce a rating
  -> generate: produce up to three strategy thoughts
  -> evaluate: score each strategy thought
  -> articulate: render the highest-scoring thought
```

The simulator logs mediator reason, rating, candidate text, and selected text
in the turn metadata. The `_memories` field is included in prompts but is not
mutated by the reviewed mediator methods.

This policy ownership is incompatible with EviCon-Lab's core constraint. In
EviCon-Lab, an LLM may render a previously selected intervention, but it must
not freely decide intervention timing, target, or action category.

### Observed metrics structure

`ConsensusTracker` is an online trajectory processor. For each participant
turn it uses an LLM attitude extractor to update that participant's latest
attitude per topic, carries unmentioned topics forward, invokes an agreement
judge for every party pair/topic, and stores overall and per-topic consensus
after every turn. Mediator turns create snapshots but do not update participant
attitudes.

`metrics/consensus.py` derives:

- Consensus Change: last-window consensus mean minus first-window mean;
- Topic-Level Efficiency: per-topic consensus delta divided by topic-mention
  turns;
- Response Latency: turns from a consensus-drop event to the next mediator
  turn;
- Mediator Effectiveness: post-intervention slope minus pre-intervention
  slope around mediator turns.

`MediatorIntelligenceJudge` supplies a separate LLM-as-judge score that is
aggregated with the trajectory metrics in `runner.py`.

### EviCon-Lab interpretation

The transcript model, deterministic simulation configuration, online tracker
shape, explicit intervention metadata, and post-hoc metric bundle are good
interface patterns. EviCon-Lab must not copy ProMediate's prompt text,
participant implementation, social mediator policy, scenario materials, or
direct Algorithm-1 port. Independently reimplement the required abstraction
with evidence IDs and held-out-probe support. Any future reuse of MIT code
requires retained copyright and license notices plus review of embedded
third-party material.

Relevant observed files: `promediate/models.py`,
`promediate/simulator/conversation.py`, `promediate/agents/generic_mediator.py`,
`promediate/agents/social_mediator.py`, `promediate/metrics/tracker.py`,
`promediate/metrics/consensus.py`, and `promediate/runner.py`.

## 3. Social-Conformity-in-Large-Language-Models

### Observed data and run records

The project uses processed multiple-choice records with item ID, question,
label-to-text options, gold answer, and a selected wrong `distractor`.
`scripts/run_attack.py` saves JSONL records containing the initial prediction,
final attack prediction, confidences, raw model outputs, peer opinions, and
intermediate step outputs. The later experiment runner also records option
log-probabilities for available answer tokens.

### Observed protocol

`experiments/2026-04-29_five_wrong_guidance/run_experiments.py` first gets a
private initial answer, then adds five wrong peer guides directed at the item's
distractor. It implements four variants:

| Protocol | Exposure and commitment behavior |
| --- | --- |
| Exp1 | All five guides appear in one user message; the model answers once. |
| Exp2 | Five guides are separate user messages in one context; only one final answer follows. |
| Exp3 | Guides arrive one at a time; each response is appended to the conversation before the next guide, creating intermediate commitments. |
| Exp4 | All five guides appear together; the model produces five self-iterations, each conditioned on previous answers, and iteration five is final. |

Exp5 is a social-label mechanism check: five wrong recommendations are either
explicitly labelled `Model 1` through `Model 5`, unlabelled, or split three
wrong versus two correct labelled recommendations. The runner writes condition,
support counts, predictions, raw prompt messages, and log-probabilities.

### Observed conformity measures

The paper-facing README defines:

- initial and final accuracy;
- conformity rate (CR), conditional on not initially selecting the target
  distractor, then ending at it;
- harmful conformity rate (HCR), initially correct then ending at the wrong
  peer target;
- beneficial revision rate (BRR), initially wrong then ending correct;
- answer-change rate.

The lower-level `src/metrics.py`, `scripts/compute_all_level_metrics.py`, and
`scripts/run_social_influence_mechanisms.py` currently divide several rates by
all valid rows, including `conformity_rate` and `wrong_conformity_rate`. This
is not the same denominator as the README's conditional CR definition.
EviCon-Lab must predeclare one denominator per metric, preserve numerator and
denominator counts in CSV, and test the implementation against hand-built
fixtures rather than importing either implementation.

### EviCon-Lab interpretation

The baseline-before-exposure structure, independent manipulation of exposure
batching/order/intermediate commitments, raw-message logging, and paired
comparison idea are directly relevant. EviCon-Lab must independently write its
protocol renderer, target-selection procedure, parser, metrics, and statistical
analysis. It must not copy the project's attack prompts, processed datasets,
raw outputs, experiment script, or log-probability implementation without a
separate provenance and rights review, even though the observed software
license is MIT.

Relevant observed files: `agent_conformity_project/src/social_experiment.py`,
`agent_conformity_project/src/metrics.py`,
`agent_conformity_project/scripts/run_attack.py`,
`agent_conformity_project/scripts/compute_all_level_metrics.py`, and
`agent_conformity_project/experiments/2026-04-29_five_wrong_guidance/run_experiments.py`.

## 4. ASVO

### Observed value-state model

ASVO does not use a fixed plain-text persona alone. Its value setup maps social
traits to selected desire dimensions and gives each selected dimension a
numeric 0--10 initial value, description, expected value, and qualitative
state rendering. A profile can separate visible desires from hidden desires.

`init_value_info_social.py` combines:

- a trait-to-desire mapping and selected desire list;
- a random qualitative degree for each trait;
- per-desire initial values and descriptions;
- a social-personality-specific sensitivity (`decrease_map`);
- expected values, with a reversal convention for deficit-like dimensions.

The active configuration selects desires such as superiority, achievement,
confidence, joyfulness, comfort, recognition, spiritual satisfaction, and
control. `hardcoded_value_state.py` maps integer values 0--10 to qualitative
descriptions for prompt context.

### Observed social orientation and update path

`desire_svo_comp.py` maps four social-personality labels to SVO ranges and
expected SVO angles: Altruistic, Prosocial, Individualistic, and Competitive.
`ValueTracker` records per-step desire values, expected values, deviations,
self/other satisfaction, estimated other-agent desires, and SVO values.

On update it observes which named agents appear in an observation, uses LLM
prompts to estimate other desires and satisfaction, calculates a context-aware
SVO quantity, re-runs a six-question allocation questionnaire, and may ask the
LLM whether and how to update expected desire values. Numeric updates are
parsed and clamped to the 0--10 range. `ValueAgent.py` injects these desire
components, tracker, social personality, observations, memory, and goal into a
Concordia agent before its MCTS action component is constructed.

### EviCon-Lab interpretation

The useful ideas are explicit numeric value dimensions, visible/hidden state,
semantic rendering separate from stored numbers, change provenance, and
separating social orientation from task output. EviCon-Lab should not adopt
ASVO's dynamic state update as its causal measure: the update itself invokes
LLM reflection and may blend social observation with unverified inference.
That would confound evidence-driven and social-driven value change.

EviCon-Lab must independently implement a declarative profile schema and a
separate change ledger. Do not copy ASVO prompts, desire component classes,
SVO questionnaire text, hardcoded qualitative strings, Concordia integration,
or simulation configuration without a separate review. If any MIT source were
ever reused, preserve its notices and verify licenses for Concordia and all
other dependencies independently.

Relevant observed files: `examples/ASVO/ASVO_agent/ValueAgent.py`,
`examples/ASVO/value_components/init_value_info_social.py`,
`examples/ASVO/value_components/desire_svo_comp.py`,
`examples/ASVO/value_components/hardcoded_value_state.py`, and
`examples/ASVO/simulation_setup.py`.

## 5. Reuse decision matrix

| Material category | EviCon-Lab decision |
| --- | --- |
| Research question, experimental factor, or metric concept | Reference with a bibliographic citation; independently formulate the implementation and test it. |
| Data-model shape, event ordering, or output-provenance pattern | Reimplement from the documented behavioral contract; do not copy source. |
| MultiAgent-Diversity implementation, data, prompts, or outputs | Do not copy because the audited local directory provides no license. |
| ProMediate code | MIT permits reuse subject to notice, but EviCon-Lab will independently implement because its mediator-control contract differs. Do not copy appendix prompts or restricted scenario material. |
| Social-Conformity code | MIT permits reuse subject to notice, but independently reimplement protocols and metrics to control denominator and provenance. Do not copy prompts, datasets, outputs, or analyses without separate review. |
| ASVO code | MIT permits reuse subject to notice, but independently implement minimal Pydantic state models rather than import Concordia-dependent components or prompt text. |
| Any upstream model output, benchmark result, figure, or reported result | Do not copy into source, tests, fixtures, or documentation as EviCon results. Store only EviCon-generated artifacts. |

## 6. Interfaces EviCon-Lab must implement itself

The following interfaces are required for the research goal and are absent or
semantically incompatible with the audited projects:

| EviCon interface | Responsibility and reason for independent implementation |
| --- | --- |
| `ValueProfile` and `ValueDimension` | Immutable value profile, scale, visibility, source, and version. Avoid ASVO's prompt-derived mutable state as ground truth. |
| `ValueProbe` and `ProbeSet` | Structured exposure prompts and held-out probes with explicit value dimensions and no leakage. |
| `EvidenceCard` and `EvidenceTracker` | Claim, source, reliability, timestamp, visibility, and linkage from a response/update to evidence. No audited project has the required evidence channel. |
| `ExposureItem`, `ExposurePlan`, and `InteractionEvent` | Separate evidence from social messages; record order, sender, recipient, authority treatment, and rendered-prompt hash. |
| `AgentResponse` and `ChangeLedger` | Capture baseline, exposure-stage, and held-out-probe answers plus explicit reasons and cited evidence IDs. |
| `LLMClient` protocol, `FakeLLM`, and cache interface | Permit deterministic tests and cache keys without importing an upstream provider layer. |
| `ExperimentConfig`, assignment, and runner | Encode Independent, Social-only, Evidence-only, Evidence+Social, and intervention arms with seed/model/prompt provenance. |
| `DiversityMetrics` and `RepresentationCoverage` | Independently calculate pairwise and structural diversity plus coverage; retain item-level inputs and missing-data rules. |
| `SocialInfluenceLossEstimator` | Estimate excess representational loss relative to a matched evidence-only counterfactual; this is not supplied by the audited code. |
| `ConformityMonitor` | Evaluate recorded signals with transparent thresholds; it must not use an LLM to select policy. |
| `MediationPolicy`, target selector, and action executor | Programmatically select `no-op`, evidence request, blind reflection, dissent solicitation, adaptive exposure, minority report, or restructure; use an LLM only to render the already selected action. |
| JSONL/CSV writer and analysis interface | Persist model, parameters, seed, prompt, cache key, evidence links, events, metrics, and uncertainty inputs. Never hard-code experimental results. |

## Implementation guardrails

1. Keep evidence visibility and social visibility as separate fields from the
   first event schema onward.
2. Freeze a participant's pre-exposure answer before a social signal is made
   visible, then evaluate held-out probes after the intervention sequence.
3. Randomize or counterbalance message order, labels, and target selection;
   record the realized assignment for every item.
4. Define metric denominators before data collection and record both counts.
5. Use `FakeLLM` and hand-constructed fixtures for protocol and metric tests;
   do not use upstream data or results as test fixtures.
6. Add provenance records before any future external material enters the
   repository, including its license, exact revision, notices, and affected
   files.

